// ==UserScript==
// @name         TMDB - Similar Quick Viewer (packed .bin loader) - Timing 4sf
// @namespace    http://tampermonkey.net/
// @version      1.2
// @description  Load a prepacked dataset.bin (remap+uint16/uintN) from GitHub, cache in IndexedDB, expose a Run button to show similar list for the current TMDB movie/tv id in JSON quickly. Timings reported to 4 significant figures.
// @author       You
// @match        https://www.themoviedb.org/*
// @grant        GM_addStyle
// @grant        GM_xmlhttpRequest
// @connect      raw.githubusercontent.com
// ==/UserScript==

/* global GM_xmlhttpRequest */

(function () {
  'use strict';

  // CONFIG
  const GITHUB_RAW_BIN_URL = 'https://raw.githubusercontent.com/hcgiub001/LB/main/dataset.bin';
  const DB_NAME = 'tmdb_packed_db_v1';
  const STORE_NAME = 'binaries';
  const STORE_KEY = 'dataset.bin';

  // UI/status constants
  const STATUS_IDLE = 'idle';
  const STATUS_LOADING = 'loading';
  const STATUS_LOADED = 'loaded';
  const STATUS_ERROR = 'error';

  // Globals
  let dataset = null; // parsed structures
  let loadState = { state: STATUS_IDLE, fromCache: false, source: null, sizeBytes: 0, error: null };

  // ---------- Small helpers ----------
  function nowMs() { return (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now(); }
  function bytesToHuman(n) { if (n == null) return 'n/a'; const u=['B','KB','MB','GB']; let i=0,v=n; while(v>=1024 && i<u.length-1){v/=1024;i++;} return v.toFixed(2)+' '+u[i]; }
  function to4sfNumber(numSeconds) {
    // Return string rounded to 4 significant figures
    if (!isFinite(numSeconds)) return String(numSeconds);
    // Use toPrecision to get 4 significant figures. Return as string to preserve formatting like 0.05783
    return Number(numSeconds).toPrecision(4).toString();
  }

  // ---------- IndexedDB helpers ----------
  function openDb() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, 1);
      req.onerror = e => reject(e.target.error);
      req.onupgradeneeded = ev => {
        const db = ev.target.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) db.createObjectStore(STORE_NAME);
      };
      req.onsuccess = ev => resolve(ev.target.result);
    });
  }

  function idbGet(key) {
    return openDb().then(db => new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const store = tx.objectStore(STORE_NAME);
      const req = store.get(key);
      req.onsuccess = e => resolve(e.target.result);
      req.onerror = e => reject(e.target.error);
    }));
  }

  function idbPut(key, value) {
    return openDb().then(db => new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      const req = store.put(value, key);
      req.onsuccess = () => resolve();
      req.onerror = e => reject(e.target.error);
    }));
  }

  // Delete helper
  function idbDelete(key) {
    return openDb().then(db => new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      const req = store.delete(key);
      req.onsuccess = () => resolve();
      req.onerror = e => reject(e.target.error);
    }));
  }

  // ---------- Fetch binary (GM_xmlhttpRequest preferred) ----------
  function fetchArrayBuffer(url) {
    return new Promise((resolve, reject) => {
      if (typeof GM_xmlhttpRequest === 'function') {
        GM_xmlhttpRequest({
          method: 'GET',
          url,
          responseType: 'arraybuffer',
          onload: res => {
            if (res.status >= 200 && res.status < 300) resolve(res.response);
            else reject(new Error('HTTP ' + res.status));
          },
          onerror: err => reject(err)
        });
      } else if (typeof fetch === 'function') {
        fetch(url).then(r => {
          if (!r.ok) throw new Error('Network error: ' + r.status);
          return r.arrayBuffer();
        }).then(resolve).catch(reject);
      } else {
        reject(new Error('No fetch available'));
      }
    });
  }

  // ---------- Binary parsing ----------
  const HEADER_SIZE = 28;

  function decodeHeader(dv) {
    const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
    if (magic !== 'SIML') throw new Error('Bad magic: ' + magic);
    const version = dv.getUint8(4);
    const endianness = dv.getUint8(5);
    const flags = dv.getUint16(6, true);
    const R = dv.getUint32(8, true);
    const E = dv.getUint32(12, true);
    const U = dv.getUint32(16, true);
    const lengths_type = dv.getUint8(20);
    const remap_index_width = dv.getUint8(21);
    const header_crc = dv.getUint32(24, true);
    return { magic, version, endianness, flags, R, E, U, lengths_type, remap_index_width, header_crc };
  }

  function createUint32ViewIfAligned(buffer, offset, count) {
    try {
      if ((offset % 4) === 0) return new Uint32Array(buffer, offset, count);
    } catch (e) {}
    // fallback copy
    const dv = new DataView(buffer);
    const out = new Uint32Array(count);
    let p = offset;
    for (let i = 0; i < count; ++i) { out[i] = dv.getUint32(p, true); p += 4; }
    return out;
  }
  function createUint16ViewIfAligned(buffer, offset, count) {
    try {
      if ((offset % 2) === 0) return new Uint16Array(buffer, offset, count);
    } catch (e) {}
    const dv = new DataView(buffer);
    const out = new Uint16Array(count);
    let p = offset;
    for (let i = 0; i < count; ++i) { out[i] = dv.getUint16(p, true); p += 2; }
    return out;
  }

  function readRemapTable(dv, offset, U) {
    const out = new Uint32Array(U);
    let p = offset;
    for (let i = 0; i < U; ++i) { out[i] = dv.getUint32(p, true); p += 4; }
    return out;
  }

  function parseDatasetBuffer(arrayBuffer) {
    const dv = new DataView(arrayBuffer);
    const header = decodeHeader(dv);
    const { R, E, U, lengths_type, remap_index_width } = header;
    let offset = HEADER_SIZE;

    // source_keys
    const sourceKeysByteOffset = offset;
    const sourceKeys = createUint32ViewIfAligned(arrayBuffer, sourceKeysByteOffset, R);
    offset += R * 4;

    // offsets
    const offsetsByteOffset = offset;
    const offsetsArr = createUint32ViewIfAligned(arrayBuffer, offsetsByteOffset, R);
    offset += R * 4;

    // lengths
    const lengthsByteOffset = offset;
    let lengthsArr = null;
    let lengths_array_bytes = 0;
    if (lengths_type === 0) {
      lengths_array_bytes = R * 1;
      try {
        lengthsArr = new Uint8Array(arrayBuffer, lengthsByteOffset, R);
      } catch (e) {
        lengthsArr = new Uint8Array(R);
        let p = lengthsByteOffset;
        for (let i = 0; i < R; ++i) { lengthsArr[i] = dv.getUint8(p); p += 1; }
      }
      offset += R * 1;
    } else {
      lengths_array_bytes = R * 2;
      lengthsArr = createUint16ViewIfAligned(arrayBuffer, lengthsByteOffset, R);
      offset += R * 2;
    }

    // remap table
    const remapTableByteOffset = offset;
    const remapArr = readRemapTable(dv, remapTableByteOffset, U); // copy
    offset += U * 4;

    // values_blob
    const valuesBlobByteOffset = offset;
    const remapWidth = remap_index_width;
    const valuesBlobBytes = E * remapWidth;

    const sizes = {
      header: HEADER_SIZE,
      source_keys: R * 4,
      offsets: R * 4,
      lengths: lengths_array_bytes,
      remap_table: U * 4,
      values_blob: valuesBlobBytes,
      total_ram_estimate: HEADER_SIZE + (R*4) + (R*4) + lengths_array_bytes + (U*4) + valuesBlobBytes
    };

    return {
      header, buffer: arrayBuffer, dv, R, E, U, remapWidth,
      sourceKeys, offsetsArr, lengthsArr, remapArr,
      valuesBlobByteOffset, valuesBlobBytes, sizes
    };
  }

  function readRemapIndexAt(parsed, entryIndex) {
    const bytePos = parsed.valuesBlobByteOffset + entryIndex * parsed.remapWidth;
    const dv = parsed.dv;
    if (parsed.remapWidth === 2) return dv.getUint16(bytePos, true);
    if (parsed.remapWidth === 3) {
      const b0 = dv.getUint8(bytePos), b1 = dv.getUint8(bytePos + 1), b2 = dv.getUint8(bytePos + 2);
      return b0 | (b1 << 8) | (b2 << 16);
    }
    return dv.getUint32(bytePos, true);
  }

  function binarySearchUint32(arr, target) {
    let lo = 0, hi = arr.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >>> 1;
      const v = arr[mid] >>> 0;
      if (v === target) return mid;
      if (v < target) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
  }

  function getSimilarForRow(parsed, rowIndex) {
    const offIndex = parsed.offsetsArr[rowIndex] >>> 0;
    const len = parsed.lengthsArr[rowIndex] >>> 0;
    const out = new Array(len);
    for (let i = 0; i < len; ++i) {
      const remapIdx = readRemapIndexAt(parsed, offIndex + i);
      const packed = parsed.remapArr[remapIdx];
      out[i] = { tmdb_id: (packed >>> 1), type: ((packed & 1) ? 'tvshow' : 'movie') };
    }
    return out;
  }

  // ---------- UI ----------
  GM_addStyle(`
    .tmdb-packer-box {
      position: fixed;
      right: 0;
      top: 50%;
      bottom: 6%;
      width: 33%;
      max-width: 520px;
      min-width: 300px;
      background: rgba(0,0,0,0.92);
      color: #fff;
      border-radius: 10px 0 0 10px;
      padding: 12px;
      z-index: 2147483647;
      font-family: Arial, sans-serif;
      box-shadow: -6px 6px 24px rgba(0,0,0,0.6);
      font-size: 13px;
      overflow: auto;
    }
    .tmdb-packer-header { display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:8px; }
    .tmdb-btn { background: #1db954; color: #fff; border: none; padding:6px 10px; border-radius:8px; cursor:pointer; font-weight:600; }
    .tmdb-small { font-size:12px; color:#cfcfcf; }
    .tmdb-status { margin-left:8px; font-size:12px; color:#9ad; }
    .tmdb-result { max-height: calc(100% - 140px); overflow:auto; background:#0a0a0a; border-radius:8px; padding:8px; margin-top:8px; font-family:monospace; white-space:pre-wrap; color:#e6e6e6; }
    .tmdb-meta { margin-top:6px; color:#bdbdbd; font-size:12px; display:block; }
    .tmdb-row { display:flex; gap:8px; align-items:center; }
    .tmdb-clear { background:#444; color:#fff; padding:4px 8px; border-radius:6px; border:0; cursor:pointer; }
  `);

  function createUI() {
    const box = document.createElement('div');
    box.className = 'tmdb-packer-box';
    box.innerHTML = `
      <div class="tmdb-packer-header">
        <div><strong>Similar Viewer</strong> <span class="tmdb-small">packed.bin</span></div>
        <div>
          <button id="tmdb_run_btn" class="tmdb-btn">Run</button>
        </div>
      </div>
      <div class="tmdb-row"><span class="tmdb-small">Status:</span> <span id="tmdb_status" class="tmdb-status">idle</span></div>
      <div class="tmdb-meta" id="tmdb_meta">Initializing...</div>
      <div style="margin-top:6px;"><button id="tmdb_clear_btn" class="tmdb-clear">Clear Cache</button></div>
      <div class="tmdb-result" id="tmdb_result">Press Run to query similar list for this title.</div>
    `;
    document.body.appendChild(box);

    const runBtn = box.querySelector('#tmdb_run_btn');
    const clearBtn = box.querySelector('#tmdb_clear_btn');
    const statusEl = box.querySelector('#tmdb_status');
    const metaEl = box.querySelector('#tmdb_meta');
    const resultEl = box.querySelector('#tmdb_result');

    function setStatus(s) { statusEl.textContent = s; }
    function setMeta(text) { metaEl.textContent = text; }
    function setResult(text) { resultEl.textContent = text; }

    runBtn.addEventListener('click', async () => {
      setStatus('running');
      const t0 = nowMs();
      try {
        await ensureDatasetLoaded();
        if (!dataset) throw new Error('Dataset unavailable');
        const p = parseTMDBUrl();
        if (!p) { setResult('Could not parse tmdb id/type from URL.'); setStatus('idle'); return; }
        const packedSourceKey = (p.id << 1) | (p.type === 'tv' ? 1 : 0);
        const rowIdx = binarySearchUint32(dataset.sourceKeys, packedSourceKey);
        if (rowIdx === -1) {
          setResult(JSON.stringify({ found: false, reason: 'No similar list for this id in dataset' }, null, 2));
          setMeta(`Queried id=${p.id} (${p.type}) - not found`);
          setStatus('idle');
          return;
        }
        const before = nowMs();
        const list = getSimilarForRow(dataset, rowIdx);
        const after = nowMs();
        const totalMs = after - t0;
        const queryMs = after - before;

        // convert to seconds and format to 4 significant figures
        const totalSec = totalMs / 1000.0;
        const querySec = queryMs / 1000.0;
        const total_s_4sf = to4sfNumber(totalSec);
        const query_s_4sf = to4sfNumber(querySec);

        const sizes = dataset.sizes;
        const info = {
          source_file: loadState.fromCache ? 'IndexedDB cache' : (loadState.source || 'network'),
          file_byteLength: loadState.sizeBytes,
          parsed_sizes: sizes,
          runtimes: {
            total_ms: Number(totalMs.toFixed(3)),
            query_ms: Number(queryMs.toFixed(3)),
            total_s: totalSec,
            query_s: querySec,
            total_s_4sf: total_s_4sf,
            query_s_4sf: query_s_4sf
          },
          result_count: list.length,
          result: list
        };
        setResult(JSON.stringify(info, null, 2));
        setMeta(`Found ${list.length} entries. Query ${query_s_4sf}s (4sf), total ${total_s_4sf}s (4sf). File: ${bytesToHuman(loadState.sizeBytes)} (${loadState.fromCache ? 'cache' : 'network'})`);
        setStatus('idle');
      } catch (err) {
        setResult('Error: ' + (err && err.message ? err.message : String(err)));
        setStatus('error');
      }
    });

    clearBtn.addEventListener('click', async () => {
      try {
        await idbDelete(STORE_KEY);
        dataset = null;
        loadState = { state: STATUS_IDLE, fromCache: false, source: null, sizeBytes: 0, error: null };
        setMeta('Cache cleared.');
        setResult('Cache cleared. Reload the page to fetch again.');
      } catch (e) {
        setResult('Clear cache error: ' + e);
      }
    });

    return { setStatus, setMeta, setResult };
  }

  function parseTMDBUrl() {
    const parts = location.pathname.split('/').filter(Boolean);
    if (parts.length >= 2) {
      const type = parts[0];
      const id = parseInt(parts[1], 10);
      if (!isNaN(id) && (type === 'movie' || type === 'tv')) return { id, type };
    }
    return null;
  }

  // ---------- Loading orchestration ----------
  async function ensureDatasetLoaded() {
    if (dataset) return dataset;
    if (loadState.state === STATUS_LOADING) {
      // wait
      while (loadState.state === STATUS_LOADING) await new Promise(r => setTimeout(r, 120));
      if (loadState.state === STATUS_LOADED) return dataset;
      throw new Error(loadState.error || 'Failed to load dataset');
    }

    loadState.state = STATUS_LOADING;
    try {
      // look for cached arrayBuffer directly
      const cached = await idbGet(STORE_KEY);
      let arrBuf = null;
      if (cached) {
        // Handle a few storage shapes: ArrayBuffer, Blob, Uint8Array, object containing arrayBuffer
        if (cached instanceof ArrayBuffer) arrBuf = cached;
        else if (cached && cached.arrayBuffer && cached.arrayBuffer instanceof ArrayBuffer) {
          arrBuf = cached.arrayBuffer;
        } else if (cached instanceof Blob) {
          arrBuf = await cached.arrayBuffer();
        } else if (cached && cached.buffer && cached.buffer instanceof ArrayBuffer) {
          arrBuf = cached.buffer;
        } else if (cached instanceof Uint8Array) {
          arrBuf = cached.buffer.slice(cached.byteOffset, cached.byteOffset + cached.byteLength);
        } else if (cached && cached.data && cached.data instanceof ArrayBuffer) {
          arrBuf = cached.data;
        } else if (cached && cached instanceof Object && cached.arrayBuffer && typeof cached.arrayBuffer === 'object') {
          // some browsers store structured clone with { arrayBuffer: ArrayBuffer }
          arrBuf = cached.arrayBuffer;
        }
      }

      if (arrBuf) {
        loadState.fromCache = true;
        loadState.sizeBytes = arrBuf.byteLength;
        loadState.source = 'idb';
        dataset = parseDatasetBuffer(arrBuf);
        loadState.state = STATUS_LOADED;
        return dataset;
      }

      // no cache -> fetch
      loadState.fromCache = false;
      const buffer = await fetchArrayBuffer(GITHUB_RAW_BIN_URL);
      if (!buffer) throw new Error('Empty network response');
      // store ArrayBuffer directly (structured clone allowed)
      try { await idbPut(STORE_KEY, buffer); } catch (e) { console.warn('IDB put failed:', e); }
      loadState.sizeBytes = buffer.byteLength;
      loadState.source = 'network';
      dataset = parseDatasetBuffer(buffer);
      loadState.state = STATUS_LOADED;
      return dataset;
    } catch (err) {
      loadState.state = STATUS_ERROR;
      loadState.error = err.message || String(err);
      throw err;
    }
  }

  // ---------- Background load ----------
  (async function backgroundLoad() {
    try {
      await ensureDatasetLoaded();
      try { document.getElementById('tmdb_meta').textContent = `Dataset ready. File: ${bytesToHuman(loadState.sizeBytes)} (${loadState.fromCache ? 'cache' : 'network'})`; } catch (e) {}
    } catch (e) {
      console.warn('Background dataset load failed:', e);
      try { document.getElementById('tmdb_meta').textContent = `Dataset load error: ${e.message || e}`; } catch (e) {}
    }
  })();

  // ---------- Build UI ----------
  const ui = createUI();

  // status updater
  (async function statusLoop() {
    while (true) {
      try {
        const st = loadState.state === STATUS_LOADING ? 'loading dataset...' : (loadState.state === STATUS_LOADED ? 'dataset ready' : (loadState.state === STATUS_ERROR ? 'error loading dataset' : 'idle'));
        const el = document.getElementById('tmdb_status');
        if (el) el.textContent = st;
      } catch (e) {}
      await new Promise(r => setTimeout(r, 700));
    }
  })();

})();
