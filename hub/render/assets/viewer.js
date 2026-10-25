/* Cocktail AV inspector: one shared WebAudio clock drives every stem and the picture.
 * No network dependencies beyond files of this bundle (served from localhost). */
(function () {
  'use strict';
  const $ = (s) => document.querySelector(s);
  const SCALE = 3;
  const COLORS = { active: [255, 214, 10], possible: [80, 220, 255] };

  // ---- RLE (row-major, background run first; mirrors vision/mask_store.py) ----
  function decodeRle(r) {
    const n = r.h * r.w, out = new Uint8Array(n);
    const fgOdd = r.foreground !== false;
    let pos = 0;
    for (let i = 0; i < r.counts.length; i++) {
      const c = r.counts[i];
      const on = (i % 2 === 1) === fgOdd;
      if (on) out.fill(1, pos, pos + c);
      pos += c;
    }
    return out;
  }
  (typeof window !== 'undefined' ? window : globalThis).__decodeRle = decodeRle;
  if (typeof document === 'undefined' || !document.getElementById('stage')) return;

  const fetchJson = async (p) => { const r = await fetch(p); if (!r.ok) throw new Error(p + ': ' + r.status); return r.json(); };
  const fetchLines = async (p) => (await (async () => { const r = await fetch(p); if (!r.ok) throw new Error(p + ': ' + r.status); return r.text(); })()).split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l));

  const S = {
    ctx: null, buffers: {}, gains: {}, nodes: [], playing: false, offset: 0, t0: 0, duration: 0,
    mode: 'stems', solo: {}, mute: {}, residual: false, selected: null, frame: 0,
    manifest: null, frames: null, masks: {}, tstate: {}, act: {}, images: [], useVideo: false,
  };

  window.__avState = S;                       // read-only inspection handle for tests

  // ---------------------------------------------------------------- audio clock
  function position() { return S.playing ? Math.max(S.offset, S.offset + (S.ctx.currentTime - S.t0)) : S.offset; }
  function ensureCtx() {
    if (!S.ctx) {
      S.ctx = new (window.AudioContext || window.webkitAudioContext)();
      const make = (key) => { const g = S.ctx.createGain(); g.connect(S.ctx.destination); S.gains[key] = g; };
      make('mixture'); make('residual'); Object.keys(S.act).forEach((id) => make('stem:' + id));
    }
    return S.ctx;
  }
  function stopNodes() { S.nodes.forEach((n) => { try { n.stop(); } catch (e) { /* already stopped */ } n.disconnect(); }); S.nodes = []; }
  function startNodes() {
    stopNodes();
    const when = S.ctx.currentTime + 0.06;       // one shared start time for every buffer
    S.t0 = when;
    Object.keys(S.buffers).forEach((key) => {
      const bs = S.ctx.createBufferSource();
      bs.buffer = S.buffers[key];
      bs.connect(S.gains[key]);
      bs.start(when, Math.min(S.offset, Math.max(S.duration - 0.001, 0)));
      S.nodes.push(bs);
    });
  }
  async function play() {
    if (!S.audioReady) { $('#banner').textContent = 'Audio is not ready yet (or unavailable in this browser).'; return; }
    ensureCtx();
    await S.ctx.resume();
    if (S.offset >= S.duration - 0.05) S.offset = 0;
    applyMix(true);
    startNodes();
    S.playing = true; $('#play').textContent = 'Pause';
    if (S.useVideo) { const vd = $('#video'); vd.currentTime = S.offset; vd.play().catch(() => {}); }
  }
  function pause() {
    if (!S.playing) return;
    S.offset = Math.min(position(), S.duration);
    stopNodes(); S.playing = false; $('#play').textContent = 'Play';
    if (S.useVideo) $('#video').pause();
  }
  function seek(t) {
    S.offset = Math.min(Math.max(t, 0), S.duration);
    if (S.playing) startNodes();                 // recreate buffer sources, reschedule every stem together
    if (S.useVideo) $('#video').currentTime = S.offset;
  }
  function applyMix(immediate) {
    if (!S.ctx) return;
    const anySolo = Object.values(S.solo).some(Boolean);
    const set = (key, v) => { const g = S.gains[key]; if (g) { if (immediate) g.gain.value = v; else g.gain.setTargetAtTime(v, S.ctx.currentTime, 0.01); } };
    Object.keys(S.act).forEach((id) => {
      let v = 0;
      if (S.mode === 'stems') v = S.mute[id] ? 0 : (anySolo ? (S.solo[id] ? 1 : 0) : 1);
      set('stem:' + id, v);
    });
    set('mixture', S.mode === 'mixture' ? 1 : 0);
    set('residual', S.residual ? 1 : 0);
  }

  // ---------------------------------------------------------------- data
  function frameAt(tUs) {
    const p = S.frames.pts_us; let lo = 0, hi = p.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (p[m] <= tUs) lo = m + 1; else hi = m; }
    return Math.max(0, lo - 1);
  }
  function assocFor(id) { return S.manifest.associations.find((a) => a.source_id === id); }
  function sourceForTrack(tid) { const a = S.manifest.associations.find((x) => x.track_id === tid); return a ? a.source_id : null; }

  function fmt(v) { return v === null || v === undefined ? 'unavailable' : (typeof v === 'number' ? v.toFixed(2) : String(v)); }
  function renderInspector() {
    const el = $('#inspector'); el.textContent = '';
    if (!S.selected) { el.textContent = 'Select a source or an object.'; el.className = 'inspector muted'; return; }
    el.className = 'inspector';
    const a = assocFor(S.selected), src = S.manifest.sources.find((s) => s.source_id === S.selected);
    const h = document.createElement('div');
    h.textContent = `${src.label} - status ${a.status}` + (a.track_id ? `, track ${a.track_id}` : ', no visual track (unattributed sound)');
    el.appendChild(h);
    const kv = [['score', fmt(a.score)], ['margin over runner-up', fmt(a.runner_up_margin)], ['causal lag (ms)', fmt(a.causal_lag_ms)],
      ['visibility', a.visibility]];
    if (a.spatial) {
      kv.push(['aliasing multimodal flag', String(a.spatial.aliasing_multimodal_flag)], ['onset flux weight', fmt(a.spatial.onset_flux_weight)],
        ['parallax sigma (rad)', fmt(a.spatial.parallax_sigma_rad)], ['bandwidth (Hz)', fmt(a.spatial.bandwidth_hz)],
        ['delay peaks (us)', a.spatial.peaks_s.map((x) => (x * 1e6).toFixed(0)).join(', ') || 'none']);
    }
    const t1 = document.createElement('table');
    kv.forEach(([k, v]) => { const tr = t1.insertRow(); const th = document.createElement('th'); th.textContent = k; tr.appendChild(th); tr.insertCell().textContent = v; });
    el.appendChild(t1);
    const t2 = document.createElement('table');
    const hr = t2.insertRow(); ['evidence', 'value', 'quality', 'weight', 'note'].forEach((x) => { const th = document.createElement('th'); th.textContent = x; hr.appendChild(th); });
    a.components.forEach((c) => { const tr = t2.insertRow(); [c.name, fmt(c.value), fmt(c.quality), fmt(c.weight), c.note].forEach((x) => { tr.insertCell().textContent = x; }); });
    el.appendChild(t2);
    if (a.learning_eligibility && a.learning_eligibility.length) {
      const p = document.createElement('div'); p.className = 'small muted';
      p.textContent = 'voice-profile learning: ' + a.learning_eligibility.map((e) => `${(e.start_us / 1e6).toFixed(1)}-${(e.end_us / 1e6).toFixed(1)}s ${e.eligible ? 'eligible' : 'BLOCKED'} (${e.reason})`).join('; ');
      el.appendChild(p);
    }
  }

  function renderSources() {
    const ul = $('#sources'); ul.textContent = '';
    S.manifest.sources.forEach((s) => {
      const a = assocFor(s.source_id), li = document.createElement('li');
      li.className = 'row' + (S.selected === s.source_id ? ' selected' : ''); li.tabIndex = 0; li.dataset.id = s.source_id;
      const nm = document.createElement('div'); nm.className = 'name'; nm.textContent = s.label;
      const chip = document.createElement('span'); chip.className = 'chip ' + a.status; chip.textContent = a.status === 'unmatched' ? 'unattributed' : (a.status === 'ambiguous' ? 'possible match' : 'matched');
      nm.appendChild(chip);
      if (a.track_id) { const t = document.createElement('span'); t.className = 'muted small'; t.textContent = ' ' + a.track_id; nm.appendChild(t); }
      const b = document.createElement('div'); b.className = 'btns';
      [['Solo', 'solo'], ['Mute', 'mute']].forEach(([lbl, key]) => {
        const bt = document.createElement('button'); bt.type = 'button'; bt.textContent = lbl; bt.setAttribute('aria-pressed', String(!!S[key][s.source_id]));
        bt.addEventListener('click', (e) => { e.stopPropagation(); S[key][s.source_id] = !S[key][s.source_id]; if (S.mode !== 'stems' && key === 'solo') setMode('stems'); applyMix(); renderSources(); });
        b.appendChild(bt);
      });
      const strip = document.createElement('canvas'); strip.className = 'strip'; strip.width = 400; strip.height = 14;
      drawStrip(strip, S.act[s.source_id]);
      strip.addEventListener('click', (e) => { e.stopPropagation(); const r = strip.getBoundingClientRect(); seek(((e.clientX - r.left) / r.width) * S.duration); });
      li.append(nm, b, strip);
      li.addEventListener('click', () => selectSource(s.source_id, true));
      li.setAttribute('role', 'button'); li.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') selectSource(s.source_id, true); });
      ul.appendChild(li);
    });
  }
  function drawStrip(c, act) {
    const g = c.getContext('2d'); g.clearRect(0, 0, c.width, c.height);
    g.fillStyle = getComputedStyle(document.body).color; g.globalAlpha = 0.65;
    (act.segments_us || []).forEach(([a, b]) => { g.fillRect((a / 1e6 / S.duration) * c.width, 2, Math.max(2, ((b - a) / 1e6 / S.duration) * c.width), c.height - 4); });
    g.globalAlpha = 1;
  }
  function setMode(m) { S.mode = m; document.querySelectorAll('input[name=mode]').forEach((r) => { r.checked = r.value === m; }); applyMix(); }
  function selectSource(id, exclusive) {
    S.selected = id;
    if (exclusive) { S.solo = { [id]: true }; S.mute = {}; setMode('stems'); }
    renderSources(); renderInspector();
  }

  // ---------------------------------------------------------------- drawing
  const stage = $('#stage'), g2 = stage.getContext('2d');
  const off = document.createElement('canvas'), og = off.getContext('2d');
  function maskAt(tid, f) { const r = S.masks[tid][f]; return r && r.rle ? r.rle : null; }
  function draw() {
    const t = position(), f = frameAt(t * 1e6); S.frame = f;
    const W = S.frames.width, H = S.frames.height;
    g2.imageSmoothingEnabled = false;
    if (S.useVideo) g2.drawImage($('#video'), 0, 0, stage.width, stage.height);
    else if (S.images[f] && S.images[f].complete) g2.drawImage(S.images[f], 0, 0, stage.width, stage.height);
    off.width = W; off.height = H;
    const img = og.createImageData(W, H), d = img.data;
    const labels = [];
    S.manifest.tracks.forEach((tr) => {
      const state = S.tstate[tr.track_id].state[f];
      if (!state || state === 'idle') return;
      const rle = maskAt(tr.track_id, f); if (!rle) return;
      const m = decodeRle(rle), col = COLORS[state === 'possible' ? 'possible' : 'active'];
      let minx = W, miny = H;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const i = y * W + x; if (!m[i]) continue;
        const edge = !(x > 0 && m[i - 1] && x < W - 1 && m[i + 1] && y > 0 && m[i - W] && y < H - 1 && m[i + W]);
        const dash = state === 'possible' && ((x >> 2) + (y >> 2)) % 2 === 1;
        d[i * 4] = col[0]; d[i * 4 + 1] = col[1]; d[i * 4 + 2] = col[2];
        d[i * 4 + 3] = edge && !dash ? 255 : (state === 'possible' ? 70 : 110);
        if (x < minx) minx = x; if (y < miny) miny = y;
      }
      const word = state === 'speaking' ? 'speaking' : (state === 'sounding' ? 'sounding' : 'possible match');
      labels.push([minx, miny, `${tr.track_id} ${word}`, col]);
    });
    og.putImageData(img, 0, 0); g2.drawImage(off, 0, 0, stage.width, stage.height);
    g2.font = '600 15px system-ui, sans-serif'; g2.textBaseline = 'bottom';
    labels.forEach(([x, y, txt, col]) => { const w = g2.measureText(txt).width + 8; g2.fillStyle = 'rgba(10,10,16,0.8)'; g2.fillRect(x * SCALE, Math.max(y * SCALE - 20, 0), w, 20); g2.fillStyle = `rgb(${col})`; g2.fillText(txt, x * SCALE + 4, Math.max(y * SCALE - 3, 17)); });
    const un = Object.values(S.act).find((a) => !a.attributed && a.gate_active[f]);
    if (un) { g2.fillStyle = 'rgba(20,10,24,0.85)'; g2.fillRect(0, stage.height - 30, stage.width, 30); g2.fillStyle = 'rgb(255,120,220)'; g2.fillText('UNATTRIBUTED SOUND: ' + un.label, 10, stage.height - 8); }
    $('#time').textContent = `${Math.min(t, S.duration).toFixed(2)} / ${S.duration.toFixed(2)} s`;
    if (!seekDrag) $('#seek').value = String(Math.round((Math.min(t, S.duration) / S.duration) * 1000));
    if (S.playing && t >= S.duration) pause();
    if (S.useVideo) { const v = $('#video'); if (!S.playing || Math.abs(v.currentTime - t) > 0.045) v.currentTime = Math.min(t, S.duration); }   // within about one frame of the audio clock
    requestAnimationFrame(draw);
  }

  // ---------------------------------------------------------------- interaction
  let seekDrag = false;
  stage.addEventListener('click', (e) => {
    const r = stage.getBoundingClientRect();
    const x = Math.floor(((e.clientX - r.left) / r.width) * S.frames.width), y = Math.floor(((e.clientY - r.top) / r.height) * S.frames.height);
    const hits = S.manifest.tracks.filter((tr) => { const rle = maskAt(tr.track_id, S.frame); return rle && decodeRle(rle)[y * rle.w + x]; }).map((tr) => tr.track_id);
    const box = $('#candidates'); box.textContent = '';
    if (hits.length === 0) { box.hidden = true; return; }
    const choose = (tid) => { box.hidden = true; const sid = sourceForTrack(tid); if (sid) selectSource(sid, true); else { S.selected = null; const el = $('#inspector'); el.className = 'inspector'; el.textContent = `${tid}: no sound source is associated with this object (silent or unmatched).`; renderSources(); } };
    if (hits.length === 1) { choose(hits[0]); return; }
    const lead = document.createElement('span'); lead.textContent = 'Overlapping masks - choose one:'; box.appendChild(lead);
    hits.forEach((tid) => { const b = document.createElement('button'); b.type = 'button'; b.textContent = tid; b.addEventListener('click', () => choose(tid)); box.appendChild(b); });
    box.hidden = false;
  });
  $('#play').addEventListener('click', () => (S.playing ? pause() : play()));
  const seekEl = $('#seek');
  seekEl.addEventListener('input', () => { seekDrag = true; S.offset = (seekEl.value / 1000) * S.duration; $('#time').textContent = `${S.offset.toFixed(2)} / ${S.duration.toFixed(2)} s`; });
  seekEl.addEventListener('change', () => { seekDrag = false; seek((seekEl.value / 1000) * S.duration); });
  document.querySelectorAll('input[name=mode]').forEach((r) => r.addEventListener('change', () => setMode(r.value)));
  $('#residual').addEventListener('change', (e) => { S.residual = e.target.checked; applyMix(); });
  $('#reset').addEventListener('click', () => { S.solo = {}; S.mute = {}; S.residual = false; $('#residual').checked = false; setMode('stems'); renderSources(); });
  document.addEventListener('keydown', (e) => { if (e.code === 'Space' && e.target.tagName !== 'BUTTON' && e.target.tagName !== 'INPUT') { e.preventDefault(); S.playing ? pause() : play(); } });

  // ---------------------------------------------------------------- boot
  (async function boot() {
    try {
      S.manifest = await fetchJson('manifest.json');
      S.frames = await fetchJson('frames.json');
      const A = S.manifest.assets;
      $('#tier').textContent = 'evidence: ' + S.manifest.evidence_tier;
      $('#banner').textContent = S.manifest.evidence_note;
      await Promise.all(S.manifest.sources.map(async (s) => { S.act[s.source_id] = await fetchJson(A['activity:' + s.source_id]); }));
      await Promise.all(S.manifest.tracks.map(async (t) => { S.masks[t.track_id] = await fetchLines(A['masks:' + t.track_id]); S.tstate[t.track_id] = await fetchJson(A['track_state:' + t.track_id]); }));
      S.duration = Math.max(...Object.values(S.act).map((x) => (x.segments_us.length ? x.segments_us[x.segments_us.length - 1][1] / 1e6 : 0)), S.frames.pts_us[S.frames.pts_us.length - 1] / 1e6);
      S.useVideo = !!A.source_video;
      if (S.useVideo) { const v = $('#video'); v.src = A.source_video; v.muted = true; }
      else S.images = S.frames.files.map((p) => { const im = new Image(); im.src = p; return im; });
      renderSources(); renderInspector();
      requestAnimationFrame(draw);                 // picture and evidence work even if audio cannot start
      try {
        ensureCtx();
        const load = async (key, path) => { const ab = await (await fetch(path)).arrayBuffer(); S.buffers[key] = await S.ctx.decodeAudioData(ab); };
        await Promise.all([load('mixture', A.mixture), load('residual', A.residual), ...S.manifest.sources.map((s) => load('stem:' + s.source_id, A['stem:' + s.source_id]))]);
        S.duration = Math.max(...Object.values(S.buffers).map((b) => b.duration));
        S.audioReady = true; applyMix(true); renderSources();
      } catch (err) { $('#banner').textContent = 'Audio unavailable: ' + err.message; }
      try { const m = await fetchJson('metrics.json'); renderMetrics(m); } catch (e) { /* metrics are optional */ }
    } catch (err) { $('#banner').textContent = 'Failed to load bundle: ' + err.message + ' (serve the folder with: python -m av_scene serve <bundle>)'; }
  })();

  function renderMetrics(m) {
    const el = $('#metrics'); el.textContent = ''; el.className = 'small';
    const rows = [];
    const walk = (o, p) => Object.entries(o).forEach(([k, v]) => { if (v && typeof v === 'object' && !Array.isArray(v)) walk(v, p + k + '.'); else if (typeof v !== 'object' && p.split('.').length <= 3) rows.push([p + k, String(typeof v === 'number' ? +v.toFixed(3) : v)]); });
    walk(m, '');
    const t = document.createElement('table');
    rows.slice(0, 40).forEach(([k, v]) => { const tr = t.insertRow(); tr.insertCell().textContent = k; tr.insertCell().textContent = v; });
    el.appendChild(t);
  }
})();
