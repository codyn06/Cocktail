/* Cocktail AV inspector: one shared WebAudio clock drives every stem and the picture.
 * No network dependencies beyond files of this bundle (served from localhost). */
(function () {
  'use strict';
  const $ = (s) => document.querySelector(s);
  // ---- video sync policy (pure; audio clock is master). drift = video.currentTime - audioTime (s).
  // |drift| >= 0.150 -> hard seek, rate 1; 0.015 <= |drift| < 0.150 while playing -> rate = clamp(1 - 0.5*drift, 0.9, 1.1);
  // |drift| < 0.015 -> rate 1; paused -> seek only when |drift| > 0.045.
  function videoSyncAction(drift, playing) {
    const a = Math.abs(drift);
    if (!playing) return { seek: a > 0.045, rate: 1 };
    if (a >= 0.150) return { seek: true, rate: 1 };
    if (a < 0.015) return { seek: false, rate: 1 };
    return { seek: false, rate: Math.min(1.1, Math.max(0.9, 1 - drift * 0.5)) };
  }
  // ---- FrameTransform: {version, frame_w, frame_h, mask_w, mask_h, stage_w, stage_h, sx, sy}; sx/sy = mask->frame scale.
  function makeFrameTransform(manifest, frames, stageW, stageH) {
    const ft = (manifest && manifest.capabilities && manifest.capabilities.frame_transform) || (frames && frames.transform) || {};
    const fw = ft.frame_w || (frames && frames.width) || 320, fh = ft.frame_h || (frames && frames.height) || 240;
    const mw = ft.mask_w || fw, mh = ft.mask_h || fh;
    return { version: 1, frame_w: fw, frame_h: fh, mask_w: mw, mask_h: mh, stage_w: stageW, stage_h: stageH, sx: fw / mw, sy: fh / mh };
  }
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

  // =============================================================== long-run (Phase 4) pure helpers
  // Active only when manifest.long_run exists. Everything here is DOM-free so node can test it.
  const LONG = { behind_s: 4, ahead_s: 10, max_buffers: 24, max_pages: 3, tl_pages: 8, lead_s: 0.06 };

  // Segments of enabled sources that intersect [t - behind, t + ahead]. Spans with no segment are silent (zero-fill):
  // they simply produce no item. `enabled` (optional) = set/array/object of source ids worth buffering.
  function segmentsToKeep(index, t, behind, ahead, enabled) {
    const lo = t - behind, hi = t + ahead, out = [];
    const want = enabled ? (Array.isArray(enabled) ? new Set(enabled) : (enabled instanceof Set ? enabled : new Set(Object.keys(enabled).filter((k) => enabled[k])))) : null;
    Object.values(index.sources || {}).forEach((src) => {
      if (want && !want.has(src.source_id)) return;
      const segs = src.segments || [], rate = src.rate_hz;
      let a = 0, b = segs.length;                       // first segment whose end > lo (segments are sorted, non-overlapping)
      while (a < b) { const m = (a + b) >> 1; if ((segs[m].start_sample + segs[m].n_samples) / rate > lo) b = m; else a = m + 1; }
      for (let i = a; i < segs.length; i++) {
        const s = segs[i];
        if (s.start_sample / rate >= hi) break;
        out.push({ key: src.source_id + '|' + s.start_sample, source_id: src.source_id, path: s.path, start_sample: s.start_sample,
          n_samples: s.n_samples, rate: rate, t0: s.start_sample / rate, t1: (s.start_sample + s.n_samples) / rate });
      }
    });
    return out;
  }

  // Place segments on ONE clock. anchor = {anchorCtx: ctx time at which media time anchorMedia plays, nowCtx}.
  // when/offset use integer sample arithmetic so segment i's end and segment i+1's start are the same float (gapless).
  function planSchedule(items, anchor) {
    const lower = Math.max(anchor.nowCtx, anchor.anchorCtx), plan = [];
    items.forEach((it) => {
      const A = anchor.anchorMedia * it.rate;
      const whenRaw = anchor.anchorCtx + (it.start_sample - A) / it.rate;
      const endWhen = anchor.anchorCtx + (it.start_sample + it.n_samples - A) / it.rate;
      if (endWhen <= lower) return;                     // entirely in the past on the shared clock
      let when = whenRaw, offset = 0;
      if (whenRaw < lower) { when = lower; offset = (lower - anchor.anchorCtx) + (A - it.start_sample) / it.rate; }
      const duration = it.n_samples / it.rate - offset;
      if (duration <= 1e-6) return;
      plan.push({ key: it.key, source_id: it.source_id, when: when, offset: offset, duration: duration, endWhen: endWhen });
    });
    return plan;
  }

  // Start every planned segment whose decoded buffer is present and that is not already active (never twice).
  function scheduleSegments(ctx, plan, buffers, gainFor, active) {
    const started = [];
    plan.forEach((p) => {
      if (active.has(p.key)) return;
      const buf = buffers.get ? buffers.get(p.key) : buffers[p.key];
      if (!buf) return;
      const node = ctx.createBufferSource();
      node.buffer = buf;
      node.connect(gainFor(p.source_id));
      node.onended = () => { const cur = active.get(p.key); if (cur && cur.node === node) active.delete(p.key); };
      node.start(p.when, p.offset, p.duration);
      active.set(p.key, { node: node, plan: p });
      started.push(p.key);
    });
    return started;
  }
  function stopAllNodes(active) {
    active.forEach((v) => { try { v.node.stop(); } catch (e) { /* not started or already stopped */ } try { v.node.disconnect(); } catch (e) { /* noop */ } });
    active.clear();
  }

  // Count-capped decoded buffer store. retain(keep, tNow) evicts everything outside the keep set, then enforces the cap by
  // dropping the buffer whose segment is farthest from tNow. put() never lets size exceed the cap.
  class BufferCache {
    constructor(cap, meta) { this.cap = cap; this.map = new Map(); this.meta = meta || (() => 0); }
    has(k) { return this.map.has(k); }
    get(k) { return this.map.get(k); }
    get size() { return this.map.size; }
    retain(keepKeys, tNow) {
      for (const k of Array.from(this.map.keys())) if (!keepKeys.has(k)) this.map.delete(k);
      this._trim(tNow);
    }
    put(k, buf, tNow) { this.map.set(k, buf); this._trim(tNow); return this.map.has(k); }
    _trim(tNow) {
      while (this.map.size > this.cap) {
        let worst = null, wd = -1;
        for (const k of this.map.keys()) { const d = Math.abs(this.meta(k) - tNow); if (d > wd) { wd = d; worst = k; } }
        this.map.delete(worst);
      }
    }
  }

  // LRU page store (frames / masks / timeline). cap pages; touch() marks use; put() evicts least recently used.
  class PageCache {
    constructor(cap) { this.cap = cap; this.map = new Map(); }
    has(k) { return this.map.has(k); }
    get(k) { if (!this.map.has(k)) return undefined; const v = this.map.get(k); this.map.delete(k); this.map.set(k, v); return v; }
    put(k, v) { this.map.delete(k); this.map.set(k, v); while (this.map.size > this.cap) this.map.delete(this.map.keys().next().value); }
    get size() { return this.map.size; }
    keys() { return Array.from(this.map.keys()); }
    values() { return Array.from(this.map.values()); }
  }

  // Which pages cover [t - behind, t + ahead], nearest first, at most `max`. pages: [{path,start_us,end_us}].
  function pagesToLoad(pages, tUs, opts) {
    const o = Object.assign({ behindUs: 0, aheadUs: 0, max: LONG.max_pages }, opts || {});
    const lo = tUs - o.behindUs, hi = tUs + o.aheadUs;
    return pages.map((p, i) => ({ p, i })).filter(({ p }) => p.end_us > lo && p.start_us <= hi && !(p.n === 0))
      .map(({ p, i }) => ({ p, i, d: (tUs >= p.start_us && tUs < p.end_us) ? -1 : Math.min(Math.abs(p.start_us - tUs), Math.abs(p.end_us - tUs)) }))
      .sort((a, b) => a.d - b.d || a.i - b.i).slice(0, o.max).map((x) => x.p.path);
  }

  // Four INDEPENDENT facts per timeline segment; never collapsed into one enum badge.
  const TONES = {
    association_status: { matched: 'ok', ambiguous: 'warn', unmatched: 'off', rejected: 'bad' },
    visibility: { visible: 'ok', occluded: 'warn', outside_view: 'off', unknown: 'muted' },
    stem_availability: { ok: 'ok', degraded: 'warn', unavailable: 'bad', not_attempted: 'muted' },
    identity_confidence: { resolved: 'ok', ambiguous: 'warn', new: 'off', none: 'muted' },
  };
  const BADGE_LABELS = { association_status: 'Association', visibility: 'Visibility', stem_availability: 'Stem', identity_confidence: 'Identity' };
  function badgesFor(seg) {
    const badges = ['association_status', 'visibility', 'stem_availability', 'identity_confidence'].map((f) => ({
      field: f, label: BADGE_LABELS[f], value: seg[f], tone: (TONES[f] || {})[seg[f]] || 'muted', text: BADGE_LABELS[f] + ': ' + String(seg[f]).replace(/_/g, ' '),
    }));
    return { badges: badges, refusal: seg.refusal_reason || null,
      summary: badges.map((b) => b.text).join('; ') + (seg.refusal_reason ? '; reason: ' + seg.refusal_reason : '') };
  }

  // ---- listen controls
  function parseDiagTrack(id) { return id.indexOf('tse:') === 0 ? id.slice(4) : null; }
  function groupFor(e) {
    if (e.role === 'diagnostic') return 'diagnostic';
    if (e.kind === 'mixture' || e.role === 'mixture') return 'mixture';
    if (e.kind === 'residual_proxy') return 'residual_proxy';
    if (e.kind === 'event') return 'event';
    if (e.kind === 'outside_view') return 'outside_view';
    return 'speech';
  }
  const GROUPS = [['speech', 'Matched speech'], ['event', 'Non-speech events'], ['outside_view', 'Outside-view voices'],
    ['residual_proxy', 'Residual playback proxy'], ['mixture', 'Original mixture'], ['diagnostic', 'Diagnostic stems (solo views, off by default)']];
  function defaultEnabled(e) { const g = groupFor(e); return g === 'speech' || g === 'event' || g === 'outside_view'; }
  function initialControlState(index, trackMap, labels) {
    const enabled = {}, layerOnTop = {};
    Object.values(index.sources || {}).forEach((e) => { enabled[e.source_id] = defaultEnabled(e); layerOnTop[e.source_id] = false; });
    return { enabled: enabled, layerOnTop: layerOnTop, trackMap: trackMap || {}, labels: labels || {}, selected: null };
  }
  // DUPLICATE-PLAYBACK RULE: an enabled diagnostic stem `tse:T` is a SOLO view of track T. It mutes the default-role stems
  // matched to T unless "layer on top" is ticked for that diagnostic. Returns {sourceId: [diagnosticIds that mute it]}.
  function diagnosticMuting(state, index) {
    const muted = {};
    const roles = {};
    Object.values((index && index.sources) || {}).forEach((e) => { roles[e.source_id] = e.role; });
    Object.keys(state.enabled).forEach((did) => {
      if (!state.enabled[did] || state.layerOnTop[did]) return;
      const track = parseDiagTrack(did);
      if (!index || roles[did] !== 'diagnostic' || track === null) return;
      ((state.trackMap || {})[track] || []).forEach((sid) => {
        if (roles[sid] === 'default') (muted[sid] = muted[sid] || []).push(did);
      });
    });
    return muted;
  }
  function effectiveGains(index, state) {
    const muted = diagnosticMuting(state, index), g = {};
    Object.keys(index.sources || {}).forEach((id) => { g[id] = state.enabled[id] && !muted[id] ? 1 : 0; });
    return g;
  }
  function controlsModel(index, state) {
    const muted = diagnosticMuting(state, index), gains = effectiveGains(index, state), buckets = {};
    Object.values(index.sources || {}).forEach((e) => {
      const g = groupFor(e);
      (buckets[g] = buckets[g] || []).push({ source_id: e.source_id, label: (state.labels && state.labels[e.source_id]) || e.source_id,
        kind: e.kind, role: e.role, enabled: !!state.enabled[e.source_id], audible: gains[e.source_id] === 1,
        mutedBy: muted[e.source_id] || [], layerOnTop: !!state.layerOnTop[e.source_id], hasLayer: g === 'diagnostic',
        nyquist_hz: e.nyquist_hz === undefined ? null : e.nyquist_hz });
    });
    return { groups: GROUPS.filter(([id]) => buckets[id]).map(([id, title]) => ({ id: id, title: title, diagnostic: id === 'diagnostic',
      items: buckets[id].sort((a, b) => (a.source_id < b.source_id ? -1 : 1)) })) };
  }
  // In-place update plan: rebuild only when the SET of controls changes; otherwise a list of property updates.
  function controlsPatch(prev, next) {
    const ids = (m) => (m ? m.groups.map((g) => g.id + ':' + g.items.map((i) => i.source_id).join(',')).join('|') : null);
    if (ids(prev) !== ids(next)) return { rebuild: true, updates: [] };
    const before = {};
    prev.groups.forEach((g) => g.items.forEach((i) => { before[i.source_id] = i; }));
    const updates = [];
    next.groups.forEach((g) => g.items.forEach((i) => {
      const b = before[i.source_id];
      ['enabled', 'layerOnTop'].forEach((f) => { if (b[f] !== i[f]) updates.push({ source_id: i.source_id, field: f, value: i[f] }); });
      if (b.audible !== i.audible || b.mutedBy.join() !== i.mutedBy.join()) updates.push({ source_id: i.source_id, field: 'note', value: mutedNote(i) });
    }));
    return { rebuild: false, updates: updates };
  }
  function mutedNote(i) { return i.mutedBy.length ? 'muted while ' + i.mutedBy.join(', ') + ' is soloed (tick "layer on top" there to hear both)' : (i.enabled ? '' : 'off'); }
  function focusKey(el) { return el && el.dataset ? { id: el.dataset.sourceId || null, ctl: el.dataset.ctl || null } : null; }
  function applyControlsPatch(patch, lookup) {          // lookup(source_id, ctl) -> element-like; only properties change
    patch.updates.forEach((u) => {
      const ctl = u.field === 'enabled' ? 'enable' : (u.field === 'layerOnTop' ? 'layer' : 'note');
      const el = lookup(u.source_id, ctl);
      if (!el) return;
      if (ctl === 'note') el.textContent = u.value; else el.checked = u.value;
    });
  }

  // Output latency: never invent a number. Only a positive finite outputLatency counts; baseLatency alone is not enough.
  function latencyInfo(ctx) {
    const ol = ctx && ctx.outputLatency, bl = ctx && ctx.baseLatency;
    const good = (v) => typeof v === 'number' && isFinite(v) && v > 0;
    if (!good(ol)) return { known: false, compensationS: 0, text: 'latency unknown (no compensation applied)' + (good(bl) ? '; browser reports only processing latency ' + (bl * 1000).toFixed(1) + ' ms' : '') };
    const total = ol + (good(bl) ? bl : 0);
    return { known: true, compensationS: total, text: 'output latency ' + (total * 1000).toFixed(0) + ' ms (video follows what is heard)' };
  }

  // Timeline segments covering time tUs, from loaded page docs ([{segments:[...]}]).
  function segmentsAt(pageDocs, tUs) {
    const out = [];
    pageDocs.forEach((d) => d.segments.forEach((s) => { if (s.start_us <= tUs && tUs < s.end_us) out.push(s); }));
    return out;
  }
  // What the user hears for a source at t: a stored segment, or silence (gap / unavailable stem) - never an error.
  // Every source with a timeline entry, including ones that never produced audio (refused in every chunk, or an
  // identity abstention): their refusal reasons are exactly what an inspector needs to see.
  function allSourceIds() {
    const ids = Object.keys(LA.idx.sources);
    ((LF.tlIndex && LF.tlIndex.sources) || []).forEach((s) => { if (ids.indexOf(s.source_id) < 0) ids.push(s.source_id); });
    return ids;
  }

  function sourceStateAt(index, id, t) {
    const src = index.sources[id];
    if (!src) return 'unknown';
    const s = (src.segments || []).find((x) => x.start_sample / src.rate_hz <= t && t < (x.start_sample + x.n_samples) / src.rate_hz);
    return s ? 'audio' : 'silent';
  }

  function prepareAudioIndex(index) {                    // sort segments once so segmentsToKeep can bisect
    const out = { version: index.version, sources: {} };
    Object.keys(index.sources || {}).forEach((id) => {
      const e = index.sources[id];
      out.sources[id] = Object.assign({}, e, { segments: (e.segments || []).slice().sort((a, b) => a.start_sample - b.start_sample) });
    });
    return out;
  }

  (typeof window !== 'undefined' ? window : globalThis).__viewerHelpers = { videoSyncAction, makeFrameTransform, LONG, segmentsToKeep, planSchedule,
    scheduleSegments, stopAllNodes, BufferCache, PageCache, pagesToLoad, badgesFor, controlsModel, controlsPatch, applyControlsPatch, focusKey,
    diagnosticMuting, effectiveGains, initialControlState, prepareAudioIndex, latencyInfo, segmentsAt, sourceStateAt, groupFor, defaultEnabled };
  if (typeof document === 'undefined' || !document.getElementById('stage')) return;

  const fetchJson = async (p) => { const r = await fetch(p); if (!r.ok) throw new Error(p + ': ' + r.status); return r.json(); };
  const fetchLines = async (p) => (await (async () => { const r = await fetch(p); if (!r.ok) throw new Error(p + ': ' + r.status); return r.text(); })()).split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l));

  const S = {
    ctx: null, buffers: {}, gains: {}, nodes: [], playing: false, offset: 0, t0: 0, duration: 0,
    mode: 'stems', solo: {}, mute: {}, residual: false, selected: null, frame: 0,
    manifest: null, frames: null, masks: {}, tstate: {}, act: {}, images: [], useVideo: false,
    long: false, latComp: 0, lat: null, curPts: 0,
  };
  // long-run state: audio engine (LA) and lazily paged frames / masks / timeline (LF)
  const LA = { idx: null, cache: null, pending: new Set(), failed: new Set(), active: new Map(), gains: {}, anchor: null, state: null, model: null, lastRun: 0, lastKeep: new Set(), kindOf: {}, behind: LONG.behind_s, ahead: LONG.ahead_s };
  const LF = { fi: null, frames: new PageCache(LONG.max_pages), masks: {}, tl: null, pending: new Set(), tlIndex: null, sig: '', lastFrame: null, lastAnn: {} };

  window.__avState = S;                       // read-only inspection handle for tests

  // ---------------------------------------------------------------- audio clock
  function position() { return S.playing ? Math.max(S.offset, S.offset + (S.ctx.currentTime - S.t0)) : S.offset; }
  function ensureCtx() {
    if (!S.ctx) {
      S.ctx = new (window.AudioContext || window.webkitAudioContext)();
      const make = (key) => { const g = S.ctx.createGain(); g.connect(S.ctx.destination); S.gains[key] = g; };
      make('mixture'); make('residual'); Object.keys(S.act).forEach((id) => make('stem:' + id));
    }
    refreshLatency();
    return S.ctx;
  }
  function refreshLatency() {
    S.lat = latencyInfo(S.ctx);
    S.latComp = S.long ? S.lat.compensationS : 0;     // legacy bundles keep their exact previous behaviour
    const el = document.getElementById('latency'); if (el && S.long && el.textContent !== S.lat.text) el.textContent = S.lat.text;
  }
  function stopNodes() { if (S.long) { stopAllNodes(LA.active); LA.anchor = null; } S.nodes.forEach((n) => { try { n.stop(); } catch (e) { /* already stopped */ } n.disconnect(); }); S.nodes = []; }
  function startNodes() {
    stopNodes();
    if (S.long) {                                     // ONE anchor for every stem: media time S.offset plays at ctx time S.t0
      S.t0 = S.ctx.currentTime + LONG.lead_s;
      LA.anchor = { anchorCtx: S.t0, anchorMedia: S.offset };
      longMaintain(true);
      return;
    }
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
    if (S.long) { longApplyGains(immediate); return; }
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
  function sourceForTrack(tid) {
    if (S.long) { const m = (LA.state.trackMap[tid] || []).filter((id) => LA.idx.sources[id] && LA.idx.sources[id].role === 'default'); return m.length ? m[0] : null; }
    const a = S.manifest.associations.find((x) => x.track_id === tid); return a ? a.source_id : null;
  }
  function trackList() { return S.long ? (S.manifest.tracks && S.manifest.tracks.length ? S.manifest.tracks : Object.keys((LF.fi && LF.fi.mask_pages) || {}).map((id) => ({ track_id: id }))) : S.manifest.tracks; }

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
    if (S.long) { selectSourceLong(id); return; }
    S.selected = id;
    if (exclusive) { S.solo = { [id]: true }; S.mute = {}; setMode('stems'); }
    renderSources(); renderInspector();
  }

  // ---------------------------------------------------------------- drawing
  const stage = $('#stage'), g2 = stage.getContext('2d');
  const off = document.createElement('canvas'), og = off.getContext('2d');
  function maskAt(tid, f) { if (S.long) return longMask(tid, S.curPts); const r = S.masks[tid][f]; return r && r.rle ? r.rle : null; }
  function draw() {
    const t = position(), tv = Math.max(0, t - (S.latComp || 0)), f = S.long ? longFrameIndex(tv * 1e6) : frameAt(tv * 1e6); S.frame = f;
    const FT = S.ft;
    g2.imageSmoothingEnabled = false;
    if (S.useVideo) g2.drawImage($('#video'), 0, 0, stage.width, stage.height);
    else if (S.long) { const im = longImage(); if (im && im.complete && im.naturalWidth > 0) g2.drawImage(im, 0, 0, stage.width, stage.height); }
    else if (S.images[f] && S.images[f].complete) g2.drawImage(S.images[f], 0, 0, stage.width, stage.height);
    const labels = [];
    off.width = FT.mask_w; off.height = FT.mask_h;
    const img = og.createImageData(FT.mask_w, FT.mask_h);
    let d = img.data;
    trackList().forEach((tr) => {
      const state = S.long ? longTrackState(tr.track_id, tv * 1e6) : S.tstate[tr.track_id].state[f];
      if (!state || state === 'idle') return;
      const rle = maskAt(tr.track_id, f); if (!rle) return;
      const W = rle.w, H = rle.h;
      if (W !== img.width || H !== img.height) return;   // mask size disagrees with the transform: skip rather than mis-draw
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
      labels.push([minx * stage.width / W, miny * stage.height / H, `${tr.track_id} ${word}`, col]);
    });
    og.putImageData(img, 0, 0); g2.drawImage(off, 0, 0, stage.width, stage.height);
    g2.font = '600 15px system-ui, sans-serif'; g2.textBaseline = 'bottom';
    labels.forEach(([x, y, txt, col]) => { const w = g2.measureText(txt).width + 8; g2.fillStyle = 'rgba(10,10,16,0.8)'; g2.fillRect(x, Math.max(y - 20, 0), w, 20); g2.fillStyle = `rgb(${col})`; g2.fillText(txt, x + 4, Math.max(y - 3, 17)); });
    const un = S.long ? longUnattributed(tv * 1e6) : Object.values(S.act).find((a) => !a.attributed && a.gate_active[f]);
    if (un) { g2.fillStyle = 'rgba(20,10,24,0.85)'; g2.fillRect(0, stage.height - 30, stage.width, 30); g2.fillStyle = 'rgb(255,120,220)'; g2.fillText('UNATTRIBUTED SOUND: ' + un.label, 10, stage.height - 8); }
    $('#time').textContent = `${Math.min(t, S.duration).toFixed(2)} / ${S.duration.toFixed(2)} s`;
    if (!seekDrag) $('#seek').value = String(Math.round((Math.min(t, S.duration) / S.duration) * 1000));
    if (S.playing && t >= S.duration) pause();
    if (S.useVideo) {
      const v = $('#video'), act = videoSyncAction(v.currentTime - tv, S.playing);     // same audio clock as the stems (tv = clock - known output latency)
      if (act.seek) v.currentTime = Math.min(tv, S.duration);
      if (v.playbackRate !== act.rate) v.playbackRate = act.rate;
    }
    if (S.long) { longMaintain(false); longUi(tv * 1e6); }
    requestAnimationFrame(draw);
  }

  // ---------------------------------------------------------------- interaction
  let seekDrag = false;
  stage.addEventListener('click', (e) => {
    const r = stage.getBoundingClientRect();
    const u = (e.clientX - r.left) / r.width, vv = (e.clientY - r.top) / r.height;
    const hits = trackList().filter((tr) => { const rle = maskAt(tr.track_id, S.frame); if (!rle) return false; if (rle.w !== S.ft.mask_w || rle.h !== S.ft.mask_h) return false; /* same rule as draw(): a mask the transform does not describe is neither drawn nor clickable */ const x = Math.min(rle.w - 1, Math.floor(u * rle.w)), y = Math.min(rle.h - 1, Math.floor(vv * rle.h)); return decodeRle(rle)[y * rle.w + x]; }).map((tr) => tr.track_id);
    const box = $('#candidates'); box.textContent = '';
    if (hits.length === 0) { box.hidden = true; return; }
    const choose = (tid) => { box.hidden = true; const sid = sourceForTrack(tid); if (sid) selectSource(sid, true); else { S.selected = null; if (S.long) { announce(tid + ': no sound source is associated with this object at this time.'); return; } const el = $('#inspector'); el.className = 'inspector'; el.textContent = `${tid}: no sound source is associated with this object (silent or unmatched).`; renderSources(); } };
    if (hits.length === 1) { choose(hits[0]); return; }
    const lead = document.createElement('span'); lead.textContent = 'Overlapping masks - choose one:'; box.appendChild(lead);
    hits.forEach((tid) => { const b = document.createElement('button'); b.type = 'button'; b.textContent = tid; b.addEventListener('click', () => choose(tid)); box.appendChild(b); });
    box.hidden = false;
  });
  $('#play').addEventListener('click', () => (S.playing ? pause() : play()));
  const seekEl = $('#seek');
  seekEl.addEventListener('input', () => { seekDrag = true; S.offset = (seekEl.value / 1000) * S.duration; $('#time').textContent = `${S.offset.toFixed(2)} / ${S.duration.toFixed(2)} s`; });
  seekEl.addEventListener('change', () => { seekDrag = false; seek((seekEl.value / 1000) * S.duration); if (S.long) longMaintain(true); });
  document.querySelectorAll('input[name=mode]').forEach((r) => r.addEventListener('change', () => setMode(r.value)));
  $('#residual').addEventListener('change', (e) => { S.residual = e.target.checked; applyMix(); });
  $('#reset').addEventListener('click', () => { S.solo = {}; S.mute = {}; S.residual = false; $('#residual').checked = false; setMode('stems'); renderSources(); });
  document.addEventListener('keydown', (e) => { if (e.code === 'Space' && e.target.tagName !== 'BUTTON' && e.target.tagName !== 'INPUT') { e.preventDefault(); S.playing ? pause() : play(); } });

  // ================================================================ long-run mode (manifest.long_run)
  const announce = (msg) => { const el = document.getElementById('live'); if (el) { el.textContent = ''; el.textContent = msg; } };
  const qs = (id) => id.replace(/["\\]/g, '\\$&');
  function longGain(id) {
    if (!LA.gains[id]) { const g = S.ctx.createGain(); g.connect(S.ctx.destination); LA.gains[id] = g; }
    return LA.gains[id];
  }
  function longApplyGains(immediate) {
    if (!LA.idx || !S.ctx) return;
    const g = effectiveGains(LA.idx, LA.state);
    Object.keys(g).forEach((id) => {
      const n = longGain(id);
      if (immediate) n.gain.value = g[id]; else n.gain.setTargetAtTime(g[id], S.ctx.currentTime, 0.01);
    });
  }
  function fetchSegment(it) {
    LA.pending.add(it.key);
    fetch(it.path).then((r) => { if (!r.ok) throw new Error(it.path + ': ' + r.status); return r.arrayBuffer(); })
      .then((ab) => S.ctx.decodeAudioData(ab))
      .then((buf) => {
        LA.pending.delete(it.key);
        if (!LA.lastKeep.has(it.key)) return;          // the window moved on while this was in flight: drop, do not cache
        LA.cache.put(it.key, buf, position());
        longMaintain(true);
      })
      .catch((err) => { LA.pending.delete(it.key); LA.failed.add(it.key); announce('Audio segment unavailable (' + err.message + '); that span stays silent.'); });
  }
  function longMaintain(force) {
    if (!LA.idx || !S.ctx) return;
    const now = performance.now();
    if (!force && now - LA.lastRun < 200) return;
    LA.lastRun = now;
    refreshLatency();
    const t = position();
    const g = effectiveGains(LA.idx, LA.state);
    const audible = Object.keys(g).filter((id) => g[id]);
    let keep = segmentsToKeep(LA.idx, t, LA.behind, LA.ahead, audible);
    keep = keep.sort((a, b) => Math.abs((a.t0 + a.t1) / 2 - t) - Math.abs((b.t0 + b.t1) / 2 - t)).slice(0, LA.cache.cap);   // never ask for more than the cap
    LA.lastKeep = new Set(keep.map((k) => k.key));
    LA.active.forEach((v, k) => LA.lastKeep.add(k));
    LA.cache.retain(LA.lastKeep, t);
    keep.forEach((it) => { if (!LA.cache.has(it.key) && !LA.pending.has(it.key) && !LA.failed.has(it.key)) fetchSegment(it); });
    if (LA.anchor) {
      const plan = planSchedule(keep.filter((k) => LA.cache.has(k.key)), { anchorCtx: LA.anchor.anchorCtx, anchorMedia: LA.anchor.anchorMedia, nowCtx: S.ctx.currentTime });
      scheduleSegments(S.ctx, plan, LA.cache, longGain, LA.active);
    }
    longLoadPages(t * 1e6);
  }

  // ---- lazily paged frames / masks / timeline (LRU)
  function pageFor(pages, tUs) {
    let best = null;
    for (const p of pages) { if (p.start_us <= tUs && (best === null || p.start_us >= best.start_us)) best = p; }
    return best;
  }
  function loadPage(cache, path, build) {
    if (cache.has(path) || LF.pending.has(path)) return;
    LF.pending.add(path);
    fetch(path).then((r) => { if (!r.ok) throw new Error(path + ': ' + r.status); return path.endsWith('.jsonl') ? r.text() : r.json(); })
      .then((doc) => { LF.pending.delete(path); cache.put(path, build(doc)); if (cache === LF.tl) onTimelineLoaded(); })
      .catch((err) => { LF.pending.delete(path); announce('Page unavailable: ' + err.message); });
  }
  function longLoadPages(tUs) {
    const opts = { behindUs: 1e6, aheadUs: 6e6, max: LONG.max_pages };
    if (LF.fi) {
      pagesToLoad(LF.fi.pages, tUs, opts).forEach((path) => loadPage(LF.frames, path, (doc) => ({ doc: doc, images: [] })));
      Object.keys(LF.fi.mask_pages || {}).forEach((tid) => {
        const cache = LF.masks[tid] || (LF.masks[tid] = new PageCache(LONG.max_pages));
        pagesToLoad(LF.fi.mask_pages[tid], tUs, opts).forEach((path) => loadPage(cache, path, (text) => {
          const byPts = new Map();
          text.split('\n').forEach((l) => { if (l.trim()) { const r = JSON.parse(l); byPts.set(r.pts_us, r.rle || null); } });
          return { byPts: byPts };
        }));
      });
    }
    if (LF.tlIndex) pagesToLoad(LF.tlIndex.pages, tUs, { behindUs: 2e6, aheadUs: 12e6, max: LONG.tl_pages }).forEach((path) => loadPage(LF.tl, path, (doc) => doc));
  }
  function longFrameIndex(tUs) {
    if (!LF.fi) return 0;
    const pg = pageFor(LF.fi.pages, tUs);
    const val = pg && LF.frames.get(pg.path);
    if (val) {
      const p = val.doc.pts_us; let lo = 0, hi = p.length;
      while (lo < hi) { const m = (lo + hi) >> 1; if (p[m] <= tUs) lo = m + 1; else hi = m; }
      const j = Math.max(0, lo - 1);
      LF.lastFrame = { val: val, j: j }; S.curPts = p[j];
      return val.doc.first_index + j;
    }
    return LF.lastFrame ? LF.lastFrame.val.doc.first_index + LF.lastFrame.j : 0;   // page still loading: hold the last picture
  }
  function longImage() {
    const lf = LF.lastFrame; if (!lf) return null;
    const mk = (j) => { if (j >= lf.val.doc.files.length) return null; if (!lf.val.images[j]) { const im = new Image(); im.src = lf.val.doc.files[j]; lf.val.images[j] = im; } return lf.val.images[j]; };
    for (let k = 1; k <= 6; k++) mk(lf.j + k);                  // small look-ahead, not the whole page
    return mk(lf.j);
  }
  function longMask(tid, pts) {
    const cache = LF.masks[tid]; if (!cache) return null;
    for (const path of cache.keys()) { const v = cache.map.get(path); if (v.byPts.has(pts)) return v.byPts.get(pts); }
    return null;
  }
  function tlDocs() { return LF.tl ? LF.tl.values() : []; }
  function longTrackState(tid, tUs) {
    let state = 'idle';
    segmentsAt(tlDocs(), tUs).forEach((s) => {
      if (s.track_id !== tid) return;
      const role = LA.idx.sources[s.source_id] && LA.idx.sources[s.source_id].role;
      if (role && role !== 'default') return;
      if (s.association_status === 'matched') state = (LA.kindOf[s.source_id] === 'speech') ? 'speaking' : 'sounding';
      else if (s.association_status === 'ambiguous' && state === 'idle') state = 'possible';
    });
    return state;
  }
  function longUnattributed(tUs) {
    const s = segmentsAt(tlDocs(), tUs).find((x) => x.association_status === 'unmatched' && x.stem_availability === 'ok' && LA.idx.sources[x.source_id] && LA.idx.sources[x.source_id].role === 'default' && LA.state.enabled[x.source_id]);
    return s ? { label: LA.state.labels[s.source_id] || s.source_id } : null;
  }
  function onTimelineLoaded() {
    let changed = false;
    const tm = LA.state.trackMap;
    tlDocs().forEach((d) => d.segments.forEach((s) => {
      const src = LA.idx.sources[s.source_id];
      if (s.association_status === 'matched' && s.track_id && src && src.role === 'default') {
        const arr = tm[s.track_id] || (tm[s.track_id] = []);
        if (arr.indexOf(s.source_id) < 0) { arr.push(s.source_id); changed = true; }
      }
    }));
    if (changed) { updateListen(); longApplyGains(false); }
    LF.sig = '';                                       // force a details refresh
  }

  // ---- listen controls (updated in place; focus survives every toggle)
  function renderListenDom(model) {
    const root = $('#listen');
    const key = focusKey(document.activeElement && root.contains(document.activeElement) ? document.activeElement : null);
    root.textContent = '';
    model.groups.forEach((grp) => {
      const fs = document.createElement('fieldset'); fs.className = 'lgroup' + (grp.diagnostic ? ' diag' : '');
      const lg = document.createElement('legend'); lg.textContent = grp.title; fs.appendChild(lg);
      if (grp.diagnostic) {
        const p = document.createElement('p'); p.className = 'muted small';
        p.textContent = 'A diagnostic stem is a SOLO view of one track: turning it on mutes the matched default stems for that track so nothing plays twice. Tick "layer on top" to hear both.';
        fs.appendChild(p);
      }
      grp.items.forEach((it) => {
        const row = document.createElement('div'); row.className = 'ctl'; row.dataset.sourceId = it.source_id;
        const lab = document.createElement('label');
        const cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = it.enabled; cb.dataset.sourceId = it.source_id; cb.dataset.ctl = 'enable';
        lab.append(cb, document.createTextNode(' ' + it.label));
        row.appendChild(lab);
        if (it.hasLayer) {
          const l2 = document.createElement('label'); l2.className = 'small';
          const c2 = document.createElement('input'); c2.type = 'checkbox'; c2.checked = it.layerOnTop; c2.dataset.sourceId = it.source_id; c2.dataset.ctl = 'layer';
          l2.append(c2, document.createTextNode(' layer on top'));
          row.appendChild(l2);
        }
        const note = document.createElement('span'); note.className = 'muted small'; note.dataset.sourceId = it.source_id; note.dataset.ctl = 'note'; note.textContent = mutedNote(it);
        row.appendChild(note);
        fs.appendChild(row);
      });
      root.appendChild(fs);
    });
    if (key && key.id) {                              // structure changed: put focus back on the same control
      const el = root.querySelector('[data-source-id="' + qs(key.id) + '"][data-ctl="' + key.ctl + '"]');
      if (el) el.focus();
    }
  }
  function listenLookup(id, ctl) { return $('#listen').querySelector('[data-source-id="' + qs(id) + '"][data-ctl="' + ctl + '"]'); }
  function updateListen() {
    const next = controlsModel(LA.idx, LA.state), patch = controlsPatch(LA.model, next);
    if (patch.rebuild) renderListenDom(next); else applyControlsPatch(patch, listenLookup);
    LA.model = next;
  }
  function selectSourceLong(id) {
    LA.state.selected = id; LA.state.enabled[id] = true;
    updateListen(); longApplyGains(false); longMaintain(true);
    announce('Selected ' + (LA.state.labels[id] || id));
    LF.sig = '';
  }

  // ---- timeline panel: four separate badges per source + refusal text (details), lanes canvas overview
  function longUi(tUs) {
    const segs = segmentsAt(tlDocs(), tUs);
    const sig = segs.map((s) => s.source_id + s.start_us + s.stem_availability + s.association_status + s.visibility).join('|') + '#' + tlDocs().length;
    if (sig === LF.sig) return;
    LF.sig = sig;
    const ul = $('#tl-details'); const bySrc = {};
    segs.forEach((s) => { bySrc[s.source_id] = s; });
    allSourceIds().forEach((id) => {
      let li = ul.querySelector('[data-source-id="' + qs(id) + '"]');
      if (!li) {                                      // created once, then only text/classes change: no focus loss
        li = document.createElement('li'); li.className = 'tlrow'; li.tabIndex = 0; li.dataset.sourceId = id;
        const nm = document.createElement('span'); nm.className = 'tlname'; nm.textContent = LA.state.labels[id] || id; li.appendChild(nm);
        ['association_status', 'visibility', 'stem_availability', 'identity_confidence'].forEach((f) => { const b = document.createElement('span'); b.dataset.field = f; li.appendChild(b); });
        const rs = document.createElement('span'); rs.className = 'refusal'; rs.dataset.field = 'refusal_reason'; li.appendChild(rs);
        ul.appendChild(li);
      }
      const s = bySrc[id];
      const state = sourceStateAt(LA.idx, id, tUs / 1e6);
      if (!s) {
        li.querySelectorAll('[data-field]').forEach((e) => { e.textContent = ''; e.className = ''; });
        const sa = li.querySelector('[data-field=stem_availability]');
        sa.textContent = state === 'audio' ? 'audio present, no timeline entry' : 'silent here'; sa.className = 'badge muted';
        li.title = (LA.state.labels[id] || id) + ': nothing recorded for this span (silent).';
        return;
      }
      const bf = badgesFor(s);
      bf.badges.forEach((b) => { const e = li.querySelector('[data-field=' + b.field + ']'); e.textContent = b.text; e.className = 'badge ' + b.tone; });
      const rs = li.querySelector('[data-field=refusal_reason]');
      rs.textContent = bf.refusal ? 'Reason: ' + bf.refusal : ''; rs.className = 'refusal' + (bf.refusal ? ' on' : '');
      li.title = (LA.state.labels[id] || id) + ': ' + bf.summary + (state === 'silent' ? '; no audio stored for this span (plays silent)' : '');
      li.setAttribute('aria-label', li.title);
      const prev = LF.lastAnn[id];
      if (prev !== undefined && prev !== s.stem_availability && LA.state.enabled[id] && (s.stem_availability === 'unavailable' || s.stem_availability === 'degraded' || prev === 'unavailable')) {
        announce((LA.state.labels[id] || id) + ' stem ' + s.stem_availability.replace(/_/g, ' ') + (s.refusal_reason ? ': ' + s.refusal_reason : ''));
      }
      LF.lastAnn[id] = s.stem_availability;
    });
    drawLanes(tUs);
  }
  function drawLanes(tUs) {
    const c = $('#tl-lanes'); if (!c || !S.duration) return;
    const rows = allSourceIds(), laneH = 5, rowH = laneH * 4 + 6;
    c.width = 900; c.height = Math.max(rows.length * rowH, 10);
    const g = c.getContext('2d'), cs = getComputedStyle(document.body);
    const css = (n, d) => (cs.getPropertyValue(n) || d).trim() || d;
    const tone = { ok: css('--ok', '#17632f'), warn: css('--amb', '#8a5a00'), off: css('--off', '#6b2b7a'), bad: '#c0392b', muted: css('--muted', '#888') };
    g.clearRect(0, 0, c.width, c.height);
    const X = (us) => (us / 1e6 / S.duration) * c.width;
    tlDocs().forEach((d) => d.segments.forEach((s) => {
      const r = rows.indexOf(s.source_id); if (r < 0) return;
      const x = X(s.start_us), w = Math.max(1.5, X(s.end_us) - x);
      ['association_status', 'visibility', 'stem_availability'].forEach((f, k) => { g.fillStyle = tone[TONES[f][s[f]] || 'muted']; g.fillRect(x, r * rowH + k * laneH, w, laneH - 1); });
      if (s.refusal_reason) { g.fillStyle = tone.bad; g.fillRect(x, r * rowH + 3 * laneH, Math.max(2, w), laneH - 1); }
    }));
    g.fillStyle = cs.color; g.fillRect(X(tUs), 0, 1.5, c.height);
  }

  async function bootLong() {
    S.long = true;
    document.body.classList.add('long');
    $('#longpanel').hidden = false;
    document.querySelectorAll('.legacy-only').forEach((e) => { e.hidden = true; });
    const A = S.manifest.assets || {}, lr = S.manifest.long_run;
    const q = new URLSearchParams(location.search);
    if (q.get('behind')) LA.behind = Math.max(0, +q.get('behind'));
    if (q.get('ahead')) LA.ahead = Math.max(1, +q.get('ahead'));
    LA.idx = prepareAudioIndex(await fetchJson(lr.audio_index));
    LF.tlIndex = await fetchJson(lr.timeline_index);
    LF.tl = new PageCache(LONG.tl_pages);
    try { LF.fi = await fetchJson(A.frames_index || 'frames/index.json'); } catch (e) { LF.fi = null; }
    const labels = {}, trackMap = {};
    LF.tlIndex.sources.forEach((s) => { labels[s.source_id] = s.label || s.source_id; LA.kindOf[s.source_id] = s.kind; });
    Object.values(LA.idx.sources).forEach((e) => { LA.kindOf[e.source_id] = e.kind; });
    (S.manifest.associations || []).forEach((a) => { if (a.status === 'matched' && a.track_id) (trackMap[a.track_id] = trackMap[a.track_id] || []).push(a.source_id); });
    LA.state = initialControlState(LA.idx, trackMap, labels);
    LA.cache = new BufferCache(LONG.max_buffers, (k) => { const i = k.lastIndexOf('|'); return +k.slice(i + 1) / LA.idx.sources[k.slice(0, i)].rate_hz; });
    S.ft = makeFrameTransform(S.manifest, LF.fi || {}, stage.width, stage.height);
    S.duration = Math.max(LF.tlIndex.n_samples / LF.tlIndex.sample_rate, ...Object.values(LA.idx.sources).map((e) => e.n_samples / e.rate_hz), 0);
    if (!S.duration && LF.fi && LF.fi.pages.length) S.duration = LF.fi.pages[LF.fi.pages.length - 1].end_us / 1e6;
    $('#tier').textContent = 'evidence: ' + S.manifest.evidence_tier;
    $('#banner').textContent = S.manifest.evidence_note;
    S.useVideo = !!A.source_video;
    if (S.useVideo) { const v = $('#video'); v.src = A.source_video; v.muted = true; }
    $('#tl-details').textContent = '';
    updateListen();
    $('#listen').addEventListener('change', (e) => {
      const t = e.target, id = t.dataset && t.dataset.sourceId, ctl = t.dataset && t.dataset.ctl;
      if (!id || !ctl) return;
      if (ctl === 'enable') LA.state.enabled[id] = t.checked; else if (ctl === 'layer') LA.state.layerOnTop[id] = t.checked; else return;
      updateListen(); longApplyGains(false); longMaintain(true);
      const m = LA.model.groups.flatMap((g) => g.items).filter((i) => i.mutedBy.length).map((i) => i.label);
      announce((LA.state.labels[id] || id) + (ctl === 'enable' ? (t.checked ? ' on' : ' off') : (t.checked ? ': layered on top' : ': solo view')) + (m.length ? '. Muted to avoid duplicate playback: ' + m.join(', ') : '.'));
    });
    try { ensureCtx(); S.audioReady = true; longApplyGains(true); } catch (err) { $('#banner').textContent = 'Audio unavailable: ' + err.message; }
    longLoadPages(0);
    requestAnimationFrame(draw);
    longMaintain(true);
    try { renderMetrics(await fetchJson('metrics.json')); } catch (e) { /* metrics are optional */ }
  }

  // ---------------------------------------------------------------- boot
  (async function boot() {
    try {
      S.manifest = await fetchJson('manifest.json');
      if (S.manifest.long_run) { await bootLong(); return; }
      S.frames = await fetchJson('frames.json');
      S.ft = makeFrameTransform(S.manifest, S.frames, stage.width, stage.height);
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
