"""Long-run viewer helpers (viewer.js) via node; evidence: mocked (fake AudioContext / fake elements, no browser).

Real-browser audio output, focus behaviour and rendering are NOT covered here; see the manual checklist at the
bottom of this file.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ASSETS = Path(__file__).resolve().parents[1] / "render" / "assets"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")

PRELUDE = """
const R=48000;
const mkIdx=(srcs)=>H.prepareAudioIndex({version:1,sources:srcs});
const src=(id,kind,role,segs,rate=R)=>({source_id:id,kind,role,rate_hz:rate,n_samples:segs.length?segs[segs.length-1][0]+segs[segs.length-1][1]:0,
  segments:segs.map(([s,n])=>({start_sample:s,n_samples:n,path:`audio/${id}/seg_${s}.wav`}))});
class FakeCtx{constructor(){this.currentTime=0;this.log=[];}
  createBufferSource(){const log=this.log;const n={connect(g){n.gain=g;},disconnect(){},stop(){log.push(['stop',n.key]);},
    start(when,offset,dur){log.push(['start',n.buffer.key,when,offset,dur]);},onended:null};return n;}}
const bufs=(items)=>{const m=new Map();items.forEach(i=>m.set(i.key,{key:i.key}));return m;};
"""


def _run(tmp_path, body):
    s = tmp_path / "c.js"
    s.write_text("const fs=require('fs'),vm=require('vm'),assert=require('assert');\n"
                 f"vm.runInThisContext(fs.readFileSync({json.dumps(str(ASSETS / 'viewer.js'))},'utf8'));\n"
                 "const H=globalThis.__viewerHelpers;\n" + PRELUDE + body + "\nconsole.log('ok');\n")
    r = subprocess.run([NODE, str(s)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


def test_seek_into_a_segment_uses_one_anchor_and_exact_offset(tmp_path):
    _run(tmp_path, """
const idx=mkIdx({a:src('a','speech','default',[[0,R*2],[R*2,R*2]]),b:src('b','event','default',[[R,R*3]])});
const keep=H.segmentsToKeep(idx,2.5,4,10);
const plan=H.planSchedule(keep,{anchorCtx:100.06,anchorMedia:2.5,nowCtx:100.0});
const byKey=Object.fromEntries(plan.map(p=>[p.key,p]));
// a's 2nd segment covers 2..4 s: starts at the anchor with a 0.5 s offset; b covers 1..4 s: also at the anchor, 1.5 s in
assert.strictEqual(byKey['a|96000'].when,100.06); assert.ok(Math.abs(byKey['a|96000'].offset-0.5)<1e-12);
assert.strictEqual(byKey['b|48000'].when,100.06); assert.ok(Math.abs(byKey['b|48000'].offset-1.5)<1e-12);
// a's 1st segment (0..2 s) is entirely before the playhead: not scheduled
assert.ok(!byKey['a|0']);
// every stem is placed against the same anchor: later starts differ only by media-time distance
const idx2=mkIdx({a:src('a','speech','default',[[R*5,R]]),b:src('b','event','default',[[R*7,R]])});
const p2=H.planSchedule(H.segmentsToKeep(idx2,2.5,4,10),{anchorCtx:10,anchorMedia:2.5,nowCtx:9.9});
assert.ok(Math.abs(p2[0].when-12.5)<1e-9 && Math.abs(p2[1].when-14.5)<1e-9 && p2[0].offset===0);
""")


def test_chunk_boundaries_are_gapless_to_the_sample(tmp_path):
    _run(tmp_path, """
const segs=[[0,1234567],[1234567,777777],[2012344,48000]];     // odd lengths, adjacent by construction
const idx=mkIdx({a:src('a','speech','default',segs)});
for (const A of [0,0.31,5.0007]) for (const anchor of [0.06,123.456789]) {
  const plan=H.planSchedule(H.segmentsToKeep(idx,A,10,100),{anchorCtx:anchor,anchorMedia:A,nowCtx:0});
  for(let i=0;i+1<plan.length;i++) assert.strictEqual(plan[i].endWhen, plan[i+1].when);   // identical floats, not just close
}
""")


def test_pause_resume_never_double_starts_a_segment(tmp_path):
    _run(tmp_path, """
const idx=mkIdx({a:src('a','speech','default',[[0,R*4],[R*4,R*4]])});
const ctx=new FakeCtx(), active=new Map(), g=()=>({});
let keep=H.segmentsToKeep(idx,1,4,10), B=bufs(keep);
let plan=H.planSchedule(keep,{anchorCtx:0.06,anchorMedia:1,nowCtx:0});
assert.strictEqual(H.scheduleSegments(ctx,plan,B,g,active).length,2);
// the top-up loop runs every 200 ms against the same anchor: nothing may start twice
ctx.currentTime=0.5; plan=H.planSchedule(keep,{anchorCtx:0.06,anchorMedia:1,nowCtx:0.5});
assert.strictEqual(H.scheduleSegments(ctx,plan,B,g,active).length,0);
assert.strictEqual(ctx.log.filter(e=>e[0]==='start').length,2);
// pause: stop everything. resume at the paused position with a NEW anchor on the same clock
H.stopAllNodes(active); assert.strictEqual(active.size,0);
const pos=1+(0.5-0.06);                                           // media time at the pause
ctx.currentTime=3; keep=H.segmentsToKeep(idx,pos,4,10); B=bufs(keep);
plan=H.planSchedule(keep,{anchorCtx:3.06,anchorMedia:pos,nowCtx:3});
H.scheduleSegments(ctx,plan,B,g,active);
const starts=ctx.log.filter(e=>e[0]==='start');
assert.strictEqual(starts.length,4);                               // 2 before pause + 2 after, 0 duplicates while running
const first=starts[2]; assert.strictEqual(first[2],3.06); assert.ok(Math.abs(first[3]-pos)<1e-9);
""")


def test_late_buffer_joins_at_the_correct_position(tmp_path):
    _run(tmp_path, """
const idx=mkIdx({a:src('a','speech','default',[[0,R*10]])});
const keep=H.segmentsToKeep(idx,0,4,10);
// decode finished 2 s after the anchor: start now, 2 s in, never replay from 0 and never in the past
const p=H.planSchedule(keep,{anchorCtx:1.06,anchorMedia:0,nowCtx:3.06})[0];
assert.strictEqual(p.when,3.06); assert.ok(Math.abs(p.offset-2.0)<1e-9); assert.ok(Math.abs(p.duration-8.0)<1e-9);
assert.strictEqual(H.planSchedule(keep,{anchorCtx:1.06,anchorMedia:0,nowCtx:11.5}).length,0);   // already over
""")


def test_missing_segment_is_silent_and_badges_say_why(tmp_path):
    _run(tmp_path, """
// stem for 'a' exists for 0-2 s and 4-6 s; the middle is a hole (stem unavailable), another source is fine throughout
const idx=mkIdx({a:src('a','speech','default',[[0,R*2],[R*4,R*2]]),b:src('b','event','default',[[0,R*6]])});
assert.strictEqual(H.sourceStateAt(idx,'a',3),'silent'); assert.strictEqual(H.sourceStateAt(idx,'a',1),'audio');
const keep=H.segmentsToKeep(idx,3,1,1);                                   // window 2..4: only b overlaps
assert.deepStrictEqual(keep.map(k=>k.source_id),['b']);
const plan=H.planSchedule(keep,{anchorCtx:1,anchorMedia:3,nowCtx:0.9});    // must not throw
assert.strictEqual(plan.length,1);
const seg={source_id:'a',association_status:'matched',visibility:'visible',stem_availability:'unavailable',refusal_reason:'separator refused: overlap',identity_confidence:'resolved'};
const b=H.badgesFor(seg);
assert.deepStrictEqual(b.badges.map(x=>x.field),['association_status','visibility','stem_availability','identity_confidence']);
assert.strictEqual(new Set(b.badges.map(x=>x.field)).size,4);               // four separate marks, not one collapsed enum
assert.strictEqual(b.badges[0].tone,'ok'); assert.strictEqual(b.badges[2].tone,'bad');
assert.ok(b.badges[2].text.includes('unavailable')); assert.strictEqual(b.refusal,'separator refused: overlap');
assert.ok(b.summary.includes('Association: matched') && b.summary.includes('Visibility: visible') && b.summary.includes('separator refused'));
// the reverse mix is just as expressible: unmatched + outside_view + a perfectly good stem
const c=H.badgesFor({association_status:'unmatched',visibility:'outside_view',stem_availability:'ok',identity_confidence:'new',refusal_reason:null});
assert.deepStrictEqual(c.badges.map(x=>x.value),['unmatched','outside_view','ok','new']); assert.strictEqual(c.refusal,null);
""")


def test_diagnostic_stem_is_a_solo_view_unless_layered(tmp_path):
    _run(tmp_path, """
const idx=mkIdx({'src_1':src('src_1','speech','default',[[0,R]]),'src_2':src('src_2','speech','default',[[0,R]]),
  'evt':src('evt','event','default',[[0,R]]),'res':src('res','residual_proxy','default',[[0,R]]),'tse:t1':src('tse:t1','tse','diagnostic',[[0,R]])});
const st=H.initialControlState(idx,{t1:['src_1'],t2:['src_2']},{});
// default mix = enabled default-role stems only; proxy and diagnostics are off
assert.deepStrictEqual(H.effectiveGains(idx,st),{'src_1':1,'src_2':1,'evt':1,'res':0,'tse:t1':0});
assert.deepStrictEqual(H.diagnosticMuting(st,idx),{});
st.enabled['tse:t1']=true;                                                  // solo view of track t1
assert.deepStrictEqual(H.effectiveGains(idx,st),{'src_1':0,'src_2':1,'evt':1,'res':0,'tse:t1':1});
assert.deepStrictEqual(H.diagnosticMuting(st,idx),{'src_1':['tse:t1']});
st.layerOnTop['tse:t1']=true;                                               // explicit layering plays both
assert.deepStrictEqual(H.effectiveGains(idx,st),{'src_1':1,'src_2':1,'evt':1,'res':0,'tse:t1':1});
st.layerOnTop['tse:t1']=false; st.enabled['tse:t1']=false;
assert.strictEqual(H.effectiveGains(idx,st)['src_1'],1);
// groups: matched speech / events / residual proxy / diagnostic are separate, diagnostic last
const m=H.controlsModel(idx,st);
assert.deepStrictEqual(m.groups.map(g=>g.id),['speech','event','residual_proxy','diagnostic']);
assert.strictEqual(m.groups[3].diagnostic,true); assert.strictEqual(m.groups[3].items[0].hasLayer,true);
assert.strictEqual(H.defaultEnabled({kind:'outside_view',role:'default'}),true); assert.strictEqual(H.groupFor({kind:'outside_view',role:'default'}),'outside_view');
""")


def test_toggle_updates_in_place_and_focus_is_kept(tmp_path):
    _run(tmp_path, """
const idx=mkIdx({'src_1':src('src_1','speech','default',[[0,R]]),'tse:t1':src('tse:t1','tse','diagnostic',[[0,R]])});
const st=H.initialControlState(idx,{t1:['src_1']},{});
const m0=H.controlsModel(idx,st);
// fake DOM: elements keyed by source id + control; a "removal" or replacement is recorded
const dom={}, calls={replaced:0};
m0.groups.forEach(g=>g.items.forEach(i=>{dom[i.source_id+'/enable']={dataset:{sourceId:i.source_id,ctl:'enable'},checked:i.enabled};
  dom[i.source_id+'/note']={dataset:{sourceId:i.source_id,ctl:'note'},textContent:''}; if(i.hasLayer) dom[i.source_id+'/layer']={dataset:{sourceId:i.source_id,ctl:'layer'},checked:false};}));
const doc={activeElement:dom['tse:t1/enable']};
const before=doc.activeElement;
st.enabled['tse:t1']=true;
const m1=H.controlsModel(idx,st), patch=H.controlsPatch(m0,m1);
assert.strictEqual(patch.rebuild,false);                                       // no DOM rebuild -> the focused checkbox survives
H.applyControlsPatch(patch,(id,ctl)=>dom[id+'/'+ctl]);
assert.strictEqual(doc.activeElement,before); assert.deepStrictEqual(H.focusKey(doc.activeElement),{id:'tse:t1',ctl:'enable'});
assert.strictEqual(dom['tse:t1/enable'].checked,true);
assert.ok(dom['src_1/note'].textContent.includes('tse:t1'));                  // the muted default stem explains itself
// layering back on clears the explanation, still no rebuild
st.layerOnTop['tse:t1']=true; const m2=H.controlsModel(idx,st), p2=H.controlsPatch(m1,m2);
assert.strictEqual(p2.rebuild,false); H.applyControlsPatch(p2,(id,ctl)=>dom[id+'/'+ctl]); assert.strictEqual(dom['src_1/note'].textContent,'');
assert.strictEqual(dom['tse:t1/layer'].checked,true);
// structural change (a new source appears) is the only case that rebuilds, and focusKey lets the renderer restore focus
const idx2=mkIdx({'src_1':src('src_1','speech','default',[[0,R]]),'tse:t1':src('tse:t1','tse','diagnostic',[[0,R]]),'src_9':src('src_9','speech','default',[[0,R]])});
const st2=H.initialControlState(idx2,{},{});
assert.strictEqual(H.controlsPatch(m2,H.controlsModel(idx2,st2)).rebuild,true);
""")


def test_decoded_buffers_never_exceed_the_cap_during_a_seek_sweep(tmp_path):
    _run(tmp_path, """
const CAP=12, segs=[]; for(let i=0;i<600;i++) segs.push([i*R*5,R*5]);       // 50 min of 5 s chunks, 3 stems
const idx=mkIdx({a:src('a','speech','default',segs),b:src('b','event','default',segs),c:src('c','outside_view','default',segs)});
const cache=new H.BufferCache(CAP,(k)=>{const i=k.lastIndexOf('|');return +k.slice(i+1)/R;});
let maxSize=0, maxKeep=0;
for(let step=0;step<2000;step++){
  const t=((step*7919)%3000)+0.37;                                            // pseudo-random seeks across the run
  let keep=H.segmentsToKeep(idx,t,4,10); maxKeep=Math.max(maxKeep,keep.length);
  keep=keep.sort((a,b)=>Math.abs((a.t0+a.t1)/2-t)-Math.abs((b.t0+b.t1)/2-t)).slice(0,CAP);
  cache.retain(new Set(keep.map(k=>k.key)),t);
  keep.forEach(k=>cache.put(k.key,{k},t)); maxSize=Math.max(maxSize,cache.size);
  assert.ok(cache.size<=CAP,'cache '+cache.size);
}
assert.ok(maxSize<=CAP && maxSize>0); assert.ok(maxKeep<600*3/10);            // the window is a tiny slice of the run
// put alone enforces the cap and evicts the farthest segment, not the nearest
const c2=new H.BufferCache(2,(k)=>+k); c2.put('0',1,0); c2.put('100',1,0); c2.put('1',1,0);
assert.deepStrictEqual([...c2.map.keys()].sort(),['0','1']);
""")


def test_pages_are_selected_nearest_first_and_lru_capped(tmp_path):
    _run(tmp_path, """
const pages=[0,1,2,3,4,5].map(i=>({path:`p${i}`,start_us:i*30e6,end_us:(i+1)*30e6,n:1}));
assert.deepStrictEqual(H.pagesToLoad(pages,45e6,{behindUs:1e6,aheadUs:6e6,max:3}),['p1']);
assert.deepStrictEqual(H.pagesToLoad(pages,59e6,{behindUs:1e6,aheadUs:6e6,max:3}),['p1','p2']);
assert.strictEqual(H.pagesToLoad(pages,0,{aheadUs:500e6,max:3}).length,3);
const c=new H.PageCache(3); ['a','b','c'].forEach(k=>c.put(k,k)); c.get('a'); c.put('d','d');
assert.deepStrictEqual(c.keys(),['c','a','d']); assert.strictEqual(c.size,3);   // b was least recently used
let max=0; for(let i=0;i<500;i++){c.put('k'+i,i); max=Math.max(max,c.size);} assert.strictEqual(max,3);
""")


def test_latency_unknown_applies_no_compensation(tmp_path):
    _run(tmp_path, """
for (const ctx of [{}, {outputLatency:undefined}, {outputLatency:0,baseLatency:0.01}, {outputLatency:NaN}, null]) {
  const l=H.latencyInfo(ctx); assert.strictEqual(l.known,false); assert.strictEqual(l.compensationS,0); assert.ok(l.text.startsWith('latency unknown'));
}
assert.ok(H.latencyInfo({outputLatency:0,baseLatency:0.01}).text.includes('processing latency'));   // reported, not used
const k=H.latencyInfo({outputLatency:0.02,baseLatency:0.005}); assert.strictEqual(k.known,true); assert.ok(Math.abs(k.compensationS-0.025)<1e-12);
// video drift keeps using the same videoSyncAction policy against the (compensated) audio clock
assert.deepStrictEqual(H.videoSyncAction(0.2,true),{seek:true,rate:1});
""")


def test_frame_transform_scaling_is_unchanged_for_paged_frame_index(tmp_path):
    _run(tmp_path, """
const fi={width:640,height:480,transform:{version:1,frame_w:640,frame_h:480,mask_w:320,mask_h:240}};
const t=H.makeFrameTransform({capabilities:{frame_transform:fi.transform}},fi,960,720);
assert.deepStrictEqual([t.sx,t.sy,t.mask_w,t.mask_h,t.stage_w],[2,2,320,240,960]);
const t2=H.makeFrameTransform({capabilities:{}},fi,960,720); assert.deepStrictEqual([t2.sx,t2.sy],[2,2]);
""")


def test_viewer_long_assets_have_no_network_dependencies_and_expose_regions():
    html = (ASSETS / "viewer.html").read_text(encoding="utf-8")
    js = (ASSETS / "viewer.js").read_text(encoding="utf-8")
    for needle in ('id="live"', 'aria-live="polite"', 'id="listen"', 'id="latency"', 'id="tl-details"', "layer on top"):
        assert needle in html + js, needle
    for text in (html, js, (ASSETS / "viewer.css").read_text(encoding="utf-8")):
        for bad in ("http://", "https://", "//cdn", "unpkg", "@import url"):
            assert bad not in text, bad


# Manual checks NOT automated (need a real browser + audio device):
#  1. Open a long bundle via `python -m av_scene serve <bundle>`; tab to a diagnostic checkbox, press Space: focus stays,
#     the matched stem's note updates and the polite live region announces the muting.
#  2. Seek across the run while playing: the Network tab shows only segments within ~[t-4 s, t+10 s].
#  3. Play across a segment boundary and a stem hole: no click at the boundary, silence in the hole, no console errors.
#  4. In a browser where AudioContext.outputLatency is undefined/0 the panel reads "latency unknown".
