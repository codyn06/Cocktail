"""viewer.js pure helpers (video sync policy, FrameTransform) via node; skipped without node."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ASSETS = Path(__file__).resolve().parents[1] / "render" / "assets"
NODE = shutil.which("node")


def _run(tmp_path, body):
    s = tmp_path / "c.js"
    s.write_text("const fs=require('fs'),vm=require('vm'),assert=require('assert');\n"
                 f"vm.runInThisContext(fs.readFileSync({json.dumps(str(ASSETS / 'viewer.js'))},'utf8'));\n"
                 "const H=globalThis.__viewerHelpers;\n" + body + "\nconsole.log('ok');\n")
    r = subprocess.run([NODE, str(s)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_video_sync_action(tmp_path):
    _run(tmp_path, """
const a=H.videoSyncAction;
assert.deepStrictEqual(a(0.2,true),{seek:true,rate:1});
assert.deepStrictEqual(a(-0.15,true),{seek:true,rate:1});
assert.deepStrictEqual(a(0.01,true),{seek:false,rate:1});
let r=a(0.1,true); assert(!r.seek && Math.abs(r.rate-0.95)<1e-9);
r=a(-0.1,true); assert(!r.seek && Math.abs(r.rate-1.05)<1e-9);
assert.deepStrictEqual(a(0.05,false),{seek:true,rate:1});
assert.deepStrictEqual(a(0.03,false),{seek:false,rate:1});
""")


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_frame_transform_fallbacks(tmp_path):
    _run(tmp_path, """
let t=H.makeFrameTransform({capabilities:{frame_transform:{frame_w:640,frame_h:480,mask_w:320,mask_h:240}}},{width:1,height:1},960,720);
assert.deepStrictEqual([t.sx,t.sy,t.frame_w,t.mask_h,t.stage_w],[2,2,640,240,960]);
t=H.makeFrameTransform({capabilities:{}},{width:320,height:240},960,720);
assert.deepStrictEqual([t.sx,t.sy,t.mask_w],[1,1,320]);
t=H.makeFrameTransform({},{width:100,height:50,transform:{frame_w:100,frame_h:50,mask_w:50,mask_h:25}},960,720);
assert.deepStrictEqual([t.sx,t.sy],[2,2]);
""")
