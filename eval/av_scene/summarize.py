"""Compact tables from <run>/results.jsonl (+ judge.jsonl): who does each candidate follow?

Run from hub/:  uv run python ../eval/av_scene/summarize.py sessions/av_matrix/run1
For the overlap trials, `follows` is the fraction of candidates whose output is closer (SI-SDR) to the
PROMPTED person than to the other person; `other_sdri` is the mean SI-SDRi vs the OTHER person.
"""
import json
import sys
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
rows = [json.loads(l) for l in (run / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
judge = {}
if (run / "judge.jsonl").exists():
    judge = {j["key"]: j for j in (json.loads(l) for l in (run / "judge.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
print("trial | prompt | anchors | vis | cand0 | best@idx | mean+-std | n>=3dB | follows | other_sdri(mean) | judge_pick(sdri)")
for r in rows:
    if not r["present"] or "other" not in r["cands"][0]:
        continue
    s, c = r["summary"], r["cond"]
    oth = np.mean([x["other"]["si_sdri"] for x in r["cands"]])
    j = judge.get(r["key"])
    print(f"{r['trial']} | {c['prompt_key']} | {c['anchor']} | {'y' if c['visual'] else 'n'} | {s['cand0_sdri']:+.1f} | "
          f"{s['best_sdri']:+.1f}@{s['best_idx']} | {s['mean_sdri']:+.1f}+-{s['std_sdri']:.1f} | {s['n_ge_3db']}/{s['n']} | "
          f"{s['frac_selection_correct']:.2f} | {oth:+.1f} | " + (f"{j['pick']}({j['pick_sdri']:+.1f})" if j else "-"))
