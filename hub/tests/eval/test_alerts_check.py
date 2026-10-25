"""Tests for eval/alerts_check.py and eval/alerts_rules.json.

Lives under hub/tests because that is where pytest runs. Label timelines are hand-made
(evidence label: simulated); the real AST labeler (--wav) is not exercised here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parents[3] / "eval"
sys.path.insert(0, str(EVAL))
import alerts_check as ac  # noqa: E402

RULES = EVAL / "alerts_rules.json"


def obs(rows):
    return [ac.Obs(t, lab, s, b) for t, lab, s, *rest in rows for b in (rest[0] if rest else None,)]


def fires(rows, transcript=(), names=()):
    return ac.evaluate(ac.load_rules(RULES, list(names)), obs(rows), list(transcript))


def ids(fs):
    return [(f.rule, f.t_s) for f in fs]


def test_shipped_rules_load_and_cover_the_three_alerts():
    r = ac.load_rules(RULES)
    assert [x["id"] for x in r.rules] == ["smoke_alarm", "siren", "wearer_name"]
    smoke = r.rules[0]
    assert "Smoke detector, smoke alarm" in smoke["classes"] and smoke["labels"] == ["alarm"]
    assert r.rules[1]["labels"] == ["siren"] and "Siren" in r.rules[1]["classes"]
    assert r.rules[2]["keywords"] == []  # the wearer's name is not invented


def test_smoke_alarm_fires_when_the_event_lasts_min_duration():
    # hops at t=3 and t=4 are active: 2 s of alarm, fire at the end of hop 4 (t=5)
    assert ids(fires([(3, "alarm", 0.9), (4, "alarm", 0.8)])) == [("smoke_alarm", 5.0)]


FINE = "Smoke detector, smoke alarm"


def test_exact_classes_raise_the_critical_smoke_alarm():
    f = fires([(3, FINE, 0.9), (4, FINE, 0.8)])
    assert len(f) == 1 and f[0].critical and f[0].spoken_text == "Smoke alarm"


def test_coarse_alarm_alone_is_downgraded_not_critical():
    """The coarse 'alarm' also holds 'Alarm' and 'Buzzer': without the exact classes it must not read as a smoke alarm."""
    f = fires([(3, "alarm", 0.9), (4, "alarm", 0.8)])
    assert len(f) == 1 and not f[0].critical and f[0].spoken_text == "Alarm sound"
    assert f[0].as_alert()["critical"] is False


def test_door_buzzer_with_exact_class_rows_present_raises_nothing():
    """Input that carries the exact classes (as --wav does): coarse 'alarm' high, but the smoke classes are low."""
    rows = [(t, lab, sc) for t in range(6) for lab, sc in (("alarm", 0.9), (FINE, 0.05), ("Fire alarm", 0.04))]
    assert fires(rows) == []


def test_siren_is_unaffected_by_the_fallback_logic():
    f = fires([(0, "siren", 0.8), (1, "siren", 0.8)])
    assert len(f) == 1 and f[0].critical and f[0].spoken_text == "Siren"


def test_coarse_fallback_needs_classes_and_labels(tmp_path):
    bad = tmp_path / "r.json"
    rule = {"id": "x", "label": "x", "critical": True, "spoken_text": "X", "detector": "sound_label", "classes": ["A"],
            "min_conf": 0.5, "min_duration_s": 1, "max_gap_s": 1, "cooldown_s": 1, "coarse_fallback": {"spoken_text": "y"}}
    bad.write_text(json.dumps({"schema": "cocktail-alert-rules/1", "rules": [rule]}))
    with pytest.raises(ValueError, match="coarse_fallback"):
        ac.load_rules(bad)


def test_one_hop_of_alarm_is_not_enough():
    assert fires([(3, "alarm", 0.99), (4, "alarm", 0.2)]) == []


def test_score_below_min_conf_never_fires():
    assert fires([(t, "alarm", 0.49) for t in range(10)]) == []


def test_short_gap_keeps_the_event_together_long_gap_breaks_it():
    # beeping alarm: active, 1 s silence, active -> one event of 3 s
    assert ids(fires([(0, "alarm", 0.9), (2, "alarm", 0.9)])) == [("smoke_alarm", 3.0)]
    # 3 s of silence between the hops (> max_gap_s 1.5): two separate 1 s events, no fire
    assert fires([(0, "alarm", 0.9), (4, "alarm", 0.9)]) == []


def test_cooldown_blocks_a_second_fire_then_a_long_event_fires_again():
    rows = [(t, "alarm", 0.9) for t in range(0, 70)]
    assert ids(fires(rows)) == [("smoke_alarm", 2.0), ("smoke_alarm", 32.0), ("smoke_alarm", 62.0)]


def test_separate_event_inside_cooldown_is_suppressed():
    rows = [(0, "alarm", 0.9), (1, "alarm", 0.9), (10, "alarm", 0.9), (11, "alarm", 0.9)]
    assert ids(fires(rows)) == [("smoke_alarm", 2.0)]


def test_exact_audioset_class_name_matches_case_insensitively():
    rows = [(0, "smoke detector, smoke alarm", 0.7), (1, "Fire alarm", 0.6)]
    assert ids(fires(rows)) == [("smoke_alarm", 2.0)]


def test_best_matching_score_per_hop_is_used():
    rows = [(0, "Smoke detector, smoke alarm", 0.2), (0, "Fire alarm", 0.9), (1, "Smoke detector, smoke alarm", 0.9)]
    assert ids(fires(rows)) == [("smoke_alarm", 2.0)]


def test_siren_and_smoke_alarm_are_independent_and_bearing_is_kept():
    rows = [(0, "siren", 0.8, -40.0), (1, "siren", 0.9, -35.0), (0, "alarm", 0.9), (1, "alarm", 0.9)]
    fs = fires(rows)
    assert {f.rule for f in fs} == {"siren", "smoke_alarm"}
    siren = next(f for f in fs if f.rule == "siren")
    assert siren.bearing_deg == -35.0 and siren.critical and siren.spoken_text == "Siren"
    alert = siren.as_alert()
    assert alert["type"] == "alert.fired" and alert["label"] == "siren" and alert["critical"] is True


def test_other_labels_do_not_trigger_anything():
    rows = [(t, lab, 0.99) for t in range(5) for lab in ("speech", "music", "other")]
    assert fires(rows) == []


def test_name_rule_needs_a_name_and_matches_whole_words():
    talk = [(5.0, "Hey Sadie, dinner is ready", 30.0), (6.0, "sadly nobody came", None)]
    assert fires([], talk) == []  # no keywords set: cannot fire
    fs = fires([], talk, names=["Sadie"])
    assert ids(fs) == [("wearer_name", 5.0)] and fs[0].bearing_deg == 30.0 and not fs[0].critical


def test_name_matching_ignores_case_accents_and_punctuation_and_respects_cooldown():
    talk = [(1.0, "SÁDIE!", None), (5.0, "sadie?", None), (12.0, "ok sadie", None)]
    assert ids(fires([], talk, names=["Sadie"])) == [("wearer_name", 1.0), ("wearer_name", 12.0)]


def test_multiword_keyword():
    talk = [(1.0, "hey there sadie jones", None)]
    assert ids(fires([], talk, names=["sadie jones"])) == [("wearer_name", 1.0)]


def test_bad_rules_are_rejected(tmp_path):
    bad = tmp_path / "r.json"
    bad.write_text(json.dumps({"schema": "cocktail-alert-rules/1", "rules": [{"id": "x", "detector": "sound_label"}]}))
    with pytest.raises(ValueError, match="missing"):
        ac.load_rules(bad)
    bad.write_text(json.dumps({"schema": "nope", "rules": []}))
    with pytest.raises(ValueError, match="schema"):
        ac.load_rules(bad)


def test_cli_with_label_and_transcript_files_and_expectation(tmp_path, capsys):
    labels = tmp_path / "labels.csv"
    labels.write_text("# t_s,label,score\nt_s,label,score,bearing_deg\n3,alarm,0.9,20\n4,alarm,0.9,20\n3,speech,0.8,\n")
    talk = tmp_path / "talk.csv"
    talk.write_text("t_s,text\n10,hello sadie\n")
    out = tmp_path / "fired.json"
    rc = ac.main(["--labels", str(labels), "--transcript", str(talk), "--name", "sadie",
                  "--expect", "smoke_alarm,wearer_name", "--json", str(out)])
    text = capsys.readouterr().out
    # the file has coarse 'alarm' rows only: the smoke alarm fires, but downgraded (not CRITICAL)
    assert rc == 0 and "PASS" in text and "CRITICAL" not in text and "bearing +20" in text and "Alarm sound" in text
    assert [a["rule"] for a in json.loads(out.read_text())] == ["smoke_alarm", "wearer_name"]
    assert ac.main(["--labels", str(labels), "--expect", "none"]) == 1


def test_cli_quiet_room_expects_none(tmp_path, capsys):
    labels = tmp_path / "quiet.csv"
    labels.write_text("t_s,label,score\n" + "".join(f"{t},other,0.9\n" for t in range(6)))
    assert ac.main(["--labels", str(labels), "--expect", "none"]) == 0


def test_cli_malformed_input_is_an_error_not_a_crash(tmp_path, capsys):
    labels = tmp_path / "bad.csv"
    labels.write_text("t_s,label,score\nsoon,alarm,high\n")
    assert ac.main(["--labels", str(labels)]) == 2
    assert "error" in capsys.readouterr().out
