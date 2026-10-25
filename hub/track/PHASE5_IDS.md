# Source identity for Phase 5: mapping, expiry and abstention (PROPOSAL)

Status: proposal from workstream B (Sadie, `hub/track`), 2026-10-25. Not a decision: the Phase 5
integration owner (events/track and av_scene: Cody; anything on the wire: Lalitha) approves or
changes it, then the agreed parts move to `.agents/decisions.md`. Evidence so far is SIMULATED
only (scripted tracker inputs, pairsim scenes with oracle separation). See "Evidence" below.

## Three identity layers and one public id

| Layer | Example | Scope | Owner |
|---|---|---|---|
| slot / stream | `s1`, Phase 4 `SlotRef` | one separator window or chunk; may swap | separate, av_scene |
| acoustic source | `src_3` | one run (Phase 4 identity store) | av_scene |
| visual track | `sam3_image:...` | one run | vision |
| **public source** | `t7` | one hub process; never reused, survives clock restarts | track |

Rule 0. Only the tracker id (`t<n>`) is sent as `Source.source_id`. `src_<n>` and visual track ids
stay in the manifest and internal tables (spec 2.2: "a model's `obj_id`, a separator slot and a
public `source_id` are not interchangeable"). No protocol change is proposed here.

## Mapping (src -> tracker id)

1. **By provenance, not by timing, in the live hub.** Phase 5 should attach the acoustic id to
   each separated stream per window (stream, `src_n`, identity confidence), so the tracker gets
   it with every observation. Pairing ids afterwards by when they are active
   (`track.compare_ids`) is the fallback for recorded outputs only: it needs clean solo speech,
   is blurred by the tracker's ~2 s window and 1.5 s hold, and cannot pair two talkers who always
   overlap (they stay unpaired, by design).
2. **Create** a mapping `src_n -> t` once `src_n` has arrived on the stream that `t` matched for
   `confirm_hits` (3) consecutive observations with identity confidence at least "medium".
   Offline pairing: clean evidence >= 2 s, else lift >= 0.3; margin >= 0.2 over the runner-up.
3. **Identity vetoes merges.** Two tracks mapped to different `src_n` with confident identity are
   never folded together (`_confirm` duplicate check, `_merge_converged`, `_dup_of`), whatever
   their bearings. This is the spec's "same direction, different speakers" gate (13).
4. **Identity splits.** An observation whose `src_n` is confidently different from the track's
   mapped `src_n` does not update that track: it goes to the track mapped to its `src_n`, or
   starts a new one.
5. **Identity rejoins.** A returning `src_n` reattaches to its mapped track while that track is
   not forgotten, even if its bearing moved beyond `gate_deg` while it was silent.
6. **Without confident identity, today's rules apply** (world bearing + label + slot), including
   their known failures (below).

## Expiry

7. A mapping lives as long as its tracker id: `forget_s` (60 s) after the last observation the id
   is dropped and its mapping with it. A `src_n` that comes back later gets a NEW tracker id (ids
   are never reused; clients can treat a new id as a new source). The manifest keeps the
   `src_n` continuity for anyone who needs it.
8. An epoch change (headset reconnect, device clock restart: `Tracker.reset`) clears the table.
   No mapping, replay result or Phase 4 batch from an older epoch is applied (spec 2.2 / Phase 5
   acceptance: "disconnected/reconnected epochs cannot mix").
9. When Phase 4 merges two acoustic ids (`src_5` absorbed into `src_2`), the tracker ids mapped
   to them are merged the same way as `_merge_converged` does (the younger folds into the older);
   when it splits one, the split rule (4) applies from then on, never retroactively.
10. A Phase 4 / replay result that arrives after its tracker id was forgotten is dropped.

## Abstention

11. **No mapping** when the pairing is ambiguous (margin < 0.2), has no contrast (always
    overlapping talkers), or the identity confidence is below "medium". The tracker id stays
    unmapped; nothing is guessed.
12. **Abstain on position, not on existence.** A talker whose direction is unsure is still a
    source: announce it with `bearing_deg` None (or a low `bearing_conf`) rather than not at all.
    Today a reading below `min_bearing_conf` (0.25) counts as no direction, and a direction-less
    source is never announced while another speech source is active: in the simulated suite this
    hid a whole talker for the whole scene in 13 of 64 talker runs. With identity, rule 4 announces
    it; without, after a sustained separate stream (proposal: `bearingless_hits` without the
    "no other same-label source active" condition once the stream's energy is independent).
13. **Do not rely on bearing confidence to abstain yet.** In the suite, bearings more than 25 deg
    wrong carried confidence 0.26-0.37 (median 0.30, n=18) and bearings within 15 deg carried
    0.27-0.50 (median 0.36, n=21): the scales overlap. Until the locator's confidence separates them
    (the locate work's "honest confidence" target), a confidence threshold cannot tell a wrong
    direction from a right one, so identity rules must not depend on it.
14. **Key per-source direction state by identity, not by slot.** The locator keeps one posterior
    per stream slot. When the separator swaps slots without noticing, the old talker's bearing is
    carried onto the new one and the tracker ids follow the slots (suite "swap": 2 switches and 2
    merged ids in 2 seeds, bearing errors 38-60 deg). Phase 5 should reset or re-key a stream's
    locator state when its acoustic id changes.

## Evidence

All SIMULATED. Nothing here is replayed from a real recording or hardware-tested: no real takes
exist on the machine this was written on (or in git), and no real Phase 4 output exists yet
(PR #58 identity task T1 pending).

Scripted tracker inputs (`uv run python -m track`, `tests/track/test_track.py`; strict xfails):

| Probe | Should | Today |
|---|---|---|
| near_talkers 8 / 15 / 20 deg | two ids | 8 deg: one id (fold at `_confirm`, `merge_deg` 12); 15, 20: ok |
| pause_beyond_forget | new id after 60 s | ok (by design) |
| two_bearingless_births | both announced | neither until directions arrive (they block each other) |
| pause_while_crossing, return_near_other | the talker who kept talking keeps its id | ids swap (forced one-to-one assignment) |
| low_conf_second_talker | announced | never announced |
| unreliable_bearings (junk at conf 0.15) | ignored | ok |
| alternating_near_talkers (10 deg, one slot) | two ids | one id (needs identity) |

Locator + tracker on pairsim scenes (`uv run python -m track.offline sessions/offline --calib
sessions/pairsim-calib16.json --seeds 2`; oracle separation, 16 cases x 2 seeds, 64 talker runs):

| Case | Unannounced | Switches | Merged | Bearing err (median of medians) |
|---|---|---|---|---|
| base | 1 | 0 | 0 | 18.0 |
| behind | 1 | 0 | 0 | 9.2 |
| near10 / near20 / near30 | 1 / 1 / 2 | 0 | 0 / 1 / 0 | 6.2 / 6.2 / 4.0 |
| pause10 / pause30 / pause70 | 1 / 0 / 1 | 0 | 0 | 14.6 / 7.8 / 16.4 |
| swap / pause_swap | 0 / 0 | 2 / 4 | 2 / 2 | 38.0 / 39.0 |
| cross_slow / cross_fast / cross_pause | 1 / 1 / 0 | 0 / 1 / 0 | 0 | 24.9 / 22.0 / 34.8 |
| no_head | 1 | 1 | 0 | 9.3 (one talker 94 deg off: front/back) |
| pose_gap (10 s) | 1 | 0 | 0 | 22.7 (no worse than base for the same seeds) |
| calib_off (+5 us) | 1 | 0 | 0 | 71.6 |

Where a bearing was wrong, the locator's own raw error was about the same (the tracker neither
caused nor fixed it), except a fast walker (15 deg/s), where the tracker's smoothing added lag
(cross_fast seed 301: median 70 deg vs the locator's 45); identity failures came from the tracker rules above, from slot swaps reaching the
locator, and from two talkers the locator put within `merge_deg` (near20, seed 300). Static
talkers: median error 7.9 deg, 73% within 25 deg (26 talker runs).

Pairing by timing (`track.compare_ids` against an ORACLE Phase 4 stand-in, every case): 14 talker
runs with a single tracker id paired, all to the right id; 0 wrong; 36 unpaired or ambiguous (no
clean solo time and too little contrast: two talkers who keep talking); 1 paired where the
tracker itself had used several ids; plus the 13 talkers never announced. Timing alone is
safe but abstains too often to be the live mapping (rule 1).

Still lacking real evidence (every row): real 3S takes with labelled positions (Shayaan's
recordings folder), a real two-person scene, real separation instead of oracle stems (slot
swaps, leakage, label noise), real Phase 4 `src_n` output on the same recording, and the client's
view of id changes on the headset.
