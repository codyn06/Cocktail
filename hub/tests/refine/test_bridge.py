"""RefinementBridge: keyed admission, one job in flight, staleness. No models, no sockets."""

from __future__ import annotations

import threading
from dataclasses import replace

import numpy as np

from refine.bridge import RefinementBridge
from refine.contracts import (CutUnavailable, LiveView, RefineConfig, RefineJob, RefineKey, RefineOutcome,
                              RefineState)

NOW = 100_000_000  # us


class Clock:
    def __init__(self) -> None:
        self.t = 10.0

    def __call__(self) -> float:
        return self.t


class Env:
    """Mutable live view + fake cut/runner + captured post queue."""

    def __init__(self, *, runner="ok", executor="inline", defer=False, cfg=None):
        self.clock = Clock()
        self.v = LiveView("t3", 1, 1, 0, ("s1",), True, NOW)
        self.cuts = 0
        self.cut_exc: Exception | None = None
        self.outcome = RefineOutcome(RefineState.accepted, "ok", track_id="k1", score=9.0, assoc_conf=0.8)
        self.runs = 0
        self.posted: list = []
        self.capture_end = NOW - 500_000
        self.events: list = []

        def run(job):
            self.runs += 1
            return self.outcome

        r = run if runner == "ok" else runner
        self.bridge = RefinementBridge(
            live_fn=lambda: self.v, cut_fn=self.cut, runner=r, config=cfg or RefineConfig(),
            post=(self.posted.append if defer else None), executor=executor, mono=self.clock)
        self.bridge.listeners.append(self.events.append)

    def cut(self, v, job_id):
        self.cuts += 1
        if self.cut_exc:
            raise self.cut_exc
        key = RefineKey(v.chosen, v.epoch, v.selection_version, v.audio_resets)
        z = np.zeros(4, np.float32)
        return RefineJob(job_id, key, ("s1",), self.capture_end - 4_000_000, self.capture_end, z, {"s1": z}, [],
                         created_mono=self.clock.t)

    def tick(self, dt=0.0):
        self.clock.t += dt
        self.bridge.tick()

    def run_to_job(self):
        self.tick()
        self.tick(1.1)  # past settle_s


def test_flow_accepted_passes_through():
    e = Env()
    e.tick()
    assert e.bridge.current().state == RefineState.pending
    e.tick(1.1)
    c = e.bridge.current()
    assert (c.state, c.track_id, c.assoc_conf, c.score) == (RefineState.accepted, "k1", 0.8, 9.0)
    assert c.audio_applied is False and c.latency_ms == 0.0
    assert [s.state for s in e.events] == [RefineState.pending, RefineState.accepted]


def test_accepted_without_track_downgraded():
    e = Env()
    e.outcome = RefineOutcome(RefineState.accepted, "x", track_id=None, assoc_conf=0.9)
    e.run_to_job()
    c = e.bridge.current()
    assert c.state == RefineState.refused and c.reason == "accepted without track or confidence"
    assert c.track_id is None and c.assoc_conf == 0.0
    e = Env()
    e.outcome = RefineOutcome(RefineState.accepted, "x", track_id="k", assoc_conf=0.0)
    e.run_to_job()
    assert e.bridge.current().state == RefineState.refused


def test_refused_and_unavailable_drop_track():
    e = Env()
    e.outcome = RefineOutcome(RefineState.refused, "no match", track_id="k9", assoc_conf=0.7, score=3.0)
    e.run_to_job()
    c = e.bridge.current()
    assert (c.state, c.track_id, c.assoc_conf, c.score) == (RefineState.refused, None, 0.0, None)


def _stale_case(mutate, expect):
    e = Env(defer=True)
    e.run_to_job()
    assert len(e.posted) == 1
    pending = e.bridge.current()
    e.v = mutate(e.v)
    e.posted.pop()()
    assert e.bridge.current() is pending or e.bridge.current() == pending
    h = e.bridge.statuses()
    assert h[0].state == RefineState.stale and expect in h[0].reason
    assert e.bridge.current().state in (RefineState.pending,)
    assert e.bridge.status_payload()["in_flight"] is False


def test_stale_selection_version():
    _stale_case(lambda v: replace(v, selection_version=2), "selection version")


def test_stale_epoch():
    _stale_case(lambda v: replace(v, epoch=2), "epoch")


def test_stale_audio_resets():
    _stale_case(lambda v: replace(v, audio_resets=1), "audio reset")


def test_stale_route_moved():
    _stale_case(lambda v: replace(v, slots=("s2",)), "route")


def test_stale_source_gone():
    _stale_case(lambda v: replace(v, source_present=False), "source gone")


def test_stale_too_old():
    _stale_case(lambda v: replace(v, now_us=NOW + 13_000_000), "old")


def test_stale_future_capture():
    e = Env(defer=True)
    e.capture_end = NOW + 5_000_000
    e.run_to_job()
    e.posted.pop()()
    assert e.bridge.statuses()[0].state == RefineState.stale
    assert "future" in e.bridge.statuses()[0].reason
    assert e.bridge.current().state == RefineState.pending


def test_selection_change_between_completion_and_publish():
    e = Env(defer=True)
    e.run_to_job()
    before = e.bridge.current()
    e.v = replace(e.v, chosen="t5", selection_version=2)
    e.tick()  # new key -> new pending
    new_pending = e.bridge.current()
    assert new_pending.key.source_id == "t5" and new_pending is not before
    e.posted.pop()()  # late result of old key
    assert e.bridge.current() == new_pending
    assert e.bridge.statuses()[0].state == RefineState.stale


def test_selection_cleared_current_none_job_not_cancelled():
    e = Env(defer=True)
    e.run_to_job()
    e.v = replace(e.v, chosen=None)
    e.tick()
    assert e.bridge.current() is None
    e.posted.pop()()
    assert e.bridge.current() is None
    assert e.bridge.statuses()[0].state == RefineState.stale


def test_cut_unavailable_retries_after_one_second():
    e = Env()
    e.cut_exc = CutUnavailable("window too short")
    e.run_to_job()
    assert e.cuts == 1
    c = e.bridge.current()
    assert (c.state, c.reason) == (RefineState.unavailable, "window too short")
    e.tick(0.5)
    assert e.cuts == 1
    e.cut_exc = None
    e.tick(0.6)
    assert e.cuts == 2 and e.bridge.current().state == RefineState.accepted


def test_cut_other_exception():
    e = Env()
    e.cut_exc = ValueError("boom")
    e.run_to_job()
    assert e.bridge.current().reason == "cut error: ValueError"
    e.tick(0.3)
    assert e.cuts == 1


def test_runner_none():
    e = Env(runner=None)
    e.run_to_job()
    c = e.bridge.current()
    assert (c.state, c.reason) == (RefineState.unavailable, "refinement models not loaded")
    assert e.cuts == 0
    n = len(e.events)
    e.tick(0.1)
    assert len(e.events) == n


def test_runner_exception():
    def bad(job):
        raise RuntimeError("x" * 300)

    e = Env(runner=bad)
    e.run_to_job()
    c = e.bridge.current()
    assert c.state == RefineState.unavailable
    assert c.reason.startswith("runner error: RuntimeError: xxx") and len(c.reason) <= 20 + 24 + 160


def test_one_job_in_flight_no_queue():
    e = Env(defer=True)
    e.run_to_job()
    assert e.cuts == 1
    e.tick(5.0)
    e.tick(5.0)
    assert e.cuts == 1 and e.bridge.status_payload()["in_flight"] is True
    assert e.bridge.current().state == RefineState.pending
    e.posted.pop()()
    assert e.bridge.current().state == RefineState.accepted


def test_refresh_after_refresh_s():
    e = Env()
    e.run_to_job()
    assert e.cuts == 1
    e.tick(3.0)
    assert e.cuts == 1
    e.tick(1.1)
    assert e.cuts == 2 and e.runs == 2
    assert e.bridge.current().job_id == 2


def test_history_bounded_newest_first():
    e = Env(cfg=RefineConfig(history=3, refresh_s=0.1, settle_s=0.0))
    for _ in range(8):
        e.tick(0.2)
    h = e.bridge.statuses()
    assert len(h) == 3 and [s.job_id for s in h] == sorted((s.job_id for s in h), reverse=True)
    p = e.bridge.status_payload()
    assert p["counts"]["accepted"] == 3 and p["current"]["state"] == "accepted"
    assert p["config"]["history"] == 3 and len(p["history"]) == 3


def test_listeners_fire_on_change_only_and_swallow_errors():
    e = Env(runner=None)
    e.bridge.listeners.insert(0, lambda s: 1 / 0)
    e.run_to_job()
    e.tick(0.1)
    e.tick(0.1)
    assert [s.state for s in e.events] == [RefineState.pending, RefineState.unavailable]


def test_real_thread_worker():
    gate, done = threading.Event(), threading.Event()
    posted: list = []

    def runner(job):
        gate.wait(2.0)
        return RefineOutcome(RefineState.accepted, "ok", track_id="k", assoc_conf=0.5)

    e = Env(runner=runner, executor="thread")
    e.bridge._post = lambda fn: (posted.append(fn), done.set())
    try:
        e.run_to_job()
        assert e.bridge.status_payload()["in_flight"] is True
        e.tick(10.0)
        assert e.cuts == 1
        gate.set()
        assert done.wait(2.0)
        posted.pop()()
        assert e.bridge.current().state == RefineState.accepted
    finally:
        gate.set()
        e.bridge.close()


def test_listeners_get_none_on_deselect_and_on_key_change():
    e = Env(defer=True)
    e.tick()
    e.v = replace(e.v, selection_version=2)  # key changes while pending
    e.tick()
    assert [s.state if s else None for s in e.events] == [RefineState.pending, None, RefineState.pending]
    e.v = replace(e.v, chosen=None)
    e.tick()
    assert e.events[-1] is None
    e.tick()
    assert len(e.events) == 4  # no repeated None when already empty


def test_job_cut_before_window_floor_is_stale_at_admission():
    e = Env(defer=True)
    e.run_to_job()
    e.v = replace(e.v, window_floor_us=e.capture_end - 1_000_000)  # route/selection changed inside the window
    e.posted.pop()()
    st = e.bridge.statuses()[0]
    assert st.state == RefineState.stale and st.reason == "route or selection changed inside the window"
