"""Phase 5 identity: a refinement/identity result from an old epoch or an old selection is stale.

Uses the bridge fakes of test_bridge.py; mocked, no models."""

from dataclasses import replace

from test_bridge import Env

from refine.contracts import RefineState


def test_old_epoch_and_old_selection_results_are_stale():
    for field, value, reason in (("epoch", 2, "epoch"), ("selection_version", 2, "selection version")):
        e = Env(defer=True)
        e.run_to_job()
        assert len(e.posted) == 1
        e.v = replace(e.v, **{field: value})  # the epoch/selection moved on while the job ran
        e.posted.pop()()
        newest = e.bridge.statuses()[0]
        assert newest.state == RefineState.stale and reason in newest.reason, (field, newest)
        assert e.bridge.status_payload()["in_flight"] is False
