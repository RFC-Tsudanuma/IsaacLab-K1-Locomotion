"""Numerical parity with the pinned C++ VisionFilter, without IsaacLab imports.

Golden values come from compiled original CVKF/selector sources, not a Python
reimplementation of their equations. See fixtures/vision_filter/README.md.
"""

import json
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
PURE_PACKAGE_PARENT = ROOT / "source/isaaclab_k1_locomotion/isaaclab_k1_locomotion"
sys.path.insert(0, str(PURE_PACKAGE_PARENT))
from direct_kick.vision_filter import HypothesisBank, VisionFilter  # noqa: E402


FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/vision_filter/cpp_oracle.json").read_text()
)


def inputs(records):
    return (
        torch.tensor([record["input"]["stamp_ns"] for record in records], dtype=torch.long),
        torch.tensor([record["input"]["global_xy"] for record in records], dtype=torch.float64),
        torch.tensor([record["input"]["local_xy"] for record in records], dtype=torch.float64),
        torch.tensor(
            [record["input"]["observed"] and record["input"]["transform_valid"] for record in records],
            dtype=torch.bool,
        ),
    )


def assert_double(actual, expected, *, context):
    torch.testing.assert_close(
        actual,
        torch.as_tensor(expected, dtype=torch.float64).reshape(actual.shape),
        rtol=1.0e-9,
        atol=1.0e-10,
        msg=context,
    )


def assert_public(filter_, records, ids):
    for row, env in zip(records, ids.tolist()):
        context = f"env={env}, stamp={row['input']['stamp_ns']}, event={row['event']}"
        expected = row["output"]
        assert filter_.status[env].item() == expected["status"], context
        assert filter_.updated[env].item() == expected["measurement_accepted"], context
        assert filter_.active[env].item() == row["confirmed_active"], context
        assert filter_.capture_count[env].item() == row["capture_count"], context
        assert filter_.stamp_ns[env].item() == row["input"]["stamp_ns"], context
        for attribute, key in (("state", "state"), ("covariance", "covariance")):
            actual = getattr(filter_, attribute)[env]
            assert actual.dtype == torch.float32
            torch.testing.assert_close(
                actual,
                torch.tensor(expected[key], dtype=torch.float32).reshape(actual.shape),
                rtol=1.0e-6,
                atol=1.0e-6,
                msg=context,
            )


def test_core_stationary_and_rolling_covariance_match_original_cpp():
    """The stationary model clamps velocity state while retaining uncertainty."""
    bank = HypothesisBank(1, "cpu")
    ids = torch.tensor([0])
    records = FIXTURE["core_analytic"]
    for step in range(2):
        sample = records[step]
        bank.step(ids, *inputs([sample]))
        for hypothesis, index in ((0, step), (1, step + 2)):
            output = records[index]["output"]
            assert_double(bank.x[0, hypothesis], output["state"], context=f"model={hypothesis}, step={step}")
            assert_double(bank.p[0, hypothesis], output["covariance"], context=f"model={hypothesis}, step={step}")
    assert bank.x[0, 0, 2:].count_nonzero().item() == 0
    assert bank.p[0, 0, 2, 2].item() > 0
    assert bank.p[0, 0, 0, 2].item() > 0


@pytest.mark.parametrize("hypothesis", [0, 1], ids=["stationary", "rolling"])
def test_individual_core_noise_missing_and_reinitialization_match_original_cpp(hypothesis):
    bank = HypothesisBank(1, "cpu")
    ids = torch.tensor([0])
    records = [row for row in FIXTURE["core"] if row["hypothesis"] == hypothesis]
    for step, row in enumerate(records):
        stamp, global_xy, local_xy, observed = inputs([row])
        if row["nis_before"] is not None:
            x, p = bank.prediction(ids, stamp)
            _, _, _, nis, _, _ = bank.innovation(x, p, global_xy, local_xy)
            assert_double(nis[0, hypothesis], row["nis_before"], context=f"core NIS model={hypothesis}, step={step}")
        bank.step(ids, stamp, global_xy, local_xy, observed)
        expected = row["output"]
        if expected["state"] is not None:
            assert_double(bank.x[0, hypothesis], expected["state"], context=f"core state model={hypothesis}, step={step}")
        assert_double(bank.p[0, hypothesis], expected["covariance"], context=f"core P model={hypothesis}, step={step}")


def test_four_hypothesis_map_and_bounce_reseed_match_original_cpp():
    bank = HypothesisBank(1, "cpu")
    ids = torch.tensor([0])
    selected_models = set()
    for step, row in enumerate(FIXTURE["bank"]):
        stamp, global_xy, local_xy, observed = inputs([row])
        expected = row["trace"]
        if step:
            x, p = bank.prediction(ids, stamp)
            _, _, _, nis, _, _ = bank.innovation(x, p, global_xy, local_xy)
            for hypothesis, value in enumerate(expected["nis"]):
                if value is not None:
                    assert_double(nis[0, hypothesis], value, context=f"NIS step={step}, model={hypothesis}")
        result = bank.step(ids, stamp, global_xy, local_xy, observed)
        selected = bank.probability[0].argmax().item()
        selected_models.add(selected)
        if selected != expected["selected"]:
            # The configured rolling/high_speed models have identical dynamics
            # and speed metadata does not gate them. Summation order can resolve
            # their tied posterior differently across Eigen/Torch. Accept that
            # label difference only after proving their states/P are equivalent.
            assert {selected, expected["selected"]} == {1, 2}, f"MAP step={step}"
            assert_double(bank.x[0, 1], bank.x[0, 2], context=f"tied model state step={step}")
            assert_double(bank.p[0, 1], bank.p[0, 2], context=f"tied model P step={step}")
        assert result.status[0].item() == row["output"]["status"]
        assert result.accepted[0].item() == row["output"]["measurement_accepted"]
        assert_double(result.state[0], row["output"]["state"], context=f"MAP state step={step}")
        assert_double(result.covariance[0], row["output"]["covariance"], context=f"MAP covariance step={step}")
        for hypothesis in range(4):
            state = expected["hypotheses"][hypothesis]["state"]
            covariance = expected["hypotheses"][hypothesis]["covariance"]
            if hypothesis == 3:
                state = expected["bounce_post_state"]
                covariance = expected["bounce_post_covariance"]
            assert_double(bank.x[0, hypothesis], state, context=f"state step={step}, model={hypothesis}")
            assert_double(bank.p[0, hypothesis], covariance, context=f"P step={step}, model={hypothesis}")
    assert {0, 3} <= selected_models
    assert selected_models & {1, 2}


def test_capture_missing_reacquisition_and_exact_timeout_match_original_cpp():
    filter_ = VisionFilter(1, "cpu")
    ids = torch.tensor([0])
    for row in FIXTURE["node"]:
        filter_.process(ids, *inputs([row]))
        assert_public(filter_, [row], ids)


@pytest.mark.parametrize("batch_together", [False, True], ids=["asynchronous_rows", "joint_batch"])
def test_environment_isolation_matches_interleaved_cpp_drivers(batch_together):
    filter_ = VisionFilter(3, "cpu")
    records = FIXTURE["interleaved"]
    width = 3 if batch_together else 1
    for start in range(0, len(records), width):
        rows = records[start : start + width]
        ids = torch.tensor([row["env"] for row in rows])
        # Untouched rows must preserve all state except the per-call updated pulse.
        inactive = torch.tensor([env for env in range(3) if env not in ids.tolist()], dtype=torch.long)
        snapshots = {
            name: getattr(filter_, name)[inactive].clone()
            for name in ("state", "covariance", "status", "stamp_ns", "capture_count", "active")
        }
        filter_.process(ids, *inputs(rows))
        assert_public(filter_, rows, ids)
        for name, previous in snapshots.items():
            assert torch.equal(getattr(filter_, name)[inactive], previous), f"other environment changed: {name}"
