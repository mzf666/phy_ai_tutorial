"""Alignment checks for pi0.6* inference and evaluation: the 14-dim static spec and camera adapter, the infer contract
and its agreement with ../backbone, the subtask schedule, the CFG identities (beta = 1 is the conditional model; the
guided velocity is the linear combination of Eq. 13), the task table / metric definitions, the time-limit loop, and the
hand-off of evaluation episodes to RECAP labels."""

import dataclasses

import numpy as np
import pytest
import torch

from pi.pi06.backbone.model import NUM_DENOISING_STEPS, AdaRMSNorm
from pi.pi06.data.data import ACTION_DIM, STATIC_IMAGE_KEYS, build_pi06_batch
from pi.pi06.infer.eval import (
    EPISODES_PER_ITERATION,
    TASKS,
    EpisodeRecord,
    StaticToyEnv,
    evaluate,
    max_episode_len,
    run_episode,
    stage_success,
    standard_error,
    success_from_quality,
    success_rate,
    task,
    throughput_per_hour,
)
from pi.pi06.infer.model import CFG_BETA_RANGE, CONTROL_HZ, LATENCY_H100_MS, STATIC_CAMERAS, STATIC_DIMS_14, guided_velocity_fn, static_delta_mask, static_obs_to_raw, static_spec, tiny_pi06_policy


@pytest.fixture(scope="module")
def policy():
    return tiny_pi06_policy(max_new_tokens=3)


@pytest.fixture(scope="module")
def raw():
    rng = np.random.default_rng(0)
    obs = {name: rng.integers(0, 256, (64, 80, 3), dtype=np.uint8) for name in STATIC_CAMERAS}
    obs["state"] = rng.uniform(-0.5, 0.5, 14).astype(np.float32)
    return static_obs_to_raw(obs, "make me an espresso")


def test_static_spec_and_adapter(raw):
    assert len(STATIC_DIMS_14) == 14 and STATIC_DIMS_14.count("left_gripper") == 1 and STATIC_DIMS_14.count("right_joint") == 6  # Fig. 5: 2 x (6 + 1)
    assert static_delta_mask() == (True,) * 6 + (False,) + (True,) * 6 + (False,)
    spec = static_spec()
    assert spec.native_dim == 14 and spec.control_hz == CONTROL_HZ == 50 and spec.action_horizon == 50 and spec.chunk_seconds == 1.0
    assert list(raw["images"]) == list(STATIC_IMAGE_KEYS) and raw["state"].shape == (1, 14) and raw["prompt"] == ["make me an espresso"]
    assert LATENCY_H100_MS == 63.0 and CFG_BETA_RANGE == (1.5, 2.5)


def test_infer_contract_matches_backbone_and_schedule(policy, raw):
    pol, robot = policy
    noise = torch.randn(1, 50, ACTION_DIM)
    out = pol.infer(raw, t_now=0.0, noise=noise)
    assert out["actions"].shape == (1, 50, 14) and out["x_0"].shape == (1, 50, ACTION_DIM) and out["hl_ran"]
    # same numbers as calling the model directly with the conditional flow layout
    obs, _ = build_pi06_batch(raw, robot.norm_stats, pol.seq, layout="flow", image_keys=STATIC_IMAGE_KEYS, action_horizon=50, delta_mask=robot.delta_mask, train=False,
                              subtasks=[out["subtask"]], advantages=[True])
    x0 = pol.model.sample_actions(obs, noise, NUM_DENOISING_STEPS)
    assert torch.allclose(x0, out["x_0"], atol=1e-5)
    # grippers absolute, joints delta: the executable action equals x_0 (identity stats) plus the state on joint dims
    a = out["actions"][0]
    x = out["x_0"][0, :, :14].numpy()
    st = raw["state"][0]
    assert np.allclose(a[:, [6, 13]], x[:, [6, 13]], atol=1e-5) and np.allclose(a[:, :6], x[:, :6] + st[:6], atol=1e-5)
    # subtask schedule: no new decode before hl_period_s elapsed
    out2 = pol.infer(raw, t_now=0.5, noise=noise)
    assert not out2["hl_ran"] and out2["subtask"] == out["subtask"] and np.allclose(out2["actions"], out["actions"])
    assert pol.infer(raw, t_now=1.0, noise=noise)["hl_ran"]
    assert {"data preprocessing", "observation forward pass", "inverse transforms", "total"} <= set(out["timing"])


def test_cfg_identities(policy, raw):
    pol, _ = policy
    vc = lambda x, t: x * 2.0
    vu = lambda x, t: x * 0.5
    x = torch.randn(1, 5, 32)
    assert guided_velocity_fn(vc, vu, 1.0) is vc  # beta = 1: the conditional model (Sec. IV-B, V-D)
    g = guided_velocity_fn(vc, vu, 2.5)(x, None)
    assert torch.allclose(g, 0.5 * x + 2.5 * (2.0 * x - 0.5 * x))  # Eq. 13 on the velocity fields
    for m in pol.model.modules():  # make the expert read the prefix (zero-init gates ignore it)
        if isinstance(m, AdaRMSNorm):
            torch.nn.init.normal_(m.modulation.weight, std=0.05)
    noise = torch.randn(1, 50, ACTION_DIM)
    pol.beta = 1.0
    ref = pol.infer(raw, t_now=0.0, noise=noise)
    pol.beta = 2.0
    out = pol.infer(raw, t_now=0.0, noise=noise)
    pol.beta = 1.0
    assert not torch.allclose(ref["x_0"], out["x_0"]) and "x2" in " ".join(out["timing"])


# ---------------------------------------------------------------- evaluation
def test_task_table_and_metrics():
    limits = {t.name: t.time_limit_s for t in TASKS}
    assert limits == {"laundry (t-shirts and shorts)": 200, "laundry (diverse items)": 500, "laundry (targeted failure removal)": 200, "cafe (double shot espresso)": 200, "box assembly": 600}
    assert task("box assembly").stages == ("pick up a box sheet", "build the box", "label the box", "place it in the crate")
    assert max_episode_len(task("box assembly")) == 600 * 50
    assert EPISODES_PER_ITERATION["laundry (t-shirts and shorts)"] == {"autonomous": 300, "corrections": 0, "robots": 4}
    assert EPISODES_PER_ITERATION["box assembly"]["autonomous"] == 600 and EPISODES_PER_ITERATION["box assembly"]["corrections"] == 360
    r = lambda ok, dur, st=4: EpisodeRecord("box assembly", 0, int(dur * 50), dur, not ok, {"a": ok}, ok, st, 1, 1, [])
    recs = [r(True, 300.0), r(False, 600.0, 2), r(True, 300.0), r(True, 600.0)]
    assert throughput_per_hour(recs) == pytest.approx(3 / (1800 / 3600))  # 3 successes in 0.5 h, the failure's time counted
    assert success_rate(recs) == 0.75 and standard_error([1, 0, 1, 1]) == pytest.approx(np.std([1, 0, 1, 1], ddof=1) / 2)
    assert stage_success(recs) == [1.0, 1.0, 0.75, 0.75]
    assert success_from_quality({"done": True, "clean": True}) and not success_from_quality({"done": True, "clean": False}) and not success_from_quality({})


def test_episode_loop_time_limit_and_labels(policy):
    pol, _ = policy
    env = StaticToyEnv()
    t = dataclasses.replace(task("box assembly"), time_limit_s=0.4)  # 20 steps: forces a timeout with need >= 3 per stage x 4 stages
    rec = run_episode(pol, env, t, 0, execute_steps=10)
    assert rec.steps == 20 and rec.timed_out and not rec.success and rec.duration_s == pytest.approx(0.4) and rec.infer_calls == 2 and rec.hl_calls == 1
    lab = rec.to_labels(max_episode_len(t))
    assert lab.num_steps == 20 and lab.max_episode_len == 20 and lab.success is False
    res = evaluate(pol, env, (dataclasses.replace(task("laundry (t-shirts and shorts)"), time_limit_s=2.0),), trials=3, execute_steps=25)
    e = res["laundry (t-shirts and shorts)"]
    assert len(e["records"]) == 3 and len(e["labels"]) == 3 and 0 <= e["success_rate"] <= 1 and e["throughput_per_hour"] >= 0 and "stage_success" not in e
