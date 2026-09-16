"""Alignment checks for pi0.5 inference and evaluation: the infer contract and the zero-chunk inverse, agreement with
../hier, the 18 / 19-dim spec and camera adapter, the rubric / language-following constants and aggregation, the
two-level loop's call counts, and the Hi Robot metric definitions."""

import math

import numpy as np
import pytest
import torch

from pi.pi05.data.data import ACTION_DIM, HL_IMAGE_KEYS, LL_IMAGE_KEYS, build_pi05_batch
from pi.pi05.infer.eval import (
    HIROBOT_TASKS,
    HIROBOT_TRIALS_PER_TASK,
    LANGUAGE_FOLLOWING,
    LANGUAGE_FOLLOWING_OOD,
    LOCATIONS,
    MOCK_HOME_TASKS,
    STANDARD_EVALS_PER_POLICY,
    TRIALS_PER_TASK,
    MockHomeToyEnv,
    evaluate_language_following,
    evaluate_mock_home,
    instruction_accuracy,
    run_hier_episode,
    task_progress,
)
from pi.pi05.infer.model import MOBILE_CAMERAS, MOBILE_DIMS_18, MOBILE_DIMS_19, mobile_delta_mask, mobile_obs_to_raw, mobile_spec, tiny_pi05_policy, to_executable_actions


@pytest.fixture(scope="module")
def policy():
    return tiny_pi05_policy(19)


@pytest.fixture(scope="module")
def hier_policy():
    return tiny_pi05_policy(19, hier=True, max_new_tokens=3)


@pytest.fixture(scope="module")
def obs():
    rng = np.random.default_rng(7)
    o = {name: rng.integers(0, 256, (48, 64, 3), dtype=np.uint8) for name in MOBILE_CAMERAS}
    o["state"] = rng.uniform(-0.5, 0.5, 19).astype(np.float32)
    return o


# ---------------------------------------------------------------- infer contract and inverse transforms
def test_infer_contract_and_zero_chunk_inverse(policy, obs):
    pol, robot = policy
    raw = mobile_obs_to_raw(obs, "pick up the plate", hl=False)
    noise = torch.randn(1, robot.action_horizon, ACTION_DIM)
    out = pol.infer(raw, noise=noise)
    assert out["actions"].shape == (1, robot.action_horizon, 19) and out["x_0"].shape == (1, robot.action_horizon, ACTION_DIM)
    assert set(out["timing"]) == {"data preprocessing", "image encoders", "observation forward pass", "x10 action forward pass (flow)", "inverse transforms", "total"}
    # a zero chunk decodes to: delta dims = the current state, absolute dims = the unnormalized 0 (= 0 with identity stats)
    zero = to_executable_actions(torch.zeros(1, robot.action_horizon, ACTION_DIM), raw["state"], robot)
    mask = np.array(robot.delta_mask)
    assert np.allclose(zero[0][:, mask], raw["state"][0][mask]) and np.allclose(zero[0][:, ~mask], 0.0, atol=1e-5)  # 5e-7 from the quantile eps
    # agrees with ../hier's sample_actions on the same observation
    o, _ = build_pi05_batch(raw, robot.norm_stats, pol.seq, layout="flow", image_keys=LL_IMAGE_KEYS, action_horizon=robot.action_horizon, delta_mask=robot.delta_mask, train=False)
    with torch.no_grad():
        x0 = pol.model.sample_actions(o, noise)
    assert torch.allclose(x0, out["x_0"], atol=1e-5)


def test_mobile_spec_and_adapter(obs):
    assert len(MOBILE_DIMS_18) == 18 and len(MOBILE_DIMS_19) == 19 and MOBILE_DIMS_19[:18] == MOBILE_DIMS_18
    dm = mobile_delta_mask(MOBILE_DIMS_19)
    assert dm[:6] == (True,) * 6 and dm[6] is False and dm[13] is False and dm[14:17] == (False,) * 3 and dm[17:] == (True, True)
    spec = mobile_spec(18)
    assert spec.native_dim == 18 and spec.chunk_seconds == 1.0  # 50 steps at 50 Hz
    hl, ll = mobile_obs_to_raw(obs, "clean", hl=True), mobile_obs_to_raw(obs, "clean", hl=False)
    assert tuple(hl["images"]) == HL_IMAGE_KEYS and tuple(ll["images"]) == LL_IMAGE_KEYS and "base_1_rgb" not in ll["images"]
    assert hl["state"].shape == (1, 19) and hl["images"]["base_0_rgb"].shape == (1, 48, 64, 3)


# ---------------------------------------------------------------- evaluation constants (paper Appendix B, C; Hi Robot Sec. 5)
def test_rubric_and_protocol_constants():
    assert [m for _, _, m, _ in MOCK_HOME_TASKS] == [8, 4, 3, 5]
    assert all(len(r) <= m for _, _, m, r in MOCK_HOME_TASKS)
    assert TRIALS_PER_TASK == 10 and len(LOCATIONS) == 12 and STANDARD_EVALS_PER_POLICY == 4 * TRIALS_PER_TASK
    assert len(LANGUAGE_FOLLOWING) == 2 and all(len(v) == 5 for v in LANGUAGE_FOLLOWING.values()) and len(LANGUAGE_FOLLOWING_OOD) == 5
    assert len(HIROBOT_TASKS) == 3 and HIROBOT_TRIALS_PER_TASK == 20
    assert instruction_accuracy([True, False, True, True]) == 0.75 and instruction_accuracy([]) == 0.0
    assert task_progress(2, 4) == 0.5


# ---------------------------------------------------------------- the loop
def test_two_level_loop_call_counts(hier_policy):
    pol, _ = hier_policy
    env = MockHomeToyEnv(seed=1)
    env.need = 10**6  # never earn a point: the episode runs to max_steps
    orig_reset = env.reset

    def reset(task, i):
        o = orig_reset(task, i)
        env.need = 10**6
        return o

    env.reset = reset
    T, k = 120, 25
    ep = run_hier_episode(pol, env, "Make Bed", 0, max_steps=T, execute_steps=k, user_messages={30: "no, the other pillow"})
    assert ep.steps == T and ep.score == 0.0 and ep.max_points == 5
    # low level: a call at every plan exhaustion plus the message; high level: t=0, every 50 steps (1 s at 50 Hz) when a plan is needed, plus the message
    assert ep.ll_calls == math.ceil(T / k) + 1  # 0, 25, 30 (message), 55, 80, 105 -> 6
    assert ep.hl_calls == 3  # 0, 30 (message), 80 (>= 1 s since 30 at a replan)
    assert len(ep.subtasks) == ep.hl_calls


def test_aggregation(hier_policy):
    pol, _ = hier_policy
    env = MockHomeToyEnv(seed=2)
    res = evaluate_mock_home(pol, env, trials_per_task=2, max_steps=20, execute_steps=10)
    assert res["n_trials"] == 8 and set(res["per_task"]) == {t for t, *_ in MOCK_HOME_TASKS}
    assert abs(res["task_progress"] - np.mean(list(res["per_task"].values()))) < 1e-9  # mean over tasks, not over trials
    for e in res["episodes"]:
        assert e.score == e.points / e.max_points
    lf = evaluate_language_following(pol, env, max_steps=12, execute_steps=6)
    assert len(lf["rows"]) == 10 and lf["chance"] == 0.2 and set(lf["per_scenario"]) == set(LANGUAGE_FOLLOWING)
