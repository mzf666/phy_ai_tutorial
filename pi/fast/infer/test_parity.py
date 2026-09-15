"""Alignment checks for pi0-FAST inference and evaluation: output contract, the inverse transform chain on a known
token sequence (analytic inverse), the DROID adapter, Table II bookkeeping, replanning rhythm, timing keys, and the
rubric aggregation. No numbers from the paper are reproduced."""

import numpy as np
import pytest
import torch

from pi.fast.data.data import paligemma_to_fast
from pi.fast.infer.eval import DROID_TASKS, DROID_TOTAL_TRIALS, DROIDToyEnv, evaluate_rubric, libero_obs_to_raw_fast, run_episode
from pi.fast.infer.model import (
    DROID_ACTION_DIM,
    DROID_ACTION_HORIZON_PAPER,
    DROID_EXECUTE_STEPS,
    DROID_STATE_DIM,
    droid_obs_to_raw,
    fast_image_slots,
    tiny_fast_policy,
    to_executable_actions,
)
from pi.fast.tokenizer.tokenizer import normalize_quantile, unnormalize_quantile
from pi.pi0.data.data import to_absolute_actions


@pytest.fixture(scope="module")
def policy():
    return tiny_fast_policy(max_decoding_steps=12)[0]


@pytest.fixture(scope="module")
def raw():
    rng = np.random.default_rng(0)
    obs = {"observation/exterior_image_1_left": rng.integers(0, 256, (180, 320, 3), dtype=np.uint8),
           "observation/wrist_image_left": rng.integers(0, 256, (180, 320, 3), dtype=np.uint8),
           "observation/joint_position": rng.uniform(-0.5, 0.5, 7).astype(np.float32),
           "observation/gripper_position": np.float32(0.3)}
    return droid_obs_to_raw(obs, "Put_the marker in the cup")


# ---------------------------------------------------------------- contract
def test_infer_output_contract(policy, raw):
    out = policy.infer(raw)
    H, d = policy.robot.action_horizon, policy.robot.native_dim
    assert out["actions"].shape == (1, H, d) and out["actions"].dtype == np.float32
    assert out["tokens"].shape == (1, policy.max_decoding_steps) and out["tokens"].dtype == torch.long
    assert 1 <= out["n_steps"] <= policy.max_decoding_steps
    assert set(out["timing"]) == {"data preprocessing", "sample_actions (prefill + decode steps)", "extract + inverse transforms", "total", "ms per decode step"}
    assert out["timing"]["total"] == pytest.approx(sum(v for k, v in out["timing"].items() if k not in ("total", "ms per decode step")))
    # random weights: no 'Action: ' marker -> zeros in normalized space -> delta dims fall back to the current state
    st = raw["state"][0]
    zeros_abs = to_absolute_actions(raw["state"], unnormalize_quantile(np.zeros((1, H, d), np.float32), policy.robot.norm_stats["actions"]), policy.robot.delta_mask)
    assert np.allclose(out["actions"], zeros_abs, atol=1e-6)
    assert np.allclose(out["actions"][0, :, :7], st[:7], atol=1e-5)


def test_inverse_chain_matches_analytic_inverse_on_a_known_postfix(policy, raw):
    """tokenize(chunk) -> the postfix a trained model would emit -> to_executable_actions == unnormalize + absolute,
    up to the FAST rounding bound 0.5/gamma * sqrt(H) in normalized units (times (q99-q01)/2 after unnormalization)."""
    rng = np.random.default_rng(1)
    robot, seq = policy.robot, policy.seq
    H, d = robot.action_horizon, robot.native_dim
    chunk = np.clip(0.3 * rng.standard_normal((H, d)), -0.9, 0.9).astype(np.float32)
    st = normalize_quantile(raw["state"][0], robot.norm_stats["state"])
    toks, m, ar, _ = seq.tokenize("put the marker in the cup", st, chunk)
    postfix = toks[m & (ar == 1)]
    acts = to_executable_actions(postfix[None], raw["state"], robot, seq)
    ref = to_absolute_actions(raw["state"], unnormalize_quantile(chunk[None], robot.norm_stats["actions"]), robot.delta_mask)
    scale = (robot.norm_stats["actions"].q99 - robot.norm_stats["actions"].q01) / 2
    assert acts.shape == (1, H, d)
    assert np.abs(acts - ref).max() <= 0.5 / 10 * np.sqrt(H) * scale.max() + 1e-6
    fast_ids = paligemma_to_fast(postfix[len(seq._action_marker) : -2])
    assert 0 <= fast_ids.min() and fast_ids.max() < seq.fast.vocab_size
    # the whole generated buffer with pads after EOS decodes the same
    padded = np.concatenate([postfix, np.zeros(50, np.int64)])
    assert np.allclose(to_executable_actions(padded[None], raw["state"], robot, seq), acts)


# ---------------------------------------------------------------- DROID adapter
def test_droid_adapter(raw):
    assert set(raw["images"]) == {"base_0_rgb", "wrist_0_rgb"}  # base_1_rgb is filled black (mask True) by build_fast_batch
    assert raw["state"].shape == (1, DROID_STATE_DIM) and raw["state"][0, 7] == pytest.approx(0.3)  # 7 joints then gripper
    assert raw["prompt"] == ["Put_the marker in the cup"]  # lowercasing / '_' handling belongs to the tokenizer
    # scalar and [1] gripper both work
    obs = {"observation/exterior_image_1_left": np.zeros((8, 8, 3), np.uint8), "observation/wrist_image_left": np.zeros((8, 8, 3), np.uint8),
           "observation/joint_position": np.arange(7, dtype=np.float32), "observation/gripper_position": np.array([0.5], np.float32)}
    assert droid_obs_to_raw(obs, "x")["state"].tolist() == [[0, 1, 2, 3, 4, 5, 6, 0.5]]
    assert DROID_ACTION_DIM == 8 and DROID_ACTION_HORIZON_PAPER == 15 and DROID_EXECUTE_STEPS == (8, 15)
    assert fast_image_slots({"images": {"base_0_rgb": 1, "left_wrist_0_rgb": 2}, "state": 0})["images"] == {"base_0_rgb": 1, "wrist_0_rgb": 2}


def test_libero_adapter_uses_fast_slot_names():
    obs = {"agentview_image": np.zeros((16, 16, 3), np.uint8), "robot0_eye_in_hand_image": np.zeros((16, 16, 3), np.uint8),
           "robot0_eef_pos": np.zeros(3, np.float32), "robot0_eef_quat": np.array([0, 0, 0, 1], np.float32), "robot0_gripper_qpos": np.zeros(2, np.float32)}
    r = libero_obs_to_raw_fast(obs, "close the drawer")
    assert set(r["images"]) == {"base_0_rgb", "wrist_0_rgb"} and r["state"].shape == (1, 8)


# ---------------------------------------------------------------- evaluation bookkeeping
def test_table_ii_bookkeeping():
    assert len(DROID_TASKS) == 17 and sum(n for _, n in DROID_TASKS) == DROID_TOTAL_TRIALS == 44  # 17 rows vs "16 tasks" in the text


def test_replanning_rhythm_and_rubric_aggregation(policy):
    env = DROIDToyEnv(seed=0)
    H, k = policy.robot.action_horizon, DROID_EXECUTE_STEPS[0]
    ep = run_episode(policy, env, 0, 0, max_steps=20, replan_steps=k)
    assert ep.infer_calls == -(-ep.steps // k)  # ceil(steps / k): one infer per executed block of k of the H predicted
    assert k < H
    out = evaluate_rubric(policy, env, DROID_TASKS[:2], max_steps=12, replan_steps=k)
    assert out["n_trials"] == 8 and set(out["per_task"]) == {DROID_TASKS[0][0], DROID_TASKS[1][0]}
    assert 0.0 <= out["task_progress"] <= 1.0
    assert out["task_progress"] == pytest.approx(np.mean([r.score for r in out["episodes"]]))
    assert all(r.score == r.points / r.max_points for r in out["episodes"])
