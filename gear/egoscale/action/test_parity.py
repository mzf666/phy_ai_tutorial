"""对齐检查: shape / 维度, 解析性质, 不变性. 不做数值精度对齐, 不承诺论文数字.

论文: EgoScale arXiv:2602.16710v1 Sec. 2.1, Sec. 3.6, Appendix D.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gear.egoscale.action.action import (
    ACTION_SPACES,
    N_JOINTS,
    N_KEYPOINTS_HUMAN,
    N_KEYPOINTS_ROBOT,
    ROT_DIM,
    ActionConfig,
    FingertipToJointMLP,
    RetargetWeights,
    ToyHand22,
    _synthetic_stream,
    action_dim,
    build_action_chunk,
    decode_se3,
    encode_se3,
    exponential_filter,
    human_keypoints_in_wrist_frame,
    paper,
    palm_scale,
    relative_wrist_motion,
    retarget_chunk,
    se3_inverse,
    tiny,
    wrist_pose_world,
)

REPS = sorted(ROT_DIM)


@pytest.fixture(scope="module")
def hand() -> ToyHand22:
    return ToyHand22()


@pytest.fixture(scope="module")
def stream(hand):
    return _synthetic_stream(tiny(), hand, seed=0)


def _random_se3(*batch, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(*batch, 3, 3, generator=g)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
    q = torch.where(torch.linalg.det(q)[..., None, None] < 0, -q, q)
    T = torch.zeros(*batch, 4, 4)
    T[..., :3, :3] = q
    T[..., :3, 3] = torch.randn(*batch, 3, generator=g) * 0.3
    T[..., 3, 3] = 1.0
    return T


# --------------------------------------------------------------------------
# shape / 维度对齐
# --------------------------------------------------------------------------
def test_structural_constants_match_the_paper():
    """论文 Sec. 2.1 / App. D 明确给出的三个数."""
    assert (N_KEYPOINTS_HUMAN, N_KEYPOINTS_ROBOT, N_JOINTS) == (21, 20, 22)


def test_hand_has_22_joints_and_20_keypoints(hand):
    assert hand.n_joints == N_JOINTS
    assert hand.limits.shape == (N_JOINTS, 2)
    assert (hand.limits[:, 0] < hand.limits[:, 1]).all()
    assert hand.fk(hand.q_rest).shape == (N_KEYPOINTS_ROBOT, 3)
    assert hand.fk_pose(hand.q_rest).shape == (N_KEYPOINTS_ROBOT, 4, 4)


def test_representation_shapes(stream, hand):
    T_wc, H_c, _ = stream
    T = T_wc.shape[0]
    assert H_c.shape == (T, 2, N_KEYPOINTS_HUMAN, 4, 4)
    W_w = wrist_pose_world(T_wc, H_c)
    assert W_w.shape == (T, 2, 4, 4)
    assert relative_wrist_motion(W_w).shape == (T, 2, 4, 4)
    assert human_keypoints_in_wrist_frame(T_wc, H_c, W_w).shape == (T, 2, N_KEYPOINTS_ROBOT, 3)


@pytest.mark.parametrize("rep", REPS)
def test_encode_width(rep):
    T = _random_se3(4, 2, seed=1)
    assert encode_se3(T, rep).shape == (4, 2, 3 + ROT_DIM[rep])


def test_action_space_dims_are_18_108_62():
    """论文 Sec. 3.6 的三种表示. rotation_6d 下双手合计 18 / 108 / 62."""
    assert [action_dim(s, "rotation_6d") for s in ACTION_SPACES] == [18, 108, 62]


def test_build_action_chunk_shapes(stream, hand):
    T_wc, H_c, q_star = stream
    T = T_wc.shape[0]
    dW = relative_wrist_motion(wrist_pose_world(T_wc, H_c))
    q_hand = torch.from_numpy(np.repeat(q_star[:, None], 2, axis=1)).float()
    tips = torch.stack(
        [
            torch.stack(
                [torch.from_numpy(hand.fk_pose(q_star[t])[[4 * i + 3 for i in range(5)]]).float()]
                * 2
            )
            for t in range(T)
        ]
    )
    for space in ACTION_SPACES:
        a = build_action_chunk(dW, space, "rotation_6d", q_hand=q_hand, fingertip_pose=tips)
        assert a.shape == (T, action_dim(space, "rotation_6d"))


def test_fingertip_mlp_param_count():
    """5 个指尖 * (3 + 6) = 45 入, 64 隐, 22 出: 45*64+64 + 64*22+22 = 4374."""
    mlp = FingertipToJointMLP("rotation_6d", 64)
    assert sum(p.numel() for p in mlp.parameters()) == 45 * 64 + 64 + 64 * 22 + 22 == 4374
    assert mlp(torch.zeros(3, 2, 45)).shape == (3, 2, N_JOINTS)


# --------------------------------------------------------------------------
# 解析性质
# --------------------------------------------------------------------------
def test_se3_inverse_is_exact():
    T = _random_se3(6, seed=2)
    assert torch.allclose(se3_inverse(T) @ T, torch.eye(4).expand(6, 4, 4), atol=1e-5)


def test_delta_w0_is_identity_and_recomposes(stream):
    """论文 Sec. 2.1: ΔW_t = (W_0)^-1 W_t, 因此 ΔW_0 = I 且 W_0 · ΔW_t = W_t."""
    T_wc, H_c, _ = stream
    W_w = wrist_pose_world(T_wc, H_c)
    dW = relative_wrist_motion(W_w)
    assert torch.allclose(dW[0], torch.eye(4).expand(2, 4, 4), atol=1e-5)
    assert torch.allclose(W_w[0:1] @ dW, W_w, atol=1e-4)


def test_delta_w_is_invariant_to_the_world_frame(stream):
    """论文 Sec. 2.1 的核心主张: 该表示 "removes dependence on absolute camera pose".

    SLAM 的世界系原点是任意的; 左乘任意 G 到所有相机位姿上, ΔW 必须一字不变.
    """
    T_wc, H_c, _ = stream
    G = _random_se3(seed=3)
    dW = relative_wrist_motion(wrist_pose_world(T_wc, H_c))
    dW_shifted = relative_wrist_motion(wrist_pose_world(G @ T_wc, H_c))
    assert torch.allclose(dW, dW_shifted, atol=1e-5)


@pytest.mark.parametrize("rep", REPS)
def test_decode_is_the_left_inverse_of_encode(rep):
    T = _random_se3(32, seed=4)
    assert torch.allclose(decode_se3(encode_se3(T, rep), rep), T, atol=1e-4)


def test_unknown_rotation_rep_is_rejected():
    """EgoScale 未披露用哪种编码, 所以不提供默认值, 乱传必须报错."""
    with pytest.raises(ValueError):
        encode_se3(_random_se3(2, seed=5), "rotation_9d")
    with pytest.raises(ValueError):
        decode_se3(torch.zeros(2, 9), "rotation_9d")


def test_exponential_filter_alpha_one_is_identity():
    """App. D 的一阶指数滤波: α = 1 时不做任何平滑."""
    q = np.random.default_rng(0).normal(size=(7, N_JOINTS))
    assert np.allclose(exponential_filter(q, 1.0), q)
    smoothed = exponential_filter(q, 0.3)
    assert np.allclose(smoothed[0], q[0])
    assert np.abs(np.diff(smoothed, axis=0)).mean() < np.abs(np.diff(q, axis=0)).mean()


def test_palm_scale_is_one_for_the_robots_own_hand(hand):
    """尺度比必须是几何量: 同一只手在不同姿态下算出来都是 1."""
    for q in (hand.q_rest, hand.limits[:, 0] + 0.05, hand.limits[:, 1] - 0.05):
        kp = hand.fk(hand.clamp(q))[None]
        assert palm_scale(kp, hand) == pytest.approx(1.0, abs=1e-9)


def test_paper_config_refuses_to_run(hand):
    """未披露的值是 None 占位, 不能被悄悄当成能跑的默认值. 见 README Sec. 8."""
    cfg = paper()
    assert cfg.rot_rep is None and cfg.alpha is None and cfg.n_frames is None
    with pytest.raises(ValueError):
        retarget_chunk(np.zeros((2, N_KEYPOINTS_ROBOT, 3)), hand, cfg)
    with pytest.raises(ValueError):
        FingertipToJointMLP("rotation_6d", cfg.mlp_hidden)


# --------------------------------------------------------------------------
# oracle: 用机器人自己的正运动学造"人手", 重定向必须还原关节角
# --------------------------------------------------------------------------
def _exact_cfg() -> ActionConfig:
    """只留位置项、不滤波, 用来检验 NLP 本身能不能求到全局解."""
    return ActionConfig(**{
        **tiny().__dict__,
        "alpha": 1.0,
        "weights": RetargetWeights(w_pos=1.0, w_smooth=0.0, w_reg=0.0),
    })


@pytest.mark.parametrize(
    "traj",
    ["static_open", "open_to_fist", "random_walk"],
)
def test_oracle_roundtrip(hand, traj):
    """三条 oracle 轨迹: 静止 / 张开到握拳 / 随机游走. 残差趋于 0, 关节角落在限位内."""
    rng = np.random.default_rng(7)
    lo, hi = hand.limits[:, 0] + 0.05, hand.limits[:, 1] - 0.05
    if traj == "static_open":
        q_star = np.repeat(lo[None], 5, axis=0)
    elif traj == "open_to_fist":
        q_star = np.stack([lo + (hi - lo) * t / 4 for t in range(5)])
    else:
        q_star = hand.clamp(hand.q_rest + 0.25 * rng.normal(size=(5, N_JOINTS)))

    kp = np.stack([hand.fk(q) for q in q_star])
    q, info = retarget_chunk(kp, hand, _exact_cfg())

    assert info["converged"].all() and info["n_fallback"] == 0
    assert info["scale"] == pytest.approx(1.0, abs=1e-9)
    assert info["residual"].max() < 1e-6, "位置项应能降到数值零"
    assert np.sqrt(((q - q_star) ** 2).mean()) < 5e-3, "关节角应还原到 oracle"
    assert (q >= hand.limits[:, 0] - 1e-9).all() and (q <= hand.limits[:, 1] + 1e-9).all()


def test_joint_limits_are_never_violated(hand):
    """App. D: "subject only to joint limits from the URDF". 拿够不着的目标也不许越界."""
    rng = np.random.default_rng(11)
    kp = rng.normal(scale=0.5, size=(6, N_KEYPOINTS_ROBOT, 3))  # 物理上不可达的目标
    q, _ = retarget_chunk(kp, hand, _exact_cfg())
    assert (q >= hand.limits[:, 0] - 1e-9).all()
    assert (q <= hand.limits[:, 1] + 1e-9).all()


def test_warm_start_does_not_move_a_converged_solution(hand):
    """warm start 只是初值: 从上一帧的解出发, 静止轨迹的解不应漂移."""
    q_star = hand.clamp(hand.q_rest + 0.1)
    kp = np.repeat(hand.fk(q_star)[None], 4, axis=0)
    q, _ = retarget_chunk(kp, hand, _exact_cfg(), q_init=q_star)
    assert np.abs(np.diff(q, axis=0)).max() < 1e-6


# --------------------------------------------------------------------------
# 分布检查
# --------------------------------------------------------------------------
def test_keypoint_error_distribution(hand):
    """随机可达姿态: FK -> 重定向 -> FK 的关键点误差分位数应在数值零附近."""
    rng = np.random.default_rng(13)
    q_star = hand.clamp(hand.q_rest + 0.3 * rng.normal(size=(12, N_JOINTS)))
    kp = np.stack([hand.fk(q) for q in q_star])
    q, info = retarget_chunk(kp, hand, _exact_cfg())
    err = np.linalg.norm(np.stack([hand.fk(qi) for qi in q]) - kp, axis=-1)
    assert np.quantile(err, 0.5) < 1e-4
    assert np.quantile(err, 0.95) < 1e-3
    assert info["converged"].mean() == 1.0
