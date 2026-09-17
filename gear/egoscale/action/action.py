"""EgoScale 的人类动作表示: 相对腕部 SE(3) 运动 + 21 关键点到 22 自由度手的重定向.

上游: EgoScale 未开源; 旋转表示的候选集合与"以矩阵为中间表示"的约定参照
      NVIDIA/Isaac-GR00T @ 4af2b622892f7dcb5aae5a3fb70bcb02dc217b96,
      gr00t/data/transform/state_action.py L29-L95 (RotationTransform).
论文: EgoScale arXiv:2602.16710v1 Sec. 2.1, Sec. 3.6, Appendix D.
许可: 上游 Isaac-GR00T 为 Apache-2.0; 本文件为 PyTorch 重写 (re-implements, does not copy).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import casadi as ca
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 结构常量. 这三个数是论文明确给出的, 不是可调项.
# ---------------------------------------------------------------------------
N_KEYPOINTS_HUMAN = 21  # 论文 Sec. 2.1: 手部姿态由 21 个关键点建模, i=1 (本仓库 i=0) 是腕部
N_KEYPOINTS_ROBOT = 20  # 论文 App. D: URDF 正运动学输出 20 个机器人关键点位姿
N_JOINTS = 22  # 论文 Sec. 2.1 / App. D: Sharpa 手的 22 自由度关节空间

ROT_DIM = {"rotation_6d": 6, "quaternion": 4, "axis_angle": 3, "euler_angles": 3}


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class RetargetWeights:
    """App. D 只写了 "a weighted combination of different objectives", 项与权重均未披露.

    见 README Sec. 8.
    """

    w_pos: float | None
    w_smooth: float | None
    w_reg: float | None


@dataclass
class ActionConfig:
    n_frames: int | None  # action chunk 长度 H; EgoScale 未披露, 见 README Sec. 8
    rot_rep: str | None  # ΔW 的旋转编码; EgoScale 未披露, 见 README Sec. 8
    weights: RetargetWeights
    alpha: float | None  # 一阶指数滤波系数; 未披露, 见 README Sec. 8
    ipopt_max_iter: int | None  # IPOPT 选项未披露, 见 README Sec. 8
    ipopt_tol: float | None
    mlp_hidden: int | None  # 指尖->关节 MLP 的宽度; 未披露, 见 README Sec. 8


def paper() -> ActionConfig:
    """论文规模的配置. 未披露的字段一律是 None (占位), 见 README Sec. 8."""
    return ActionConfig(
        n_frames=None,
        rot_rep=None,
        weights=RetargetWeights(w_pos=None, w_smooth=None, w_reg=None),
        alpha=None,
        ipopt_max_iter=None,
        ipopt_tol=None,
        mlp_hidden=None,
    )


def tiny() -> ActionConfig:
    """CPU 上几秒钟跑通完整代码路径. 下面每个值都不是论文值, 只为让代码路径可执行."""
    return ActionConfig(
        n_frames=8,  # tiny only, not a paper value
        rot_rep="rotation_6d",  # tiny only, not a paper value
        weights=RetargetWeights(
            w_pos=1.0,  # tiny only, not a paper value
            w_smooth=1e-4,  # tiny only, not a paper value
            w_reg=1e-5,  # tiny only, not a paper value
        ),
        alpha=0.6,  # tiny only, not a paper value
        ipopt_max_iter=200,  # tiny only, not a paper value
        ipopt_tol=1e-8,  # tiny only, not a paper value
        mlp_hidden=64,  # tiny only, not a paper value
    )


# ---------------------------------------------------------------------------
# 1. SE(3) 基本运算与旋转编码
#    编码候选集合来自 Isaac-GR00T@4af2b62 state_action.py L32; 具体用哪一个 EgoScale
#    未披露, 因此所有函数都要求显式传入 rot_rep.
# ---------------------------------------------------------------------------
def se3_inverse(T: torch.Tensor) -> torch.Tensor:
    """(..., 4, 4) 齐次变换的解析逆: [R|t]^-1 = [R^T | -R^T t]."""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = R.transpose(-1, -2)
    out = torch.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -(Rt @ t.unsqueeze(-1)).squeeze(-1)
    out[..., 3, 3] = 1.0
    return out


def wrist_pose_world(T_wc: torch.Tensor, H_c: torch.Tensor) -> torch.Tensor:
    """论文 Sec. 2.1: W_t^w = T_t^{w<-c} H_t^{c,1}, 本仓库 0-based 所以取 H_c[..., 0, :, :].

    T_wc: (T, 4, 4) 相机位姿; H_c: (T, 2, 21, 4, 4) 相机系下的手部关键点位姿.
    返回 (T, 2, 4, 4) 的世界系腕部位姿.
    """
    assert H_c.shape[-3] == N_KEYPOINTS_HUMAN, f"需要 {N_KEYPOINTS_HUMAN} 个关键点"
    wrist_c = H_c[:, :, 0]  # (T, 2, 4, 4), 腕部是第 0 个关键点
    return T_wc[:, None] @ wrist_c


def relative_wrist_motion(W_w: torch.Tensor) -> torch.Tensor:
    """论文 Sec. 2.1: ΔW_t = (W_0^w)^{-1} W_t^w, 参考帧是 action chunk 的第 0 帧.

    正文措辞是"相邻帧之间", 但公式写的是第 0 帧; 本仓库按公式, 见 README Sec. 1.4 第 1 条.
    """
    W0_inv = se3_inverse(W_w[0:1])  # (1, 2, 4, 4)
    return W0_inv @ W_w


def _matrix_to_rotation_6d(R: torch.Tensor) -> torch.Tensor:
    """取前两行展平 (pytorch3d 的约定, 上游 RotationTransform 经它转换)."""
    return R[..., :2, :].reshape(*R.shape[:-2], 6)


def _rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt: 前两行正交化后叉乘补第三行, 因此 decode(encode(R)) == R."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / a1.norm(dim=-1, keepdim=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = b2 / b2.norm(dim=-1, keepdim=True)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def _matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """实部在前 (w, x, y, z), 与上游依赖的 pytorch3d 一致."""
    m = R.reshape(*R.shape[:-2], 9).unbind(-1)
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = m
    trace = m00 + m11 + m22
    w = torch.sqrt(torch.clamp(1.0 + trace, min=1e-12)) / 2.0
    x = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-12)) / 2.0
    y = torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=1e-12)) / 2.0
    z = torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=1e-12)) / 2.0
    # 用最大的分量定符号, 避免开方后丢符号
    quat = torch.stack((w, x, y, z), dim=-1)
    signs = torch.stack(
        (
            torch.ones_like(w),
            torch.sign(m21 - m12),
            torch.sign(m02 - m20),
            torch.sign(m10 - m01),
        ),
        dim=-1,
    )
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    quat = quat * signs
    return quat / quat.norm(dim=-1, keepdim=True)


def _quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def _matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    q = _matrix_to_quaternion(R)
    w = torch.clamp(q[..., 0], -1.0, 1.0)
    angle = 2.0 * torch.acos(w)
    s = torch.sqrt(torch.clamp(1.0 - w * w, min=1e-12))
    small = s < 1e-6
    scale = torch.where(small, torch.full_like(s, 2.0), angle / s)
    return q[..., 1:] * scale.unsqueeze(-1)


def _axis_angle_to_matrix(v: torch.Tensor) -> torch.Tensor:
    angle = v.norm(dim=-1, keepdim=True)
    axis = v / torch.clamp(angle, min=1e-12)
    half = angle * 0.5
    q = torch.cat((torch.cos(half), axis * torch.sin(half)), dim=-1)
    return _quaternion_to_matrix(q)


def _matrix_to_euler_xyz(R: torch.Tensor) -> torch.Tensor:
    """R = Rx(a) Ry(b) Rz(c) 的 XYZ 约定 (与 pytorch3d 的 "XYZ" 一致)."""
    b = torch.asin(torch.clamp(R[..., 0, 2], -1.0, 1.0))
    a = torch.atan2(-R[..., 1, 2], R[..., 2, 2])
    c = torch.atan2(-R[..., 0, 1], R[..., 0, 0])
    return torch.stack((a, b, c), dim=-1)


def _euler_xyz_to_matrix(e: torch.Tensor) -> torch.Tensor:
    a, b, c = e.unbind(-1)
    ca_, sa = torch.cos(a), torch.sin(a)
    cb, sb = torch.cos(b), torch.sin(b)
    cc, sc = torch.cos(c), torch.sin(c)
    return torch.stack(
        (
            cb * cc, -cb * sc, sb,
            sa * sb * cc + ca_ * sc, -sa * sb * sc + ca_ * cc, -sa * cb,
            -ca_ * sb * cc + sa * sc, ca_ * sb * sc + sa * cc, ca_ * cb,
        ),
        dim=-1,
    ).reshape(*e.shape[:-1], 3, 3)


_TO_VEC = {
    "rotation_6d": _matrix_to_rotation_6d,
    "quaternion": _matrix_to_quaternion,
    "axis_angle": _matrix_to_axis_angle,
    "euler_angles": _matrix_to_euler_xyz,
}
_TO_MAT = {
    "rotation_6d": _rotation_6d_to_matrix,
    "quaternion": _quaternion_to_matrix,
    "axis_angle": _axis_angle_to_matrix,
    "euler_angles": _euler_xyz_to_matrix,
}


def encode_se3(T: torch.Tensor, rot_rep: str) -> torch.Tensor:
    """(..., 4, 4) -> (..., 3 + R). rot_rep 必须显式给出: EgoScale 未披露用哪一种."""
    if rot_rep not in _TO_VEC:
        raise ValueError(f"rot_rep 必须是 {sorted(_TO_VEC)} 之一, 收到 {rot_rep!r}")
    return torch.cat((T[..., :3, 3], _TO_VEC[rot_rep](T[..., :3, :3])), dim=-1)


def decode_se3(vec: torch.Tensor, rot_rep: str) -> torch.Tensor:
    """(..., 3 + R) -> (..., 4, 4), encode_se3 的左逆."""
    if rot_rep not in _TO_MAT:
        raise ValueError(f"rot_rep 必须是 {sorted(_TO_MAT)} 之一, 收到 {rot_rep!r}")
    R = _TO_MAT[rot_rep](vec[..., 3:])
    T = torch.zeros(*vec.shape[:-1], 4, 4, dtype=vec.dtype, device=vec.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = vec[..., :3]
    T[..., 3, 3] = 1.0
    return T


# ---------------------------------------------------------------------------
# 2. 22 自由度手的 URDF 式正运动学
#    论文 App. D 只给了三条结构事实: 22 个关节, 20 个关键点位姿, 关节限位来自 URDF.
#    真实 Sharpa Wave 的 URDF 不公开, 下面的连杆长度/限位/每指关节数都是玩具值,
#    见 README Sec. 8.
# ---------------------------------------------------------------------------
# 每指: (名字, 关节轴序列, 每个关节之后的连杆长度 m, 掌上基座位置 m, 基座绕 z 的朝向 rad)
# 轴 "z" 是外展/内收, "y" 是屈伸. 关节数 5+4+4+4+5 = 22, 关键点数 5*4 = 20.
_FINGERS = (
    ("thumb", ("z", "y", "z", "y", "y"), (0.030, 0.038, 0.032, 0.026, 0.022), (0.020, -0.030, 0.0), -1.05),
    ("index", ("z", "y", "y", "y"), (0.090, 0.040, 0.026, 0.020), (0.010, 0.022, 0.0), 0.18),
    ("middle", ("z", "y", "y", "y"), (0.092, 0.044, 0.028, 0.022), (0.000, 0.008, 0.0), 0.02),
    ("ring", ("z", "y", "y", "y"), (0.088, 0.041, 0.026, 0.020), (-0.008, -0.006, 0.0), -0.12),
    ("pinky", ("z", "y", "z", "y", "y"), (0.080, 0.034, 0.000, 0.022, 0.018), (-0.016, -0.020, 0.0), -0.28),
)
# 每个轴类型的关节限位 (rad). 玩具值, 见 README Sec. 8.
_LIMITS_BY_AXIS = {"z": (-0.40, 0.40), "y": (-0.10, 1.60)}


def _rot(axis: str, angle):
    """casadi 符号与数值共用的单轴旋转矩阵."""
    c, s = ca.cos(angle), ca.sin(angle)
    if axis == "z":
        return ca.vertcat(ca.horzcat(c, -s, 0), ca.horzcat(s, c, 0), ca.horzcat(0, 0, 1))
    if axis == "y":
        return ca.vertcat(ca.horzcat(c, 0, s), ca.horzcat(0, 1, 0), ca.horzcat(-s, 0, c))
    raise ValueError(axis)


class ToyHand22:
    """22 自由度手的正运动学: q (22,) -> 20 个关键点位姿.

    正运动学用 casadi 符号搭一次, 数值求值与 NLP 目标共用同一个表达式, 保证两者严格一致.
    """

    n_joints = N_JOINTS
    n_keypoints = N_KEYPOINTS_ROBOT

    def __init__(self) -> None:
        q = ca.MX.sym("q", N_JOINTS)
        positions, rotations, limits = [], [], []
        j = 0
        for _name, axes, lengths, base_p, base_yaw in _FINGERS:
            R = _rot("z", base_yaw)
            p = ca.DM(base_p)
            chain_p, chain_R = [], []
            for axis, length in zip(axes, lengths):
                R = R @ _rot(axis, q[j])
                p = p + R @ ca.DM([length, 0.0, 0.0])
                limits.append(_LIMITS_BY_AXIS[axis])
                chain_p.append(p)
                chain_R.append(R)
                j += 1
            # App. D 的 20 个关键点 = 每指末端的 4 个 (MCP / PIP / DIP / TIP)
            positions.extend(chain_p[-4:])
            rotations.extend(chain_R[-4:])
        assert j == N_JOINTS, j
        assert len(positions) == N_KEYPOINTS_ROBOT, len(positions)

        self.limits = np.asarray(limits, dtype=np.float64)  # (22, 2)
        self.q_rest = self.limits.mean(axis=1)
        self._q_sym = q
        self._p_sym = ca.horzcat(*positions).T  # (20, 3)
        self._R_sym = rotations
        self._fk_fn = ca.Function("fk", [q], [self._p_sym])

    def fk(self, q: np.ndarray) -> np.ndarray:
        """(22,) 弧度 -> (20, 3) 米, 腕部坐标系下的关键点位置."""
        return np.asarray(self._fk_fn(np.asarray(q, dtype=np.float64))).reshape(
            N_KEYPOINTS_ROBOT, 3
        )

    def fk_pose(self, q: np.ndarray) -> np.ndarray:
        """(22,) -> (20, 4, 4). App. D 的 "positions and quaternions", 这里给齐次矩阵."""
        fn = ca.Function("fk_pose", [self._q_sym], [self._p_sym, *self._R_sym])
        out = fn(np.asarray(q, dtype=np.float64))
        p = np.asarray(out[0]).reshape(N_KEYPOINTS_ROBOT, 3)
        T = np.zeros((N_KEYPOINTS_ROBOT, 4, 4))
        T[:, 3, 3] = 1.0
        T[:, :3, 3] = p
        for k in range(N_KEYPOINTS_ROBOT):
            T[k, :3, :3] = np.asarray(out[k + 1])
        return T

    def clamp(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.limits[:, 0], self.limits[:, 1])


# ---------------------------------------------------------------------------
# 3. 逐帧重定向 (论文 App. D)
# ---------------------------------------------------------------------------
def human_keypoints_in_wrist_frame(
    T_wc: torch.Tensor, H_c: torch.Tensor, W_w: torch.Tensor
) -> torch.Tensor:
    """21 个相机系关键点 -> 20 个腕部系关键点位置 (去掉腕部本身).

    腕部已经由 ΔW 单独监督, 不在手指目标里重复计一次, 见 README Sec. 1.4 第 3 条.
    返回 (T, 2, 20, 3).
    """
    world_kp = T_wc[:, None, None] @ H_c  # (T, 2, 21, 4, 4)
    in_wrist = se3_inverse(W_w)[:, :, None] @ world_kp
    return in_wrist[:, :, 1:, :3, 3]


def _middle_finger_length(kp: np.ndarray) -> float:
    """中指的总骨长: MCP->PIP->DIP->TIP 三段之和.

    用骨长而不是"指尖到腕部的直线距离", 因为前者与手的姿态严格无关: 相邻关键点之间
    是刚性连杆, 距离恒等于连杆长度; 而腕部到 MCP 的距离会随掌指关节转动而变, 所以
    这一段不算进去. 尺度比必须是几何量, 不能随帧变化.
    """
    return float(np.linalg.norm(np.diff(kp[..., 8:12, :], axis=-2), axis=-1).sum(-1).mean())


def palm_scale(kp_human: np.ndarray, hand: ToyHand22) -> float:
    """手掌尺度比 s = 机器人中指骨长 / 人手中指骨长, 用来消除人手与机器人手的尺寸差.

    App. D 只提到 "kinematic consistency", 没有定义尺度项; 这个定义是本仓库的, 见 README Sec. 8.
    """
    robot = _middle_finger_length(hand.fk(hand.q_rest))
    human = _middle_finger_length(kp_human)
    return robot / max(human, 1e-9)


def _build_solver(
    hand: ToyHand22, weights: RetargetWeights, cfg: ActionConfig, kp_weight: np.ndarray
):
    """App. D: "solve a nonlinear program over the 22 joint angles, subject only to joint
    limits from the URDF, and minimize a weighted combination of different objectives".
    目标函数的项与权重未披露, 见 README Sec. 8.
    """
    for name, value in (("w_pos", weights.w_pos), ("w_smooth", weights.w_smooth), ("w_reg", weights.w_reg)):
        if value is None:
            raise ValueError(f"{name} 未披露 (见 README Sec. 8); paper() 配置无法求解, 请用 tiny()")
    q = hand._q_sym
    target = ca.MX.sym("target", N_KEYPOINTS_ROBOT, 3)
    q_prev = ca.MX.sym("q_prev", N_JOINTS)
    q_rest = ca.DM(hand.q_rest)

    # kp_weight 决定哪些关键点被监督: 全 1 是论文默认的 "retarget the 21 keypoints";
    # 只在 5 个指尖上为 1 就是 Sec. 3.6 的 fingertip 分支所能提供的全部信息.
    diff = hand._p_sym - target
    cost = weights.w_pos * ca.sum1(ca.DM(kp_weight) * ca.sum2(diff * diff))
    cost += weights.w_smooth * ca.sumsqr(q - q_prev)
    cost += weights.w_reg * ca.sumsqr(q - q_rest)

    nlp = {"x": q, "f": cost, "p": ca.vertcat(ca.reshape(target, -1, 1), q_prev)}
    opts = {
        "print_time": 0,
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "ipopt.max_iter": cfg.ipopt_max_iter,  # 未披露, 见 README Sec. 8
        "ipopt.tol": cfg.ipopt_tol,  # 未披露, 见 README Sec. 8
    }
    return ca.nlpsol("retarget", "ipopt", nlp, opts)


def exponential_filter(q: np.ndarray, alpha: float) -> np.ndarray:
    """App. D: "smoothed using a first-order exponential filter". 系数未披露, 见 README Sec. 8.

    q̃_t = α q_t + (1-α) q̃_{t-1}, q̃_0 = q_0. α = 1 时是恒等.
    """
    out = np.empty_like(q)
    out[0] = q[0]
    for t in range(1, len(q)):
        out[t] = alpha * q[t] + (1.0 - alpha) * out[t - 1]
    return out


def retarget_chunk(
    kp_human: np.ndarray,
    hand: ToyHand22,
    cfg: ActionConfig,
    q_init: np.ndarray | None = None,
    kp_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """(T, 20, 3) 腕部系人手关键点 -> (T, 22) 关节角, 逐帧 NLP + warm start + 指数滤波.

    kp_weight: (20,) 每个关键点的监督权重, 默认全 1 (论文 App. D 的默认). 只把 5 个指尖
    置 1 就退化成 Sec. 3.6 的 fingertip 表示所携带的信息量.
    """
    assert kp_human.ndim == 3 and kp_human.shape[1:] == (N_KEYPOINTS_ROBOT, 3), kp_human.shape
    if cfg.alpha is None:
        raise ValueError("alpha 未披露 (见 README Sec. 8); paper() 配置无法求解, 请用 tiny()")
    if kp_weight is None:
        kp_weight = np.ones(N_KEYPOINTS_ROBOT)
    solver = _build_solver(hand, cfg.weights, cfg, np.asarray(kp_weight, float).reshape(-1, 1))
    scale = palm_scale(kp_human, hand)

    q_prev = hand.q_rest.copy() if q_init is None else hand.clamp(np.asarray(q_init, float))
    raw = np.empty((len(kp_human), N_JOINTS))
    residual = np.empty(len(kp_human))
    converged = np.empty(len(kp_human), dtype=bool)
    n_fallback = 0
    for t, target in enumerate(kp_human):
        sol = solver(
            x0=q_prev,  # App. D: warm-started from the previous frame's solution
            lbx=hand.limits[:, 0],  # App. D: subject only to joint limits from the URDF
            ubx=hand.limits[:, 1],
            p=np.concatenate([(scale * target).reshape(-1, order="F"), q_prev]),
        )
        ok = solver.stats()["success"]
        converged[t] = ok
        if ok:
            q_prev = np.asarray(sol["x"]).ravel()
        else:
            # 上游未说求解失败怎么办; 本仓库保持上一帧的解 (等价于 warm start 原地不动)
            n_fallback += 1
        raw[t] = q_prev
        residual[t] = float(sol["f"])
    return exponential_filter(raw, cfg.alpha), {
        "raw": raw,
        "residual": residual,
        "converged": converged,
        "n_fallback": n_fallback,
        "scale": scale,
    }


# ---------------------------------------------------------------------------
# 4. 三种动作空间 (论文 Sec. 3.6, 图 8)
# ---------------------------------------------------------------------------
ACTION_SPACES = ("wrist_only", "fingertip", "full")
_FINGERTIP_IDX = [4 * i + 3 for i in range(5)]  # 每指第 4 个关键点是 TIP


def action_dim(space: str, rot_rep: str) -> int:
    """双手合计的动作维度. rotation_6d 下分别是 18 / 108 / 62."""
    r = ROT_DIM[rot_rep]
    per_hand = {"wrist_only": 3 + r, "fingertip": (3 + r) * 6, "full": 3 + r + N_JOINTS}[space]
    return 2 * per_hand


def build_action_chunk(
    dW: torch.Tensor,
    space: str,
    rot_rep: str,
    q_hand: torch.Tensor | None = None,
    fingertip_pose: torch.Tensor | None = None,
) -> torch.Tensor:
    """把 ΔW 与手指监督拼成一个 chunk 的动作向量 (T, D).

    dW: (T, 2, 4, 4); q_hand: (T, 2, 22); fingertip_pose: (T, 2, 5, 4, 4).
    """
    if space not in ACTION_SPACES:
        raise ValueError(f"space 必须是 {ACTION_SPACES} 之一, 收到 {space!r}")
    wrist = encode_se3(dW, rot_rep)  # (T, 2, 3+R)
    if space == "wrist_only":
        per_hand = wrist
    elif space == "fingertip":
        assert fingertip_pose is not None, "fingertip 空间需要 5 个指尖位姿"
        tips = encode_se3(fingertip_pose, rot_rep)  # (T, 2, 5, 3+R)
        per_hand = torch.cat((wrist, tips.flatten(start_dim=2)), dim=-1)
    else:
        assert q_hand is not None, "full 空间需要 22 个重定向关节角"
        per_hand = torch.cat((wrist, q_hand.to(wrist.dtype)), dim=-1)
    return per_hand.flatten(start_dim=1)  # 左右手拼接 -> (T, D)


class FingertipToJointMLP(nn.Module):
    """论文 Sec. 3.6 的指尖表示分支: 5 个指尖 SE(3) -> 22 个关节命令.

    论文只写了 "an MLP", 层数/宽度/激活均未披露, 见 README Sec. 8.
    """

    def __init__(self, rot_rep: str, hidden: int | None) -> None:
        super().__init__()
        if hidden is None:
            raise ValueError("MLP 宽度未披露 (见 README Sec. 8); paper() 配置无法实例化")
        in_dim = 5 * (3 + ROT_DIM[rot_rep])
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, N_JOINTS)
        )

    def forward(self, tips: torch.Tensor) -> torch.Tensor:
        return self.net(tips)


# ---------------------------------------------------------------------------
# 5. 一次 tiny 运行
# ---------------------------------------------------------------------------
def _synthetic_stream(cfg: ActionConfig, hand: ToyHand22, seed: int = 0):
    """造一段第一视角流: 相机在世界里走, 手相对相机做"张开->握拳"的插值.

    人手关键点由机器人自己的正运动学生成, 这样 oracle 关节角已知.
    """
    g = torch.Generator().manual_seed(seed)
    T = cfg.n_frames
    q_open = hand.clamp(hand.limits[:, 0] + 0.05)
    q_fist = hand.clamp(hand.limits[:, 1] - 0.05)
    q_star = np.stack([q_open + (q_fist - q_open) * (t / (T - 1)) for t in range(T)])

    T_wc = torch.eye(4).repeat(T, 1, 1)
    ang = torch.linspace(0.0, 0.6, T)
    T_wc[:, :3, :3] = _euler_xyz_to_matrix(torch.stack((ang * 0.3, ang, ang * 0.5), dim=-1))
    T_wc[:, :3, 3] = torch.stack((ang * 0.4, torch.sin(ang), ang * 0.1), dim=-1)

    H_c = torch.eye(4).repeat(T, 2, N_KEYPOINTS_HUMAN, 1, 1)
    for t in range(T):
        pose = torch.from_numpy(hand.fk_pose(q_star[t])).float()  # (20, 4, 4)
        for h in range(2):
            wrist = torch.eye(4)
            wrist[:3, 3] = torch.tensor([0.25 * (1 if h else -1), 0.1, 0.45]) + 0.02 * torch.randn(
                3, generator=g
            )
            H_c[t, h, 0] = wrist
            H_c[t, h, 1:] = wrist @ pose
    return T_wc, H_c, q_star


def main() -> None:
    torch.manual_seed(0)
    cfg = tiny()
    hand = ToyHand22()
    print(f"config: tiny(), H={cfg.n_frames}, rot_rep={cfg.rot_rep!r}")
    print(f"ToyHand22: n_joints={hand.n_joints}, n_keypoints={hand.n_keypoints}, "
          f"limits {hand.limits.min():.2f}..{hand.limits.max():.2f} rad")

    T_wc, H_c, q_star = _synthetic_stream(cfg, hand)
    print(f"[1] raw streams        T_wc {tuple(T_wc.shape)}  H_c {tuple(H_c.shape)}")

    W_w = wrist_pose_world(T_wc, H_c)
    print(f"[2] wrist in world     W_w  {tuple(W_w.shape)}  t0 left = {W_w[0,0,:3,3].numpy().round(4)}")

    dW = relative_wrist_motion(W_w)
    print(f"[3] relative motion    dW   {tuple(dW.shape)}  dW[0] is identity = "
          f"{torch.allclose(dW[0], torch.eye(4).expand(2,4,4), atol=1e-5)}  "
          f"dW[-1] trans = {dW[-1,0,:3,3].numpy().round(4)}")

    vec = encode_se3(dW, cfg.rot_rep)
    print(f"[4] encode SE(3)       vec  {tuple(vec.shape)}  first row = {vec[1,0].numpy().round(3)}")

    kp = human_keypoints_in_wrist_frame(T_wc, H_c, W_w)
    print(f"[5] keypoints in wrist kp   {tuple(kp.shape)}  tip of middle = "
          f"{kp[0,0,11].numpy().round(4)}")

    t0 = time.time()
    q_left, info = retarget_chunk(kp[:, 0].double().numpy(), hand, cfg)
    dt = (time.time() - t0) / cfg.n_frames
    print(f"[6] retarget (NLP)     q    {q_left.shape}  converged={info['converged'].all()}  "
          f"scale={info['scale']:.3f}  residual[-1]={info['residual'][-1]:.3e}  "
          f"{1e3*dt:.1f} ms/frame")
    print(f"    joint RMSE vs oracle = {np.sqrt(((q_left - q_star)**2).mean()):.4f} rad "
          f"(tiny 的平滑项与 alpha=0.6 的滤波都会让它偏离真值)")
    exact = ActionConfig(**{**cfg.__dict__, "alpha": 1.0,
                            "weights": RetargetWeights(w_pos=1.0, w_smooth=0.0, w_reg=0.0)})
    q_exact, info_exact = retarget_chunk(kp[:, 0].double().numpy(), hand, exact)
    print(f"    去掉平滑与滤波后    = {np.sqrt(((q_exact - q_star)**2).mean()):.6f} rad, "
          f"residual max = {info_exact['residual'].max():.2e}  (机制自检: oracle 可还原)")
    print(f"    joints within limits = "
          f"{bool((q_left >= hand.limits[:,0] - 1e-9).all() and (q_left <= hand.limits[:,1] + 1e-9).all())}")

    q_right, _ = retarget_chunk(kp[:, 1].double().numpy(), hand, cfg)
    q_hand = torch.from_numpy(np.stack((q_left, q_right), axis=1)).float()  # (T, 2, 22)

    tips = torch.zeros(cfg.n_frames, 2, 5, 4, 4)
    for t in range(cfg.n_frames):
        for h in range(2):
            tips[t, h] = torch.from_numpy(hand.fk_pose(q_hand[t, h].numpy())[_FINGERTIP_IDX]).float()

    for space in ACTION_SPACES:
        a = build_action_chunk(dW, space, cfg.rot_rep, q_hand=q_hand, fingertip_pose=tips)
        print(f"[7] action chunk       {space:11s} {tuple(a.shape)}  "
              f"dim matches action_dim() = {a.shape[-1] == action_dim(space, cfg.rot_rep)}")

    mlp = FingertipToJointMLP(cfg.rot_rep, cfg.mlp_hidden)
    tip_vec = encode_se3(tips, cfg.rot_rep).flatten(start_dim=2)
    out = mlp(tip_vec)
    print(f"[8] fingertip -> joints {tuple(tip_vec.shape)} -> {tuple(out.shape)}  "
          f"params = {sum(p.numel() for p in mlp.parameters())}")

    back = decode_se3(vec, cfg.rot_rep)
    W_back = W_w[0:1] @ back
    print(f"[9] inverse            decode(encode(dW)) == dW : "
          f"{torch.allclose(back, dW, atol=1e-5)}   W_0 @ dW == W_w : "
          f"{torch.allclose(W_back, W_w, atol=1e-4)}")


if __name__ == "__main__":
    main()
