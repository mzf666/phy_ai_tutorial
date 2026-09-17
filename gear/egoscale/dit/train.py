"""flow matching 的训练侧: Beta 时间步采样, 线性加噪, 速度目标, action_mask 加权的 MSE.

上游: NVIDIA/Isaac-GR00T @ 4af2b622892f7dcb5aae5a3fb70bcb02dc217b96,
      gr00t/model/action_head/flow_matching_action_head.py L256-L258 (sample_time),
      L270-L347 (训练前向与 loss).
论文: EgoScale arXiv:2602.16710v1 Sec. 2.3; GR00T N1 arXiv:2503.14734v2 Sec. 2.1 式 (1).
许可: 上游 Isaac-GR00T 为 Apache-2.0; 本文件为 PyTorch 重写 (re-implements, does not copy).

优化器、学习率与三阶段 curriculum 在 ../train.
"""

from __future__ import annotations

import torch
from torch.distributions import Beta

from gear.egoscale.dit.model import ActionExpert, DiTConfig, tiny


def sample_time(batch_size: int, cfg: DiTConfig, device=None,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """上游 L256-L258: u ~ Beta(1.5, 1), tau = (s - u) / s, s = 0.999.

    Beta(1.5, 1) 把质量堆在 u 接近 1 的一侧, 变换之后 tau 就偏向 0, 也就是**高噪声端**.
    与 pi0 / pi0.5 用的是同一个分布 (arXiv:2410.24164v1).
    """
    for name in ("noise_beta_alpha", "noise_beta_beta", "noise_s"):
        if getattr(cfg, name) is None:
            raise ValueError(f"{name} 未披露 (见 README Sec. 8); paper() 配置无法采样")
    dist = Beta(torch.tensor(cfg.noise_beta_alpha), torch.tensor(cfg.noise_beta_beta))
    if generator is None:
        u = dist.sample((batch_size,))
    else:  # Beta.sample 不接受 generator, 用解析逆变换以保证测试可复现
        u = _beta_icdf(torch.rand(batch_size, generator=generator),
                       cfg.noise_beta_alpha, cfg.noise_beta_beta)
    u = u.to(device)
    return (cfg.noise_s - u) / cfg.noise_s


def _beta_icdf(p: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """Beta(alpha, 1) 的解析逆 CDF: F(x) = x^alpha, 所以 x = p^(1/alpha).

    只在 beta == 1 时成立 —— 上游的 noise_beta_beta 正是 1.0, 所以够用; 其他取值报错而不是
    悄悄退化成别的分布.
    """
    if beta != 1.0:
        raise ValueError(f"解析逆 CDF 只对 beta == 1 成立, 收到 {beta}")
    return p.pow(1.0 / alpha)


def add_noise(action: torch.Tensor, noise: torch.Tensor,
              tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """上游 L307-L308.

    A_tau = (1 - tau) * eps + tau * A;  velocity = A - eps.

    注意符号: GR00T N1 论文式 (1) 写的回归目标是 `eps - A`, 与上游代码相反. 本仓库按上游,
    因为上游自洽: d/dtau A_tau = A - eps, 而采样是 A <- A + dt * V 的前向 Euler.
    见 README Sec. 1.x 第 2 条.
    """
    t = tau[:, None, None]
    return (1 - t) * noise + t * action, action - noise


def discretize(tau: torch.Tensor, cfg: DiTConfig) -> torch.Tensor:
    """上游 L311: 连续 tau -> [0, num_timestep_buckets) 的整数桶."""
    return (tau * cfg.num_timestep_buckets).long()


def masked_mse(pred: torch.Tensor, target: torch.Tensor,
               action_mask: torch.Tensor) -> torch.Tensor:
    """上游 L341-L343: 分母只数真实维度, 跨本体的 padding 维不计入.

    写成 .mean() 的话, padding 比例高的本体会因为大量"预测 0"的容易维而显得 loss 更低,
    不同本体的 loss 就不可比了. 见 ../data 的 figs/padding.png.
    """
    sq = torch.nn.functional.mse_loss(pred, target, reduction="none") * action_mask
    return sq.sum() / action_mask.sum()


def flow_matching_loss(
    model: ActionExpert,
    phi: torch.Tensor,
    phi_mask: torch.Tensor,
    state: torch.Tensor,
    action: torch.Tensor,
    action_mask: torch.Tensor,
    embodiment_id: torch.Tensor,
    has_proprio: torch.Tensor,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """一次训练前向 (上游 L270-L347). 只有这一个目标: 没有 CE, 没有辅助项."""
    cfg = model.cfg
    b = action.shape[0]
    noise = torch.randn(action.shape, device=action.device, dtype=action.dtype,
                        generator=generator)
    tau = sample_time(b, cfg, device=action.device, generator=generator).to(action.dtype)
    noisy, velocity = add_noise(action, noise, tau)
    bucket = discretize(tau, cfg)

    state_feats = model.encode_state(state, embodiment_id, has_proprio)
    tokens = model.build_tokens(state_feats,
                                model.encode_action(noisy, bucket, embodiment_id))
    pred = model.velocity(tokens, phi, bucket, embodiment_id, phi_mask)
    return {"loss": masked_mse(pred, velocity, action_mask), "tau": tau,
            "bucket": bucket, "pred": pred, "target": velocity}


def oracle_integrate(action: torch.Tensor, noise: torch.Tensor, k: int) -> torch.Tensor:
    """用真实速度场 (A - eps) 做 k 步前向 Euler, 从噪声出发.

    路径是直线, 速度沿路径恒定, 所以任意 k 都精确还原 A —— 这正是 K = 4 就够用的原因.
    """
    x = noise.clone()
    v = action - noise
    for _ in range(k):
        x = x + v / k
    return x


def main() -> None:
    torch.manual_seed(0)
    cfg = tiny()
    model = ActionExpert(cfg)
    g = torch.Generator().manual_seed(0)
    b, s_len = 4, 19

    phi = torch.randn(b, s_len, cfg.backbone_embedding_dim)
    phi_mask = torch.ones(b, s_len, dtype=torch.bool)
    state = torch.randn(b, cfg.state_horizon, cfg.max_state_dim)
    action = torch.randn(b, cfg.action_horizon, cfg.max_action_dim)
    action_mask = torch.zeros_like(action, dtype=torch.bool)
    native = [62, 62, 32, 62]  # r1pro / human / g1 / human 的原生维度 (见 ../data)
    for i, n in enumerate(native):
        action_mask[i, :, :n] = True
    emb_id = torch.tensor([2, 0, 3, 1])
    has_proprio = torch.tensor([True, False, True, False])

    out = flow_matching_loss(model, phi, phi_mask, state, action, action_mask, emb_id,
                             has_proprio, generator=g)
    print(f"config: tiny(), Beta({cfg.noise_beta_alpha}, {cfg.noise_beta_beta}), "
          f"s={cfg.noise_s}, {cfg.num_timestep_buckets} buckets, K={cfg.num_inference_timesteps}")
    print(f"[1] tau sampled     {out['tau'].numpy().round(4)}")
    print(f"[2] discretized     {out['bucket'].tolist()}  (in [0, {cfg.num_timestep_buckets}))")
    noisy, vel = add_noise(action, torch.randn_like(action), out["tau"])
    print(f"[3] A_tau           {tuple(noisy.shape)}   velocity target {tuple(vel.shape)}")
    print(f"[4] pred            {tuple(out['pred'].shape)}")
    print(f"[5] mask True/row   {action_mask[:, 0].sum(-1).tolist()} of {cfg.max_action_dim}")
    print(f"[6] masked loss     {out['loss'].item():.4f}")

    naive = torch.nn.functional.mse_loss(out["pred"], out["target"])
    print(f"    naive .mean()   {naive.item():.4f}  (wrong denominator, see ../data)")

    out["loss"].backward()
    grads = {n: p.grad.abs().sum().item() for n, p in model.named_parameters()
             if p.grad is not None}
    print(f"[7] backward ok, {len(grads)} tensors got gradients; "
          f"state_placeholder grad = {model.state_placeholder.grad.abs().sum():.4e} "
          f"(non-zero because samples 1 and 3 are human demos)")

    eps = torch.randn_like(action)
    for k in (1, 4, 50):
        err = (oracle_integrate(action, eps, k) - action).abs().max()
        print(f"[8] oracle Euler K={k:<3d} max |A_hat - A| = {err:.2e}  "
              f"(linear path: exact for any K)")

    taus = sample_time(20000, cfg, generator=torch.Generator().manual_seed(1))
    print(f"[9] tau distribution: mean {taus.mean():.4f}, median {taus.median():.4f}, "
          f"q10 {taus.quantile(0.1):.4f}, q90 {taus.quantile(0.9):.4f}")
    print(f"    mean < 0.5 -> training mass sits on the high-noise end")


if __name__ == "__main__":
    main()
