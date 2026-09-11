"""Shape and analytic-property checks for the pi0 flow-matching objective and sampler.

References: openpi @ 215abfb217dbac7d5f1273282331b9b1866c0479 (src/openpi/models/pi0.py L188-L279),
pi0 paper arXiv:2410.24164v1 Sec. III, Appendix B. Run on CPU: `uv run pytest pi/pi0/flow_matching -q`.
"""

import torch

from pi.pi0.action_expert import model as A
from pi.pi0.flow_matching import model as M
from pi.pi0.flow_matching import train as T

B, H, D = 2, A.ACTION_HORIZON, A.ACTION_DIM
P = 3 * 256 + 48


# ---------------------------------------------------------------- timestep distribution (pi0.py L197)
def test_sample_timestep_range_and_bias_towards_noise():
    g = torch.Generator().manual_seed(0)
    t = T.sample_timestep(200_000, generator=g)
    assert t.shape == (200_000,)
    assert t.min() >= 0.001 and t.max() <= 1.0
    # Beta(1.5, 1) has mean 1.5 / 2.5 = 0.6 -> after the affine map 0.6 * 0.999 + 0.001 = 0.6004
    assert abs(t.mean().item() - 0.6004) < 0.005
    assert (t > 0.5).float().mean() > 0.6  # most samples are on the noisy side (paper Fig. 14)


def test_sample_timestep_matches_torch_beta():
    """The inverse-CDF draw must follow the same law as torch.distributions.Beta(1.5, 1) (a distribution check)."""
    g = torch.Generator().manual_seed(1)
    ours = (T.sample_timestep(100_000, generator=g) - 0.001) / 0.999
    torch.manual_seed(1)
    ref = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0)).sample((100_000,))
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        assert abs(ours.quantile(q).item() - ref.quantile(q).item()) < 0.01


# ---------------------------------------------------------------- interpolation (pi0.py L198-L200)
def test_interpolate_endpoints_and_constant_velocity():
    a, n = torch.randn(B, H, D), torch.randn(B, H, D)
    x0, u0 = T.interpolate(a, n, torch.zeros(B))
    x1, u1 = T.interpolate(a, n, torch.ones(B))
    torch.testing.assert_close(x0, a)  # t = 0: clean actions
    torch.testing.assert_close(x1, n)  # t = 1: pure noise
    torch.testing.assert_close(u0, n - a)
    torch.testing.assert_close(u1, n - a)  # target does not depend on t
    xh, _ = T.interpolate(a, n, torch.full((B,), 0.5))
    torch.testing.assert_close(xh, 0.5 * (a + n))


# ---------------------------------------------------------------- sampler (pi0.py L216-L279)
def test_euler_with_oracle_velocity_recovers_actions_exactly():
    """Along the linear path the true velocity is constant (noise - actions), so 10 Euler steps from noise land on
    the actions with no discretization error. Also checks the step count and the time grid 1.0 ... 0.1."""
    a, n = torch.randn(B, H, D), torch.randn(B, H, D)
    seen = []

    def oracle(x_t, t):
        seen.append(t[0].item())
        return n - a

    x0 = M.sample_actions(oracle, n, num_steps=10)
    torch.testing.assert_close(x0, a, atol=1e-5, rtol=1e-5)
    assert len(seen) == 10
    torch.testing.assert_close(torch.tensor(seen), torch.tensor([1.0 - 0.1 * i for i in range(10)]), atol=1e-6, rtol=0)
    seen.clear()
    M.sample_actions(oracle, n, num_steps=4)
    assert len(seen) == 4


def test_sample_actions_shape_with_tiny_model():
    torch.manual_seed(0)
    vlm_cfg, exp_cfg = A.tiny_experts()
    llm, proj = A.MoEGemma((vlm_cfg, exp_cfg)).eval(), A.ActionProjections(exp_cfg).eval()
    prefix_emb, prefix_mask = torch.randn(B, P, vlm_cfg.width), torch.ones(B, P, dtype=torch.bool)
    prefix_ar = torch.zeros(P, dtype=torch.bool)
    with torch.no_grad():
        _, kv = llm([prefix_emb, None], prefix_mask.long().cumsum(1) - 1, A.make_attn_mask(prefix_mask, prefix_ar))
        v = M.make_velocity_fn(llm, proj, kv, prefix_mask, torch.randn(B, D))
        x0 = M.sample_actions(v, torch.randn(B, H, D))
    assert x0.shape == (B, H, D) and torch.isfinite(x0).all()


# ---------------------------------------------------------------- loss (pi0.py L212-L214)
def test_compute_loss_shape_and_zero_for_oracle():
    a, n, t = torch.randn(B, H, D), torch.randn(B, H, D), T.sample_timestep(B)
    loss = T.compute_loss(lambda x_t, t: n - a, a, n, t)
    assert loss.shape == (B, H)
    torch.testing.assert_close(loss, torch.zeros(B, H))
    loss2 = T.compute_loss(lambda x_t, t: torch.zeros_like(x_t), a, n, t)
    torch.testing.assert_close(loss2, (n - a).pow(2).mean(-1))  # includes every one of the 32 dims, padded or not


def test_train_and_inference_velocity_agree():
    """make_train_velocity_fn (joint forward, no cache) and make_velocity_fn (cached prefix) are the same function."""
    torch.manual_seed(0)
    vlm_cfg, exp_cfg = A.tiny_experts()
    llm, proj = A.MoEGemma((vlm_cfg, exp_cfg)).eval(), A.ActionProjections(exp_cfg).eval()
    prefix_emb, prefix_mask = torch.randn(B, P, vlm_cfg.width), torch.ones(B, P, dtype=torch.bool)
    prefix_mask[:, 512:768] = False
    prefix_ar = torch.zeros(P, dtype=torch.bool)
    state, x_t, t = torch.randn(B, D), torch.randn(B, H, D), T.sample_timestep(B)
    with torch.no_grad():
        _, kv = llm([prefix_emb, None], prefix_mask.long().cumsum(1) - 1, A.make_attn_mask(prefix_mask, prefix_ar))
        v_inf = M.make_velocity_fn(llm, proj, kv, prefix_mask, state)(x_t, t)
        v_train = T.make_train_velocity_fn(llm, proj, prefix_emb, prefix_mask, prefix_ar, state)(x_t, t)
    torch.testing.assert_close(v_inf, v_train, atol=1e-5, rtol=1e-5)
