"""对齐检查: token 布局与参数量, flow matching 的端点与 oracle 积分, 本体隔离, 时间步分布.

论文: EgoScale arXiv:2602.16710v1 Sec. 2.3, App. D.1.
上游: Isaac-GR00T@4af2b62 gr00t/model/action_head/flow_matching_action_head.py,
      gr00t/model/action_head/cross_attention_dit.py.
"""

from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

from gear.egoscale.dit.model import (
    N15_SOURCES,
    ActionExpert,
    CategorySpecificLinear,
    CategorySpecificMLP,
    DiTConfig,
    MultiEmbodimentActionEncoder,
    n15,
    paper,
    tiny,
)
from gear.egoscale.dit.train import (
    add_noise,
    discretize,
    flow_matching_loss,
    masked_mse,
    oracle_integrate,
    sample_time,
)

S_LEN = 19


def _n(m) -> int:
    return sum(p.numel() for p in m.parameters())


@pytest.fixture
def cfg():
    return tiny()


def _batch(cfg, b=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    phi = torch.randn(b, S_LEN, cfg.backbone_embedding_dim, generator=g)
    phi_mask = torch.ones(b, S_LEN, dtype=torch.bool)
    state = torch.randn(b, cfg.state_horizon, cfg.max_state_dim, generator=g)
    action = torch.randn(b, cfg.action_horizon, cfg.max_action_dim, generator=g)
    action_mask = torch.ones_like(action, dtype=torch.bool)
    emb_id = torch.arange(b) % cfg.max_num_embodiments
    has_proprio = torch.ones(b, dtype=torch.bool)
    return phi, phi_mask, state, action, action_mask, emb_id, has_proprio


# --------------------------------------------------------------------------
# shape / 参数量
# --------------------------------------------------------------------------
def test_token_layout(cfg):
    """上游 L325-L327: [state(Ts) | future_tokens(N) | action(H)]."""
    m = ActionExpert(cfg).eval()
    phi, phi_mask, state, _, _, emb_id, has_proprio = _batch(cfg)
    sf = m.encode_state(state, emb_id, has_proprio)
    af = m.encode_action(torch.randn(3, cfg.action_horizon, cfg.max_action_dim),
                         torch.zeros(3, dtype=torch.long), emb_id)
    tokens = m.build_tokens(sf, af)
    expected = cfg.state_horizon + cfg.num_target_vision_tokens + cfg.action_horizon
    assert tokens.shape == (3, expected, cfg.input_embedding_dim)


def test_velocity_takes_only_the_last_h_tokens(cfg):
    """上游 L344-L345: state token 与 future token 的输出被丢弃."""
    m = ActionExpert(cfg).eval()
    phi, phi_mask, state, _, _, emb_id, has_proprio = _batch(cfg)
    sf = m.encode_state(state, emb_id, has_proprio)
    af = m.encode_action(torch.randn(3, cfg.action_horizon, cfg.max_action_dim),
                         torch.zeros(3, dtype=torch.long), emb_id)
    v = m.velocity(m.build_tokens(sf, af), phi, torch.zeros(3, dtype=torch.long), emb_id,
                   phi_mask)
    assert v.shape == (3, cfg.action_horizon, cfg.max_action_dim)


def test_interleaved_blocks(cfg):
    """上游 L227-L231 / L281-L288: idx % 2 == 1 的层不接 encoder_hidden_states."""
    m = ActionExpert(cfg)
    assert len(m.dit.blocks) == cfg.num_layers
    assert m.dit.n_cross == cfg.num_layers // 2
    assert [b.cross for b in m.dit.blocks] == [True, False] * (cfg.num_layers // 2)

    off = ActionExpert(replace(cfg, interleave_self_attention=False))
    assert off.dit.n_cross == cfg.num_layers  # 全是 cross-attention


def test_category_specific_param_counts(cfg):
    """每个本体一套权重: C * (in*out + out). 用不到的槽位也占参数."""
    c, i, o = cfg.max_num_embodiments, 7, 5
    lin = CategorySpecificLinear(c, i, o)
    assert _n(lin) == c * (i * o + o)
    assert lin.W.shape == (c, i, o) and lin.b.shape == (c, o)

    mlp = CategorySpecificMLP(c, i, o, 3)
    assert _n(mlp) == c * (i * o + o) + c * (o * 3 + 3)

    enc = MultiEmbodimentActionEncoder(cfg.max_action_dim, cfg.input_embedding_dim, c)
    w, d = cfg.input_embedding_dim, cfg.max_action_dim
    assert _n(enc) == c * (d * w + w) + c * (2 * w * w + w) + c * (w * w + w)


def test_sample_shape(cfg):
    m = ActionExpert(cfg).eval()
    phi, phi_mask, state, _, _, emb_id, has_proprio = _batch(cfg)
    a = m.sample(phi, phi_mask, state, torch.ones_like(state, dtype=torch.bool), emb_id,
                 has_proprio)
    assert a.shape == (3, cfg.action_horizon, cfg.max_action_dim)


# --------------------------------------------------------------------------
# flow matching 的解析性质
# --------------------------------------------------------------------------
def test_noise_path_endpoints():
    """上游 L307: A_tau = (1-tau)*eps + tau*A. tau=0 得噪声, tau=1 得动作."""
    a = torch.randn(4, 8, 6)
    e = torch.randn(4, 8, 6)
    at0, _ = add_noise(a, e, torch.zeros(4))
    at1, _ = add_noise(a, e, torch.ones(4))
    assert torch.allclose(at0, e, atol=1e-6)
    assert torch.allclose(at1, a, atol=1e-6)


def test_velocity_target_sign_follows_upstream_not_the_paper():
    """上游 L308 是 A - eps; GR00T N1 论文式 (1) 写的是 eps - A. 见 README Sec. 1.x 第 2 条."""
    a = torch.randn(3, 5, 4)
    e = torch.randn(3, 5, 4)
    _, v = add_noise(a, e, torch.rand(3))
    assert torch.allclose(v, a - e)
    assert not torch.allclose(v, e - a)


@pytest.mark.parametrize("k", [1, 2, 4, 50])
def test_oracle_euler_is_exact_for_any_k(k):
    """路径是直线, 速度沿路径恒定, 所以理想速度场下任意步数都精确. 这是 K=4 够用的原因."""
    a = torch.randn(4, 8, 6)
    e = torch.randn(4, 8, 6)
    assert torch.allclose(oracle_integrate(a, e, k), a, atol=1e-4)


def test_discretization_range(cfg):
    """桶落在 [-1, num_timestep_buckets). 下界是 -1 而不是 0, 见 test_tau_can_be_slightly_negative."""
    tau = sample_time(60000, cfg, generator=torch.Generator().manual_seed(0))
    b = discretize(tau, cfg)
    assert b.min() >= -1 and b.max() < cfg.num_timestep_buckets
    assert b.dtype == torch.long
    assert (b >= 0).float().mean() > 0.99


def test_tau_can_be_slightly_negative(cfg):
    """s = 0.999 < 1, 所以 u > s 时 tau = (s - u)/s 会落到 (s-1)/s = -0.001.

    这是上游 L256-L258 的实际行为: 约 P(u > 0.999) = 1 - 0.999^1.5 ~ 0.15% 的样本 tau < 0,
    离散化后桶是 -1. 上游的时间步编码是正弦函数而不是查表, 所以负数不会崩; 但它确实意味着
    有极小一部分训练样本落在路径的 tau < 0 那一侧 (比纯噪声还远一点). 见 README Sec. 1.x.
    """
    tau = sample_time(200000, cfg, generator=torch.Generator().manual_seed(9))
    lower = (cfg.noise_s - 1.0) / cfg.noise_s
    assert tau.min() >= lower - 1e-6
    assert tau.max() <= 1.0
    frac = (tau < 0).float().mean().item()
    assert 0.0 < frac < 0.01
    assert abs(frac - (1 - cfg.noise_s ** cfg.noise_beta_alpha)) < 0.002


def test_masked_mse_matches_plain_mse_when_all_true():
    p = torch.randn(3, 5, 7)
    t = torch.randn(3, 5, 7)
    full = torch.ones_like(p, dtype=torch.bool)
    assert torch.allclose(masked_mse(p, t, full),
                          torch.nn.functional.mse_loss(p, t), atol=1e-6)

    partial = torch.zeros_like(p, dtype=torch.bool)
    partial[..., :4] = True
    assert torch.allclose(masked_mse(p, t, partial),
                          torch.nn.functional.mse_loss(p[..., :4], t[..., :4]), atol=1e-6)


def test_loss_is_zero_for_a_perfect_prediction():
    p = torch.randn(3, 5, 7)
    mask = torch.ones_like(p, dtype=torch.bool)
    assert masked_mse(p, p.clone(), mask).item() == pytest.approx(0.0, abs=1e-9)


def test_sample_time_rejects_undisclosed_config(cfg):
    with pytest.raises(ValueError):
        sample_time(4, replace(cfg, noise_s=None))


# --------------------------------------------------------------------------
# 占位 token 与本体隔离
# --------------------------------------------------------------------------
def test_placeholder_replaces_the_state_token(cfg):
    """论文 Sec. 2.3: has_proprio=False 时 state 的数值完全不影响输出."""
    m = ActionExpert(cfg).eval()
    _, _, state, _, _, emb_id, _ = _batch(cfg)
    human = torch.zeros(3, dtype=torch.bool)
    with torch.no_grad():
        a = m.encode_state(state, emb_id, human)
        b = m.encode_state(torch.randn_like(state) * 100, emb_id, human)
    assert torch.allclose(a, b)
    assert torch.allclose(a[0], m.state_placeholder[0].expand_as(a[0]))

    robot = torch.ones(3, dtype=torch.bool)
    with torch.no_grad():
        c = m.encode_state(state, emb_id, robot)
    assert not torch.allclose(c, a)


def test_placeholder_before_encoder_is_not_implemented(cfg):
    """论文 Sec. 2.3 没说注入在 encoder 前还是后; 另一条路只在 ledger 里登记, 不猜实现."""
    m = ActionExpert(replace(cfg, placeholder_before_encoder=True)).eval()
    _, _, state, _, _, emb_id, has_proprio = _batch(cfg)
    with pytest.raises(NotImplementedError):
        m.encode_state(state, emb_id, has_proprio)


def test_embodiment_adapters_are_isolated(cfg):
    """改一个本体槽位的适配器权重, 不能影响另一个本体的输出."""
    m = ActionExpert(cfg).eval()
    phi, phi_mask, state, _, _, _, _ = _batch(cfg, b=1)
    emb0 = torch.tensor([0])
    emb1 = torch.tensor([1])
    hp = torch.ones(1, dtype=torch.bool)

    with torch.no_grad():
        before = m.encode_state(state, emb0, hp).clone()
        m.state_encoder.layer1.W.data[1] += 5.0  # 只动本体 1 的权重
        after = m.encode_state(state, emb0, hp)
        other = m.encode_state(state, emb1, hp)
    assert torch.allclose(before, after)
    assert not torch.allclose(before, other)


def test_shared_dit_is_not_embodiment_specific(cfg):
    """论文 Sec. 2.3: DiT 与 backbone 完全共享, 只有两头分本体."""
    m = ActionExpert(cfg)
    names = [n for n, _ in m.dit.named_parameters()]
    assert names, "DiT 必须有参数"
    assert not any("W" in n and p.dim() == 3 for n, p in m.dit.named_parameters()), \
        "DiT 里不应出现按本体索引的 (C, in, out) 权重"


# --------------------------------------------------------------------------
# phi 掩码: 上游从不应用
# --------------------------------------------------------------------------
def test_phi_mask_is_ignored_by_default_and_honoured_when_switched_on(cfg):
    """上游 DiT.forward 两个调用点都硬写 encoder_attention_mask=None
    (cross_attention_dit.py L284, L291), 且 BasicTransformerBlock 里那行是注释掉的 (L167).

    默认照抄上游 (掩码无效); 打开 apply_phi_mask 后才真正生效. 见 README Sec. 1.x 第 1 条.
    """
    phi, phi_mask, state, _, _, emb_id, has_proprio = _batch(cfg, b=1)
    phi_mask[0, -6:] = False
    phi2 = phi.clone()
    phi2[0, -6:] = torch.randn_like(phi2[0, -6:]) * 10

    def run(c):
        m = ActionExpert(c).eval()
        torch.manual_seed(1)
        sf = m.encode_state(state, emb_id, has_proprio)
        af = m.encode_action(torch.zeros(1, c.action_horizon, c.max_action_dim),
                             torch.zeros(1, dtype=torch.long), emb_id)
        tok = m.build_tokens(sf, af)
        t = torch.zeros(1, dtype=torch.long)
        with torch.no_grad():
            return (m.velocity(tok, phi, t, emb_id, phi_mask),
                    m.velocity(tok, phi2, t, emb_id, phi_mask))

    torch.manual_seed(7)
    a, b = run(cfg)  # 上游默认: 掩码不生效
    assert not torch.allclose(a, b, atol=1e-5)

    torch.manual_seed(7)
    c, d = run(replace(cfg, apply_phi_mask=True))
    assert torch.allclose(c, d, atol=1e-5)


def test_apply_phi_mask_requires_a_mask(cfg):
    m = ActionExpert(replace(cfg, apply_phi_mask=True)).eval()
    phi, _, state, _, _, emb_id, has_proprio = _batch(cfg, b=1)
    sf = m.encode_state(state, emb_id, has_proprio)
    af = m.encode_action(torch.zeros(1, cfg.action_horizon, cfg.max_action_dim),
                         torch.zeros(1, dtype=torch.long), emb_id)
    with pytest.raises(AssertionError):
        m.velocity(m.build_tokens(sf, af), phi, torch.zeros(1, dtype=torch.long), emb_id, None)


# --------------------------------------------------------------------------
# 训练前向
# --------------------------------------------------------------------------
def test_training_forward_and_backward(cfg):
    m = ActionExpert(cfg)
    args = _batch(cfg, b=4, seed=2)
    phi, phi_mask, state, action, action_mask, emb_id, has_proprio = args
    has_proprio = torch.tensor([True, False, True, False])
    out = flow_matching_loss(m, phi, phi_mask, state, action, action_mask, emb_id,
                             has_proprio, generator=torch.Generator().manual_seed(3))
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
    assert out["pred"].shape == out["target"].shape == action.shape
    out["loss"].backward()
    assert m.state_placeholder.grad is not None
    assert m.state_placeholder.grad.abs().sum() > 0, "batch 里有人类样本, 占位 token 必须有梯度"
    assert m.dit.proj_out_2.weight.grad is not None


# --------------------------------------------------------------------------
# 分布检查
# --------------------------------------------------------------------------
def test_tau_distribution_matches_the_transformed_beta(cfg):
    """经验分位数与 Beta(1.5,1) 经 tau = (s-u)/s 变换后的解析分位数一致."""
    tau = sample_time(60000, cfg, generator=torch.Generator().manual_seed(5))
    # u = p^(1/alpha) 的 p 分位数 -> tau = (s - u)/s, 且变换是递减的
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        u_q = (1 - q) ** (1 / cfg.noise_beta_alpha)
        expected = (cfg.noise_s - u_q) / cfg.noise_s
        assert abs(tau.quantile(q).item() - expected) < 0.02, q
    assert tau.mean().item() < 0.5, "Beta(1.5,1) 变换后质量偏向高噪声端"
    assert tau.max() <= 1.0


def test_paper_config_is_all_placeholders():
    cfg = paper()
    structural = [f.name for f in fields(DiTConfig)
                  if f.name not in ("apply_phi_mask", "placeholder_before_encoder")]
    assert all(getattr(cfg, n) is None for n in structural)
    assert cfg.apply_phi_mask is False  # 上游默认: 掩码不生效
    with pytest.raises(TypeError):
        ActionExpert(cfg)


def test_n15_values_all_carry_a_source():
    c = n15()
    for key, src in N15_SOURCES.items():
        assert getattr(c, key) is not None and src
    assert (c.num_layers, c.num_attention_heads, c.attention_head_dim) == (16, 32, 48)
    assert c.num_attention_heads * c.attention_head_dim == c.input_embedding_dim == 1536
    assert c.hidden_size == 1024 and c.backbone_embedding_dim == 2048
    assert c.state_horizon + c.num_target_vision_tokens + c.action_horizon == 49
