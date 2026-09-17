"""对齐检查: token 数与参数量, pixel shuffle 的重排性质, 屏蔽的有效性.

不加载任何真实权重, 不复现任何下游数字.
论文: EgoScale arXiv:2602.16710v1 Sec. 2.3; GR00T N1 arXiv:2503.14734v2 Sec. 2.1.
上游: Isaac-GR00T@4af2b62 gr00t/model/backbone/eagle_backbone.py L29-L133 等.
"""

from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

from gear.egoscale.backbone.model import (
    N15_SOURCES,
    BackboneConfig,
    LanguageModel,
    VisionLanguageBackbone,
    VLPostProcess,
    build_connector,
    n15,
    paper,
    pixel_shuffle,
    tiny,
)


def _n(m) -> int:
    return sum(p.numel() for p in m.parameters())


def _inputs(cfg, b=2, lang=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(b, cfg.n_views, 3, *cfg.image_hw, generator=g)
    view_mask = torch.ones(b, cfg.n_views, dtype=torch.bool)
    token_ids = torch.randint(0, cfg.vocab, (b, lang), generator=g)
    attn = torch.ones(b, lang, dtype=torch.bool)
    return images, view_mask, token_ids, attn


# --------------------------------------------------------------------------
# pixel shuffle
# --------------------------------------------------------------------------
def test_pixel_shuffle_shape():
    """上游 modeling_eagle2_5_vl.py L297-L309: (N,w,h,c) -> (N,w*s,h*s,c/s^2)."""
    x = torch.randn(2, 8, 8, 16)
    y = pixel_shuffle(x, 0.5)
    assert y.shape == (2, 4, 4, 64)
    assert y.numel() == x.numel()


def test_pixel_shuffle_is_a_rearrangement_not_a_resize():
    """元素多重集不变: 它把四个相邻 patch 拼成一个更宽的 token, 不做任何插值或丢弃."""
    x = torch.arange(2 * 8 * 8 * 16, dtype=torch.float32).reshape(2, 8, 8, 16)
    y = pixel_shuffle(x, 0.5)
    assert torch.equal(torch.sort(x.flatten()).values, torch.sort(y.flatten()).values)


def test_pixel_shuffle_upstream_example():
    """上游注释里的例子 (L327-L333): [B,1024,1024] -> [B,16,16,4096] -> [B,256,4096]."""
    vit = torch.randn(1, 1024, 1024)
    h = w = int(vit.shape[1] ** 0.5)
    y = pixel_shuffle(vit.reshape(1, h, w, -1), 0.5)
    assert y.shape == (1, 16, 16, 4096)
    assert y.reshape(1, -1, y.shape[-1]).shape == (1, 256, 4096)


@pytest.mark.parametrize("shuffle,expected_factor", [(False, 1), (True, 4)])
def test_tokens_per_view(shuffle, expected_factor):
    cfg = replace(tiny(), use_pixel_shuffle=shuffle)
    m = VisionLanguageBackbone(cfg)
    assert m.tokens_per_view == m.vision.n_patch // expected_factor


# --------------------------------------------------------------------------
# connector
# --------------------------------------------------------------------------
def test_connector_variants_param_counts():
    """上游 L138-L156 的三种形态. 只有 2 层那种带 LayerNorm."""
    cfg = tiny()
    d_in_ps = cfg.d_vit * 4  # pixel shuffle 下输入宽度 x4
    two = build_connector(replace(cfg, mlp_connector_layers=2, use_pixel_shuffle=True))
    assert _n(two) == 2 * d_in_ps + (d_in_ps * cfg.d_llm + cfg.d_llm) + \
        (cfg.d_llm * cfg.d_llm + cfg.d_llm)
    assert isinstance(two[0], torch.nn.LayerNorm)

    one_ps = build_connector(replace(cfg, mlp_connector_layers=1, use_pixel_shuffle=True))
    assert _n(one_ps) == d_in_ps * cfg.d_llm + cfg.d_llm
    assert len(one_ps) == 1

    one_no = build_connector(replace(cfg, mlp_connector_layers=1, use_pixel_shuffle=False))
    assert _n(one_no) == cfg.d_vit * cfg.d_llm + cfg.d_llm


def test_unsupported_connector_depth_is_rejected():
    with pytest.raises(ValueError):
        build_connector(replace(tiny(), mlp_connector_layers=3))


# --------------------------------------------------------------------------
# select_layer: 上层被物理删除
# --------------------------------------------------------------------------
def test_select_layer_drops_layers_and_parameters():
    """上游 eagle_backbone.py L59-L60 在构造时 pop 掉上层, 省的是参数量不只是一次前向."""
    cfg = tiny()
    kept = LanguageModel(cfg)
    full = LanguageModel(replace(cfg, select_layer=cfg.llm_layers_total))
    assert len(kept.layers) == cfg.select_layer
    assert kept.n_dropped == cfg.llm_layers_total - cfg.select_layer

    per_layer = _n(full.layers[0])
    assert _n(full) - _n(kept) == kept.n_dropped * per_layer


def test_select_layer_out_of_range_is_rejected():
    with pytest.raises(AssertionError):
        LanguageModel(replace(tiny(), select_layer=99))


# --------------------------------------------------------------------------
# VL 后处理
# --------------------------------------------------------------------------
def test_vlln_switch_controls_both_pieces():
    """上游 L199-L206: use_vlln 同时决定 LayerNorm 与自注意力, 不能只关一个."""
    off = VLPostProcess(replace(tiny(), use_vlln=False))
    assert isinstance(off.vlln, torch.nn.Identity) and len(off.attn) == 0
    x = torch.randn(2, 5, tiny().d_llm)
    assert torch.equal(off(x), x)

    on = VLPostProcess(tiny())
    assert isinstance(on.vlln, torch.nn.LayerNorm) and len(on.attn) == tiny().vl_attn_layers


def test_vl_attention_inner_dim_must_match_backbone_width():
    """N1.5 的 32 x 64 = 2048 正好等于 hidden_size; 不相等说明配置读错了."""
    with pytest.raises(AssertionError):
        VLPostProcess(replace(tiny(), vl_attn_head_dim=7))
    n = n15()
    assert n.vl_attn_heads * n.vl_attn_head_dim == n.d_llm == 2048


# --------------------------------------------------------------------------
# 前向: shape 与屏蔽
# --------------------------------------------------------------------------
def test_forward_shapes():
    cfg = tiny()
    m = VisionLanguageBackbone(cfg).eval()
    images, view_mask, ids, attn = _inputs(cfg)
    out = m(images, view_mask, ids, attn)
    s = cfg.n_views * m.tokens_per_view + ids.shape[1]
    assert out["backbone_features"].shape == (2, s, cfg.d_llm)
    assert out["backbone_attention_mask"].shape == (2, s)
    assert out["backbone_attention_mask"].all()


def test_masked_camera_slot_is_isolated_inside_the_llm():
    """填黑的槽位仍然占 token (序列长度固定), 在 LLM 的 attention 里被屏蔽, 见 README Sec. 8."""
    cfg = tiny()
    m = VisionLanguageBackbone(cfg).eval()
    images, view_mask, ids, attn = _inputs(cfg, b=1)
    view_mask[0, 1:] = False  # 只有头部相机是真的

    with torch.no_grad():
        ha, valid = m.encode_sequence(images, view_mask, ids, attn)
        images2 = images.clone()
        images2[0, 1:] = torch.randn_like(images2[0, 1:])  # 改掉被屏蔽槽位的像素
        hb, _ = m.encode_sequence(images2, view_mask, ids, attn)

    v = valid[0]
    assert torch.allclose(ha[0][v], hb[0][v], atol=1e-5)
    assert not torch.allclose(ha[0][~v], hb[0][~v], atol=1e-5)  # 被屏蔽位置本身会变


def test_left_padded_text_is_isolated_inside_the_llm():
    """上游 tokenizer 的 padding_side = 'left' (transforms.py L51)."""
    cfg = tiny()
    m = VisionLanguageBackbone(cfg).eval()
    images, view_mask, ids, attn = _inputs(cfg, b=1)
    attn[0, :3] = False

    with torch.no_grad():
        ha, valid = m.encode_sequence(images, view_mask, ids, attn)
        ids2 = ids.clone()
        ids2[0, :3] = (ids2[0, :3] + 17) % cfg.vocab  # 改掉 pad 位置的 token id
        hb, _ = m.encode_sequence(images, view_mask, ids2, attn)

    v = valid[0]
    assert torch.allclose(ha[0][v], hb[0][v], atol=1e-5)


def test_vl_self_attention_is_unmasked_upstream_so_isolation_leaks():
    """上游的 SelfAttentionTransformer.forward 只吃 hidden_states, 不吃任何 mask
    (cross_attention_dit.py L358-L375; 调用点 flow_matching_action_head.py L263-L269).

    后果: 无效位 (left padding 与填黑相机) 会经由这几层自注意力泄漏到有效位上, 屏蔽只在下游
    DiT 的 cross-attention 里靠 encoder_attention_mask 生效. 这是上游的实际行为, 不是本仓库
    的 bug; 见 README Sec. 1.x. 关掉 use_vlln 后泄漏消失, 这正好说明泄漏来自这几层.
    """
    cfg = tiny()
    m = VisionLanguageBackbone(cfg).eval()
    images, view_mask, ids, attn = _inputs(cfg, b=1)
    view_mask[0, 1:] = False

    with torch.no_grad():
        a = m(images, view_mask, ids, attn)
        images2 = images.clone()
        images2[0, 1:] = torch.randn_like(images2[0, 1:])
        b = m(images2, view_mask, ids, attn)
    v = a["backbone_attention_mask"][0]
    assert not torch.allclose(a["backbone_features"][0][v], b["backbone_features"][0][v],
                              atol=1e-5), "泄漏本该发生; 若此断言失败说明后处理被加了 mask"

    off = VisionLanguageBackbone(replace(cfg, use_vlln=False)).eval()
    with torch.no_grad():
        c = off(images, view_mask, ids, attn)
        d = off(images2, view_mask, ids, attn)
    assert torch.allclose(c["backbone_features"][0][v], d["backbone_features"][0][v], atol=1e-5)


def test_freezing_switches_match_upstream_semantics():
    """上游 L65-L81 把 mlp1 (connector) 归在 visual 一侧; L83-L94 把冻结部分切到 eval."""
    m = VisionLanguageBackbone(tiny())
    m.set_trainable_parameters(tune_llm=False, tune_visual=True)
    assert not any(p.requires_grad for p in m.llm.parameters())
    assert all(p.requires_grad for p in m.vision.parameters())
    assert all(p.requires_grad for p in m.connector.parameters())

    m.train()
    m.set_frozen_modules_to_eval_mode()
    assert not m.llm.training and m.vision.training

    m.set_trainable_parameters(tune_llm=True, tune_visual=False)
    assert all(p.requires_grad for p in m.llm.parameters())
    assert not any(p.requires_grad for p in m.connector.parameters())


# --------------------------------------------------------------------------
# 配置的出处
# --------------------------------------------------------------------------
def test_paper_config_is_all_placeholders():
    """EgoScale 没披露 backbone 的任何结构参数. 见 README Sec. 8."""
    cfg = paper()
    assert all(getattr(cfg, f.name) is None for f in fields(BackboneConfig))
    with pytest.raises(TypeError):
        VisionLanguageBackbone(cfg)


def test_n15_values_all_carry_a_source():
    """n15() 里每个非 None 的字段都必须在 N15_SOURCES 里有出处, 否则就是猜的."""
    cfg = n15()
    named = {f.name for f in fields(BackboneConfig) if getattr(cfg, f.name) is not None}
    # project_to_dim 与 use_pixel_shuffle 的值是 None / False, 单独检查
    assert named | {"project_to_dim", "use_pixel_shuffle"} <= set(N15_SOURCES) | named
    for key in N15_SOURCES:
        assert hasattr(cfg, key)
    assert (cfg.select_layer, cfg.d_llm, cfg.vl_attn_layers) == (12, 2048, 4)
    assert cfg.project_to_dim is None and cfg.use_pixel_shuffle is False
    # checkpoint 的 config 没给视觉塔结构, 这些必须仍然是 None
    assert cfg.d_vit is None and cfg.patch is None and cfg.image_hw is None


# --------------------------------------------------------------------------
# 分布检查
# --------------------------------------------------------------------------
def test_activations_do_not_blow_up():
    cfg = tiny()
    m = VisionLanguageBackbone(cfg).eval()
    images, view_mask, ids, attn = _inputs(cfg, b=4, seed=3)
    with torch.no_grad():
        f = m(images, view_mask, ids, attn)["backbone_features"]
    assert torch.isfinite(f).all()
    assert 0.1 < f.std().item() < 10.0
