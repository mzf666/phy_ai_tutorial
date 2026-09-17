"""视觉语言 backbone 的最小实现: patch 化, pixel shuffle, connector, 中间层抽取, VL 后处理.

上游: NVIDIA/Isaac-GR00T @ 4af2b622892f7dcb5aae5a3fb70bcb02dc217b96,
      gr00t/model/backbone/eagle_backbone.py L29-L133 (EagleBackbone),
      gr00t/model/backbone/eagle2_hg_model/modeling_eagle2_5_vl.py L138-L156, L297-L339,
      gr00t/model/action_head/flow_matching_action_head.py L199-L206, L263-L269.
论文: EgoScale arXiv:2602.16710v1 Sec. 2.3; GR00T N1 arXiv:2503.14734v2 Sec. 2.1.
许可: 上游 Isaac-GR00T 为 Apache-2.0; 本文件为 PyTorch 重写 (re-implements, does not copy).

只有推理路径. 冻结开关的语义在这里, 但优化器与 curriculum 在 ../train.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BackboneConfig:
    image_hw: tuple[int, int] | None
    patch: int | None
    n_views: int | None
    d_vit: int | None
    vit_layers: int | None
    vit_heads: int | None
    use_pixel_shuffle: bool | None
    downsample_ratio: float | None
    mlp_connector_layers: int | None
    d_llm: int | None
    llm_layers_total: int | None
    select_layer: int | None  # 取第几层的隐状态; 之后的层在构造时被删掉
    llm_heads: int | None
    vocab: int | None
    project_to_dim: int | None  # None 表示 Identity (上游 eagle_backbone.py L53-L56)
    use_vlln: bool | None
    vl_attn_layers: int | None
    vl_attn_heads: int | None
    vl_attn_head_dim: int | None


# 已发布的 nvidia/GR00T-N1.5-3B 的 config.json (2026-09-17 访问) 里能读到的值.
# 这些是 GR00T N1.5 的值, 不是 EgoScale 的披露值, 见 README Sec. 8.
N15_SOURCES = {
    "select_layer": "GR00T-N1.5-3B config.json backbone_cfg.select_layer = 12",
    "d_llm": "GR00T-N1.5-3B config.json hidden_size = 2048",
    "project_to_dim": "GR00T-N1.5-3B config.json backbone_cfg.project_to_dim = null",
    "use_vlln": "GR00T-N1.5-3B config.json action_head_cfg.use_vlln",
    "vl_attn_layers": "GR00T-N1.5-3B config.json vl_self_attention_cfg.num_layers = 4",
    "vl_attn_heads": "GR00T-N1.5-3B config.json vl_self_attention_cfg.num_attention_heads = 32",
    "vl_attn_head_dim": "GR00T-N1.5-3B config.json vl_self_attention_cfg.attention_head_dim = 64",
    "use_pixel_shuffle": "eagle_path '...1mlp_nops' -> no pixel shuffle",
    "mlp_connector_layers": "eagle_path '...1mlp_nops' -> 1 层 connector",
}


def paper() -> BackboneConfig:
    """EgoScale 没有披露 backbone 的任何结构参数, 全是 None 占位. 见 README Sec. 8."""
    return BackboneConfig(**{f.name: None for f in fields(BackboneConfig)})


def n15() -> BackboneConfig:
    """GR00T N1.5 已发布 checkpoint 里能读到的值. 逐值出处见 N15_SOURCES.

    没出现在 N15_SOURCES 里的字段仍然是 None: checkpoint 的 config 没给视觉塔的结构.
    """
    cfg = paper()
    cfg.select_layer = 12
    cfg.d_llm = 2048
    cfg.project_to_dim = None
    cfg.use_vlln = True
    cfg.vl_attn_layers = 4
    cfg.vl_attn_heads = 32
    cfg.vl_attn_head_dim = 64
    cfg.use_pixel_shuffle = False
    cfg.mlp_connector_layers = 1
    return cfg


def tiny() -> BackboneConfig:
    """CPU 上几秒钟跑通完整链路. 下面的结构值都不是论文值, 只为让代码路径可执行."""
    return BackboneConfig(
        image_hw=(32, 32),  # tiny only, not a paper value (上游 224x224)
        patch=8,  # tiny only, not a paper value
        n_views=3,  # 论文 Sec. 2.5: 头部 + 双腕三路相机
        d_vit=48,  # tiny only, not a paper value
        vit_layers=2,  # tiny only, not a paper value
        vit_heads=4,  # tiny only, not a paper value
        use_pixel_shuffle=True,  # tiny 打开以便走通这条分支 (N1.5 是关的)
        downsample_ratio=0.5,  # 上游 configuration_eagle2_5_vl.py L45 的默认
        mlp_connector_layers=2,  # tiny 用 2 层以便走通 LayerNorm 分支
        d_llm=64,  # tiny only, not a paper value (N1.5 是 2048)
        llm_layers_total=6,  # tiny only, not a paper value
        select_layer=4,  # tiny only, not a paper value (N1.5 是 12)
        llm_heads=4,  # tiny only, not a paper value
        vocab=256,  # tiny only, not a paper value
        project_to_dim=None,  # 与 N1.5 一致: Identity
        use_vlln=True,  # N1.5 是 True
        vl_attn_layers=2,  # tiny only (N1.5 是 4)
        vl_attn_heads=4,  # tiny only (N1.5 是 32)
        vl_attn_head_dim=16,  # tiny only (N1.5 是 64)
    )


# ---------------------------------------------------------------------------
# 1. 视觉塔
# ---------------------------------------------------------------------------
class Block(nn.Module):
    """标准 pre-LN Transformer 块. 上游用的是 SigLIP2 / Qwen3 的真实实现, 这里只保结构."""

    def __init__(self, d: int, heads: int, causal: bool = False) -> None:
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.causal = causal

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        h = self.n1(x)
        mask = None
        if self.causal:
            n = x.shape[1]
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), 1)
        a, _ = self.attn(h, h, h, need_weights=False, attn_mask=mask,
                         key_padding_mask=key_padding_mask)
        x = x + a
        return x + self.mlp(self.n2(x))


class VisionTower(nn.Module):
    """patch 化 + 若干 Transformer 块. 权重视为给定, 这里只复现结构与 token 数."""

    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        h, w = cfg.image_hw
        assert h % cfg.patch == 0 and w % cfg.patch == 0
        self.grid = (h // cfg.patch, w // cfg.patch)
        self.n_patch = self.grid[0] * self.grid[1]
        self.proj = nn.Conv2d(3, cfg.d_vit, cfg.patch, cfg.patch)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patch, cfg.d_vit))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(Block(cfg.d_vit, cfg.vit_heads) for _ in range(cfg.vit_layers))
        self.norm = nn.LayerNorm(cfg.d_vit)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) -> (N, n_patch, d_vit)."""
        x = self.proj(images).flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


def pixel_shuffle(x: torch.Tensor, scale_factor: float = 0.5) -> torch.Tensor:
    """上游 modeling_eagle2_5_vl.py L297-L309 的空间到通道重排, 逐行照搬其 view/permute 顺序.

    (N, w, h, c) -> (N, w*s, h*s, c/s^2). s = 0.5 时 token 数 /4, 通道 x4.
    这是重排而不是插值: 元素本身一个不少, 只是被重新分组.
    """
    n, w, h, c = x.size()
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(n, int(h * scale_factor), int(w * scale_factor),
               int(c / (scale_factor * scale_factor)))
    return x.permute(0, 2, 1, 3).contiguous()


def build_connector(cfg: BackboneConfig) -> nn.Module:
    """上游 modeling_eagle2_5_vl.py L138-L156 的三种形态."""
    inv = int(1 / cfg.downsample_ratio) ** 2 if cfg.use_pixel_shuffle else 1
    d_in = cfg.d_vit * inv
    if cfg.mlp_connector_layers == 2:
        # 上游 L142-L147: 只有这一种带 LayerNorm
        return nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, cfg.d_llm), nn.GELU(),
                             nn.Linear(cfg.d_llm, cfg.d_llm))
    if cfg.mlp_connector_layers == 1:
        return nn.Sequential(nn.Linear(d_in, cfg.d_llm))  # 上游 L149-L155
    raise ValueError(f"上游只实现了 1 层与 2 层 connector, 收到 {cfg.mlp_connector_layers}")


# ---------------------------------------------------------------------------
# 2. LLM 与中间层抽取
# ---------------------------------------------------------------------------
class LanguageModel(nn.Module):
    """只保留前 select_layer 层. 上游 eagle_backbone.py L59-L60 在构造时就把上层 pop 掉."""

    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        assert 0 < cfg.select_layer <= cfg.llm_layers_total, \
            f"select_layer {cfg.select_layer} 必须落在 (0, {cfg.llm_layers_total}]"
        self.embed = nn.Embedding(cfg.vocab, cfg.d_llm)
        self.n_dropped = cfg.llm_layers_total - cfg.select_layer
        # 被删掉的层在这里根本不构造: 省的是参数量与显存, 不只是一次前向
        self.layers = nn.ModuleList(
            Block(cfg.d_llm, cfg.llm_heads, causal=True) for _ in range(cfg.select_layer)
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        for blk in self.layers:
            x = blk(x, key_padding_mask=key_padding_mask)
        return x


class VLPostProcess(nn.Module):
    """上游把它放在 action head 里 (flow_matching_action_head.py L199-L206, L263-L269).

    use_vlln 同时控制 LayerNorm 与自注意力两者: 关掉时两者一起退化成 Identity, 不能只关一个.
    """

    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        d = cfg.project_to_dim or cfg.d_llm
        if cfg.use_vlln:
            inner = cfg.vl_attn_heads * cfg.vl_attn_head_dim
            assert inner == d, f"VL self-attention 的 inner dim {inner} 必须等于 backbone 宽度 {d}"
            self.vlln: nn.Module = nn.LayerNorm(d)
            self.attn: nn.Module = nn.ModuleList(
                Block(d, cfg.vl_attn_heads) for _ in range(cfg.vl_attn_layers)
            )
        else:
            self.vlln, self.attn = nn.Identity(), nn.ModuleList()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vlln(x)
        for blk in self.attn:  # 上游 SelfAttentionTransformer 的前向不带 mask
            x = blk(x)
        return x


class VisionLanguageBackbone(nn.Module):
    """图像 x 三路 + 指令 -> 条件向量 phi_t. 论文 Sec. 2.3."""

    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision = VisionTower(cfg)
        self.connector = build_connector(cfg)
        self.llm = LanguageModel(cfg)
        # 上游 eagle_backbone.py L53-L56: project_to_dim 为 None 时是 Identity
        self.project = nn.Linear(cfg.d_llm, cfg.project_to_dim) if cfg.project_to_dim \
            else nn.Identity()
        self.post = VLPostProcess(cfg)
        self.tune_llm, self.tune_visual = True, True

    @property
    def tokens_per_view(self) -> int:
        if not self.cfg.use_pixel_shuffle:
            return self.vision.n_patch
        s = self.cfg.downsample_ratio
        return int(self.vision.n_patch * s * s)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """(B, V, 3, H, W) -> (B, V * tokens_per_view, d_llm)."""
        b, v = images.shape[:2]
        x = self.vision(images.flatten(0, 1))  # (B*V, n_patch, d_vit)
        if self.cfg.use_pixel_shuffle:
            gh, gw = self.vision.grid
            x = pixel_shuffle(x.reshape(x.shape[0], gh, gw, -1), self.cfg.downsample_ratio)
            x = x.reshape(x.shape[0], -1, x.shape[-1])  # 上游 L330-L333
        x = self.connector(x)
        return x.reshape(b, v * x.shape[1], -1)

    def encode_sequence(
        self,
        images: torch.Tensor,
        view_mask: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """图像 token + 文本 token -> LLM 第 select_layer 层的隐状态, 以及有效位 mask.

        LLM 这一段是**带 mask** 的: 无效位既不做 key 也不做 value.
        """
        img = self.encode_images(images)
        txt = self.llm.embed(token_ids)
        seq = torch.cat((img, txt), dim=1)

        # 填黑的相机槽位仍然占 token (序列长度必须固定), 但在 attention 里被屏蔽.
        # 上游没有 per-view mask 的概念, 这一步是本仓库的处理, 见 README Sec. 8.
        img_valid = view_mask.repeat_interleave(self.tokens_per_view, dim=1)
        valid = torch.cat((img_valid, attention_mask), dim=1)

        h = self.llm(seq, key_padding_mask=~valid)
        return self.project(h), valid  # 上游 eagle_backbone.py L112

    def forward(
        self,
        images: torch.Tensor,
        view_mask: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        h, valid = self.encode_sequence(images, view_mask, token_ids, attention_mask)
        # 注意: 上游的 SelfAttentionTransformer.forward 只吃 hidden_states, **不吃任何 mask**
        # (cross_attention_dit.py L358-L375; 调用点 flow_matching_action_head.py L263-L269).
        # 因此无效位 (left padding 与填黑相机) 会通过这几层自注意力泄漏到有效位上; 屏蔽只在
        # 下游 DiT 的 cross-attention 里靠 encoder_attention_mask 生效. 这是上游的实际行为,
        # 本仓库照做并在 README Sec. 1.x 记录.
        return {"backbone_features": self.post(h), "backbone_attention_mask": valid}

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool) -> None:
        """上游 eagle_backbone.py L65-L81. 三阶段的冻结表见 ../train."""
        self.tune_llm, self.tune_visual = tune_llm, tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.llm.requires_grad_(False)
        if not tune_visual:
            self.vision.requires_grad_(False)
            self.connector.requires_grad_(False)  # 上游把 mlp1 归在 visual 一侧

    def set_frozen_modules_to_eval_mode(self) -> None:
        """上游 L83-L94. 不能省: HF Trainer 每步都调 model.train(), 冻结部分的 dropout 会复活."""
        if self.training:
            if not self.tune_llm:
                self.llm.eval()
            if not self.tune_visual:
                self.vision.eval()
                self.connector.eval()


# ---------------------------------------------------------------------------
# 3. 一次 tiny 运行
# ---------------------------------------------------------------------------
def _n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main() -> None:
    torch.manual_seed(0)
    cfg = tiny()
    model = VisionLanguageBackbone(cfg).eval()
    b, lang_len = 2, 7

    print(f"config: tiny(), image {cfg.image_hw} patch {cfg.patch} -> grid {model.vision.grid}, "
          f"V={cfg.n_views}, d_vit={cfg.d_vit}, d_llm={cfg.d_llm}")
    print(f"pixel_shuffle={cfg.use_pixel_shuffle} (ratio {cfg.downsample_ratio}) -> "
          f"{model.vision.n_patch} patches/view -> {model.tokens_per_view} tokens/view")
    print(f"LLM: {cfg.llm_layers_total} layers total, select_layer={cfg.select_layer} -> "
          f"{len(model.llm.layers)} kept, {model.llm.n_dropped} dropped at construction")

    images = torch.randn(b, cfg.n_views, 3, *cfg.image_hw)
    view_mask = torch.tensor([[True, True, True], [True, False, False]])
    token_ids = torch.randint(0, cfg.vocab, (b, lang_len))
    attn = torch.ones(b, lang_len, dtype=torch.bool)
    attn[1, :2] = False  # left padding, 上游 tokenizer 的 padding_side = "left"

    with torch.no_grad():
        img = model.encode_images(images)
        print(f"\n[1] images        {tuple(images.shape)}")
        print(f"[2] vision tower  -> (B*V, {model.vision.n_patch}, {cfg.d_vit})")
        print(f"[3] pixel shuffle -> (B*V, {model.tokens_per_view}, "
              f"{cfg.d_vit * int(1 / cfg.downsample_ratio) ** 2})")
        print(f"[4] connector     -> img {tuple(img.shape)}")
        out = model(images, view_mask, token_ids, attn)
        feats, mask = out["backbone_features"], out["backbone_attention_mask"]
        print(f"[5] + text {lang_len} tokens -> seq len "
              f"{cfg.n_views * model.tokens_per_view} + {lang_len} = {feats.shape[1]}")
        print(f"[6] LLM {len(model.llm.layers)} layers + project + VL post -> "
              f"{tuple(feats.shape)}")
        print(f"[7] attention mask valid per sample = {mask.sum(1).tolist()} / {mask.shape[1]}")
        print(f"    (sample 1 has 2 black camera slots and 2 left-pad text tokens)")
        print(f"    feature std = {feats.std().item():.3f}")

    print("\nparameters:")
    for name, mod in (("vision tower", model.vision), ("connector", model.connector),
                      ("llm (kept layers)", model.llm), ("project", model.project),
                      ("vl post-process", model.post)):
        print(f"  {name:20s} {_n_params(mod):>10,}")
    print(f"  {'total':20s} {_n_params(model):>10,}")

    full = tiny()
    full.select_layer = full.llm_layers_total
    dropped = _n_params(VisionLanguageBackbone(full)) - _n_params(model)
    print(f"\nkeeping only {cfg.select_layer}/{cfg.llm_layers_total} LLM layers saves "
          f"{dropped:,} parameters (they are never constructed)")

    model.set_trainable_parameters(tune_llm=False, tune_visual=True)
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"stage II freezing (tune_llm=False, tune_visual=True): "
          f"{tr:,} / {_n_params(model):,} parameters trainable")

    print("\nGR00T N1.5 disclosed values (NOT EgoScale's, see README Sec. 8):")
    n = n15()
    for k, src in N15_SOURCES.items():
        print(f"  {k:22s} = {getattr(n, k)!r:8}  <- {src}")


if __name__ == "__main__":
    main()
