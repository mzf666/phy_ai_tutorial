"""共享的 flow-matching DiT action expert + 本体专属的两头适配器. 只有推理路径.

上游: NVIDIA/Isaac-GR00T @ 4af2b622892f7dcb5aae5a3fb70bcb02dc217b96,
      gr00t/model/action_head/flow_matching_action_head.py L30-L98, L166-L256, L349-L404,
      gr00t/model/action_head/cross_attention_dit.py L31-L67, L70-L188, L191-L307.
论文: EgoScale arXiv:2602.16710v1 Sec. 2.3, App. D.1; GR00T N1 arXiv:2503.14734v2 Sec. 2.1.
许可: 上游 Isaac-GR00T 为 Apache-2.0; 本文件为 PyTorch 重写 (re-implements, does not copy).

时间步分布、速度目标与 loss 在 train.py.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiTConfig:
    # --- 序列 ---
    state_horizon: int | None
    action_horizon: int | None  # H
    num_target_vision_tokens: int | None  # future_tokens 的个数
    max_state_dim: int | None
    max_action_dim: int | None
    max_num_embodiments: int | None  # C
    # --- 宽度 ---
    input_embedding_dim: int | None  # W, DiT 的 inner_dim
    backbone_embedding_dim: int | None  # cross_attention_dim
    hidden_size: int | None  # DiT 的 output_dim, 也是 action decoder 的输入宽度
    # --- DiT ---
    num_layers: int | None
    num_attention_heads: int | None
    attention_head_dim: int | None
    dropout: float | None
    interleave_self_attention: bool | None
    # --- flow matching ---
    noise_beta_alpha: float | None
    noise_beta_beta: float | None
    noise_s: float | None
    num_timestep_buckets: int | None
    num_inference_timesteps: int | None  # K
    add_pos_embed: bool | None
    max_seq_len: int | None
    # --- 本仓库的显式开关 (上游没有这两个) ---
    apply_phi_mask: bool = False  # 上游从不应用 phi 的掩码, 见 README Sec. 1.x 第 1 条
    placeholder_before_encoder: bool = False  # 占位 token 注入在 state_encoder 前还是后


# 已发布的 nvidia/GR00T-N1.5-3B 的 config.json (2026-09-17 访问). 这些是 GR00T N1.5 的值,
# 不是 EgoScale 的披露值, 见 README Sec. 8.
N15_SOURCES = {
    "action_horizon": "action_head_cfg.action_horizon = 16",
    "num_target_vision_tokens": "action_head_cfg.num_target_vision_tokens = 32",
    "max_state_dim": "action_head_cfg.max_state_dim = 64",
    "max_action_dim": "action_head_cfg.max_action_dim = 32",
    "input_embedding_dim": "action_head_cfg.input_embedding_dim = 1536",
    "backbone_embedding_dim": "action_head_cfg.backbone_embedding_dim = 2048",
    "hidden_size": "action_head_cfg.hidden_size = 1024",
    "num_layers": "diffusion_model_cfg.num_layers = 16",
    "num_attention_heads": "diffusion_model_cfg.num_attention_heads = 32",
    "attention_head_dim": "diffusion_model_cfg.attention_head_dim = 48",
    "dropout": "diffusion_model_cfg.dropout = 0.2",
    "interleave_self_attention": "diffusion_model_cfg.interleave_self_attention = true",
    "noise_beta_alpha": "action_head_cfg.noise_beta_alpha = 1.5",
    "noise_beta_beta": "action_head_cfg.noise_beta_beta = 1.0",
    "noise_s": "action_head_cfg.noise_s = 0.999",
    "num_timestep_buckets": "action_head_cfg.num_timestep_buckets = 1000",
    "num_inference_timesteps": "action_head_cfg.num_inference_timesteps = 4",
    "add_pos_embed": "action_head_cfg.add_pos_embed = true",
}


def paper() -> DiTConfig:
    """EgoScale 没披露 DiT 的任何结构参数, 全是 None 占位. 见 README Sec. 8."""
    return DiTConfig(**{f.name: None for f in fields(DiTConfig)
                        if f.name not in ("apply_phi_mask", "placeholder_before_encoder")})


def n15() -> DiTConfig:
    """GR00T N1.5 已发布 checkpoint 的值. 逐值出处见 N15_SOURCES."""
    cfg = paper()
    cfg.state_horizon = 1  # 上游 data_config.py L165: observation_indices = [0]
    cfg.action_horizon = 16
    cfg.num_target_vision_tokens = 32
    cfg.max_state_dim = 64
    cfg.max_action_dim = 32
    cfg.max_num_embodiments = 32  # 上游 flow_matching_action_head.py L137
    cfg.input_embedding_dim = 1536
    cfg.backbone_embedding_dim = 2048
    cfg.hidden_size = 1024
    cfg.num_layers = 16
    cfg.num_attention_heads = 32
    cfg.attention_head_dim = 48
    cfg.dropout = 0.2
    cfg.interleave_self_attention = True
    cfg.noise_beta_alpha = 1.5
    cfg.noise_beta_beta = 1.0
    cfg.noise_s = 0.999
    cfg.num_timestep_buckets = 1000
    cfg.num_inference_timesteps = 4
    cfg.add_pos_embed = True
    cfg.max_seq_len = 1024  # 上游 flow_matching_action_head.py L122
    return cfg


def tiny() -> DiTConfig:
    """CPU 上几秒钟跑通. 结构值都不是论文值, 只为让代码路径可执行; flow matching 的分布参数
    是上游披露值, 原样保留."""
    return DiTConfig(
        state_horizon=1,  # 上游 observation_indices = [0]
        action_horizon=8,  # tiny only, not a paper value (N1.5 是 16)
        num_target_vision_tokens=4,  # tiny only (N1.5 是 32)
        max_state_dim=64,  # tiny only (与 ../data 的 tiny 对齐)
        max_action_dim=64,  # tiny only (与 ../data 的 tiny 对齐; N1.5 是 32)
        max_num_embodiments=4,  # tiny only (上游是 32)
        input_embedding_dim=48,  # tiny only (N1.5 是 1536)
        backbone_embedding_dim=64,  # tiny only (N1.5 是 2048; 要等于 ../backbone 的 d_llm)
        hidden_size=32,  # tiny only (N1.5 是 1024)
        num_layers=4,  # tiny only (N1.5 是 16)
        num_attention_heads=4,  # tiny only (N1.5 是 32)
        attention_head_dim=12,  # tiny only (4 x 12 = 48 = input_embedding_dim)
        dropout=0.0,  # tiny only, 关掉以便测试可复现 (N1.5 是 0.2)
        interleave_self_attention=True,  # N1.5 的值
        noise_beta_alpha=1.5,  # 上游披露值
        noise_beta_beta=1.0,  # 上游披露值
        noise_s=0.999,  # 上游披露值
        num_timestep_buckets=1000,  # 上游披露值
        num_inference_timesteps=4,  # GR00T N1 的 K = 4
        add_pos_embed=True,  # N1.5 的值
        max_seq_len=1024,  # 上游披露值
    )


# ---------------------------------------------------------------------------
# 1. 本体专属适配器 (上游 flow_matching_action_head.py L30-L98)
# ---------------------------------------------------------------------------
def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class CategorySpecificLinear(nn.Module):
    """每个本体一套权重, 前向按 embodiment_id 取 (上游 L30-L42).

    参数量是 C * (in*out + out): 用不到的本体槽位一样占参数.
    """

    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_categories = num_categories
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        return torch.bmm(x, self.W[cat_ids]) + self.b[cat_ids].unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    """两层, 中间 ReLU (上游 L45-L54)."""

    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int,
                 output_dim: int) -> None:
        super().__init__()
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.relu(self.layer1(x, cat_ids)), cat_ids)


class SinusoidalPositionalEncoding(nn.Module):
    """(B, T) 的时间步 -> (B, T, w) 的正弦编码 (上游 action_encoder.py L24-L52)."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        half = self.embedding_dim // 2
        exponent = -torch.arange(half, dtype=torch.float, device=timesteps.device) * (
            math.log(10000.0) / half
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class MultiEmbodimentActionEncoder(nn.Module):
    """noisy action + 时间步 -> token (上游 L56-L98). W1 -> concat(tau) -> W2 + swish -> W3."""

    def __init__(self, action_dim: int, hidden_size: int, num_embodiments: int) -> None:
        super().__init__()
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor,
                cat_ids: torch.Tensor) -> torch.Tensor:
        b, t, _ = actions.shape
        if timesteps.dim() != 1 or timesteps.shape[0] != b:
            raise ValueError("timesteps 必须是 (B,), 才能沿 T 复制 (上游 L80-L86)")
        a_emb = self.W1(actions, cat_ids)
        tau_emb = self.pos_encoding(timesteps.unsqueeze(1).expand(-1, t)).to(a_emb.dtype)
        x = swish(self.W2(torch.cat([a_emb, tau_emb], dim=-1), cat_ids))
        return self.W3(x, cat_ids)


# ---------------------------------------------------------------------------
# 2. DiT (上游 cross_attention_dit.py L31-L307)
# ---------------------------------------------------------------------------
class TimestepEncoder(nn.Module):
    """离散桶 -> 正弦投影 (256 通道) -> MLP (上游 L31-L41 用 diffusers 的 Timesteps)."""

    def __init__(self, embedding_dim: int, num_channels: int = 256) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.mlp = nn.Sequential(nn.Linear(num_channels, embedding_dim), nn.SiLU(),
                                 nn.Linear(embedding_dim, embedding_dim))

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.num_channels // 2
        # diffusers 的 Timesteps 用 downscale_freq_shift=1, flip_sin_to_cos=True
        exponent = -math.log(10000.0) * torch.arange(
            half, dtype=torch.float32, device=timesteps.device
        ) / half
        emb = timesteps.float().unsqueeze(-1) * exponent.exp()
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)  # flip_sin_to_cos
        return self.mlp(emb)


class AdaLayerNorm(nn.Module):
    """上游 L44-L67. 注意 chunk 顺序是 scale 在前 shift 在后 (L64)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, eps, elementwise_affine=False)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(self.silu(temb)).chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    """上游 BasicTransformerBlock (L70-L188): AdaLN -> attention -> LayerNorm -> FF, 两处残差.

    cross_attention_dim 为 None 时退化成纯自注意力 —— 这正是 interleave_self_attention
    在奇数层做的事 (上游 L227-L231, L281-L288).
    """

    def __init__(self, dim: int, heads: int, head_dim: int, dropout: float,
                 cross_attention_dim: int | None) -> None:
        super().__init__()
        self.cross = cross_attention_dim is not None
        inner = heads * head_dim
        assert inner == dim, f"heads*head_dim ({inner}) 必须等于 DiT 宽度 ({dim})"
        self.norm1 = AdaLayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True,
            kdim=cross_attention_dim, vdim=cross_attention_dim,
        )
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(approximate="tanh"),
                                nn.Dropout(dropout), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor, temb: torch.Tensor,
                encoder_hidden_states: torch.Tensor | None = None,
                encoder_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x, temb)
        kv = encoder_hidden_states if self.cross else h
        a, _ = self.attn(h, kv, kv, need_weights=False,
                         key_padding_mask=encoder_key_padding_mask if self.cross else None)
        x = x + a
        return x + self.ff(self.norm3(x))


class DiT(nn.Module):
    """上游 L191-L307. 偶数层 cross, 奇数层 self (当 interleave_self_attention 为真)."""

    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        dim = cfg.num_attention_heads * cfg.attention_head_dim
        self.dim = dim
        self.timestep_encoder = TimestepEncoder(dim)
        self.blocks = nn.ModuleList()
        for idx in range(cfg.num_layers):
            use_self = idx % 2 == 1 and cfg.interleave_self_attention
            self.blocks.append(DiTBlock(
                dim, cfg.num_attention_heads, cfg.attention_head_dim, cfg.dropout,
                None if use_self else cfg.backbone_embedding_dim,
            ))
        self.norm_out = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.proj_out_1 = nn.Linear(dim, 2 * dim)
        self.proj_out_2 = nn.Linear(dim, cfg.hidden_size)

    @property
    def n_cross(self) -> int:
        return sum(b.cross for b in self.blocks)

    def forward(self, x: torch.Tensor, phi: torch.Tensor, timestep: torch.Tensor,
                phi_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        temb = self.timestep_encoder(timestep)
        for blk in self.blocks:
            x = blk(x, temb, phi, phi_key_padding_mask)
        # 输出头, 上游 L299-L305: 这里 chunk 的顺序是 shift 在前, 与 AdaLayerNorm 相反
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        x = self.norm_out(x) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(x)


# ---------------------------------------------------------------------------
# 3. action expert (上游 FlowmatchingActionHead L166-L256, L349-L404)
# ---------------------------------------------------------------------------
class ActionExpert(nn.Module):
    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        w, c = cfg.input_embedding_dim, cfg.max_num_embodiments
        self.dit = DiT(cfg)
        self.state_encoder = CategorySpecificMLP(c, cfg.max_state_dim, w, w)
        self.action_encoder = MultiEmbodimentActionEncoder(cfg.max_action_dim, w, c)
        self.action_decoder = CategorySpecificMLP(c, cfg.hidden_size, cfg.hidden_size,
                                                  cfg.max_action_dim)
        self.future_tokens = nn.Embedding(cfg.num_target_vision_tokens, w)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)  # 上游 L197
        # 论文 Sec. 2.3: 人类演示没有 q_t, 换成一个可学习的占位 token
        self.state_placeholder = nn.Parameter(torch.zeros(1, 1, w))
        nn.init.normal_(self.state_placeholder, mean=0.0, std=0.02)
        if cfg.add_pos_embed:
            self.position_embedding = nn.Embedding(cfg.max_seq_len, w)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)  # 上游 L208

    # -- 组件 ---------------------------------------------------------------
    def encode_state(self, state: torch.Tensor, embodiment_id: torch.Tensor,
                     has_proprio: torch.Tensor) -> torch.Tensor:
        """(B, Ts, max_state_dim) -> (B, Ts, W), 人类样本换成占位 token.

        论文 Sec. 2.3 没说占位注入在 encoder 前还是后; 默认在后, 见 README Sec. 8.
        """
        if self.cfg.placeholder_before_encoder:
            raise NotImplementedError(
                "占位 token 注入在 encoder 之前这条路只在 README Sec. 8 里登记, 未实现"
            )
        feats = self.state_encoder(state, embodiment_id)
        ph = self.state_placeholder.expand(feats.shape[0], feats.shape[1], -1)
        return torch.where(has_proprio[:, None, None], feats, ph)

    def encode_action(self, noisy: torch.Tensor, t_bucket: torch.Tensor,
                      embodiment_id: torch.Tensor) -> torch.Tensor:
        feats = self.action_encoder(noisy, t_bucket, embodiment_id)
        if self.cfg.add_pos_embed:  # 上游 L318-L321
            pos = torch.arange(feats.shape[1], device=feats.device)
            feats = feats + self.position_embedding(pos).unsqueeze(0)
        return feats

    def build_tokens(self, state_feats: torch.Tensor,
                     action_feats: torch.Tensor) -> torch.Tensor:
        """[state(Ts), future_tokens(N), action(H)] (上游 L325-L327)."""
        fut = self.future_tokens.weight.unsqueeze(0).expand(state_feats.shape[0], -1, -1)
        return torch.cat((state_feats, fut, action_feats), dim=1)

    def velocity(self, tokens: torch.Tensor, phi: torch.Tensor, t_bucket: torch.Tensor,
                 embodiment_id: torch.Tensor,
                 phi_mask: torch.Tensor | None = None) -> torch.Tensor:
        """DiT + action decoder, 取最后 H 个 token (上游 L336-L345)."""
        kpm = None
        if self.cfg.apply_phi_mask:
            # 上游从不走这条路: DiT.forward 的两个调用点都硬写 encoder_attention_mask=None
            # (cross_attention_dit.py L284, L291), 且 L167 那行本身是注释掉的.
            # 见 README Sec. 1.x 第 1 条.
            assert phi_mask is not None, "apply_phi_mask=True 时必须给 phi_mask"
            kpm = ~phi_mask
        out = self.dit(tokens, phi, t_bucket, kpm)
        pred = self.action_decoder(out, embodiment_id)
        return pred[:, -self.cfg.action_horizon:]

    # -- 冻结开关 (上游 L217-L254) ------------------------------------------
    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool) -> None:
        """上游 L217-L238. tune_projector 一起控制四者, 不能只冻其中一个."""
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            self.state_placeholder.requires_grad_(False)
            if self.cfg.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.dit.requires_grad_(False)

    def set_frozen_modules_to_eval_mode(self) -> None:
        """上游 L240-L254. 不能省: HF Trainer 每步都调 model.train()."""
        if not self.training:
            return
        if not getattr(self, "tune_projector", True):
            self.state_encoder.eval()
            self.action_encoder.eval()
            self.action_decoder.eval()
            if self.cfg.add_pos_embed:
                self.position_embedding.eval()
        if not getattr(self, "tune_diffusion_model", True):
            self.dit.eval()

    # -- 推理 ---------------------------------------------------------------
    @torch.no_grad()
    def sample(self, phi: torch.Tensor, phi_mask: torch.Tensor, state: torch.Tensor,
               state_mask: torch.Tensor, embodiment_id: torch.Tensor,
               has_proprio: torch.Tensor) -> torch.Tensor:
        """K 步前向 Euler (上游 L349-L404). tau 从 0 走到 (K-1)/K, 最后一步不在 tau = 1 上."""
        cfg = self.cfg
        b, device = phi.shape[0], phi.device
        state_feats = self.encode_state(state, embodiment_id, has_proprio)
        actions = torch.randn(b, cfg.action_horizon, cfg.max_action_dim, device=device,
                              dtype=phi.dtype)
        k = cfg.num_inference_timesteps
        for step in range(k):
            tau = step / float(k)
            bucket = torch.full((b,), int(tau * cfg.num_timestep_buckets), device=device,
                                dtype=torch.long)
            tokens = self.build_tokens(state_feats,
                                       self.encode_action(actions, bucket, embodiment_id))
            v = self.velocity(tokens, phi, bucket, embodiment_id, phi_mask)
            actions = actions + (1.0 / k) * v
        return actions


# ---------------------------------------------------------------------------
# 4. 一次 tiny 运行
# ---------------------------------------------------------------------------
def _n(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main() -> None:
    torch.manual_seed(0)
    cfg = tiny()
    m = ActionExpert(cfg).eval()
    b, s_len = 3, 19

    print(f"config: tiny(), W={cfg.input_embedding_dim}, DiT {cfg.num_layers} layers "
          f"({cfg.num_attention_heads}x{cfg.attention_head_dim}), "
          f"cross_dim={cfg.backbone_embedding_dim}, out={cfg.hidden_size}")
    print(f"embodiment slots C={cfg.max_num_embodiments}, H={cfg.action_horizon}, "
          f"future_tokens={cfg.num_target_vision_tokens}, K={cfg.num_inference_timesteps}")
    print(f"DiT blocks: {m.dit.n_cross} cross / {len(m.dit.blocks) - m.dit.n_cross} self "
          f"(interleave={cfg.interleave_self_attention})")

    phi = torch.randn(b, s_len, cfg.backbone_embedding_dim)
    phi_mask = torch.ones(b, s_len, dtype=torch.bool)
    phi_mask[1, -4:] = False
    state = torch.randn(b, cfg.state_horizon, cfg.max_state_dim)
    state_mask = torch.ones_like(state, dtype=torch.bool)
    emb_id = torch.tensor([0, 2, 3])
    has_proprio = torch.tensor([False, True, True])  # 样本 0 是人类演示

    with torch.no_grad():
        sf = m.encode_state(state, emb_id, has_proprio)
        print(f"\n[1] phi           {tuple(phi.shape)}  valid per sample "
              f"{phi_mask.sum(1).tolist()}")
        print(f"[2] state encode  {tuple(state.shape)} -> {tuple(sf.shape)}")
        print(f"    sample 0 (human) uses the placeholder token: "
              f"{torch.allclose(sf[0], m.state_placeholder[0].expand_as(sf[0]))}")
        noisy = torch.randn(b, cfg.action_horizon, cfg.max_action_dim)
        bucket = torch.full((b,), 250, dtype=torch.long)
        af = m.encode_action(noisy, bucket, emb_id)
        print(f"[3] action encode {tuple(noisy.shape)} -> {tuple(af.shape)} "
              f"(+ position_embedding)")
        tokens = m.build_tokens(sf, af)
        print(f"[4] token layout  [state {cfg.state_horizon} | future "
              f"{cfg.num_target_vision_tokens} | action {cfg.action_horizon}] "
              f"-> {tuple(tokens.shape)}")
        v = m.velocity(tokens, phi, bucket, emb_id, phi_mask)
        print(f"[5] DiT + decoder -> take last {cfg.action_horizon} tokens -> {tuple(v.shape)}")

        t0 = time.perf_counter()
        a = m.sample(phi, phi_mask, state, state_mask, emb_id, has_proprio)
        dt = 1e3 * (time.perf_counter() - t0)
        print(f"[6] sample K={cfg.num_inference_timesteps} Euler steps -> {tuple(a.shape)}  "
              f"{dt:.1f} ms (CPU, batch {b})")
        taus = [step / cfg.num_inference_timesteps for step in range(cfg.num_inference_timesteps)]
        print(f"    tau visited: {taus}  (note: never tau = 1.0)")

    print("\nparameters:")
    for name, mod in (("DiT", m.dit), ("state_encoder", m.state_encoder),
                      ("action_encoder", m.action_encoder),
                      ("action_decoder", m.action_decoder),
                      ("future_tokens", m.future_tokens),
                      ("position_embedding", m.position_embedding)):
        print(f"  {name:20s} {_n(mod):>10,}")
    print(f"  {'state_placeholder':20s} {m.state_placeholder.numel():>10,}")
    print(f"  {'total':20s} {_n(m):>10,}")
    per_emb = _n(m.state_encoder) + _n(m.action_encoder) + _n(m.action_decoder)
    print(f"  adapters scale with C={cfg.max_num_embodiments}: {per_emb:,} params "
          f"({per_emb / cfg.max_num_embodiments:,.0f} per embodiment slot)")

    print("\nGR00T N1.5 disclosed values (NOT EgoScale's, see README Sec. 8):")
    n = n15()
    for k, src in N15_SOURCES.items():
        print(f"  {k:26s} = {getattr(n, k)!r:8}  <- {src}")


if __name__ == "__main__":
    main()
