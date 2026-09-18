"""pi0.6* value function (inference): the same VLA architecture on a smaller Gemma 3 backbone, a 201-way head over
discretised normalised returns, and the expectation readout V(o, ell) = sum_b p(b | o, ell) v(b).

Minimal PyTorch re-implementation. Sources of truth:
  paper    pi0.6* arXiv:2511.14759v2 Sec. IV-A (distributional value function p_phi(V | o_t, ell) over B = 201 bins,
           "same architecture as the VLA policy, but with a smaller VLM backbone", V = sum_b p(V = b | o) v(b)),
           Sec. V-C ("smaller 670M parameter VLM backbone that is also initialized from Gemma 3", same language inputs
           as the VLA, values normalised to (-1, 0)), Fig. 3-4, Fig. 13
  gemma    google-deepmind/gemma @ 0513283af5afffa27390b6ede2facc35d0f16e08 _gemma.py Gemma3_1B L169-L193 (the nearest
           disclosed Gemma 3 size; 670M itself matches no released variant, README Sec. 8)
  openpi   no value-function code exists upstream (@ 215abfb)
Licenses: Apache-2.0 (gemma, openpi pieces). Re-implements, does not copy.

Inference only (repo rule): the head, the readout position and the expectation. Eq. 1 (the CE), the advantage
estimators, the thresholds and the indicator are training-side and live in train.py.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn

from pi.pi0.vlm.model import ViTConfig
from pi.pi06.backbone.model import GEMMA3_1B, SIGLIP_400M_448, Gemma3Config, Gemma3VLM, tiny_gemma3, tiny_vit448
from pi.pi06.data.data import NUM_BINS, Pi06Observation, bin_values

VALUE_BACKBONE_PARAMS = 670e6  # paper Sec. V-C: "a smaller 670M parameter VLM backbone that is also initialized from Gemma 3"
VALUE_BACKBONE = GEMMA3_1B  # nearest disclosed Gemma 3 size (698M non-embedding, report Table 1); the exact 670M config is undisclosed (README Sec. 8)


def tiny_value_backbone() -> Gemma3Config:
    """Narrower than the tiny policy backbone (32 vs 64), as the paper's value backbone is smaller than the policy's."""
    return dataclasses.replace(tiny_gemma3(), width=32, mlp_dim=64, num_heads=2, num_kv_heads=1)


# ======================================================================================
# 1. The value function. Sec. IV-A: p_phi(V | o_t, ell) in Delta^B, B = 201. The body is the policy's Gemma3VLM (vision,
#    embedder, single-expert stack); the head is one linear layer to 201 logits read at the LAST valid token of the
#    prefix (text is causal, so that token has seen every image and text token; the readout position is undisclosed,
#    README Sec. 8). The input is the "value" layout of ../data: Task + metadata + State, no subtask, no advantage.
# ======================================================================================
class ValueFunction(nn.Module):
    def __init__(self, vit_cfg: ViTConfig = SIGLIP_400M_448, cfg: Gemma3Config = VALUE_BACKBONE, num_bins: int = NUM_BINS):
        super().__init__()
        self.vlm = Gemma3VLM(vit_cfg, (cfg,))
        self.value_head = nn.Linear(cfg.width, num_bins)
        self.register_buffer("bin_values", torch.from_numpy(bin_values(num_bins)), persistent=False)  # v(b), -1 .. 0
        self.num_bins = num_bins

    def readout_index(self, valid: torch.Tensor) -> torch.Tensor:
        """i64[B]: index of the last valid column of the prefix (images | text)."""
        ar = torch.arange(valid.shape[1], device=valid.device)
        return (valid.long() * ar).amax(1)

    def forward(self, obs: Pi06Observation) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (value logits f32[B, 201], prefix hidden states f32[B, S, W] for the web co-training head in train.py)."""
        h, _cache, valid, _n_img = self.vlm.forward_prefix(obs)
        idx = self.readout_index(valid)
        feat = h[torch.arange(h.shape[0], device=h.device), idx]  # [B, W]
        return self.value_head(feat), h

    def distribution(self, obs: Pi06Observation) -> torch.Tensor:
        """p_phi(V = b | o, ell) f32[B, 201]."""
        return torch.softmax(self(obs)[0].float(), -1)

    @torch.no_grad()
    def value(self, obs: Pi06Observation) -> torch.Tensor:
        """V^{pi_ref}(o, ell) = sum_b p(b) v(b) f32[B] in [-1, 0] (Sec. IV-A)."""
        return self.distribution(obs) @ self.bin_values


def tiny_value_function() -> ValueFunction:
    return ValueFunction(tiny_vit448(), tiny_value_backbone())


def value_param_count(cfg: Gemma3Config = VALUE_BACKBONE, num_bins: int = NUM_BINS) -> dict[str, int]:
    from pi.pi06.backbone.model import backbone_param_count

    pc = backbone_param_count(cfg)
    return {**pc, "value_head": cfg.width * num_bins + num_bins}


# ======================================================================================
# 2. One value read with the tiny config.  uv run python -m pi.pi06.value.model
# ======================================================================================
def main():
    import numpy as np

    from pi.pi06.data.data import STATIC_IMAGE_KEYS, build_pi06_batch, tiny_pi06_tokenizer, unit_stats

    torch.manual_seed(0)
    B, H, d = 2, 10, 7
    vf = tiny_value_function().eval()
    n = lambda m: sum(p.numel() for p in m.parameters())
    pc = value_param_count()
    print(f"tiny value function: backbone {vf.vlm.cfg}")
    print(f"params: vlm {n(vf.vlm):,}  value_head {n(vf.value_head):,}")
    print(f"paper: Gemma 3 1B body as the stand-in for '670M' (Sec. V-C): non-embedding {pc['non_embedding']:,} + embedding {pc['embedding']:,}; head {pc['value_head']:,}")
    rng = np.random.default_rng(0)
    raw = {"images": {k: rng.integers(0, 256, (B, 96, 128, 3), dtype=np.uint8) for k in ("base_0_rgb", "left_wrist_0_rgb")},
           "state": rng.uniform(-0.5, 0.5, (B, d)).astype(np.float32), "prompt": ["make a double espresso", "fold the shirt"]}
    seq = tiny_pi06_tokenizer(H, d)
    obs, _ = build_pi06_batch(raw, unit_stats(d), seq, layout="value", image_keys=STATIC_IMAGE_KEYS, action_horizon=H, delta_mask=None, train=False, metadata=["speed: fast", None])
    with torch.no_grad():
        h, _, valid, n_img = vf.vlm.forward_prefix(obs)
        idx = vf.readout_index(valid)
        print(f"\n[prefix] layout value: {int(obs.token_mask[0].sum())} text tokens (no Subtask / Advantage), h {tuple(h.shape)}, readout index {idx.tolist()} (= last valid column; {n_img} image cols)")
        logits, _ = vf(obs)
        p = torch.softmax(logits, -1)
        print(f"[head]   logits {tuple(logits.shape)} -> softmax; sample 0: argmax bin {int(p[0].argmax())}, max p {float(p[0].max()):.4f} (untrained ~ uniform: 1/201 = {1/201:.4f})")
        v = vf.value(obs)
        print(f"[value]  V = sum_b p(b) v(b) = {[f'{x:.4f}' for x in v.tolist()]}  (uniform over [-1, 0] -> -0.5; range (-1, 0), Sec. V-C)")
        onehot = torch.zeros(1, 201)
        onehot[0, 160] = 30.0
        print(f"[check]  a peaked distribution at bin 160 -> V = {float(torch.softmax(onehot, -1) @ vf.bin_values):.4f} = v(160) = {float(vf.bin_values[160]):.4f} = '32 of T_max steps left' when T_max = 40")
    print("\n../value/train.py: Eq. 1 CE on value_bin, advantage = R - V (pre-training) or n-step (N = 50), thresholds, indicators")


if __name__ == "__main__":
    main()
