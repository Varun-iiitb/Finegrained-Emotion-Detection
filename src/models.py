"""DG-DQA model — DQF fusion body (+ open-vocab head added in a later prompt).

Implements the DQF (dual-query) fusion over THREE modality SEQUENCES
(text / audio / visual) as cached by extract_features.py (see CLAUDE.md §5).
A missing/masked modality is just a length-1 zero token, so the same code path
handles full sequences, length-1 "vectors", and missing-modality robustness.

Pipeline (per CLAUDE.md §7 / Prompt 5):
  per modality:  Linear -> Conv1D(temporal) -> Q-UA
  Q-UA:          learnable queries refined by MHSA(+LayerNorm) over the features,
                 broadcast-added to the features, then a Transformer encoder.
  Q-CA (pairs):  a shared learnable query per pair {t-a, t-v, a-v} attends to
                 each modality (both directions) -> concat -> Linear.
  Global:        stack [CLS] + the 3 pair features -> MHSA -> fused [CLS].
  Granularity:   exposes {t,a,v, ta,tv,av, tav} so aux losses can attach.

Random per-modality masking (`random_modality_mask`) zeros whole modalities at
train time for missing-modality robustness.
"""

import argparse
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

MODS = ("t", "a", "v")
PAIRS: List[Tuple[str, str]] = [("t", "a"), ("t", "v"), ("a", "v")]
PAIR_NAME = {("t", "a"): "ta", ("t", "v"): "tv", ("a", "v"): "av"}


def masked_mean(x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """Mean over the time axis ignoring padded positions. x:[B,T,d] pad:[B,T] (True=pad)."""
    valid = (~pad).float().unsqueeze(-1)            # [B,T,1]
    return (x * valid).sum(1) / valid.sum(1).clamp_min(1.0)


# ---------------------------------------------------------------------------
# Q-UA : unimodal dual-query refinement
# ---------------------------------------------------------------------------

class QUA(nn.Module):
    """Learnable queries refine over the modality features (MHSA + LayerNorm),
    are broadcast-added back, then a Transformer encoder contextualizes.
    Returns the contextualized sequence and a masked-mean summary."""

    def __init__(self, d: int, heads: int, n_query: int, n_layers: int, ff: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_query, d) * 0.02)
        self.refine = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers,
                                             enable_nested_tensor=False)

    def forward(self, H: torch.Tensor, pad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = H.size(0)
        q = self.query.unsqueeze(0).expand(B, -1, -1)          # [B,nq,d]
        refined, _ = self.refine(q, H, H, key_padding_mask=pad)
        refined = self.ln(q + refined)                          # [B,nq,d]
        summary = refined.mean(1, keepdim=True)                 # [B,1,d]
        U = self.encoder(H + summary, src_key_padding_mask=pad)  # [B,T,d]
        return U, masked_mean(U, pad)


# ---------------------------------------------------------------------------
# Q-CA : pairwise cross-modal shared-query attention
# ---------------------------------------------------------------------------

class QCA(nn.Module):
    """A shared learnable query attends to modality A and modality B (both
    directions); the two pooled results are concatenated and projected."""

    def __init__(self, d: int, heads: int, n_query: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_query, d) * 0.02)
        self.attn_a = nn.MultiheadAttention(d, heads, batch_first=True)
        self.attn_b = nn.MultiheadAttention(d, heads, batch_first=True)
        self.proj = nn.Linear(2 * d, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, Ua, pada, Ub, padb) -> torch.Tensor:
        B = Ua.size(0)
        q = self.query.unsqueeze(0).expand(B, -1, -1)
        ca, _ = self.attn_a(q, Ua, Ua, key_padding_mask=pada)   # [B,nq,d]
        cb, _ = self.attn_b(q, Ub, Ub, key_padding_mask=padb)
        feat = self.proj(torch.cat([ca.mean(1), cb.mean(1)], -1))   # [B,d]
        return self.ln(feat)


# ---------------------------------------------------------------------------
# Global fusion over the pair features + [CLS]
# ---------------------------------------------------------------------------

class GlobalFusion(nn.Module):
    def __init__(self, d: int, heads: int, n_layers: int, ff: int):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, ff, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers,
                                             enable_nested_tensor=False)

    def forward(self, pair_feats: List[torch.Tensor]) -> torch.Tensor:
        B = pair_feats[0].size(0)
        cls = self.cls.unsqueeze(0).expand(B, -1, -1)           # [B,1,d]
        tokens = torch.stack(pair_feats, 1)                     # [B,P,d]
        out = self.encoder(torch.cat([cls, tokens], 1))         # [B,1+P,d]
        return out[:, 0]                                        # fused CLS [B,d]


# ---------------------------------------------------------------------------
# DQF body
# ---------------------------------------------------------------------------

class DQFBody(nn.Module):
    """Full DQF fusion body. forward() returns a dict of granularity features:
    {t,a,v (unimodal), ta,tv,av (pairwise), tav (fused [CLS])}, all [B, d_model].
    """

    def __init__(self, d_text: int, d_audio: int, d_visual: int,
                 d_model: int = 256, n_heads: int = 4, n_layers: int = 6,
                 n_query: int = 8, ff: Optional[int] = None,
                 global_layers: int = 2):
        super().__init__()
        ff = ff or 4 * d_model
        d_in = {"t": d_text, "a": d_audio, "v": d_visual}
        self.proj = nn.ModuleDict({m: nn.Linear(d_in[m], d_model) for m in MODS})
        self.conv = nn.ModuleDict({m: nn.Conv1d(d_model, d_model, 3, padding=1)
                                   for m in MODS})
        self.qua = nn.ModuleDict({m: QUA(d_model, n_heads, n_query, n_layers, ff)
                                  for m in MODS})
        self.qca = nn.ModuleDict({PAIR_NAME[p]: QCA(d_model, n_heads, n_query)
                                  for p in PAIRS})
        self.glob = GlobalFusion(d_model, n_heads, global_layers, ff)
        self.d_model = d_model

    @staticmethod
    def random_modality_mask(B: int, device, p_drop: float = 0.3,
                             max_drop: int = 2) -> torch.Tensor:
        """[B,3] float keep-mask (1=keep, 0=drop) for {t,a,v}; never drops all."""
        keep = torch.ones(B, 3, device=device)
        for b in range(B):
            if torch.rand(()) < p_drop:
                k = int(torch.randint(1, max_drop + 1, ()).item())
                idx = torch.randperm(3, device=device)[:k]
                keep[b, idx] = 0.0
                if keep[b].sum() == 0:                          # safety: keep ≥1
                    keep[b, torch.randint(0, 3, ())] = 1.0
        return keep

    def forward(self, feats: Dict[str, torch.Tensor], pads: Dict[str, torch.Tensor],
                modality_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """feats/pads keyed by {'t','a','v'}: feats[m]=[B,T_m,d_in], pads[m]=[B,T_m]
        (True=pad). modality_mask: optional [B,3] keep-mask for {t,a,v}."""
        U, gran = {}, {}
        for i, m in enumerate(MODS):
            H = self.proj[m](feats[m])                          # [B,T,d]
            H = H * (~pads[m]).unsqueeze(-1)                    # zero padded positions
            H = self.conv[m](H.transpose(1, 2)).transpose(1, 2)  # temporal Conv1D
            if modality_mask is not None:
                H = H * modality_mask[:, i].view(-1, 1, 1)       # drop whole modality
            U[m], gran[m] = self.qua[m](H, pads[m])
        pair_feats = []
        for p in PAIRS:
            name = PAIR_NAME[p]
            f = self.qca[name](U[p[0]], pads[p[0]], U[p[1]], pads[p[1]])
            gran[name] = f
            pair_feats.append(f)
        gran["tav"] = self.glob(pair_feats)
        return gran


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def _rand_modality(B: int, t_max: int, d: int, device):
    """Random [B,T,d] sequence + pad mask with variable real lengths per sample."""
    T = int(torch.randint(1, t_max + 1, ()).item())
    x = torch.randn(B, T, d, device=device)
    pad = torch.zeros(B, T, dtype=torch.bool, device=device)
    for b in range(B):                                          # vary real length
        real = int(torch.randint(1, T + 1, ()).item())
        pad[b, real:] = True
    return x, pad


def main() -> int:
    ap = argparse.ArgumentParser(description="DQF body self-check")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # real cached dims (CLAUDE.md §5): text=1024, audio=768, visual=768
    body = DQFBody(d_text=1024, d_audio=768, d_visual=768,
                   d_model=args.d_model, n_heads=args.heads,
                   n_layers=args.layers).to(device).eval()
    n_params = sum(p.numel() for p in body.parameters()) / 1e6
    print(f"DQFBody: d_model={args.d_model} layers={args.layers} heads={args.heads} "
          f"| trainable params={n_params:.2f}M  | device={device}")

    B = args.batch
    feats, pads = {}, {}
    for m, (tmax, d) in {"t": (20, 1024), "a": (64, 768), "v": (16, 768)}.items():
        feats[m], pads[m] = _rand_modality(B, tmax, d, device)

    def run(tag, mm):
        with torch.no_grad():
            out = body(feats, pads, modality_mask=mm)
        keys = ["t", "a", "v", "ta", "tv", "av", "tav"]
        shapes = "  ".join(f"{k}{tuple(out[k].shape)}" for k in keys)
        print(f"[{tag}] fused tav={tuple(out['tav'].shape)}")
        print(f"        granularity: {shapes}")

    run("full", None)
    mm1 = torch.ones(B, 3, device=device); mm1[:, 1] = 0          # drop audio
    run("drop-1(audio)", mm1)
    mm2 = torch.ones(B, 3, device=device); mm2[:, 1] = 0; mm2[:, 2] = 0  # drop a+v
    run("drop-2(audio,visual)", mm2)
    run("random-mask", DQFBody.random_modality_mask(B, device))
    print("\n[selfcheck] DQF body forward OK (full + masked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
