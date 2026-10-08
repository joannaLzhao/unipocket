"""
UniPocket network definition and checkpoint loading.

The architecture is identical to the one trained in ``bench_esm2emb_only.py``:
a shared MLP backbone over per-residue ESM2 embeddings feeding two independent
sigmoid heads

    CRY  -- cryptic-pocket-lining residue   (trained on CryptoBench)
    LIG  -- ligand-binding residue          (trained on RCSB holo structures)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

CKPT_FILENAME = "unipocket_esm2_650M.pt"
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def _default_checkpoint() -> str:
    """
    Locate the shipped checkpoint.

    Looked up in order: ``$UNIPOCKET_CKPT``, the repository's ``checkpoints/``
    directory (the editable-install and git-clone case), then ``checkpoints/``
    inside the installed package. The repo path is returned as the fallback so
    the error message points at where the file is expected to be.
    """
    env = os.environ.get("UNIPOCKET_CKPT")
    if env:
        return env
    repo = os.path.join(os.path.dirname(_PKG_DIR), "checkpoints", CKPT_FILENAME)
    packaged = os.path.join(_PKG_DIR, "checkpoints", CKPT_FILENAME)
    return packaged if (not os.path.isfile(repo) and os.path.isfile(packaged)) else repo


DEFAULT_CKPT = _default_checkpoint()


class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.norm(x + self.net(x)))


class MultiHeadPocketNet(nn.Module):
    """Shared backbone + two task heads (ligand / cryptic)."""

    def __init__(self, in_dim: int = 1280, compress: int = 256, wide: int = 512,
                 head_hidden: int = 128, dropout: float = 0.15):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, compress), nn.LayerNorm(compress), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(compress, wide),   nn.LayerNorm(wide),    nn.GELU(), nn.Dropout(dropout),
            ResBlock(wide, dropout),
            ResBlock(wide, dropout),
            nn.Linear(wide, compress),   nn.GELU(), nn.Dropout(dropout),
        )
        self.lig_fc  = nn.Sequential(nn.Linear(compress, head_hidden), nn.GELU(), nn.Dropout(dropout))
        self.lig_out = nn.Linear(head_hidden, 1)
        self.cry_fc  = nn.Sequential(nn.Linear(compress, head_hidden), nn.GELU(), nn.Dropout(dropout))
        self.cry_out = nn.Linear(head_hidden, 1)

    def forward_lig(self, x):
        return self.lig_out(self.lig_fc(self.backbone(x))).squeeze(-1)

    def forward_cry(self, x):
        return self.cry_out(self.cry_fc(self.backbone(x))).squeeze(-1)

    def forward(self, x):
        """Return (cry_logits, lig_logits) from one backbone pass."""
        h = self.backbone(x)
        return (self.cry_out(self.cry_fc(h)).squeeze(-1),
                self.lig_out(self.lig_fc(h)).squeeze(-1))


class UniPocket:
    """Thin wrapper that owns the weights and scores embedding matrices."""

    def __init__(self, net: MultiHeadPocketNet, device: torch.device,
                 in_dim: int, meta: Optional[Dict[str, Any]] = None):
        self.net = net
        self.device = device
        self.in_dim = in_dim
        self.meta = meta or {}

    @torch.no_grad()
    def score(self, emb: np.ndarray, batch_size: int = 8192):
        """(L, in_dim) float array -> (cry, lig) probabilities, each (L,)."""
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[1] != self.in_dim:
            raise ValueError(
                f"expected embeddings of shape (L, {self.in_dim}), got {emb.shape}")
        cry_out, lig_out = [], []
        for start in range(0, len(emb), batch_size):
            x = torch.from_numpy(emb[start:start + batch_size]).to(self.device)
            cry, lig = self.net(x)
            cry_out.append(torch.sigmoid(cry.float()).cpu().numpy())
            lig_out.append(torch.sigmoid(lig.float()).cpu().numpy())
        if not cry_out:
            empty = np.zeros(0, np.float32)
            return empty, empty.copy()
        return (np.concatenate(cry_out).astype(np.float32),
                np.concatenate(lig_out).astype(np.float32))


def load_unipocket(ckpt_path: str = DEFAULT_CKPT,
                   device: Optional[torch.device] = None) -> UniPocket:
    """Load the trained UniPocket head from a checkpoint file."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"UniPocket checkpoint not found: {ckpt_path}\n"
            "See checkpoints/README.md for where to get it.")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        state, cfg = raw["state_dict"], dict(raw.get("cfg", {}))
        in_dim = int(raw.get("in_dim", 1280))
        meta = {k: v for k, v in raw.items() if k != "state_dict"}
    else:                                     # bare state_dict
        state, cfg, in_dim, meta = raw, {}, 1280, {}

    net = MultiHeadPocketNet(
        in_dim=in_dim,
        compress=int(cfg.get("compress", 256)),
        wide=int(cfg.get("wide", 512)),
        head_hidden=int(cfg.get("head_hidden", 128)),
        dropout=float(cfg.get("dropout", 0.15)),
    )
    net.load_state_dict(state, strict=True)   # dropout is inert in eval()
    net.to(device).eval()
    return UniPocket(net, device, in_dim, meta)
