"""
Per-residue ESM2 embeddings.

UniPocket was trained on layer-33 representations of ``esm2_t33_650M_UR50D``
(1280-d per residue), taken from the raw model with no normalisation, so this
module reproduces exactly that: tokens in, ``representations[33]`` out, BOS and
EOS stripped.

ESM2 has a 1024-token context (1022 residues once BOS/EOS are accounted for).
Longer chains are handled by ``long_mode``:

``window``   (default) overlapping windows; each residue takes its embedding
             from the window in which it sits furthest from the edges, so every
             residue is embedded with as much flanking context as available.
``truncate`` keep the first 1022 residues only, which is what the published
             benchmark runs did; the rest get no score.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

DEFAULT_ESM_MODEL = "esm2_t33_650M_UR50D"
MAX_RESIDUES = 1022          # 1024-token context minus BOS and EOS
_LAYER_FOR_MODEL = {"esm2_t33_650M_UR50D": 33, "esm2_t36_3B_UR50D": 36,
                    "esm2_t30_150M_UR50D": 30, "esm2_t12_35M_UR50D": 12,
                    "esm2_t6_8M_UR50D": 6}


def _windows(length: int, size: int, stride: int) -> List[Tuple[int, int]]:
    """Window start/end pairs covering ``length``, the last one flush to the end."""
    if length <= size:
        return [(0, length)]
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return [(s, s + size) for s in starts]


class ESM2Embedder:
    """Lazily-loaded ESM2 encoder producing (L, D) float32 arrays."""

    def __init__(self, model_name: str = DEFAULT_ESM_MODEL,
                 device: Optional[torch.device] = None,
                 repr_layer: Optional[int] = None,
                 max_len: int = MAX_RESIDUES,
                 stride: Optional[int] = None,
                 long_mode: str = "window",
                 fp16: bool = False):
        if long_mode not in ("window", "truncate"):
            raise ValueError("long_mode must be 'window' or 'truncate'")
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.repr_layer = repr_layer if repr_layer is not None else _LAYER_FOR_MODEL.get(model_name, 33)
        self.max_len = int(max_len)
        self.stride = int(stride) if stride else max(1, self.max_len // 2)
        self.long_mode = long_mode
        self.fp16 = fp16 and self.device.type == "cuda"
        self._model = None
        self._batch_converter = None
        self.embed_dim: Optional[int] = None

    # -- loading ----------------------------------------------------------
    def load(self) -> "ESM2Embedder":
        if self._model is not None:
            return self
        import esm  # imported lazily so --help works without torch hub downloads

        loader = getattr(esm.pretrained, self.model_name, None)
        if loader is None:
            raise ValueError(f"unknown ESM2 model '{self.model_name}'")
        model, alphabet = loader()
        model = model.to(self.device).eval()
        if self.fp16:
            model = model.half()
        self._model = model
        self._batch_converter = alphabet.get_batch_converter()
        self.embed_dim = int(model.embed_dim)
        return self

    # -- core -------------------------------------------------------------
    @torch.no_grad()
    def _forward(self, batch: Sequence[Tuple[str, str]]) -> List[np.ndarray]:
        """Embed sequences that all fit in the context window."""
        self.load()
        _, _, tokens = self._batch_converter(list(batch))
        out = self._model(tokens.to(self.device),
                          repr_layers=[self.repr_layer], return_contacts=False)
        reps = out["representations"][self.repr_layer]
        return [reps[i, 1:len(seq) + 1].float().cpu().numpy().astype(np.float32)
                for i, (_, seq) in enumerate(batch)]

    def embed(self, seq: str) -> np.ndarray:
        """Embed one sequence of any length -> (L, D), L == len(seq) in window mode."""
        if not seq:
            return np.zeros((0, self.embed_dim or 1280), np.float32)
        if len(seq) <= self.max_len:
            return self._forward([("query", seq)])[0]
        if self.long_mode == "truncate":
            return self._forward([("query", seq[:self.max_len])])[0]

        wins = _windows(len(seq), self.max_len, self.stride)
        emb = np.zeros((len(seq), self.embed_dim or 1280), np.float32)
        # Distance of each residue from its window's nearest edge; a residue is
        # written by the window that gives it the most context on both sides.
        best = np.full(len(seq), -1.0, np.float32)
        for start, end in wins:
            win_emb = self._forward([("query", seq[start:end])])[0]
            if emb.shape[1] != win_emb.shape[1]:
                emb = np.zeros((len(seq), win_emb.shape[1]), np.float32)
            local = np.arange(end - start)
            centrality = np.minimum(local, (end - start) - 1 - local).astype(np.float32)
            take = np.nonzero(centrality > best[start:end])[0]
            emb[start + take] = win_emb[take]
            best[start + take] = centrality[take]
        return emb

    def embed_many(self, items: Iterable[Tuple[str, str]],
                   max_tokens_per_batch: int = 8192,
                   progress: bool = False) -> Dict[str, np.ndarray]:
        """
        Embed many (id, sequence) pairs, batching the short ones together.

        Batches are capped by padded token count rather than sequence count so
        memory use stays flat whether the inputs are peptides or full proteins.
        """
        items = list(items)
        order = sorted(range(len(items)), key=lambda i: len(items[i][1]))
        results: Dict[str, np.ndarray] = {}

        iterator = order
        if progress:
            from tqdm import tqdm
            iterator = tqdm(order, desc="ESM2", unit="seq")

        batch: List[Tuple[str, str]] = []

        def flush():
            if batch:
                for (name, _), emb in zip(batch, self._forward(batch)):
                    results[name] = emb
                batch.clear()

        for i in iterator:
            name, seq = items[i]
            if len(seq) > self.max_len:
                flush()
                results[name] = self.embed(seq)
                continue
            longest = max([len(seq)] + [len(s) for _, s in batch])
            if batch and (len(batch) + 1) * (longest + 2) > max_tokens_per_batch:
                flush()
            batch.append((name, seq))
        flush()
        return results
