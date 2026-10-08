"""
End-to-end prediction: a structure or a sequence in, per-residue scores out.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from .embed import DEFAULT_ESM_MODEL, ESM2Embedder
from .model import DEFAULT_CKPT, UniPocket, load_unipocket
from .structure import (ResidueRecord, extract_residues, group_by_chain,
                        load_structure, sequence_of)

HEADS = ("cry", "lig")


@dataclass
class Prediction:
    """Scores for one scoring unit (a whole structure, one chain, or one sequence)."""

    entry_id: str
    sequence: str
    cry: np.ndarray
    lig: np.ndarray
    source: str = ""
    kind: str = "structure"                 # "structure" or "sequence"
    residues: Optional[List[ResidueRecord]] = None
    structure: Optional[Any] = None         # Bio.PDB Structure, for B-factor output
    model_idx: int = 0
    fmt: str = "pdb"
    n_input_residues: int = 0               # before any truncation
    embedding: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.n_input_residues:
            self.n_input_residues = len(self.sequence)

    @property
    def n_scored(self) -> int:
        return int(len(self.cry))

    @property
    def truncated(self) -> bool:
        return self.n_scored < self.n_input_residues

    @property
    def chains(self) -> List[str]:
        if not self.residues:
            return []
        seen: List[str] = []
        for r in self.residues:
            if r.chain_id not in seen:
                seen.append(r.chain_id)
        return seen

    def scores(self, head: str) -> np.ndarray:
        if head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}, got {head!r}")
        return self.cry if head == "cry" else self.lig

    def summary(self, top_pct: float = 0.05, threshold: float = 0.5) -> Dict[str, Any]:
        """One flat record per prediction, for the run-level summary table."""
        row: Dict[str, Any] = {
            "entry_id": self.entry_id,
            "source": os.path.basename(self.source) if self.source else "",
            "kind": self.kind,
            "chains": "".join(self.chains),
            "n_residues": self.n_scored,
            "n_input_residues": self.n_input_residues,
            "truncated": self.truncated,
        }
        for head in HEADS:
            s = self.scores(head)
            if len(s) == 0:
                row.update({f"{head}_mean": np.nan, f"{head}_max": np.nan,
                            f"{head}_top{int(top_pct * 100)}pct_mean": np.nan,
                            f"{head}_n_above_{threshold:g}": 0})
                continue
            k = max(1, int(np.ceil(top_pct * len(s))))
            row[f"{head}_mean"] = round(float(s.mean()), 4)
            row[f"{head}_max"] = round(float(s.max()), 4)
            row[f"{head}_top{int(top_pct * 100)}pct_mean"] = round(
                float(np.sort(s)[::-1][:k].mean()), 4)
            row[f"{head}_n_above_{threshold:g}"] = int((s >= threshold).sum())
        return row

    def top_residues(self, head: str, k: int = 10) -> List[Dict[str, Any]]:
        """The k highest-scoring residues for one head, best first."""
        s = self.scores(head)
        order = np.argsort(s)[::-1][:k]
        out = []
        for rank, i in enumerate(order, start=1):
            rec = self.residues[i] if self.residues else None
            out.append({
                "rank": rank,
                "index": int(i) + 1,
                "chain": rec.chain_id if rec else "",
                "resseq": rec.resseq if rec else int(i) + 1,
                "icode": rec.icode.strip() if rec else "",
                "resname": rec.resname if rec else "",
                "aa": self.sequence[i] if i < len(self.sequence) else "",
                "score": round(float(s[i]), 4),
            })
        return out


class PocketPredictor:
    """Holds the ESM2 encoder and the UniPocket head; reuse it across inputs."""

    def __init__(self,
                 ckpt: str = DEFAULT_CKPT,
                 esm_model: str = DEFAULT_ESM_MODEL,
                 device: Optional[torch.device] = None,
                 long_mode: str = "window",
                 fp16: bool = False,
                 embedder: Optional[ESM2Embedder] = None,
                 unipocket: Optional[UniPocket] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unipocket = unipocket or load_unipocket(ckpt, self.device)
        self.embedder = embedder or ESM2Embedder(
            esm_model, device=self.device, long_mode=long_mode, fp16=fp16)
        self.long_mode = long_mode

    # -- sequences --------------------------------------------------------
    def predict_sequence(self, entry_id: str, seq: str,
                         source: str = "", keep_embedding: bool = False,
                         residues: Optional[List[ResidueRecord]] = None,
                         structure: Optional[Any] = None,
                         model_idx: int = 0, fmt: str = "pdb",
                         kind: str = "sequence") -> Prediction:
        emb = self.embedder.embed(seq)
        return self._assemble(entry_id, seq, emb, source, kind, residues,
                              structure, model_idx, fmt, keep_embedding)

    def _assemble(self, entry_id, seq, emb, source, kind, residues,
                  structure, model_idx, fmt, keep_embedding) -> Prediction:
        cry, lig = self.unipocket.score(emb)
        n_scored = len(cry)
        return Prediction(
            entry_id=entry_id,
            sequence=seq[:n_scored],
            cry=cry, lig=lig,
            source=source, kind=kind,
            residues=list(residues[:n_scored]) if residues is not None else None,
            structure=structure, model_idx=model_idx, fmt=fmt,
            n_input_residues=len(seq),
            embedding=emb if keep_embedding else None,
        )

    def predict_fasta_entries(self, entries: Sequence,
                              source: str = "", keep_embedding: bool = False,
                              progress: bool = False) -> List[Prediction]:
        """Batched scoring of many (id, sequence) pairs."""
        embeddings = self.embedder.embed_many(entries, progress=progress)
        return [self._assemble(name, seq, embeddings[name], source, "sequence",
                               None, None, 0, "pdb", keep_embedding)
                for name, seq in entries]

    # -- structures -------------------------------------------------------
    def predict_structure(self, path: str,
                          per_chain: bool = False,
                          chains: Optional[Sequence[str]] = None,
                          model_idx: int = 0,
                          standard_only: bool = False,
                          require_ca: bool = True,
                          entry_id: Optional[str] = None,
                          keep_embedding: bool = False) -> List[Prediction]:
        """
        Score one PDB / mmCIF file.

        By default every protein chain of the selected model is concatenated
        into a single sequence and scored in one pass, which is how the model
        was evaluated in the paper. ``per_chain=True`` scores each chain
        independently instead, so a chain's scores do not depend on which other
        chains happen to be in the file.

        ``require_ca=False`` also keeps residues whose coordinates are only
        partly modelled, matching the published benchmark scripts.
        """
        entry_id = entry_id or os.path.splitext(os.path.basename(path))[0]
        structure, fmt = load_structure(path, entry_id)
        records = extract_residues(structure, model_idx=model_idx, chains=chains,
                                   standard_only=standard_only,
                                   require_ca=require_ca)
        if not records:
            raise ValueError(f"no protein residues found in {path}"
                             + (f" for chains {','.join(chains)}" if chains else ""))

        groups = ([(f"{entry_id}_{cid}", recs)
                   for cid, recs in group_by_chain(records).items()]
                  if per_chain else [(entry_id, records)])

        preds = []
        for name, recs in groups:
            seq = sequence_of(recs)
            emb = self.embedder.embed(seq)
            preds.append(self._assemble(name, seq, emb, path, "structure", recs,
                                        structure, model_idx, fmt, keep_embedding))
        return preds
