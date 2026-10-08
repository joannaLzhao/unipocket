"""
Output writers: NumPy ``.npz``, tabular CSV, plain-text score columns,
B-factor-annotated PDB/mmCIF, and a PyMOL session script.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .predict import HEADS, Prediction
from .structure import write_scored_structure

CSV_HEADER = ["index", "chain", "resseq", "icode", "resname", "aa",
              "cry_score", "lig_score"]


def _rows(pred: Prediction) -> Iterable[List]:
    for i in range(pred.n_scored):
        rec = pred.residues[i] if pred.residues else None
        yield [
            i + 1,
            rec.chain_id if rec else "",
            rec.resseq if rec else i + 1,
            rec.icode.strip() if rec else "",
            rec.resname if rec else "",
            pred.sequence[i] if i < len(pred.sequence) else "",
            round(float(pred.cry[i]), 4),
            round(float(pred.lig[i]), 4),
        ]


def write_csv(pred: Prediction, path: str) -> str:
    """One row per residue, with the structure identifiers needed to map back."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        w.writerows(_rows(pred))
    return path


def write_npz(pred: Prediction, path: str, with_embedding: bool = False) -> str:
    """
    Arrays indexed identically to the CSV rows.

    Keys: ``cry``, ``lig``, ``aa``, ``chain``, ``resseq``, ``icode``,
    ``resname``, ``sequence``, ``entry_id`` (+ ``embedding`` if requested).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    n = pred.n_scored
    recs = pred.residues
    data = {
        "cry": pred.cry.astype(np.float32),
        "lig": pred.lig.astype(np.float32),
        "aa": np.array(list(pred.sequence[:n]), dtype="<U1"),
        "chain": np.array([r.chain_id for r in recs] if recs else [""] * n, dtype="<U4"),
        "resseq": np.array([r.resseq for r in recs] if recs else np.arange(1, n + 1), dtype=np.int32),
        "icode": np.array([r.icode.strip() for r in recs] if recs else [""] * n, dtype="<U1"),
        "resname": np.array([r.resname for r in recs] if recs else [""] * n, dtype="<U3"),
        "sequence": np.array(pred.sequence[:n]),
        "entry_id": np.array(pred.entry_id),
        "n_input_residues": np.array(pred.n_input_residues),
        "truncated": np.array(pred.truncated),
    }
    if with_embedding and pred.embedding is not None:
        data["embedding"] = pred.embedding.astype(np.float32)
    np.savez_compressed(path, **data)
    return path


def write_txt(pred: Prediction, path: str) -> str:
    """
    Whitespace-aligned text table with a ``#``-commented header.

    Readable by eye and by ``numpy.loadtxt(path, usecols=(6, 7))``.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"# UniPocket per-residue scores for {pred.entry_id}\n")
        fh.write(f"# source: {pred.source}\n")
        fh.write(f"# residues scored: {pred.n_scored} of {pred.n_input_residues}\n")
        fh.write("# cry = cryptic-pocket score, lig = ligand-binding score "
                 "(both sigmoid probabilities in [0,1])\n")
        fh.write("#{:>6s} {:>5s} {:>7s} {:>5s} {:>7s} {:>3s} {:>9s} {:>9s}\n".format(
            "index", "chain", "resseq", "icode", "resname", "aa", "cry", "lig"))
        for row in _rows(pred):
            fh.write(" {:>6d} {:>5s} {:>7d} {:>5s} {:>7s} {:>3s} {:>9.4f} {:>9.4f}\n".format(
                row[0], row[1] or "-", row[2], row[3] or "-", row[4] or "-",
                row[5] or "-", row[6], row[7]))
    return path


def write_structure_scores(pred: Prediction, out_dir: str, head: str,
                           scale: float = 100.0, fmt: Optional[str] = None,
                           suffix: Optional[str] = None) -> Optional[str]:
    """Write the input structure with ``head`` scores in the B-factor column."""
    if pred.structure is None or pred.residues is None:
        return None
    fmt = fmt or pred.fmt
    ext = "cif" if fmt == "cif" else "pdb"
    name = suffix or f"{pred.entry_id}_{head}.{ext}"
    out_path = os.path.join(out_dir, name)
    return write_scored_structure(pred.structure, pred.model_idx, pred.residues,
                                  pred.scores(head), out_path,
                                  scale=scale, fmt=fmt)


def write_pymol_script(pred: Prediction, out_dir: str, head: str,
                       structure_file: str, scale: float = 100.0,
                       top_k: int = 10) -> str:
    """A ``.pml`` that loads the scored structure and colours by B-factor."""
    path = os.path.join(out_dir, f"{pred.entry_id}_{head}.pml")
    tops = pred.top_residues(head, top_k)
    sel = " or ".join(
        f"(chain {t['chain']} and resi {t['resseq']}{t['icode']})" if t["chain"]
        else f"resi {t['resseq']}" for t in tops) or "none"
    head_name = "cryptic pocket" if head == "cry" else "ligand binding"
    with open(path, "w") as fh:
        fh.write(f"# UniPocket {head_name} scores for {pred.entry_id}\n")
        fh.write(f"# B-factor column = score x {scale:g}\n")
        fh.write(f"load {os.path.basename(structure_file)}, {pred.entry_id}_{head}\n")
        fh.write("hide everything\nshow cartoon\n")
        fh.write(f"spectrum b, blue_white_red, {pred.entry_id}_{head} and name CA, "
                 f"0, {scale:g}\n")
        fh.write(f"select top_{head}, {sel}\n")
        fh.write(f"show sticks, top_{head} and not hydro\n")
        fh.write(f"color yellow, top_{head} and elem C\n")
        fh.write("deselect\nzoom\n")
    return path


def write_top_residues(pred: Prediction, path: str, k: int = 25) -> str:
    """Ranked top-k residue table for both heads."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["head", "rank", "index", "chain", "resseq", "icode",
                    "resname", "aa", "score"])
        for head in HEADS:
            for t in pred.top_residues(head, k):
                w.writerow([head, t["rank"], t["index"], t["chain"], t["resseq"],
                            t["icode"], t["resname"], t["aa"], t["score"]])
    return path


def write_summary(rows: Sequence[Dict], path: str) -> str:
    """Run-level table, one row per scored unit, sorted by cryptic-pocket max."""
    if not rows:
        return path
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    ordered = sorted(rows, key=lambda r: (r.get("cry_max") or -1), reverse=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(ordered)
    return path
