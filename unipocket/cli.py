"""
Command-line entry point.

    unipocket 1ake                      # fetch from RCSB and score
    unipocket my_model.pdb --per-chain
    unipocket structures/ -o results
    unipocket seqs.fasta --formats csv,npz
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List, Optional, Sequence, Tuple

import torch

from . import __version__
from .embed import DEFAULT_ESM_MODEL
from .model import DEFAULT_CKPT
from .predict import HEADS, PocketPredictor, Prediction
from .structure import fetch_pdb, is_pdb_id, read_fasta
from .writers import (write_csv, write_npz, write_pymol_script, write_summary,
                      write_top_residues, write_txt, write_structure_scores)

STRUCTURE_EXT = (".pdb", ".ent", ".cif", ".mmcif", ".pdb1")
FASTA_EXT = (".fasta", ".fa", ".faa", ".fas", ".seq")
ALL_FORMATS = ("csv", "npz", "txt", "pdb", "pml", "top")
DEFAULT_FORMATS = "csv,npz,pdb"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="unipocket",
        description="Per-residue cryptic-pocket (CRY) and ligand-binding (LIG) "
                    "scores from an ESM2-650M sequence embedding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Inputs may be PDB/mmCIF files, directories of them, FASTA files, "
               "or bare 4-character RCSB entry IDs.")
    p.add_argument("inputs", nargs="+", help="structure file(s), directory, FASTA, or PDB ID")
    p.add_argument("-o", "--out-dir", default="unipocket_out", help="output directory")

    g = p.add_argument_group("structure handling")
    g.add_argument("--per-chain", action="store_true",
                   help="score each chain separately instead of all chains concatenated")
    g.add_argument("--chains", default=None,
                   help="comma-separated chain IDs to keep (default: all protein chains)")
    g.add_argument("--model-idx", type=int, default=0,
                   help="model index to use for multi-model files (NMR, trajectories)")
    g.add_argument("--standard-only", action="store_true",
                   help="drop modified residues such as MSE instead of mapping "
                        "them to their parent amino acid")
    g.add_argument("--keep-ca-less", action="store_true",
                   help="also score protein residues with no CA atom "
                        "(partly modelled side chains); off by default")

    g = p.add_argument_group("model")
    g.add_argument("--ckpt", default=DEFAULT_CKPT, help="UniPocket checkpoint")
    g.add_argument("--esm-model", default=DEFAULT_ESM_MODEL,
                   help="ESM2 variant (must match the checkpoint's input dim)")
    g.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    g.add_argument("--fp16", action="store_true", help="run ESM2 in half precision (CUDA only)")
    g.add_argument("--long-mode", default="window", choices=("window", "truncate"),
                   help="how to handle chains longer than 1022 residues")

    g = p.add_argument_group("output")
    g.add_argument("--formats", default=DEFAULT_FORMATS,
                   help="comma-separated subset of: " + ",".join(ALL_FORMATS))
    g.add_argument("--heads", default="cry,lig",
                   help="heads to write scored structures for")
    g.add_argument("--bfactor-scale", type=float, default=100.0,
                   help="multiply scores by this before writing the B-factor column")
    g.add_argument("--threshold", type=float, default=0.5,
                   help="score cutoff used for the n_above_* summary counts")
    g.add_argument("--top-k", type=int, default=25, help="rows per head in the top-residue table")
    g.add_argument("--save-embeddings", action="store_true",
                   help="also store the 1280-d ESM2 embedding in the .npz")
    g.add_argument("--summary", default="summary.csv", help="run-level summary file name")
    g.add_argument("--cache-dir", default=None,
                   help="where downloaded RCSB files are kept (default: <out-dir>/rcsb_cache)")
    g.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--version", action="version", version=f"unipocket {__version__}")
    return p


def resolve_inputs(tokens: Sequence[str], cache_dir: str,
                   log=print) -> List[Tuple[str, str]]:
    """Expand CLI tokens into a list of (kind, path) with kind in {structure, fasta}."""
    out: List[Tuple[str, str]] = []
    for token in tokens:
        if os.path.isdir(token):
            found = sorted(f for ext in STRUCTURE_EXT
                           for f in glob.glob(os.path.join(token, "*" + ext)))
            if not found:
                log(f"  ! no structure files in {token}")
            out.extend(("structure", f) for f in found)
        elif os.path.isfile(token):
            ext = os.path.splitext(token)[1].lower()
            if ext in FASTA_EXT:
                out.append(("fasta", token))
            else:
                out.append(("structure", token))
        elif is_pdb_id(token):
            log(f"  fetching {token.upper()} from RCSB ...")
            out.append(("structure", fetch_pdb(token, cache_dir)))
        else:
            raise FileNotFoundError(f"no such file, directory, or PDB ID: {token}")
    return out


def emit(pred: Prediction, dirs: dict, formats: Sequence[str], heads: Sequence[str],
         args, log=print) -> None:
    """Write every requested output for one prediction."""
    if "csv" in formats:
        write_csv(pred, os.path.join(dirs["per_residue"], f"{pred.entry_id}.csv"))
    if "npz" in formats:
        write_npz(pred, os.path.join(dirs["per_residue"], f"{pred.entry_id}.npz"),
                  with_embedding=args.save_embeddings)
    if "txt" in formats:
        write_txt(pred, os.path.join(dirs["per_residue"], f"{pred.entry_id}.txt"))
    if "top" in formats:
        write_top_residues(pred, os.path.join(dirs["per_residue"],
                                              f"{pred.entry_id}_top.csv"), args.top_k)
    # A .pml is useless without the structure it loads, so "pml" implies "pdb".
    if {"pdb", "pml"} & set(formats):
        if pred.structure is None:
            log(f"  ! {pred.entry_id}: sequence input, skipping scored-structure output")
            return
        for head in heads:
            struct_path = write_structure_scores(
                pred, dirs["structures"], head, scale=args.bfactor_scale)
            if "pml" in formats and struct_path:
                write_pymol_script(pred, dirs["structures"], head, struct_path,
                                   scale=args.bfactor_scale)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    log = (lambda *a, **k: None) if args.quiet else print

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    bad = [f for f in formats if f not in ALL_FORMATS]
    if bad:
        print(f"error: unknown output format(s): {', '.join(bad)}", file=sys.stderr)
        return 2
    heads = [h.strip().lower() for h in args.heads.split(",") if h.strip()]
    bad = [h for h in heads if h not in HEADS]
    if bad:
        print(f"error: unknown head(s): {', '.join(bad)}", file=sys.stderr)
        return 2
    chains = [c.strip() for c in args.chains.split(",")] if args.chains else None

    out_dir = os.path.abspath(args.out_dir)
    dirs = {"root": out_dir,
            "per_residue": os.path.join(out_dir, "per_residue"),
            "structures": os.path.join(out_dir, "structures")}
    os.makedirs(dirs["per_residue"], exist_ok=True)
    if {"pdb", "pml"} & set(formats):
        os.makedirs(dirs["structures"], exist_ok=True)
    cache_dir = args.cache_dir or os.path.join(out_dir, "rcsb_cache")

    log("Resolving inputs ...")
    try:
        items = resolve_inputs(args.inputs, cache_dir, log=log)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not items:
        print("error: nothing to do", file=sys.stderr)
        return 2
    log(f"  {len(items)} input(s)")

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    log(f"Device: {device}")
    log("Loading UniPocket + ESM2 (first run downloads ~2.5 GB of ESM2 weights) ...")
    try:
        predictor = PocketPredictor(ckpt=args.ckpt, esm_model=args.esm_model,
                                    device=device, long_mode=args.long_mode,
                                    fp16=args.fp16)
        predictor.embedder.load()
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    log("  ready.")

    summary_rows, failures = [], []
    for kind, path in items:
        name = os.path.basename(path)
        try:
            if kind == "fasta":
                entries = read_fasta(path)
                log(f"[{name}] {len(entries)} sequence(s)")
                preds = predictor.predict_fasta_entries(
                    entries, source=path, keep_embedding=args.save_embeddings,
                    progress=not args.quiet)
            else:
                preds = predictor.predict_structure(
                    path, per_chain=args.per_chain, chains=chains,
                    model_idx=args.model_idx, standard_only=args.standard_only,
                    require_ca=not args.keep_ca_less,
                    keep_embedding=args.save_embeddings)
                log(f"[{name}] {sum(p.n_scored for p in preds)} residues, "
                    f"{len(preds)} scoring unit(s)")
        except Exception as exc:                       # keep going through a batch
            failures.append((name, str(exc)))
            log(f"  ! {name}: {exc}")
            continue

        for pred in preds:
            if pred.truncated:
                log(f"  ! {pred.entry_id}: scored {pred.n_scored} of "
                    f"{pred.n_input_residues} residues (--long-mode truncate)")
            emit(pred, dirs, formats, heads, args, log=log)
            summary_rows.append(pred.summary(threshold=args.threshold))

    summary_path = os.path.join(out_dir, args.summary)
    write_summary(summary_rows, summary_path)

    log("")
    log("=" * 60)
    log(f"Scored {len(summary_rows)} unit(s) from {len(items)} input(s)")
    log(f"  per-residue scores : {dirs['per_residue']}")
    if {"pdb", "pml"} & set(formats):
        log(f"  scored structures  : {dirs['structures']}")
    log(f"  summary            : {summary_path}")
    if failures:
        log(f"  {len(failures)} input(s) failed:")
        for name, err in failures:
            log(f"    {name}: {err}")
    if summary_rows and not args.quiet:
        top = sorted(summary_rows, key=lambda r: r.get("cry_max") or -1, reverse=True)[:5]
        log("")
        log("Top entries by CRY max:")
        for r in top:
            log(f"  {r['entry_id']:<24s} n={r['n_residues']:<6d} "
                f"cry_max={r['cry_max']:.3f}  lig_max={r['lig_max']:.3f}")
    return 1 if failures and not summary_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
