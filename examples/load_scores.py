#!/usr/bin/env python
"""
How to consume UniPocket output from your own analysis code.

    python examples/load_scores.py examples/output/per_residue/1ake.npz
"""
import sys

import numpy as np


def main(npz_path: str) -> None:
    z = np.load(npz_path, allow_pickle=False)

    cry = z["cry"]            # (L,) cryptic-pocket probability
    lig = z["lig"]            # (L,) ligand-binding probability
    chain = z["chain"]        # (L,) chain id as written in the input file
    resseq = z["resseq"]      # (L,) author residue number
    aa = z["aa"]              # (L,) one-letter code

    print(f"{str(z['entry_id'])}: {len(cry)} residues scored")
    print(f"  cryptic : mean {cry.mean():.3f}  max {cry.max():.3f}")
    print(f"  ligand  : mean {lig.mean():.3f}  max {lig.max():.3f}")

    # The top 5% of residues by cryptic score: the usual hot-spot definition.
    k = max(1, int(np.ceil(0.05 * len(cry))))
    top = np.argsort(cry)[::-1][:k]
    print(f"\nTop {k} cryptic-pocket residues:")
    for i in sorted(top):
        print(f"  {chain[i]}/{aa[i]}{resseq[i]:<5d} cry={cry[i]:.3f} lig={lig[i]:.3f}")

    # A PyMOL selection string for the same set.
    sel = " or ".join(f"(chain {chain[i]} and resi {resseq[i]})" for i in sorted(top))
    print(f"\nPyMOL:\n  select cryptic_top, {sel}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
