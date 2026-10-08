#!/usr/bin/env python
"""
Run UniPocket without installing the package.

    python scripts/predict.py 1ake -o results
    python scripts/predict.py my_model.pdb --per-chain --formats csv,npz,pdb,pml

Identical to the `unipocket` console script; it just puts the repository root
on sys.path first.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unipocket.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
