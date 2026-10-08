#!/usr/bin/env bash
# Minimal smoke test: fetch adenylate kinase (1AKE) and score it.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/predict.py 1ake \
    -o examples/output \
    --formats csv,npz,txt,pdb,pml,top
echo
echo "Open the result in PyMOL with:"
echo "  pymol examples/output/structures/1ake_cry.pml"
