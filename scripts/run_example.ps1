# Minimal smoke test: fetch adenylate kinase (1AKE) and score it.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
python scripts/predict.py 1ake -o examples/output --formats csv,npz,txt,pdb,pml,top
Write-Host ""
Write-Host "Open the result in PyMOL with:"
Write-Host "  pymol examples/output/structures/1ake_cry.pml"
