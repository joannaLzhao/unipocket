# UniPocket

Per-residue **cryptic-pocket** and **ligand-binding** prediction for any protein
structure, from sequence alone.

UniPocket is a small two-head network over ESM2-650M per-residue embeddings. It
needs no MSA, no docking, no molecular dynamics and no pocket-detection
preprocessing — give it a PDB file and it returns, for every residue, two
probabilities:

| Head | Question it answers | Trained on |
|---|---|---|
| **CRY** | Does this residue line a *cryptic* pocket — one that is closed in the apo structure and opens on ligand binding? | CryptoBench |
| **LIG** | Does this residue contact a ligand in a holo structure? | RCSB holo complexes (5 Å cutoff) |

On the CryptoBench held-out test set the CRY head reaches **0.816 ROC-AUC** and
**6.4× enrichment in the top 5%** of residues, against 0.593 (fpocket), 0.618
(P2Rank) and 0.499 (PocketMiner) on the same 193 proteins.

---

## Install

```bash
git clone <repo-url> && cd unipocket

# conda (recommended — handles the CUDA build of PyTorch)
conda env create -f environment.yml
conda activate unipocket
pip install -e .

# or pip, with PyTorch installed first from pytorch.org
pip install -r requirements.txt
pip install -e .
```

Requires Python ≥ 3.9. A GPU is optional: a 400-residue protein takes ~1 s on a
consumer GPU and ~15 s on CPU, nearly all of it inside ESM2.

On first use `fair-esm` downloads the ESM2-650M weights (~2.5 GB) to the Torch
hub cache. The UniPocket head itself (6.6 MB) ships in `checkpoints/` — see
[checkpoints/README.md](checkpoints/README.md). Point `--ckpt` or the
`UNIPOCKET_CKPT` environment variable elsewhere to use a different head.

## Run a brand-new PDB

```bash
# an RCSB entry, fetched automatically
unipocket 1ake -o results

# your own file (PDB or mmCIF), each chain scored separately
unipocket my_structure.pdb --per-chain -o results

# a whole directory, everything written out
unipocket structures/ -o results --formats csv,npz,txt,pdb,pml,top

# sequences only, no structure needed
unipocket sequences.fasta -o results --formats csv,npz
```

Without installing the package, `python scripts/predict.py ...` takes the same
arguments.

### Output layout

```
results/
├── summary.csv                  one row per scored chain/structure
├── per_residue/
│   ├── 1ake.csv                 index, chain, resseq, icode, resname, aa, cry_score, lig_score
│   ├── 1ake.npz                 the same columns as NumPy arrays
│   ├── 1ake.txt                 aligned plain text, np.loadtxt-friendly
│   └── 1ake_top.csv             top-k residues per head, ranked
├── structures/
│   ├── 1ake_cry.pdb             input coordinates, B-factor = CRY score × 100
│   ├── 1ake_lig.pdb             input coordinates, B-factor = LIG score × 100
│   └── 1ake_cry.pml             PyMOL script: colour by score, show the top hits
└── rcsb_cache/                  anything downloaded, so reruns are offline
```

`--formats` selects which of `csv,npz,txt,pdb,pml,top` are written (default
`csv,npz,pdb`). Every file is indexed consistently: row *i* of the CSV, element
*i* of each `.npz` array, and the residue carrying the *i*-th B-factor are the
same residue.

### Looking at the result

```bash
pymol results/structures/1ake_cry.pml
```

or in any viewer, colour by B-factor on a 0–100 ramp. The scale factor is
`--bfactor-scale` (default 100, the pLDDT convention; use `1` for raw
probabilities).


## How scores are produced

```
PDB / mmCIF ──▶ residue extraction ──▶ sequence ──▶ ESM2-650M layer 33 ──▶ UniPocket ──▶ CRY, LIG
                (Bio.PDB, CA-bearing                 (1280-d per residue)   (1.7 M params)
                 protein residues)
```

1. **Residue extraction.** Protein residues of the selected model that have a Cα
   atom, in file order. Waters, ions and ligands are dropped. Modified residues
   such as MSE are mapped to their parent amino acid so they do not break the
   chain (`--standard-only` drops them instead); residues with no Cα are treated
   as unresolved and skipped (`--keep-ca-less` keeps them).
2. **Embedding.** The concatenated sequence goes through ESM2-650M; layer-33
   representations are taken raw, with no normalisation, exactly as in training.
3. **Scoring.** The two heads share one backbone pass; outputs are sigmoids.
4. **Mapping back.** Scores are written against the original chain ID, residue
   number and insertion code, so nothing has to be re-derived downstream.

### Chains

By default all protein chains are concatenated into one sequence and scored in
a single pass, which is how the model was evaluated in the paper. ESM2 is
context-sensitive, so a residue's score then depends slightly on what else is in
the file: in 1AKE the two identical copies of adenylate kinase differ by ~0.001.
Use `--per-chain` to score each chain on its own and get results independent of
the rest of the assembly. Use `--chains A,B` to keep only some chains.

### Chains longer than 1022 residues

ESM2 has a 1024-token context. Longer chains are handled by `--long-mode`:

- `window` (default) — overlapping 1022-residue windows with a 511 stride; each
  residue takes its embedding from the window where it sits furthest from an
  edge, so every residue is scored with as much flanking context as exists.
- `truncate` — keep the first 1022 residues only and leave the rest unscored.
  This is what the published benchmark runs did; use it to reproduce them.

`summary.csv` flags any entry where `n_residues < n_input_residues`.

---

## Reproducing the published numbers

```bash
unipocket <input> --standard-only --long-mode truncate --keep-ca-less
```

with all chains concatenated (i.e. without `--per-chain`). That combination
reproduces the residue selection of the benchmark scripts exactly, and was
checked to be **bit-identical** — max absolute difference 0.0 on both heads —
against the reference implementation in
`supplementary/gen_pocketminer_external.py`, both on 1AKE and on a
deliberately awkward variant carrying a selenomethionine and a residue with no
Cα. The defaults differ from that path only in ways the paper never exercised:
they recover MSE as methionine, drop unresolved residues, and window past the
1022-residue limit instead of truncating.

