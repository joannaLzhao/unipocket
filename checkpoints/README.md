# Checkpoints

## `unipocket_esm2_650M.pt` (6.6 MB)

The trained UniPocket head used for every number in the paper.

| | |
|---|---|
| Training script | `bench_esm2emb_only.py` (run tag `esm2_emb_only_20260408_203019`) |
| Input | ESM2-650M (`esm2_t33_650M_UR50D`) layer-33 per-residue representations, 1280-d, unnormalised |
| Architecture | shared MLP backbone (1280 → 256 → 512 → 2× ResBlock → 256) + two heads (128 → 1) |
| Parameters | 1,711,106 (1.71 M) |
| SHA-256 | `60e98189ca8aad0d16c3318f7087d10bbecef3742ead3357e6c7246bc8e1c925` |

The file is a `torch.save` dict with keys `state_dict`, `in_dim`, `cfg`, plus the
validation/test metrics and the dataset splits recorded at training time:

```python
import torch
ckpt = torch.load("checkpoints/unipocket_esm2_650M.pt", weights_only=False)
ckpt["cfg"]              # hyperparameters
ckpt["test_cry_macro"]   # CryptoBench test metrics for the CRY head
ckpt["a_split"]          # the RCSB train/val/test PDB IDs
```

### Held-out performance recorded in the checkpoint

| Head | Test set | ROC-AUC | AP | EF@1% | EF@5% |
|---|---|---|---|---|---|
| CRY | CryptoBench test (193 proteins) | 0.816 | 0.410 | 8.01 | 6.37 |
| LIG | RCSB holo test (420 proteins) | 0.640 | 0.113 | 6.13 | 3.81 |

Macro-averaged per protein. The CRY head is the one the paper benchmarks
against fpocket, P2Rank and PocketMiner.

## ESM2 weights

Not stored here. `fair-esm` downloads `esm2_t33_650M_UR50D` (~2.5 GB) into the
Torch hub cache on first use:

- Linux/macOS: `~/.cache/torch/hub/checkpoints/`
- Windows: `%USERPROFILE%\.cache\torch\hub\checkpoints\`

Set `TORCH_HOME` to move that cache somewhere else.
