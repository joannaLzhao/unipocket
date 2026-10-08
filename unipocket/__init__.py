"""
UniPocket - per-residue cryptic-pocket and ligand-binding prediction from
ESM2-650M sequence embeddings.

Typical use::

    from unipocket import PocketPredictor

    predictor = PocketPredictor()
    for pred in predictor.predict_structure("1ake.pdb"):
        print(pred.entry_id, pred.cry.max(), pred.lig.max())
"""

__version__ = "0.1.0"

from .embed import DEFAULT_ESM_MODEL, ESM2Embedder
from .model import DEFAULT_CKPT, MultiHeadPocketNet, UniPocket, load_unipocket
from .predict import HEADS, PocketPredictor, Prediction
from .structure import (ResidueRecord, extract_residues, fetch_pdb,
                        load_structure, read_fasta, sequence_of,
                        write_scored_structure)
from .writers import (write_csv, write_npz, write_pymol_script, write_summary,
                      write_top_residues, write_txt, write_structure_scores)

__all__ = [
    "__version__",
    "PocketPredictor", "Prediction", "HEADS",
    "UniPocket", "MultiHeadPocketNet", "load_unipocket", "DEFAULT_CKPT",
    "ESM2Embedder", "DEFAULT_ESM_MODEL",
    "ResidueRecord", "load_structure", "extract_residues", "sequence_of",
    "fetch_pdb", "read_fasta", "write_scored_structure",
    "write_csv", "write_npz", "write_txt", "write_structure_scores",
    "write_pymol_script", "write_top_residues", "write_summary",
]
