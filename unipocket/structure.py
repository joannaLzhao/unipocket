"""
Structure handling: PDB / mmCIF in, scored PDB / mmCIF out.

The residue list produced here defines the coordinate system for every
downstream array: ``scores[i]`` always refers to ``residues[i]``.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import MMCIFIO, MMCIFParser, PDBIO, PDBParser
from Bio.PDB.Polypeptide import is_aa

# Non-standard residues that are part of the polypeptide chain and are mapped
# onto their parent amino acid so they do not create artificial chain breaks.
MODIFIED_TO_STANDARD: Dict[str, str] = {
    "MSE": "M",  # selenomethionine (by far the most common)
    "SEC": "C", "PYL": "K", "SEP": "S", "TPO": "T", "PTR": "Y",
    "CSO": "C", "CME": "C", "CSD": "C", "OCS": "C", "CSS": "C",
    "HYP": "P", "KCX": "K", "LLP": "K", "MLY": "K", "M3L": "K",
    "ALY": "K", "FME": "M", "CGU": "E", "PCA": "E",
    "DAL": "A", "ABA": "A", "AIB": "A", "NLE": "L", "ORN": "K",
    "TYS": "Y", "SNN": "N", "MEN": "N",
    "SMC": "C", "MHO": "M", "TRQ": "W", "TRN": "W",
}

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.{ext}"


@dataclass(frozen=True)
class ResidueRecord:
    """One scored residue, keyed so it can be found again in the structure."""
    chain_id: str
    resseq: int
    icode: str
    resname: str
    aa: str
    het_flag: str

    @property
    def key(self) -> Tuple[str, str, int, str]:
        return (self.chain_id, self.het_flag, self.resseq, self.icode)

    @property
    def label(self) -> str:
        return f"{self.chain_id}{self.resseq}{self.icode.strip()}"


def three_to_one(resname: str, allow_modified: bool = True) -> Optional[str]:
    """3-letter residue name -> 1-letter code, or None if not a protein residue."""
    name = resname.strip().upper()
    std = protein_letters_3to1.get(name.capitalize())
    if std:
        return std
    if allow_modified:
        return MODIFIED_TO_STANDARD.get(name)
    return None


def is_pdb_id(token: str) -> bool:
    """True for a bare 4-character RCSB entry ID such as ``1ake``."""
    return len(token) == 4 and token[0].isdigit() and token.isalnum()


def fetch_pdb(pdb_id: str, cache_dir: str) -> str:
    """Download an RCSB entry (PDB format, mmCIF fallback) and return the path."""
    os.makedirs(cache_dir, exist_ok=True)
    pdb_id = pdb_id.lower()
    for ext in ("pdb", "cif"):
        dest = os.path.join(cache_dir, f"{pdb_id}.{ext}")
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            return dest
    last_err: Optional[Exception] = None
    for ext in ("pdb", "cif"):
        url = RCSB_URL.format(pdb_id=pdb_id.upper(), ext=ext)
        dest = os.path.join(cache_dir, f"{pdb_id}.{ext}")
        try:
            urllib.request.urlretrieve(url, dest)
            return dest
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            last_err = exc
            if os.path.exists(dest):
                os.remove(dest)
    raise RuntimeError(f"could not download {pdb_id} from RCSB: {last_err}")


def _parser_for(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".cif", ".mmcif"):
        return MMCIFParser(QUIET=True), "cif"
    return PDBParser(QUIET=True, PERMISSIVE=True), "pdb"


def load_structure(path: str, structure_id: Optional[str] = None):
    """Parse a PDB or mmCIF file; returns (structure, format)."""
    parser, fmt = _parser_for(path)
    structure_id = structure_id or os.path.splitext(os.path.basename(path))[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structure = parser.get_structure(structure_id, path)
    return structure, fmt


def extract_residues(structure, model_idx: int = 0,
                     chains: Optional[Sequence[str]] = None,
                     standard_only: bool = False,
                     require_ca: bool = True) -> List[ResidueRecord]:
    """
    Collect protein residues from one model, in file order.

    Residues without a CA atom are skipped by default: they are not resolved in
    the deposited coordinates, so there is nothing to colour and no side chain
    the model could be asked about.
    """
    model_ids = [m.id for m in structure]
    if model_idx >= len(model_ids):
        raise ValueError(f"model index {model_idx} out of range "
                         f"(structure has {len(model_ids)} model(s))")
    model = structure[model_ids[model_idx]]

    wanted = set(chains) if chains else None
    records: List[ResidueRecord] = []
    for chain in model:
        if wanted is not None and chain.id not in wanted:
            continue
        for res in chain:
            het_flag, resseq, icode = res.get_id()
            resname = res.get_resname().strip().upper()
            if standard_only:
                if not is_aa(res, standard=True):
                    continue
                aa = three_to_one(resname, allow_modified=False)
            else:
                aa = three_to_one(resname, allow_modified=True)
            if aa is None or aa == "X":
                continue
            if require_ca and "CA" not in res:
                continue
            records.append(ResidueRecord(
                chain_id=chain.id, resseq=int(resseq), icode=icode,
                resname=resname, aa=aa, het_flag=het_flag))
    return records


def sequence_of(records: Iterable[ResidueRecord]) -> str:
    return "".join(r.aa for r in records)


def group_by_chain(records: Sequence[ResidueRecord]) -> Dict[str, List[ResidueRecord]]:
    """Chain id -> residues, preserving file order within each chain."""
    out: Dict[str, List[ResidueRecord]] = {}
    for r in records:
        out.setdefault(r.chain_id, []).append(r)
    return out


def write_scored_structure(structure, model_idx: int,
                           records: Sequence[ResidueRecord],
                           values: Sequence[float],
                           out_path: str,
                           scale: float = 100.0,
                           default: float = 0.0,
                           fmt: str = "pdb") -> str:
    """
    Write the structure with the B-factor column replaced by ``values * scale``.

    Residues that were not scored (ligands, waters, unresolved) get ``default``,
    so a 0-100 colour ramp in PyMOL or ChimeraX reads directly as the score.
    """
    if len(records) != len(values):
        raise ValueError("records and values must have the same length")
    lookup = {r.key: float(v) * scale for r, v in zip(records, values)}

    model = structure[[m.id for m in structure][model_idx]]
    for chain in model:
        for res in chain:
            het_flag, resseq, icode = res.get_id()
            val = lookup.get((chain.id, het_flag, int(resseq), icode), default)
            for atom in res.get_unpacked_list():
                atom.set_bfactor(val)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    io = MMCIFIO() if fmt == "cif" else PDBIO()
    io.set_structure(model)
    io.save(out_path)
    return out_path


def read_fasta(path: str) -> List[Tuple[str, str]]:
    """Minimal FASTA reader -> list of (id, sequence)."""
    entries: List[Tuple[str, str]] = []
    name, chunks = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    entries.append((name, "".join(chunks)))
                parts = line[1:].split()
                name = parts[0] if parts else f"seq{len(entries) + 1}"
                chunks = []
            else:
                chunks.append(line.upper().replace(" ", ""))
    if name is not None:
        entries.append((name, "".join(chunks)))
    return entries
