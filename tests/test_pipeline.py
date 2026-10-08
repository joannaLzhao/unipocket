"""
Tests for the UniPocket inference pipeline.

The ESM2-dependent tests are skipped automatically when ``fair-esm`` is not
installed, so the structure/IO/model layers can still be checked offline.
Run with:  pytest -q
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from unipocket import (DEFAULT_CKPT, PocketPredictor, Prediction,
                       extract_residues, load_structure, load_unipocket,
                       sequence_of, write_csv, write_npz, write_txt)
from unipocket.embed import _windows
from unipocket.structure import three_to_one, write_scored_structure

# A four-residue fragment with one selenomethionine, written as a HETATM.
TINY_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 10.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 10.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 10.00           O
ATOM      5  N   GLY A   2       3.332   1.550   0.000  1.00 20.00           N
ATOM      6  CA  GLY A   2       3.970   2.850   0.000  1.00 20.00           C
ATOM      7  C   GLY A   2       5.480   2.700   0.000  1.00 20.00           C
ATOM      8  O   GLY A   2       6.000   1.590   0.000  1.00 20.00           O
HETATM    9  N   MSE A   3       6.150   3.840   0.000  1.00 30.00           N
HETATM   10  CA  MSE A   3       7.600   3.900   0.000  1.00 30.00           C
HETATM   11  C   MSE A   3       8.150   5.320   0.000  1.00 30.00           C
HETATM   12  O   MSE A   3       7.400   6.300   0.000  1.00 30.00           O
ATOM     13  N   LYS B   7       9.470   5.450   0.000  1.00 40.00           N
ATOM     14  CA  LYS B   7      10.110   6.750   0.000  1.00 40.00           C
ATOM     15  C   LYS B   7      11.620   6.600   0.000  1.00 40.00           C
ATOM     16  O   LYS B   7      12.140   5.490   0.000  1.00 40.00           O
HETATM   17  O   HOH A 101      20.000  20.000  20.000  1.00 50.00           O
END
"""


@pytest.fixture
def tiny_pdb(tmp_path):
    path = tmp_path / "tiny.pdb"
    path.write_text(TINY_PDB)
    return str(path)


def has_esm() -> bool:
    try:
        import esm  # noqa: F401
        return True
    except ImportError:
        return False


# ── residue handling ──────────────────────────────────────────────────────
def test_three_to_one_standard_and_modified():
    assert three_to_one("ALA") == "A"
    assert three_to_one("MSE") == "M"                          # mapped
    assert three_to_one("MSE", allow_modified=False) is None   # dropped
    assert three_to_one("HOH") is None


def test_extract_residues_skips_water_and_keeps_mse(tiny_pdb):
    structure, fmt = load_structure(tiny_pdb)
    assert fmt == "pdb"
    recs = extract_residues(structure)
    assert sequence_of(recs) == "AGMK"
    assert [r.chain_id for r in recs] == ["A", "A", "A", "B"]
    assert [r.resseq for r in recs] == [1, 2, 3, 7]


def test_extract_residues_standard_only(tiny_pdb):
    structure, _ = load_structure(tiny_pdb)
    assert sequence_of(extract_residues(structure, standard_only=True)) == "AGK"


def test_extract_residues_chain_filter(tiny_pdb):
    structure, _ = load_structure(tiny_pdb)
    assert sequence_of(extract_residues(structure, chains=["B"])) == "K"


def test_require_ca_controls_unresolved_residues(tmp_path):
    """A residue with no CA is unresolved: skipped by default, kept on request."""
    kept = [l for l in TINY_PDB.splitlines()
            if not (l[12:16].strip() == "CA" and l[22:26].strip() == "2")]
    path = tmp_path / "gap.pdb"
    path.write_text(os.linesep.join(kept) + os.linesep)
    structure, _ = load_structure(str(path))
    assert sequence_of(extract_residues(structure)) == "AMK"
    assert sequence_of(extract_residues(structure, require_ca=False)) == "AGMK"


# ── B-factor output ───────────────────────────────────────────────────────
def test_bfactor_written_and_unscored_zeroed(tiny_pdb, tmp_path):
    structure, _ = load_structure(tiny_pdb)
    recs = extract_residues(structure)
    scores = np.array([0.10, 0.25, 0.50, 0.99], np.float32)
    out = str(tmp_path / "scored.pdb")
    write_scored_structure(structure, 0, recs, scores, out, scale=100.0)

    reloaded, _ = load_structure(out)
    model = reloaded[0]
    got = [model[r.chain_id][(r.het_flag, r.resseq, r.icode)]["CA"].get_bfactor()
           for r in recs]
    assert np.allclose(got, scores * 100.0, atol=0.01)
    # water was never scored, so it must carry the default
    assert model["A"][("W", 101, " ")]["O"].get_bfactor() == pytest.approx(0.0)


# ── model ─────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not os.path.isfile(DEFAULT_CKPT), reason="checkpoint not present")
def test_checkpoint_loads_strictly_and_scores_in_range():
    net = load_unipocket(DEFAULT_CKPT, torch.device("cpu"))
    assert net.in_dim == 1280
    cry, lig = net.score(np.random.randn(37, 1280).astype(np.float32))
    assert cry.shape == lig.shape == (37,)
    assert np.all((cry >= 0) & (cry <= 1)) and np.all((lig >= 0) & (lig <= 1))


@pytest.mark.skipif(not os.path.isfile(DEFAULT_CKPT), reason="checkpoint not present")
def test_scoring_is_batch_size_invariant():
    """Chunking only changes float32 GEMM tiling, never the reported score."""
    net = load_unipocket(DEFAULT_CKPT, torch.device("cpu"))
    emb = np.random.randn(100, 1280).astype(np.float32)
    a_cry, a_lig = net.score(emb, batch_size=1024)
    b_cry, b_lig = net.score(emb, batch_size=7)
    assert np.allclose(a_cry, b_cry, atol=1e-5)
    assert np.allclose(a_lig, b_lig, atol=1e-5)
    # the rounding written to disk (4 dp) is unaffected
    assert np.array_equal(a_cry.round(4), b_cry.round(4))


# ── windowing ─────────────────────────────────────────────────────────────
def test_windows_cover_every_residue():
    for length in (10, 1022, 1023, 1500, 4000):
        wins = _windows(length, 1022, 511)
        covered = np.zeros(length, bool)
        for s, e in wins:
            covered[s:e] = True
        assert covered.all(), length
        assert wins[-1][1] == length
        assert all(e - s <= 1022 for s, e in wins)


# ── writers ───────────────────────────────────────────────────────────────
def _fake_prediction(tiny_pdb) -> Prediction:
    structure, fmt = load_structure(tiny_pdb)
    recs = extract_residues(structure)
    return Prediction(
        entry_id="tiny", sequence=sequence_of(recs),
        cry=np.array([0.1, 0.9, 0.4, 0.7], np.float32),
        lig=np.array([0.8, 0.2, 0.6, 0.3], np.float32),
        source=tiny_pdb, residues=recs, structure=structure, fmt=fmt)


def test_npz_round_trip_matches_csv(tiny_pdb, tmp_path):
    pred = _fake_prediction(tiny_pdb)
    npz_path = write_npz(pred, str(tmp_path / "tiny.npz"))
    csv_path = write_csv(pred, str(tmp_path / "tiny.csv"))
    write_txt(pred, str(tmp_path / "tiny.txt"))

    z = np.load(npz_path, allow_pickle=False)
    assert np.allclose(z["cry"], pred.cry)
    assert "".join(z["aa"]) == "AGMK"
    assert list(z["chain"]) == ["A", "A", "A", "B"]
    assert list(z["resseq"]) == [1, 2, 3, 7]

    rows = [l.split(",") for l in open(csv_path).read().splitlines()[1:]]
    assert [r[5] for r in rows] == ["A", "G", "M", "K"]
    assert [float(r[6]) for r in rows] == pytest.approx(list(pred.cry), abs=1e-4)

    # plain-text scores stay loadable as a numeric array
    txt = np.loadtxt(str(tmp_path / "tiny.txt"), usecols=(6, 7))
    assert np.allclose(txt[:, 0], pred.cry, atol=1e-4)


def test_summary_fields(tiny_pdb):
    row = _fake_prediction(tiny_pdb).summary()
    assert row["n_residues"] == 4 and row["truncated"] is False
    assert row["chains"] == "AB"
    assert row["cry_max"] == pytest.approx(0.9, abs=1e-4)
    assert row["cry_n_above_0.5"] == 2


def test_top_residues_are_ranked(tiny_pdb):
    tops = _fake_prediction(tiny_pdb).top_residues("cry", k=2)
    assert [t["resseq"] for t in tops] == [2, 7]
    assert tops[0]["score"] >= tops[1]["score"]


# ── end to end (needs ESM2 weights) ───────────────────────────────────────
@pytest.mark.skipif(not has_esm() or not os.path.isfile(DEFAULT_CKPT),
                    reason="needs fair-esm and the checkpoint")
def test_end_to_end_structure(tiny_pdb, tmp_path):
    predictor = PocketPredictor(device=torch.device("cpu"))
    (pred,) = predictor.predict_structure(tiny_pdb)
    assert pred.n_scored == 4
    assert pred.sequence == "AGMK"
    assert np.all((pred.cry >= 0) & (pred.cry <= 1))
    assert not pred.truncated

    per_chain = predictor.predict_structure(tiny_pdb, per_chain=True)
    assert [p.entry_id for p in per_chain] == ["tiny_A", "tiny_B"]


@pytest.mark.skipif(not has_esm() or not os.path.isfile(DEFAULT_CKPT),
                    reason="needs fair-esm and the checkpoint")
def test_window_mode_matches_single_pass_for_short_sequences():
    """A sequence inside the context window must bypass windowing entirely."""
    from unipocket.embed import ESM2Embedder
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    dev = torch.device("cpu")
    win = ESM2Embedder(device=dev, long_mode="window").load()
    trunc = ESM2Embedder(device=dev, long_mode="truncate").load()
    assert np.allclose(win.embed(seq), trunc.embed(seq))


@pytest.mark.skipif(not has_esm() or not os.path.isfile(DEFAULT_CKPT),
                    reason="needs fair-esm and the checkpoint")
def test_cli_writes_every_requested_format(tiny_pdb, tmp_path):
    from unipocket.cli import main

    out = tmp_path / "out"
    rc = main([tiny_pdb, "-o", str(out), "--device", "cpu", "-q",
               "--formats", "csv,npz,txt,pdb,pml,top"])
    assert rc == 0
    for rel in ("summary.csv",
                "per_residue/tiny.csv", "per_residue/tiny.npz",
                "per_residue/tiny.txt", "per_residue/tiny_top.csv",
                "structures/tiny_cry.pdb", "structures/tiny_lig.pdb",
                "structures/tiny_cry.pml"):
        assert (out / rel).is_file(), rel

    # the B-factor column must agree with the CSV it was written from
    z = np.load(out / "per_residue" / "tiny.npz")
    scored, _ = load_structure(str(out / "structures" / "tiny_cry.pdb"))
    bfac = [a.get_bfactor() for a in scored[0].get_atoms() if a.get_id() == "CA"]
    assert np.allclose(bfac, z["cry"] * 100.0, atol=0.01)
