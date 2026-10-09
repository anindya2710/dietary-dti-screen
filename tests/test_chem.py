"""
Structural QC tests.

The headline test is :meth:`TestInvalidSmiles.test_deeppurpose_silently_zero_fills`,
which documents *why* this filter exists: DeepPurpose does not raise on invalid SMILES,
it returns an all-zero fingerprint and scores the molecule anyway.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dti_screen.chem import MAX_SEQ_DRUG, canonicalize, validate_library

VALID = {
    "caffeine": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "quercetin": "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
    "glycine": "NCC(=O)O",
}
# rutin: a real dietary glycoside whose canonical SMILES is 122 chars (> MAX_SEQ_DRUG)
RUTIN = (
    "C[C@@H]1O[C@@H](OC[C@H]2O[C@@H](Oc3c(-c4ccc(O)c(O)c4)oc4cc(O)cc(O)c4c3=O)"
    "[C@H](O)[C@@H](O)[C@@H]2O)[C@H](O)[C@H](O)[C@H]1O"
)


class TestCanonicalize:
    def test_canonicalizes_equivalent_smiles_identically(self):
        assert canonicalize("C1=CC=CC=C1") == canonicalize("c1ccccc1")

    @pytest.mark.parametrize("bad", ["not_a_smiles_at_all", "C1CC", "", "   ", None, 42])
    def test_returns_none_for_invalid_input(self, bad):
        assert canonicalize(bad) is None


class TestValidateLibrary:
    def test_keeps_valid_molecules(self):
        df = pd.DataFrame({"name": list(VALID), "SMILES": list(VALID.values())})
        clean, rejected, report = validate_library(df, name_col="name", verbose=False)
        assert len(clean) == 3
        assert rejected.empty
        assert report["valid_molecules"] == 3

    def test_drops_blank_and_unparseable(self):
        df = pd.DataFrame(
            {
                "name": ["ok", "blank", "whitespace", "junk", "badring", "none"],
                "SMILES": [
                    VALID["caffeine"],
                    "",
                    "   ",
                    "not_a_smiles_at_all",
                    "C1CC",
                    None,
                ],
            }
        )
        clean, rejected, report = validate_library(df, name_col="name", verbose=False)
        assert len(clean) == 1
        assert clean.iloc[0]["compound_name"] == "ok"
        assert report["dropped_blank"] == 3      # "", "   ", None
        assert report["dropped_unparseable"] == 2
        assert len(rejected) == 5

    def test_rejected_rows_carry_a_reason(self):
        df = pd.DataFrame({"name": ["a", "b"], "SMILES": [VALID["glycine"], "nope!!"]})
        _, rejected, _ = validate_library(df, name_col="name", verbose=False)
        assert "_reject_reason" in rejected.columns
        assert rejected["_reject_reason"].notna().all()

    def test_deduplicates_on_canonical_structure(self):
        """Differently written SMILES for one molecule collapse to a single row."""
        df = pd.DataFrame(
            {"name": ["benzene_a", "benzene_b"], "SMILES": ["c1ccccc1", "C1=CC=CC=C1"]}
        )
        clean, _, report = validate_library(df, name_col="name", verbose=False)
        assert len(clean) == 1
        assert report["dropped_duplicates"] == 1

    def test_dedupe_can_be_disabled(self):
        df = pd.DataFrame({"name": ["a", "b"], "SMILES": ["c1ccccc1", "C1=CC=CC=C1"]})
        clean, _, report = validate_library(df, name_col="name", dedupe=False, verbose=False)
        assert len(clean) == 2
        assert report["dropped_duplicates"] == 0

    def test_long_smiles_is_flagged_not_dropped(self):
        """rutin exceeds MAX_SEQ_DRUG; it must survive but carry the truncation flag."""
        df = pd.DataFrame({"name": ["rutin", "caffeine"], "SMILES": [RUTIN, VALID["caffeine"]]})
        clean, _, report = validate_library(df, name_col="name", verbose=False)
        assert len(clean) == 2
        assert report["flagged_cnn_truncated"] == 1
        row = clean.loc[clean["compound_name"] == "rutin"].iloc[0]
        assert row["cnn_smiles_truncated"] is True or bool(row["cnn_smiles_truncated"])
        assert len(row["canonical_smiles"]) > MAX_SEQ_DRUG
        other = clean.loc[clean["compound_name"] == "caffeine"].iloc[0]
        assert not other["cnn_smiles_truncated"]

    def test_computes_descriptors(self):
        df = pd.DataFrame({"name": ["caffeine"], "SMILES": [VALID["caffeine"]]})
        clean, _, _ = validate_library(df, name_col="name", verbose=False)
        row = clean.iloc[0]
        assert row["formula"] == "C8H10N4O2"
        assert row["MW"] == pytest.approx(194.19, abs=0.05)
        assert row["n_heavy_atoms"] == 14

    def test_autogenerates_names_when_no_name_column(self):
        df = pd.DataFrame({"SMILES": list(VALID.values())})
        clean, _, _ = validate_library(df, name_col=None, verbose=False)
        assert clean["compound_name"].tolist() == ["CMPD_00000", "CMPD_00001", "CMPD_00002"]

    def test_missing_smiles_column_raises_with_available_columns(self):
        df = pd.DataFrame({"moldb_smiles": ["CCO"], "name": ["ethanol"]})
        with pytest.raises(KeyError, match="moldb_smiles"):
            validate_library(df, smiles_col="SMILES", verbose=False)

    def test_alternate_smiles_column_name(self):
        """FooDB-style exports use moldb_smiles rather than SMILES."""
        df = pd.DataFrame({"moldb_smiles": ["CCO"], "name": ["ethanol"]})
        clean, _, _ = validate_library(
            df, smiles_col="moldb_smiles", name_col="name", verbose=False
        )
        assert clean.iloc[0]["formula"] == "C2H6O"

    def test_all_invalid_raises(self):
        df = pd.DataFrame({"SMILES": ["junk", "C1CC", ""]})
        with pytest.raises(ValueError, match="No valid molecules"):
            validate_library(df, verbose=False)

    def test_example_csv_is_fully_valid(self):
        """Every shipped example molecule must survive QC unchanged."""
        from pathlib import Path

        csv = Path(__file__).resolve().parents[1] / "data" / "example_ingredients.csv"
        df = pd.read_csv(csv)
        clean, rejected, report = validate_library(df, name_col="name", verbose=False)
        assert rejected.empty
        assert report["dropped_duplicates"] == 0
        assert len(clean) == len(df)

    def test_example_csv_formulas_match_rdkit(self):
        """The CSV's own formula column must agree with RDKit's recomputation."""
        from pathlib import Path

        csv = Path(__file__).resolve().parents[1] / "data" / "example_ingredients.csv"
        df = pd.read_csv(csv)
        clean, _, _ = validate_library(df, name_col="name", verbose=False)
        merged = clean.merge(df, on="name", suffixes=("_calc", "_csv"))
        assert (merged["formula_calc"] == merged["formula_csv"]).all()


@pytest.mark.needs_deeppurpose
class TestInvalidSmiles:
    def test_deeppurpose_silently_zero_fills(self):
        """
        Documents the failure mode this module exists to prevent: DeepPurpose does not
        raise on invalid SMILES, it returns an all-zero fingerprint, so the molecule
        receives a confident-looking score indistinguishable from a real one.
        """
        pytest.importorskip("torch")
        from DeepPurpose.utils import smiles2morgan

        assert smiles2morgan("not_a_smiles_at_all").sum() == 0.0
        assert smiles2morgan("C1CC").sum() == 0.0
        assert smiles2morgan(VALID["caffeine"]).sum() > 0.0
