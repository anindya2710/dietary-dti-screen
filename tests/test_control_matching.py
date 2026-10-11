"""
Regression tests for structure-based control matching.

Found in external review. The pipeline tags a user's row as a control by comparing
canonical SMILES, but the benchmark used to *select* controls by compound name. A
library containing caffeine under any other name — "1,3,7-trimethylxanthine", a vendor
catalogue number, a typo — was therefore flagged as a control and then silently dropped
from the AUC, losing one of four ADORA2A actives with no warning.

Structure is the identity; the name is a label the user chose. These tests pin that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dti_screen.benchmark import benchmark_target
from dti_screen.controls import (
    control_roles,
    control_roles_by_smiles,
    role_series,
    verify_control_structures,
)
from dti_screen.crosstarget import mw_baseline_comparison, own_active_ranks

CAFFEINE_INPUT = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
CAFFEINE_CANON = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"


class TestRolesBySmiles:
    def test_every_control_has_a_smiles_key(self):
        assert len(control_roles_by_smiles()) == len(control_roles())

    def test_keys_are_canonical(self):
        from rdkit import Chem

        for smi in control_roles_by_smiles():
            assert Chem.MolToSmiles(Chem.MolFromSmiles(smi)) == smi

    def test_caffeine_maps_to_adora2a(self):
        assert control_roles_by_smiles()[CAFFEINE_CANON] == "ADORA2A"


class TestRoleSeries:
    def test_matches_by_structure_when_name_differs(self):
        """The exact defect: right molecule, wrong name."""
        df = pd.DataFrame(
            {"compound_name": ["1,3,7-trimethylxanthine"], "canonical_smiles": [CAFFEINE_CANON]}
        )
        assert role_series(df).iloc[0] == "ADORA2A"

    def test_falls_back_to_name_without_a_smiles_column(self):
        df = pd.DataFrame({"compound_name": ["caffeine", "harmine"]})
        assert role_series(df).tolist() == ["ADORA2A", "MAOA"]

    def test_structure_wins_over_a_misleading_name(self):
        """A row named 'harmine' but structurally caffeine is caffeine."""
        df = pd.DataFrame(
            {"compound_name": ["harmine"], "canonical_smiles": [CAFFEINE_CANON]}
        )
        assert role_series(df).iloc[0] == "ADORA2A"

    def test_non_controls_are_nan(self):
        df = pd.DataFrame({"compound_name": ["widgetol"], "canonical_smiles": ["CCCCCCCCO"]})
        assert pd.isna(role_series(df).iloc[0])

    def test_all_19_controls_resolve_by_structure_alone(self):
        ctrl = verify_control_structures()
        df = pd.DataFrame(
            {
                "compound_name": [f"anon_{i}" for i in range(len(ctrl))],
                "canonical_smiles": ctrl["SMILES"],
            }
        )
        assert role_series(df).notna().sum() == len(ctrl)


def _frame(names):
    """Predictions-shaped frame over the 19 controls, with caller-supplied names."""
    ctrl = verify_control_structures()
    g = pd.DataFrame(
        {
            "compound_name": names,
            "canonical_smiles": ctrl["SMILES"].tolist(),
            "MW": ctrl["MW"].tolist(),
            "pKd_aggregate": np.linspace(9.0, 4.0, len(ctrl)),
            "target": "ADORA2A",
            "is_control": True,
        }
    )
    g = g.sort_values("pKd_aggregate", ascending=False).reset_index(drop=True)
    g["rank"] = np.arange(1, len(g) + 1)
    return g


class TestBenchmarkUsesStructure:
    def setup_method(self):
        ctrl = verify_control_structures()
        self.real = ctrl["compound_name"].tolist()
        self.renamed = ["1,3,7-trimethylxanthine" if n == "caffeine" else n for n in self.real]

    def test_renaming_a_control_does_not_drop_it(self):
        a = benchmark_target(_frame(self.real), "ADORA2A", control_roles(), verbose=False)
        b = benchmark_target(_frame(self.renamed), "ADORA2A", control_roles(), verbose=False)
        assert a["n_actives"] == 4
        assert b["n_actives"] == 4, "renamed caffeine was silently dropped"
        assert a["auc"] == b["auc"]

    def test_renaming_every_control_still_works(self):
        anon = [f"CMPD_{i:03d}" for i in range(len(self.real))]
        out = benchmark_target(_frame(anon), "ADORA2A", control_roles(), verbose=False)
        assert out["n_actives"] == 4
        assert out["n_decoys"] == 6

    def test_mw_baseline_unaffected_by_renaming(self):
        a = mw_baseline_comparison(_frame(self.real))
        b = mw_baseline_comparison(_frame(self.renamed))
        assert a["n_actives"].iloc[0] == b["n_actives"].iloc[0] == 4
        assert a["model_auc"].iloc[0] == b["model_auc"].iloc[0]

    def test_own_active_ranks_unaffected_by_renaming(self):
        a = own_active_ranks(_frame(self.real))
        b = own_active_ranks(_frame(self.renamed))
        assert a["n_actives"].iloc[0] == b["n_actives"].iloc[0] == 4
        assert a["median_rank"].iloc[0] == b["median_rank"].iloc[0]


class TestCrossTargetPivotsOnStructure:
    def test_duplicate_names_are_not_silently_averaged(self):
        """
        Two distinct molecules sharing a compound_name must stay distinct. Pivoting on
        name would average their affinities before correlating.
        """
        from dti_screen.crosstarget import cross_target_correlation

        rows = []
        for tgt, off in (("A", 0.0), ("B", 0.5)):
            for smi, pk in (("CCO", 5.0), ("CCCO", 7.0), ("CCCCO", 6.0), ("CCCCCO", 8.0)):
                rows.append(
                    {
                        "compound_name": "isomer",  # deliberately identical
                        "canonical_smiles": smi,
                        "pKd_aggregate": pk + off,
                        "target": tgt,
                        "is_control": False,
                    }
                )
        out = cross_target_correlation(pd.DataFrame(rows))
        assert out["n_molecules"].iloc[0] == 4, "distinct structures were collapsed by name"

    def test_falls_back_to_name_when_no_smiles_column(self):
        from dti_screen.crosstarget import cross_target_correlation

        rows = []
        for tgt, off in (("A", 0.0), ("B", 0.5)):
            for name, pk in (("a", 5.0), ("b", 7.0), ("c", 6.0)):
                rows.append(
                    {"compound_name": name, "pKd_aggregate": pk + off,
                     "target": tgt, "is_control": False}
                )
        out = cross_target_correlation(pd.DataFrame(rows))
        assert out["n_molecules"].iloc[0] == 3
        assert out["spearman_rho"].iloc[0] == pytest.approx(1.0)
