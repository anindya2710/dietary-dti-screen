"""
Control-set integrity.

These tests guard the benchmark's foundation. A wrong control structure would not crash
anything -- it would quietly produce a meaningless AUC, which is worse.
"""

from __future__ import annotations

import pytest

from dti_screen.controls import (
    CONTROL_COMPOUNDS,
    control_roles,
    verify_control_structures,
)


class TestVerifyControlStructures:
    def test_all_controls_match_literature_formulas(self):
        df = verify_control_structures()
        assert len(df) == len(CONTROL_COMPOUNDS)

    def test_every_target_has_actives_and_decoys_exist(self):
        df = verify_control_structures()
        counts = df.groupby("control_for").size().to_dict()
        for target in ("MAOA", "ADORA2A", "CB1"):
            assert counts.get(target, 0) >= 4, f"{target} needs >= 4 actives"
        assert counts.get("decoy", 0) >= 5

    def test_known_formulas_spot_check(self):
        df = verify_control_structures().set_index("compound_name")
        # Independently checkable reference values.
        assert df.loc["caffeine", "formula"] == "C8H10N4O2"
        assert df.loc["harmine", "formula"] == "C13H12N2O"
        assert df.loc["rimonabant", "formula"] == "C22H21Cl3N4O"
        assert df.loc["delta9-THC", "formula"] == "C21H30O2"
        assert df.loc["sucrose", "formula"] == "C12H22O11"
        assert df.loc["CP-55940", "formula"] == "C24H40O3"

    def test_molecular_weights_are_sane(self):
        df = verify_control_structures()
        assert df["MW"].between(50, 600).all()

    def test_no_duplicate_names(self):
        df = verify_control_structures()
        assert df["compound_name"].is_unique

    def test_canonical_smiles_are_unique(self):
        """Two controls must not be the same molecule written differently."""
        df = verify_control_structures()
        assert df["SMILES"].is_unique

    def test_a_wrong_formula_is_caught(self):
        """
        The assertion must actually fire. Regression guard: an earlier draft of this set
        had an extra hydroxyl on CP-55940 (C24H40O4 vs C24H40O3) and this check is what
        found it.
        """
        broken = [("caffeine", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "C99H99N9O9", "ADORA2A", "x")]
        with pytest.raises(AssertionError, match="expected C99H99N9O9"):
            verify_control_structures(broken)

    def test_an_unparseable_control_is_caught(self):
        broken = [("nonsense", "not_a_smiles", "C1H1", "MAOA", "x")]
        with pytest.raises(AssertionError, match="unparseable"):
            verify_control_structures(broken)

    def test_sucrose_is_a_disaccharide(self):
        """
        Regression guard: an earlier draft used a monosaccharide SMILES labelled
        'sucrose', which the formula check caught (C6H12O6 vs C12H22O11).
        """
        df = verify_control_structures().set_index("compound_name")
        assert df.loc["sucrose", "MW"] == pytest.approx(342.30, abs=0.05)


class TestControlRoles:
    def test_maps_every_control(self):
        roles = control_roles()
        assert len(roles) == len(CONTROL_COMPOUNDS)

    def test_role_values_are_targets_or_decoy(self):
        assert set(control_roles().values()) <= {"MAOA", "ADORA2A", "CB1", "decoy"}

    def test_spot_check_assignments(self):
        roles = control_roles()
        assert roles["harmine"] == "MAOA"
        assert roles["ZM-241385"] == "ADORA2A"
        assert roles["rimonabant"] == "CB1"
        assert roles["D-glucose"] == "decoy"
