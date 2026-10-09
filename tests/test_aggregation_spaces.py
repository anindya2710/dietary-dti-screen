"""
Regression tests for the aggregation-space correction.

An earlier version of screen.py claimed that aggregating in pKd space was "the
equivalent" of DeepPurpose's nM-space aggregation. An external audit identified this as
false, and it is: averaging in pKd is a geometric mean in nM, while DeepPurpose's
convert_y=True branch takes an arithmetic mean in nM, which is dominated by the
weakest-binding model.

These tests pin both branches to DeepPurpose's actual arithmetic and pin the
non-equivalence itself, so the claim cannot quietly return.
"""

from __future__ import annotations

import numpy as np
import pytest

from dti_screen.screen import (
    AGGREGATION_SPACES,
    aggregate_pkd,
    aggregation_sensitivity,
    nM_to_pkd,
    pkd_to_nM,
)

# 4 models x 6 molecules
PKD = np.array(
    [
        [8.9, 7.2, 6.1, 5.4, 4.9, 3.2],
        [8.1, 7.9, 5.2, 6.0, 4.1, 3.8],
        [9.4, 6.8, 6.9, 4.8, 5.5, 2.9],
        [7.6, 8.4, 5.8, 5.9, 4.4, 3.5],
    ]
)


class TestPkdSpaceMatchesDeepPurpose:
    """space='pkd' must equal DeepPurpose's convert_y=False branch, verbatim."""

    def test_agg_mean_max(self):
        expected = (PKD.max(axis=0) + PKD.mean(axis=0)) / 2.0
        np.testing.assert_allclose(aggregate_pkd(PKD, "agg_mean_max", space="pkd"), expected)

    def test_mean(self):
        np.testing.assert_allclose(aggregate_pkd(PKD, "mean", space="pkd"), PKD.mean(axis=0))

    def test_max_effect(self):
        np.testing.assert_allclose(
            aggregate_pkd(PKD, "max_effect", space="pkd"), PKD.max(axis=0)
        )


class TestNmSpaceMatchesDeepPurpose:
    """space='nm' must equal DeepPurpose's convert_y=True branch (oneliner's default)."""

    def test_agg_mean_max(self):
        nm = pkd_to_nM(PKD)
        expected_nm = (nm.min(axis=0) + nm.mean(axis=0)) / 2.0
        np.testing.assert_allclose(
            aggregate_pkd(PKD, "agg_mean_max", space="nm"), nM_to_pkd(expected_nm)
        )

    def test_mean_is_arithmetic_in_nM_not_pkd(self):
        nm = pkd_to_nM(PKD)
        np.testing.assert_allclose(
            aggregate_pkd(PKD, "mean", space="nm"), nM_to_pkd(nm.mean(axis=0))
        )

    def test_max_effect_becomes_min_in_nM(self):
        """Stronger binding is a smaller nM value, so 'max effect' is the minimum."""
        nm = pkd_to_nM(PKD)
        np.testing.assert_allclose(
            aggregate_pkd(PKD, "max_effect", space="nm"), nM_to_pkd(nm.min(axis=0))
        )


class TestTheTwoSpacesAreNotEquivalent:
    """The central regression: these branches are different estimators."""

    def test_mean_differs_between_spaces(self):
        a = aggregate_pkd(PKD, "mean", space="pkd")
        b = aggregate_pkd(PKD, "mean", space="nm")
        assert not np.allclose(a, b)

    def test_pkd_mean_is_geometric_mean_in_nM(self):
        """Averaging in log space == geometric mean in linear space."""
        nm = pkd_to_nM(PKD)
        geometric = np.exp(np.log(nm).mean(axis=0))
        np.testing.assert_allclose(
            pkd_to_nM(aggregate_pkd(PKD, "mean", space="pkd")), geometric, rtol=1e-9
        )

    def test_nm_mean_exceeds_pkd_mean_in_nM(self):
        """AM >= GM, so the nM branch always reports weaker or equal binding."""
        nm_branch = pkd_to_nM(aggregate_pkd(PKD, "mean", space="nm"))
        pkd_branch = pkd_to_nM(aggregate_pkd(PKD, "mean", space="pkd"))
        assert np.all(nm_branch >= pkd_branch - 1e-9)

    def test_rank_order_can_invert(self):
        """
        Pins the audit finding. Over random ensembles the two branches disagree on rank
        order in the overwhelming majority of cases; here we require that a disagreement
        is findable at all, which is what makes the old "equivalent" claim false.
        """
        rng = np.random.default_rng(0)
        disagreements = 0
        for _ in range(300):
            m = rng.uniform(3.0, 10.0, size=(4, 8))
            a = np.argsort(-aggregate_pkd(m, "agg_mean_max", space="pkd"))
            b = np.argsort(-aggregate_pkd(m, "agg_mean_max", space="nm"))
            if not np.array_equal(a, b):
                disagreements += 1
        assert disagreements > 150, (
            f"expected frequent rank disagreement, saw {disagreements}/300"
        )


class TestAggregationSensitivity:
    def test_reports_rho_and_order_flag(self):
        s = aggregation_sensitivity(PKD)
        assert set(s) == {"spearman_rho", "same_order", "pkd_space", "nm_space"}
        assert -1.0 <= s["spearman_rho"] <= 1.0
        assert isinstance(s["same_order"], bool)

    def test_identical_models_give_identical_spaces(self):
        """With zero ensemble disagreement, AM == GM and the branches coincide."""
        flat = np.tile(np.array([8.0, 7.0, 6.0, 5.0]), (4, 1))
        s = aggregation_sensitivity(flat)
        np.testing.assert_allclose(s["pkd_space"], s["nm_space"], rtol=1e-9)
        assert s["same_order"] is True


class TestValidation:
    def test_unknown_space_rejected(self):
        with pytest.raises(ValueError, match="unknown space"):
            aggregate_pkd(PKD, "mean", space="log10")

    def test_unknown_aggregation_rejected(self):
        with pytest.raises(ValueError, match="unknown aggregation"):
            aggregate_pkd(PKD, "median", space="pkd")

    def test_spaces_constant(self):
        assert AGGREGATION_SPACES == ("pkd", "nm")

    def test_config_rejects_bad_space(self):
        from dti_screen.config import ScreenConfig

        with pytest.raises(ValueError, match="aggregation_space must be"):
            ScreenConfig(aggregation_space="nanomolar").validate()
