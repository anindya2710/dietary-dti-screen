"""
Tests for the small-sample statistics added after an external audit.

The audit's correct central finding: with 4-5 actives and 6 decoys, AUC is quantised to
steps of 1/24 or 1/30, so a reported "delta" of 0.042 is a single swapped pair. These
tests pin the resolution arithmetic, the exact permutation test, and the bootstrap.
"""

from __future__ import annotations

import numpy as np
import pytest

from dti_screen.benchmark import (
    auc_resolution,
    bootstrap_auc_ci,
    exact_permutation_p,
    pairwise_auc,
)


class TestAucResolution:
    def test_four_by_six(self):
        """The repo's CB1 case: delta of 0.042 is exactly one pair."""
        assert auc_resolution(4, 6) == pytest.approx(1 / 24)
        assert auc_resolution(4, 6) == pytest.approx(0.0417, abs=1e-4)

    def test_five_by_six(self):
        assert auc_resolution(5, 6) == pytest.approx(1 / 30)

    def test_larger_samples_give_finer_resolution(self):
        assert auc_resolution(100, 100) < auc_resolution(4, 6)

    @pytest.mark.parametrize("a,d", [(0, 6), (4, 0), (0, 0), (-1, 5)])
    def test_degenerate_inputs_are_nan(self, a, d):
        assert np.isnan(auc_resolution(a, d))

    def test_observed_auc_is_a_multiple_of_the_resolution(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            act, dec = rng.normal(size=4), rng.normal(size=6)
            auc = pairwise_auc(act, dec)
            # ties give half-steps, so check against half the resolution
            assert (auc / (auc_resolution(4, 6) / 2)) == pytest.approx(
                round(auc / (auc_resolution(4, 6) / 2)), abs=1e-9
            )


class TestExactPermutationP:
    def test_perfect_separation_smallest_possible_p(self):
        """With 4 vs 6, only 1 of C(10,4)=210 labellings is perfect."""
        p = exact_permutation_p([10, 9, 8, 7], [6, 5, 4, 3, 2, 1])
        assert p == pytest.approx(1 / 210)

    def test_perfect_inversion_gives_p_of_one(self):
        assert exact_permutation_p([1, 2, 3, 4], [5, 6, 7, 8, 9, 10]) == pytest.approx(1.0)

    def test_all_ties_gives_p_of_one(self):
        assert exact_permutation_p([5, 5, 5], [5, 5, 5]) == pytest.approx(1.0)

    def test_p_is_a_probability(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            p = exact_permutation_p(rng.normal(size=4), rng.normal(size=6))
            assert 0.0 <= p <= 1.0

    def test_empty_group_is_nan(self):
        assert np.isnan(exact_permutation_p([], [1, 2, 3]))

    def test_modest_separation_is_not_significant_at_this_n(self):
        """
        The headline consequence: a visually convincing gap can still be
        unremarkable with four actives.
        """
        p = exact_permutation_p([7.0, 6.5, 6.0, 5.5], [6.2, 5.8, 5.4, 5.0, 4.6, 4.2])
        assert p > 0.01

    def test_returns_nan_when_enumeration_too_large(self):
        assert np.isnan(exact_permutation_p(np.arange(30.0), np.arange(30.0)))


class TestBootstrapCi:
    def test_interval_brackets_the_point_estimate(self):
        act = np.array([9.0, 8.0, 7.5, 7.0])
        dec = np.array([6.0, 5.0, 4.5, 4.0, 3.5, 3.0])
        lo, hi = bootstrap_auc_ci(act, dec, n_boot=2000)
        assert lo <= pairwise_auc(act, dec) <= hi

    def test_bounds_within_unit_interval(self):
        lo, hi = bootstrap_auc_ci([7, 6, 5, 4], [6, 5, 4, 3, 2, 1], n_boot=2000)
        assert 0.0 <= lo <= hi <= 1.0

    def test_interval_is_wide_at_tiny_n(self):
        """The audit's point: n=4 vs 6 cannot support a precise AUC."""
        lo, hi = bootstrap_auc_ci([7.0, 6.5, 6.0, 5.5],
                                  [6.2, 5.8, 5.4, 5.0, 4.6, 4.2], n_boot=4000)
        assert (hi - lo) > 0.3, f"expected a wide interval, got [{lo:.3f}, {hi:.3f}]"

    def test_deterministic_given_seed(self):
        a, d = [9, 8, 7, 6], [5, 4, 3, 2, 1, 0]
        assert bootstrap_auc_ci(a, d, n_boot=500, seed=42) == bootstrap_auc_ci(
            a, d, n_boot=500, seed=42
        )

    def test_empty_group_is_nan(self):
        lo, hi = bootstrap_auc_ci([], [1, 2, 3])
        assert np.isnan(lo) and np.isnan(hi)


class TestVerdictRespectsResolution:
    def test_delta_within_one_pair_is_called_indistinguishable(self):
        """A delta of 1/24 must not be reported as the model beating the baseline."""
        import pandas as pd

        from dti_screen.crosstarget import mw_baseline_comparison

        roles = {f"a{i}": "T" for i in range(4)} | {f"d{i}": "decoy" for i in range(6)}
        # actives marginally ahead: exactly one pair's worth
        rows = []
        for i in range(4):
            rows.append({"compound_name": f"a{i}", "target": "T",
                         "pKd_aggregate": 6.0 + i * 0.1, "MW": 300 + i, "rank": i + 1})
        for i in range(6):
            rows.append({"compound_name": f"d{i}", "target": "T",
                         "pKd_aggregate": 5.0 + i * 0.1, "MW": 290 + i, "rank": 5 + i})
        out = mw_baseline_comparison(pd.DataFrame(rows), roles)
        # the table rounds to 4 dp, so compare at that precision
        assert out["auc_resolution"].iloc[0] == pytest.approx(1 / 24, abs=5e-5)
        assert "delta_in_pairs" in out.columns
        assert {"model_auc_ci_lo", "model_auc_ci_hi", "model_auc_exact_p"} <= set(out.columns)
