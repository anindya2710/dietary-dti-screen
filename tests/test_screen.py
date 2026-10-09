"""Unit conversion, aggregation and benchmark arithmetic."""

from __future__ import annotations

import numpy as np
import pytest

from dti_screen.benchmark import interpret_auc, interpret_benchmark, pairwise_auc
from dti_screen.screen import aggregate_pkd, nM_to_pkd, pkd_to_nM


class TestUnitConversion:
    def test_matches_deeppurpose_formula(self):
        """Must equal DeepPurpose convert_y_unit(y, 'p', 'nM') == 10**(-y)/1e-9."""
        pkd = np.array([5.0, 7.0, 9.0])
        np.testing.assert_allclose(pkd_to_nM(pkd), np.power(10.0, -pkd) / 1e-9)

    def test_known_values(self):
        # pKd 9 -> 1 nM; pKd 6 -> 1000 nM
        assert pkd_to_nM(9.0)[()] == pytest.approx(1.0)
        assert pkd_to_nM(6.0)[()] == pytest.approx(1000.0)

    def test_round_trip(self):
        pkd = np.array([4.5, 6.25, 8.75, 10.0])
        np.testing.assert_allclose(nM_to_pkd(pkd_to_nM(pkd)), pkd, rtol=1e-12)

    def test_direction_is_inverted(self):
        """Higher pKd must map to lower nM -- the sort-direction trap."""
        pkd = np.array([5.0, 6.0, 7.0, 8.0])
        nm = pkd_to_nM(pkd)
        assert np.all(np.diff(nm) < 0)


class TestAggregation:
    def setup_method(self):
        # 3 models x 4 molecules
        self.m = np.array(
            [
                [5.0, 6.0, 7.0, 8.0],
                [6.0, 6.0, 6.0, 6.0],
                [7.0, 9.0, 5.0, 4.0],
            ]
        )

    def test_mean(self):
        np.testing.assert_allclose(aggregate_pkd(self.m, "mean"), self.m.mean(axis=0))

    def test_max_effect(self):
        np.testing.assert_allclose(aggregate_pkd(self.m, "max_effect"), self.m.max(axis=0))

    def test_agg_mean_max_is_midpoint_of_max_and_mean(self):
        """Mirrors oneliner's (min+mean)/2 in nM space, inverted for pKd space."""
        expected = (self.m.max(axis=0) + self.m.mean(axis=0)) / 2.0
        np.testing.assert_allclose(aggregate_pkd(self.m, "agg_mean_max"), expected)
        # Worked through by hand, column by column:
        #   col0 [5,6,7] -> max 7, mean 6 -> 6.5
        #   col1 [6,6,9] -> max 9, mean 7 -> 8.0
        #   col2 [7,6,5] -> max 7, mean 6 -> 6.5
        #   col3 [8,6,4] -> max 8, mean 6 -> 7.0
        np.testing.assert_allclose(expected, [6.5, 8.0, 6.5, 7.0])

    def test_single_model_is_identity(self):
        single = np.array([[5.0, 6.0, 7.0]])
        for how in ("mean", "max_effect", "agg_mean_max"):
            np.testing.assert_allclose(aggregate_pkd(single, how), single[0])

    def test_nan_from_one_model_is_ignored(self):
        m = np.array([[5.0, np.nan], [7.0, 8.0]])
        np.testing.assert_allclose(aggregate_pkd(m, "mean"), [6.0, 8.0])

    def test_unknown_aggregation_raises(self):
        with pytest.raises(ValueError, match="unknown aggregation"):
            aggregate_pkd(self.m, "median")


class TestPairwiseAuc:
    def test_perfect_separation(self):
        assert pairwise_auc([9, 8, 7], [3, 2, 1]) == 1.0

    def test_perfect_inversion(self):
        assert pairwise_auc([1, 2, 3], [7, 8, 9]) == 0.0

    def test_all_ties_is_one_half(self):
        assert pairwise_auc([5, 5], [5, 5]) == 0.5

    def test_half_credit_for_ties(self):
        # one active at 5 vs decoys {4, 5, 6}: 1 win, 1 tie, 1 loss -> 1.5/3
        assert pairwise_auc([5], [4, 5, 6]) == pytest.approx(0.5)

    def test_empty_group_is_nan(self):
        assert np.isnan(pairwise_auc([], [1, 2]))
        assert np.isnan(pairwise_auc([1, 2], []))

    def test_ignores_nan_values(self):
        assert pairwise_auc([9, np.nan], [1, 2]) == 1.0

    def test_matches_sklearn(self):
        """Cross-check against an independent implementation when available."""
        sk = pytest.importorskip("sklearn.metrics")
        rng = np.random.default_rng(0)
        actives = rng.normal(7.0, 1.0, 25)
        decoys = rng.normal(5.5, 1.0, 40)
        y_true = np.r_[np.ones_like(actives), np.zeros_like(decoys)]
        y_score = np.r_[actives, decoys]
        assert pairwise_auc(actives, decoys) == pytest.approx(
            sk.roc_auc_score(y_true, y_score)
        )


class TestInterpretAuc:
    @pytest.mark.parametrize(
        "auc, fragment",
        [
            (0.95, "signal"),
            (0.80, "signal"),
            (0.70, "caution"),
            (0.60, "caution"),
            (0.50, "noise"),
            (0.10, "noise"),
            (float("nan"), "not computable"),
        ],
    )
    def test_verdicts(self, auc, fragment):
        assert fragment in interpret_auc(auc)


class TestInterpretBenchmark:
    """The reported verdict must respect significance, not just AUC magnitude."""

    def test_perfect_auc_but_small_n_is_still_flagged(self):
        v = interpret_benchmark(1.000, 0.0048, 4, 6)
        assert "significant separation" in v
        assert "MW baseline" in v

    def test_high_auc_without_significance_is_not_called_signal(self):
        v = interpret_benchmark(0.767, 0.0887, 5, 6)
        assert "NOT significant" in v
        assert "signal" not in v.lower()

    def test_non_significant_mentions_sample_size(self):
        assert "4x6" in interpret_benchmark(0.708, 0.176, 4, 6)

    def test_missing_p_degrades_gracefully(self):
        v = interpret_benchmark(0.9, float("nan"), 4, 6)
        assert "significance not computed" in v

    def test_nan_auc(self):
        assert "not computable" in interpret_benchmark(float("nan"), 0.1, 0, 6)
