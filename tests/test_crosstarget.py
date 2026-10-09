"""
Diagnostics tests.

Each diagnostic is checked against a synthetic prediction table with a *known*
structure, so a regression shows up as a wrong verdict rather than a wrong-looking
number. Two fixtures matter:

* ``ligand_only_df`` -- scores depend solely on the molecule. Cross-target rho must be
  1.0 and the MW baseline must explain everything.
* ``target_specific_df`` -- each target has its own ordering. Cross-target rho must drop
  and the model must beat the size baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dti_screen.controls import verify_control_structures
from dti_screen.crosstarget import (
    cross_target_correlation,
    load_predictions,
    mw_baseline_comparison,
    own_active_ranks,
    run_analysis,
    size_correlation,
)

TARGETS = ["MAOA", "ADORA2A", "CB1"]


def _frame(score_fn):
    """Build a predictions-shaped frame from score_fn(name, target, mw)."""
    ctrl = verify_control_structures()
    rows = []
    for target in TARGETS:
        recs = []
        for _, c in ctrl.iterrows():
            recs.append(
                {
                    "compound_name": c["compound_name"],
                    "canonical_smiles": c["SMILES"],
                    "formula": c["formula"],
                    "MW": c["MW"],
                    "pKd_aggregate": score_fn(c["compound_name"], target, c["MW"]),
                    "is_control": True,
                    "target": target,
                }
            )
        g = pd.DataFrame(recs).sort_values("pKd_aggregate", ascending=False)
        g["rank"] = np.arange(1, len(g) + 1)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def ligand_only_df():
    """Score is a function of the molecule alone -- the pathological case."""
    return _frame(lambda name, target, mw: mw / 100.0)


@pytest.fixture
def target_specific_df():
    """Each target orders molecules differently, independent of MW."""
    roles = dict(zip(*(lambda d: (d["compound_name"], d["control_for"]))(
        verify_control_structures())))

    def score(name, target, mw):
        # that target's own actives score high; everything else low
        return 9.0 if roles[name] == target else 4.0

    return _frame(score)


class TestCrossTargetCorrelation:
    def test_ligand_only_gives_perfect_correlation(self, ligand_only_df):
        out = cross_target_correlation(ligand_only_df)
        assert len(out) == 3  # 3 target pairs
        assert np.allclose(out["spearman_rho"], 1.0)

    def test_target_specific_gives_low_correlation(self, target_specific_df):
        out = cross_target_correlation(target_specific_df)
        assert (out["spearman_rho"] < 0.5).all()

    def test_controls_only_scope_is_labelled(self, ligand_only_df):
        out = cross_target_correlation(ligand_only_df, controls_only=True)
        assert set(out["scope"]) == {"controls_only"}

    def test_all_target_pairs_covered(self, ligand_only_df):
        out = cross_target_correlation(ligand_only_df)
        pairs = {(r.target_a, r.target_b) for r in out.itertuples()}
        assert len(pairs) == 3
        for a, b in pairs:
            assert a < b  # canonical ordering, no duplicated pairs

    def test_raises_with_a_single_target(self, ligand_only_df):
        one = ligand_only_df[ligand_only_df["target"] == "MAOA"]
        with pytest.raises(ValueError, match="at least 2|too few"):
            cross_target_correlation(one)


class TestSizeCorrelation:
    def test_detects_pure_size_dependence(self, ligand_only_df):
        out = size_correlation(ligand_only_df)
        assert np.allclose(out["spearman_pkd_vs_mw"], 1.0)

    def test_one_row_per_target(self, ligand_only_df):
        assert sorted(size_correlation(ligand_only_df)["target"]) == sorted(TARGETS)


class TestMwBaseline:
    def test_size_only_model_does_not_beat_mw(self, ligand_only_df):
        out = mw_baseline_comparison(ligand_only_df)
        assert (out["delta"].abs() < 1e-9).all()
        # verdict is now resolution-aware: a zero delta is within one swapped pair
        assert out["verdict"].str.contains("indistinguishable").all()
        assert (out["delta_in_pairs"].abs() < 1e-9).all()

    def test_target_specific_model_beats_mw(self, target_specific_df):
        out = mw_baseline_comparison(target_specific_df)
        assert (out["model_auc"] == 1.0).all()
        assert (out["delta"] > 0.05).any()
        assert out["verdict"].str.contains("exceeds the size baseline").any()
        # a real effect must be worth more than one swapped pair
        assert (out.loc[out["delta"] > 0.05, "delta_in_pairs"] > 1.0).all()

    def test_reports_the_confound_itself(self, ligand_only_df):
        out = mw_baseline_comparison(ligand_only_df)
        assert {"median_mw_active", "median_mw_decoy"} <= set(out.columns)
        assert (out["n_decoys"] == 6).all()

    def test_auc_bounds(self, ligand_only_df):
        out = mw_baseline_comparison(ligand_only_df)
        for col in ("model_auc", "mw_only_auc"):
            assert out[col].between(0.0, 1.0).all()


class TestOwnActiveRanks:
    def test_target_specific_actives_rank_top(self, target_specific_df):
        out = own_active_ranks(target_specific_df)
        assert (out["best_rank"] == 1).all()

    def test_reports_rank_spread(self, ligand_only_df):
        out = own_active_ranks(ligand_only_df)
        assert (out["worst_rank"] >= out["median_rank"]).all()
        assert (out["median_rank"] >= out["best_rank"]).all()
        assert (out["n_screened"] == 19).all()


class TestLoadAndRun:
    def test_missing_file_raises_actionable_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="dti-screen run"):
            load_predictions(str(tmp_path / "all_targets_predictions.csv"))

    def test_missing_columns_raises(self, tmp_path):
        p = tmp_path / "all_targets_predictions.csv"
        pd.DataFrame({"compound_name": ["x"], "target": ["MAOA"]}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="missing expected columns"):
            load_predictions(str(p))

    def test_end_to_end_writes_all_diagnostics(self, tmp_path, ligand_only_df):
        pred = tmp_path / "predictions"
        pred.mkdir()
        ligand_only_df.to_csv(pred / "all_targets_predictions.csv", index=False)

        out = run_analysis(str(pred), str(tmp_path / "diag"), verbose=False)
        assert set(out) == {
            "cross_target_all",
            "cross_target_controls",
            "size_correlation",
            "mw_baseline",
            "own_active_ranks",
            "aggregation_sensitivity",
        }
        for name in out:
            assert (tmp_path / "diag" / f"diagnostic_{name}.csv").exists()
