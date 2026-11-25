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
    SUMMARY_PKD_COLUMNS,
    aggregation_sensitivity_table,
    cross_target_correlation,
    load_predictions,
    model_pkd_columns,
    mw_baseline_comparison,
    own_active_ranks,
    run_analysis,
    size_correlation,
)
from dti_screen.screen import aggregate_pkd

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
            "size_correlation_controls",
            "mw_baseline",
            "own_active_ranks",
            "aggregation_sensitivity",
        }
        for name in out:
            assert (tmp_path / "diag" / f"diagnostic_{name}.csv").exists()


# ---------------------------------------------------------------------------------
# Regression tests for the third audit round (2026-11-25). Each one fails against the
# code as it stood before that round; see README, "A note on how these conclusions
# were checked".
# ---------------------------------------------------------------------------------


def _frame_with_model_columns(model_suffix: str = "bindingdb", aggregation: str = "agg_mean_max"):
    """
    A predictions-shaped frame carrying four per-model pKd columns.

    ``model_suffix`` stands in for the checkpoint family, so the same frame can be
    built as a BindingDB ensemble or as the Davis/KIBA ensemble that
    ``config/default.yaml`` also allows.
    """
    rng = np.random.default_rng(11)
    ctrl = verify_control_structures()
    rows = []
    for target in TARGETS:
        n = len(ctrl)
        per_model = rng.uniform(4.0, 9.0, size=(4, n))
        g = pd.DataFrame(
            {
                "compound_name": ctrl["compound_name"].to_numpy(),
                "canonical_smiles": ctrl["SMILES"].to_numpy(),
                "formula": ctrl["formula"].to_numpy(),
                "MW": ctrl["MW"].to_numpy(),
                "is_control": True,
                "target": target,
                "aggregation": aggregation,
            }
        )
        for i, encoder in enumerate(("cnn_cnn", "morgan_cnn", "morgan_aac", "daylight_aac")):
            g[f"pKd_{encoder}_{model_suffix}"] = per_model[i]
        g["pKd_mean"] = per_model.mean(axis=0)
        g["pKd_std"] = per_model.std(axis=0)
        g["pKd_aggregate"] = aggregate_pkd(per_model, aggregation, space="pkd")
        g["pKd_aggregate_alt"] = aggregate_pkd(per_model, aggregation, space="nm")
        g = g.sort_values("pKd_aggregate", ascending=False)
        g["rank"] = np.arange(1, len(g) + 1)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


class TestModelColumnSelection:
    """Per-model columns were selected by checkpoint name, which matched only BindingDB."""

    def test_summary_columns_are_excluded(self):
        df = _frame_with_model_columns()
        cols = model_pkd_columns(df)
        assert len(cols) == 4
        assert not SUMMARY_PKD_COLUMNS & set(cols)

    def test_non_bindingdb_checkpoints_are_found(self):
        df = _frame_with_model_columns(model_suffix="davis")
        assert len(model_pkd_columns(df)) == 4

    def test_diagnostic_5_survives_a_non_bindingdb_ensemble(self):
        """It used to return an empty table, so the diagnostic vanished without a word."""
        out = aggregation_sensitivity_table(_frame_with_model_columns(model_suffix="davis"))
        assert len(out) == len(TARGETS)
        assert out["n_models"].eq(4).all()


class TestAggregationSensitivityUsesTheRunsAggregation:
    """Diagnostic 5 assumed agg_mean_max whatever the screen actually ran with."""

    def test_max_effect_is_identical_in_both_spaces(self):
        # max_effect is nanmax in pKd and nanmin in nM, which is the same molecule
        # either way, so the two spaces must agree exactly.
        out = aggregation_sensitivity_table(_frame_with_model_columns(), "max_effect")
        assert np.allclose(out["spearman_pkd_vs_nm"], 1.0)
        assert out["identical_ranking"].all()

    def test_agg_mean_max_is_not_identical(self):
        out = aggregation_sensitivity_table(_frame_with_model_columns(), "agg_mean_max")
        assert (out["spearman_pkd_vs_nm"] < 1.0).all()

    def test_run_analysis_reads_the_aggregation_from_the_data(self, tmp_path):
        pred = tmp_path / "predictions"
        pred.mkdir()
        _frame_with_model_columns(aggregation="max_effect").to_csv(
            pred / "all_targets_predictions.csv", index=False
        )
        out = run_analysis(str(pred), str(tmp_path / "diag"), verbose=False)
        sens = out["aggregation_sensitivity"]
        assert np.allclose(sens["spearman_pkd_vs_nm"], 1.0), (
            "a max_effect run was reported under agg_mean_max"
        )
        assert sens["identical_ranking"].all()


class TestSizeCorrelationScope:
    """The README reported a controls-only rho that no code path produced."""

    def test_controls_only_is_labelled_and_counted(self, ligand_only_df):
        df = ligand_only_df.copy()
        # add one non-control molecule per target, so the two scopes differ in size
        extra = df[df["target"] == df["target"].iloc[0]].head(1).copy()
        rows = []
        for target in TARGETS:
            e = extra.copy()
            e["target"] = target
            e["compound_name"] = "library_molecule"
            e["canonical_smiles"] = "CCO"
            e["is_control"] = False
            rows.append(e)
        df = pd.concat([df, *rows], ignore_index=True)

        all_scope = size_correlation(df, controls_only=False)
        ctrl_scope = size_correlation(df, controls_only=True)

        assert set(all_scope["scope"]) == {"all_molecules"}
        assert set(ctrl_scope["scope"]) == {"controls_only"}
        assert (ctrl_scope["n_molecules"] < all_scope["n_molecules"]).all()
        assert ctrl_scope["n_molecules"].eq(int(df["is_control"].sum() / len(TARGETS))).all()

    def test_controls_only_without_the_column_returns_nothing(self, ligand_only_df):
        """is_control is optional, so a missing column must not relabel the library."""
        df = ligand_only_df.drop(columns=["is_control"])
        out = size_correlation(df, controls_only=True)
        assert out.empty
        assert list(out.columns) == ["target", "spearman_pkd_vs_mw", "n_molecules", "scope"]
