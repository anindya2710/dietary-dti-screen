"""
End-to-end pipeline test.

Pretrained checkpoints are replaced with **randomly initialized models of the same
architectures**, so the full code path runs -- featurization, prediction, aggregation,
sorting, CSV writing, benchmarking -- without a ~500 MB download in CI. Predicted values
are meaningless under random weights; every assertion here is about pipeline behaviour
(shape, ordering, merge integrity, arithmetic), never about predicted affinity.

Requires torch + DeepPurpose, so the whole module is marked ``needs_deeppurpose``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.needs_deeppurpose

pytest.importorskip("torch")
pytest.importorskip("DeepPurpose")

ENCODINGS = {
    "cnn_cnn_bindingdb": ("CNN", "CNN"),
    "morgan_cnn_bindingdb": ("Morgan", "CNN"),
    "morgan_aac_bindingdb": ("Morgan", "AAC"),
    "daylight_aac_bindingdb": ("Daylight", "AAC"),
}

LIBRARY = [
    ("caffeine", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"),
    ("quercetin", "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12"),
    ("curcumin", "COc1cc(/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2)ccc1O"),
    ("resveratrol", "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1"),
    ("piperine", "O=C(/C=C/C=C/c1ccc2c(c1)OCO2)N1CCCCC1"),
    ("tyramine", "NCCc1ccc(O)cc1"),
    ("oleic acid", r"CCCCCCCC/C=C\CCCCCCCC(=O)O"),
    ("dup_glucose_a", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"),
    ("dup_glucose_b", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"),  # duplicate
    ("BAD_unparseable", "not_a_smiles_at_all"),                    # must be rejected
    ("BAD_ring", "C1CC"),                                          # must be rejected
    ("BAD_blank", ""),                                             # must be rejected
    (
        "rutin_long",  # 122-char canonical SMILES -> flagged, not dropped
        "C[C@@H]1O[C@@H](OC[C@H]2O[C@@H](Oc3c(-c4ccc(O)c(O)c4)oc4cc(O)cc(O)c4c3=O)"
        "[C@H](O)[C@@H](O)[C@@H]2O)[C@H](O)[C@H](O)[C@H]1O",
    ),
]

#: Deliberately unavailable, to prove the ensemble degrades instead of dying.
BROKEN_MODEL = "deliberately_missing_model"


@pytest.fixture
def fake_ensemble(monkeypatch):
    """Patch model loading to build random-weight models of the real architectures."""
    from DeepPurpose import DTI as dp_models
    from DeepPurpose import utils as dp_utils

    from dti_screen import pipeline

    def _load_ensemble(model_names, work_dir="dp_work", quiet=True, verbose=True):
        loaded, failures = {}, {}
        for name in model_names:
            enc = ENCODINGS.get(name)
            if enc is None:
                failures[name] = "RuntimeError: simulated checkpoint download failure"
                continue
            drug, target = enc
            cfg = dp_utils.generate_config(
                drug_encoding=drug,
                target_encoding=target,
                cls_hidden_dims=[64, 32],
                train_epoch=1,
                batch_size=256,
            )
            loaded[name] = dp_models.model_initialize(**cfg)
        return loaded, failures

    # pipeline.py does `from .models import load_ensemble`, so patch the bound name.
    monkeypatch.setattr(pipeline, "load_ensemble", _load_ensemble)
    return _load_ensemble


@pytest.fixture
def run_output(tmp_path, patched_uniprot, fake_ensemble):
    """Run the whole pipeline once and hand back the outputs."""
    from dti_screen.config import ScreenConfig
    from dti_screen.pipeline import run_screen

    csv = tmp_path / "ingredients.csv"
    pd.DataFrame(LIBRARY, columns=["name", "SMILES"]).to_csv(csv, index=False)

    cfg = ScreenConfig(
        input_csv=str(csv),
        smiles_col="SMILES",
        name_col="name",
        output_dir=str(tmp_path / "predictions"),
        work_dir=str(tmp_path / "dp_work"),
        cache_dir=str(tmp_path / "cache"),
        ensemble=[*ENCODINGS, BROKEN_MODEL],
        include_controls=True,
    )
    return run_screen(cfg, verbose=False), cfg


class TestOutputs:
    def test_one_csv_per_target_plus_combined(self, run_output):
        out, cfg = run_output
        import os

        for label in ("MAOA", "ADORA2A", "CB1"):
            assert os.path.exists(os.path.join(cfg.output_dir, f"{label}_predictions.csv"))
        assert os.path.exists(os.path.join(cfg.output_dir, "all_targets_predictions.csv"))
        assert os.path.exists(os.path.join(cfg.output_dir, "control_benchmark.csv"))
        assert os.path.exists(os.path.join(cfg.output_dir, "rejected_molecules.csv"))

    def test_all_three_targets_screened(self, run_output):
        out, _ = run_output
        assert set(out["results"]) == {"MAOA", "ADORA2A", "CB1"}

    def test_broken_model_degraded_the_ensemble_without_failing(self, run_output):
        out, _ = run_output
        assert BROKEN_MODEL in out["failures"]["models"]
        for res in out["results"].values():
            assert res["n_models"].iloc[0] == 4

    def test_invalid_molecules_never_reach_the_results(self, run_output):
        out, _ = run_output
        for res in out["results"].values():
            assert not res["compound_name"].astype(str).str.startswith("BAD_").any()

    def test_rejected_file_captures_all_three_bad_rows(self, run_output):
        out, cfg = run_output
        import os

        rej = pd.read_csv(os.path.join(cfg.output_dir, "rejected_molecules.csv"))
        assert len(rej) == 3

    def test_duplicate_structure_collapsed(self, run_output):
        out, _ = run_output
        for res in out["results"].values():
            assert (res["compound_name"] == "dup_glucose_b").sum() == 0
            assert (res["compound_name"] == "dup_glucose_a").sum() == 1

    def test_long_smiles_flagged_but_retained(self, run_output):
        out, _ = run_output
        for res in out["results"].values():
            row = res.loc[res["compound_name"] == "rutin_long"]
            assert len(row) == 1
            assert bool(row["cnn_smiles_truncated"].iloc[0])

    def test_controls_were_appended(self, run_output):
        out, _ = run_output
        from dti_screen.controls import CONTROL_COMPOUNDS

        for res in out["results"].values():
            # caffeine is in both the library and the control set -> tagged, not duplicated
            assert (res["compound_name"] == "caffeine").sum() == 1
            assert int(res["is_control"].sum()) == len(CONTROL_COMPOUNDS)


class TestOrderingAndArithmetic:
    def test_sorted_by_ascending_nM(self, run_output):
        out, _ = run_output
        for res in out["results"].values():
            assert res["predicted_Kd_nM"].is_monotonic_increasing

    def test_ascending_nM_equals_descending_pkd(self, run_output):
        out, _ = run_output
        for res in out["results"].values():
            assert res["pKd_aggregate"].is_monotonic_decreasing

    def test_rank_is_contiguous_from_one(self, run_output):
        out, _ = run_output
        for res in out["results"].values():
            assert res["rank"].tolist() == list(range(1, len(res) + 1))

    def test_nM_column_matches_the_conversion_formula(self, run_output):
        out, _ = run_output
        for res in out["results"].values():
            expected = np.power(10.0, -res["pKd_aggregate"].to_numpy()) / 1e-9
            np.testing.assert_allclose(expected, res["predicted_Kd_nM"].to_numpy(), rtol=1e-9)

    def test_aggregate_matches_max_plus_mean_over_two(self, run_output):
        out, _ = run_output
        for res in out["results"].values():
            cols = [f"pKd_{m}" for m in ENCODINGS]
            mat = res[cols].to_numpy(float)
            expected = (mat.max(axis=1) + mat.mean(axis=1)) / 2.0
            np.testing.assert_allclose(expected, res["pKd_aggregate"].to_numpy(), rtol=1e-9)

    def test_smiles_present_on_every_row(self, run_output):
        out, _ = run_output
        for res in out["results"].values():
            assert res["canonical_smiles"].notna().all()
            assert (res["canonical_smiles"].str.len() > 0).all()

    def test_predictions_are_finite_and_positive(self, run_output):
        out, _ = run_output
        for res in out["results"].values():
            assert res["predicted_Kd_nM"].notna().all()
            assert (res["predicted_Kd_nM"] > 0).all()


class TestBenchmark:
    def test_benchmark_covers_every_target(self, run_output):
        out, _ = run_output
        assert set(out["benchmark"]["target"]) == {"MAOA", "ADORA2A", "CB1"}

    def test_benchmark_counts_actives_and_decoys(self, run_output):
        out, _ = run_output
        for _, row in out["benchmark"].iterrows():
            assert row["n_actives"] >= 4
            assert row["n_decoys"] >= 5

    def test_auc_in_unit_interval(self, run_output):
        out, _ = run_output
        aucs = out["benchmark"]["auc"].to_numpy(float)
        assert np.all((aucs >= 0.0) & (aucs <= 1.0))

    def test_every_target_gets_a_verdict(self, run_output):
        out, _ = run_output
        assert out["benchmark"]["verdict"].notna().all()


class TestConfigValidation:
    def test_mpnn_model_is_rejected_with_an_explanation(self):
        from dti_screen.config import ScreenConfig

        cfg = ScreenConfig(ensemble=["mpnn_cnn_bindingdb"])
        with pytest.raises(ValueError, match="dgl"):
            cfg.validate()

    def test_unknown_aggregation_is_rejected(self):
        from dti_screen.config import ScreenConfig

        with pytest.raises(ValueError, match="aggregation must be"):
            ScreenConfig(aggregation="median").validate()

    def test_missing_input_file_raises_clearly(self, tmp_path, patched_uniprot):
        from dti_screen.config import ScreenConfig
        from dti_screen.pipeline import run_screen

        cfg = ScreenConfig(
            input_csv=str(tmp_path / "nope.csv"),
            output_dir=str(tmp_path / "out"),
            work_dir=str(tmp_path / "work"),
        )
        with pytest.raises(FileNotFoundError, match="not found"):
            run_screen(cfg, verbose=False)
