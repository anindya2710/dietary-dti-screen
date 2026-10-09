"""
End-to-end orchestration.

One entry point, :func:`run_screen`, shared by the CLI and the Colab notebook so that
both run identical code. Each stage isolates its own failures: a dead accession, an
unavailable checkpoint or one model erroring on one target all degrade the run rather
than ending it.
"""

from __future__ import annotations

import os

import pandas as pd

from .benchmark import benchmark_summary, benchmark_target
from .chem import validate_library
from .config import ScreenConfig
from .controls import control_roles, verify_control_structures
from .models import load_ensemble
from .screen import screen_target

#: Column order for the per-target CSVs.
BASE_COLUMNS = [
    "rank",
    "compound_name",
    "canonical_smiles",
    "formula",
    "MW",
    "predicted_Kd_nM",
    "pKd_aggregate",
    "pKd_mean",
    "pKd_std",
]
TAIL_COLUMNS = [
    "n_models",
    "aggregation",
    "aggregation_space",
    "cnn_smiles_truncated",
    "is_control",
    "target",
]


def _first_line(message: str, limit: int = 160) -> str:
    """Condense a multi-line exception string for a one-line error summary."""
    line = str(message).strip().splitlines()[0] if str(message).strip() else ""
    return line if len(line) <= limit else line[: limit - 3] + "..."


def build_screening_set(cfg: ScreenConfig, verbose: bool = True):
    """
    Load and validate the library, optionally appending the verified control set.

    Returns ``(screening_set, rejected, controls_df)``. Controls already present in the
    user's library are tagged in place rather than duplicated, compared on canonical
    structure so a differently written SMILES for the same molecule still matches.
    """
    controls_df = verify_control_structures()
    if verbose:
        print(f"Verified {len(controls_df)} control structures against expected formulas.")

    if not os.path.exists(cfg.input_csv):
        raise FileNotFoundError(
            f"Input library not found: {cfg.input_csv}\n"
            f"Provide a CSV with a {cfg.smiles_col!r} column, or point --input elsewhere."
        )

    raw = pd.read_csv(cfg.input_csv)
    if verbose:
        print(f"Loaded {cfg.input_csv}: {len(raw):,} rows, columns = {list(raw.columns)}\n")

    library, rejected, _ = validate_library(
        raw,
        smiles_col=cfg.smiles_col,
        name_col=cfg.name_col,
        dedupe=cfg.dedupe,
        label="your library",
        verbose=verbose,
    )
    library["is_control"] = False

    if cfg.include_controls:
        ctrl, _, _ = validate_library(
            controls_df[["compound_name", "SMILES"]].copy(),
            smiles_col="SMILES",
            name_col="compound_name",
            dedupe=False,
            label="control set",
            verbose=verbose,
        )
        ctrl["is_control"] = True

        overlap = set(library["canonical_smiles"]) & set(ctrl["canonical_smiles"])
        if overlap:
            if verbose:
                print(
                    f"\n  {len(overlap)} control compound(s) already in your library; "
                    f"tagging your rows instead of duplicating them."
                )
            library.loc[library["canonical_smiles"].isin(overlap), "is_control"] = True
            ctrl = ctrl[~ctrl["canonical_smiles"].isin(overlap)]

        library = pd.concat([library, ctrl], ignore_index=True)

    if verbose:
        n_ctrl = int(library["is_control"].sum())
        print(
            f"\nScreening set: {len(library):,} molecules "
            f"({n_ctrl} control, {len(library) - n_ctrl} library)"
        )
    return library, rejected, controls_df


def run_screen(cfg: ScreenConfig, verbose: bool = True) -> dict:
    """
    Run the full pipeline and write one CSV per target.

    Returns a dict with ``results``, ``benchmark``, ``targets``, ``written`` and a
    ``failures`` breakdown by stage.
    """
    from .targets import fetch_targets  # lazy: avoids importing requests for --help

    cfg.validate()
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.work_dir, exist_ok=True)

    failures: dict[str, object] = {}

    # --- 1. library -------------------------------------------------------------------
    library, rejected, _ = build_screening_set(cfg, verbose=verbose)
    if len(rejected):
        path = os.path.join(cfg.output_dir, "rejected_molecules.csv")
        rejected.to_csv(path, index=False)
        if verbose:
            print(f"Rejected rows -> {path} ({len(rejected)} rows). Inspect these.")

    # --- 2. targets -------------------------------------------------------------------
    if verbose:
        print("\nFetching target sequences from UniProt")
        print("-" * 46)
    target_records, target_failures = fetch_targets(
        cfg.targets, cache_dir=cfg.cache_dir, verbose=verbose
    )
    if target_failures:
        failures["targets"] = target_failures
    if not target_records:
        raise RuntimeError(
            f"None of the {len(cfg.targets)} target sequences could be retrieved, so no "
            f"screen is possible.\n"
            f"Sequences come from rest.uniprot.org - check that it is reachable, then "
            f"retry (successful fetches are cached in {cfg.cache_dir}).\n"
            f"First failure ({next(iter(target_failures))}): "
            f"{_first_line(next(iter(target_failures.values())))}\n"
            f"Per-target failures are listed above."
        )

    # --- 3. ensemble ------------------------------------------------------------------
    if verbose:
        print("\nLoading pretrained ensemble")
        print("-" * 46)
    ensemble, model_failures = load_ensemble(
        cfg.ensemble, work_dir=cfg.work_dir, quiet=cfg.quiet_deeppurpose, verbose=verbose
    )
    if model_failures:
        failures["models"] = model_failures
    if not ensemble:
        raise RuntimeError(
            f"None of the {len(cfg.ensemble)} pretrained checkpoints could be loaded, so "
            f"no predictions are possible.\n"
            f"Checkpoints come from dataverse.harvard.edu - check that it is reachable, "
            f"then retry.\n"
            f"First failure ({next(iter(model_failures))}): "
            f"{_first_line(next(iter(model_failures.values())))}\n"
            f"Per-model failures are listed above."
        )
    if verbose and model_failures:
        print(f"Degraded ensemble - missing: {list(model_failures)}")

    # --- 4. screen --------------------------------------------------------------------
    results: dict[str, pd.DataFrame] = {}
    screen_failures: dict[str, object] = {}
    for label, rec in target_records.items():
        if verbose:
            print(
                f"\n=== {label} ({rec.uniprot_id}, {rec.length} aa) x "
                f"{len(library):,} molecules x {len(ensemble)} models ==="
            )
        try:
            res, fails = screen_target(
                label,
                rec.sequence,
                library,
                ensemble,
                aggregation=cfg.aggregation,
                aggregation_space=cfg.aggregation_space,
                work_dir=cfg.work_dir,
                quiet=cfg.quiet_deeppurpose,
                verbose=verbose,
            )
            results[label] = res
            if fails:
                screen_failures[label] = fails
        except Exception as exc:  # noqa: BLE001
            screen_failures[label] = repr(exc)
            if verbose:
                print(f"  [FAIL] {label}: {type(exc).__name__}: {exc}")
    if screen_failures:
        failures["screen"] = screen_failures
    if not results:
        raise RuntimeError(f"No target could be screened. {screen_failures}")

    # --- 5. write ---------------------------------------------------------------------
    written = []
    for label, res in results.items():
        cols = [c for c in BASE_COLUMNS if c in res.columns]
        cols += [c for c in res.columns if c.startswith("pKd_") and c not in cols]
        cols += [c for c in TAIL_COLUMNS if c in res.columns]
        path = os.path.join(cfg.output_dir, f"{label}_predictions.csv")
        res[cols].to_csv(path, index=False)
        written.append(path)
        if verbose:
            print(f"\n{path}  ({len(res):,} rows)")

    combined = pd.concat(results.values(), ignore_index=True)
    combined_path = os.path.join(cfg.output_dir, "all_targets_predictions.csv")
    combined.to_csv(combined_path, index=False)
    written.append(combined_path)
    if verbose:
        print(f"{combined_path}  ({len(combined):,} rows)")

    # --- 6. benchmark -----------------------------------------------------------------
    bench = pd.DataFrame()
    if cfg.include_controls:
        roles = control_roles()
        rows = [benchmark_target(res, label, roles, verbose=verbose)
                for label, res in results.items()]
        bench = benchmark_summary(rows, verbose=verbose)
        if not bench.empty:
            bench_path = os.path.join(cfg.output_dir, "control_benchmark.csv")
            bench.to_csv(bench_path, index=False)
            written.append(bench_path)
            if verbose:
                print(f"\n{bench_path}")

    return {
        "results": results,
        "benchmark": bench,
        "targets": target_records,
        "library": library,
        "written": written,
        "failures": failures,
    }
