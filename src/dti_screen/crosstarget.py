"""
Post-screen diagnostics: is the model using the protein at all?

The control benchmark in :mod:`dti_screen.benchmark` answers "are actives separated
from decoys?". It does not answer *why*, and there are two cheap ways to pass it that
have nothing to do with target recognition:

1. **A target-independent ligand prior.** If the model ranks molecules the same way
   regardless of which protein it is shown, it is not doing drug-*target* prediction.
   Measured here as the rank correlation of the same molecules across different targets.
   Three unrelated proteins producing one ranking is a stronger negative result than a
   scrambled-sequence control, because the inputs are real.

2. **A size/drug-likeness confound.** Dietary decoys are systematically smaller than
   drug-like actives, so ranking on molecular weight alone can score a high AUC. The
   honest comparison is model AUC against an **MW-only baseline** on the identical
   active/decoy split. If the model does not beat sorting by molecular weight, its AUC
   is measuring the confound, not the biology.

Point 2 is a design limitation of this project's own benchmark, and it is reported here
rather than hidden. The standard remedy is property-matched decoys (DUD-E style: sample
decoys matched to the actives' MW and logP distributions).
"""

from __future__ import annotations

import os
from itertools import combinations

import numpy as np
import pandas as pd

from .benchmark import (
    auc_resolution,
    bootstrap_auc_ci,
    exact_permutation_p,
    pairwise_auc,
)
from .controls import control_roles


def load_predictions(path: str) -> pd.DataFrame:
    """Load the combined predictions CSV written by the screen."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run the screen first (`dti-screen run`), which writes "
            f"all_targets_predictions.csv into the output directory."
        )
    df = pd.read_csv(path)
    missing = {"compound_name", "target", "pKd_aggregate", "MW", "rank"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {sorted(missing)}")
    if df["target"].nunique() < 2:
        raise ValueError(
            f"{path} contains only {df['target'].nunique()} target(s); cross-target "
            f"analysis needs at least 2."
        )
    return df


def _spearman(x, y) -> float:
    """Spearman rho without a hard scipy dependency at import time."""
    try:
        from scipy.stats import spearmanr

        return float(spearmanr(x, y).statistic)
    except Exception:
        # Pearson on ranks is exactly Spearman (ties handled by average ranking).
        xr = pd.Series(x).rank().to_numpy()
        yr = pd.Series(y).rank().to_numpy()
        return float(np.corrcoef(xr, yr)[0, 1])


def cross_target_correlation(df: pd.DataFrame, controls_only: bool = False) -> pd.DataFrame:
    """
    Rank correlation of the same molecules between every pair of targets.

    High values mean the ranking barely depends on which protein was supplied -- the
    central diagnostic for ligand-only behaviour.
    """
    sub = df[df["is_control"]] if (controls_only and "is_control" in df.columns) else df
    wide = sub.pivot_table(index="compound_name", columns="target", values="pKd_aggregate")
    wide = wide.dropna()
    if wide.shape[1] < 2:
        raise ValueError(
            f"cross-target correlation needs at least 2 targets, got {wide.shape[1]}"
        )
    if len(wide) < 3:
        raise ValueError("too few molecules shared across targets to correlate")

    rows = []
    for a, b in combinations(sorted(wide.columns), 2):
        rows.append(
            {
                "target_a": a,
                "target_b": b,
                "spearman_rho": round(_spearman(wide[a], wide[b]), 4),
                "n_molecules": len(wide),
                "scope": "controls_only" if controls_only else "all_molecules",
            }
        )
    return pd.DataFrame(rows)


def size_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Rank correlation between predicted affinity and molecular weight, per target."""
    rows = []
    for target, g in df.groupby("target"):
        g = g.dropna(subset=["MW", "pKd_aggregate"])
        rows.append(
            {
                "target": target,
                "spearman_pkd_vs_mw": round(_spearman(g["MW"], g["pKd_aggregate"]), 4),
                "n_molecules": len(g),
            }
        )
    return pd.DataFrame(rows).sort_values("target").reset_index(drop=True)


def mw_baseline_comparison(df: pd.DataFrame, roles: dict[str, str] | None = None) -> pd.DataFrame:
    """
    Model AUC vs an MW-only baseline on the identical active/decoy split.

    ``delta`` is what the model contributes beyond molecular weight. A delta near zero
    (or negative) means that target's AUC is explained by the size confound.
    """
    roles = roles or control_roles()
    rows = []
    for target, g in df.groupby("target"):
        g = g[g["compound_name"].isin(roles)].copy()
        mapped = g["compound_name"].map(roles)
        act = g[mapped == target]
        dec = g[mapped == "decoy"]
        if act.empty or dec.empty:
            continue

        model_auc = pairwise_auc(act["pKd_aggregate"], dec["pKd_aggregate"])
        mw_auc = pairwise_auc(act["MW"], dec["MW"])
        delta = model_auc - mw_auc
        step = auc_resolution(len(act), len(dec))
        ci_lo, ci_hi = bootstrap_auc_ci(act["pKd_aggregate"], dec["pKd_aggregate"])
        p_exact = exact_permutation_p(act["pKd_aggregate"], dec["pKd_aggregate"])

        # The verdict must respect the measurement's own resolution. With 4 actives and
        # 6 decoys the AUC can only move in steps of 1/24 = 0.042, so a delta of 0.042
        # is a single swapped pair. Any threshold finer than `step` is reading noise.
        if abs(delta) <= step:
            verdict = (
                f"delta ({delta:+.3f}) is within one swapped pair ({step:.3f}); "
                f"indistinguishable from the molecular-weight baseline"
            )
        elif delta > step:
            verdict = f"model exceeds the size baseline by {delta / step:.1f} pairs"
        else:
            verdict = f"model is WORSE than the size baseline by {abs(delta) / step:.1f} pairs"

        rows.append(
            {
                "target": target,
                "model_auc": round(model_auc, 4),
                "model_auc_ci_lo": round(ci_lo, 4),
                "model_auc_ci_hi": round(ci_hi, 4),
                "model_auc_exact_p": round(p_exact, 4) if np.isfinite(p_exact) else np.nan,
                "mw_only_auc": round(mw_auc, 4),
                "delta": round(delta, 4),
                "auc_resolution": round(step, 4),
                "delta_in_pairs": round(delta / step, 2) if step else np.nan,
                "n_actives": len(act),
                "n_decoys": len(dec),
                "median_mw_active": round(float(act["MW"].median()), 1),
                "median_mw_decoy": round(float(dec["MW"].median()), 1),
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows).sort_values("target").reset_index(drop=True)


def aggregation_sensitivity_table(df: pd.DataFrame, how: str = "agg_mean_max") -> pd.DataFrame:
    """
    How much does the choice of aggregation space change the ranking, per target?

    Recomputes both DeepPurpose branches from the per-model ``pKd_<model>`` columns in
    the output CSV. A Spearman well below 1.0 means the reported ranking depends on
    ``convert_y`` -- a flag that reads like a units setting but silently swaps an
    arithmetic mean in nM for a geometric one.
    """
    from .screen import aggregation_sensitivity

    model_cols = [c for c in df.columns if c.startswith("pKd_") and c.endswith("bindingdb")]
    if not model_cols:
        return pd.DataFrame()

    rows = []
    for target, g in df.groupby("target"):
        matrix = g[model_cols].to_numpy(float).T  # (n_models, n_molecules)
        s = aggregation_sensitivity(matrix, how)
        rows.append(
            {
                "target": target,
                "spearman_pkd_vs_nm": round(s["spearman_rho"], 4),
                "identical_ranking": s["same_order"],
                "n_models": len(model_cols),
                "n_molecules": len(g),
            }
        )
    return pd.DataFrame(rows).sort_values("target").reset_index(drop=True)


def own_active_ranks(df: pd.DataFrame, roles: dict[str, str] | None = None) -> pd.DataFrame:
    """Where each target's own known actives land in that target's ranking."""
    roles = roles or control_roles()
    rows = []
    for target, g in df.groupby("target"):
        mapped = g["compound_name"].map(roles)
        act = g[mapped == target]
        if act.empty:
            continue
        rows.append(
            {
                "target": target,
                "n_actives": len(act),
                "n_screened": len(g),
                "best_rank": int(act["rank"].min()),
                "median_rank": int(act["rank"].median()),
                "worst_rank": int(act["rank"].max()),
                "ranks": ", ".join(str(int(r)) for r in sorted(act["rank"])),
            }
        )
    return pd.DataFrame(rows).sort_values("target").reset_index(drop=True)


def run_analysis(
    predictions_dir: str = "predictions",
    output_dir: str | None = None,
    verbose: bool = True,
    space: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run every diagnostic, print a report, and write the tables as CSVs.

    ``space`` re-derives ``pKd_aggregate`` from the per-model columns in the chosen
    aggregation space before running anything, so every conclusion in this report can be
    re-checked under DeepPurpose's other ``convert_y`` branch. Use it to establish
    whether a finding is robust to that flag rather than assuming it is.
    """
    output_dir = output_dir or predictions_dir
    os.makedirs(output_dir, exist_ok=True)
    df = load_predictions(os.path.join(predictions_dir, "all_targets_predictions.csv"))
    roles = control_roles()

    if space is not None:
        from .screen import aggregate_pkd

        model_cols = [c for c in df.columns if c.startswith("pKd_") and c.endswith("bindingdb")]
        if not model_cols:
            raise ValueError(
                "--space needs the per-model pKd_<model> columns; re-run the screen "
                "with a current version to produce them."
            )
        how = str(df["aggregation"].iloc[0]) if "aggregation" in df.columns else "agg_mean_max"
        parts = []
        for _, g in df.groupby("target", sort=False):
            g = g.copy()
            g["pKd_aggregate"] = aggregate_pkd(g[model_cols].to_numpy(float).T, how, space=space)
            g = g.sort_values("pKd_aggregate", ascending=False)
            g["rank"] = range(1, len(g) + 1)
            parts.append(g)
        df = pd.concat(parts, ignore_index=True)
        if verbose:
            print(f"[re-derived pKd_aggregate in {space!r} space before analysis]\n")

    xt_all = cross_target_correlation(df, controls_only=False)
    xt_ctrl = cross_target_correlation(df, controls_only=True)
    size = size_correlation(df)
    base = mw_baseline_comparison(df, roles)
    ranks = own_active_ranks(df, roles)
    aggsens = aggregation_sensitivity_table(df)

    if verbose:
        print("=" * 76)
        print("DIAGNOSTIC 1 - Is the ranking target-dependent?")
        print("=" * 76)
        print(pd.concat([xt_all, xt_ctrl]).to_string(index=False))
        peak = pd.concat([xt_all, xt_ctrl])["spearman_rho"].max()
        print(
            f"\n  Highest cross-target rho = {peak:.3f}.\n"
            f"  Near 1.0 means the same molecules are ranked the same way regardless of\n"
            f"  which protein was supplied -- i.e. ligand-driven, not target-driven.\n"
            f"  This subsumes a scrambled-sequence control: these proteins are real and\n"
            f"  unrelated, and the ranking still barely moves."
        )

        print("\n" + "=" * 76)
        print("DIAGNOSTIC 2 - Is the score a proxy for molecular size?")
        print("=" * 76)
        print(size.to_string(index=False))

        print("\n" + "=" * 76)
        print("DIAGNOSTIC 3 - Does the model beat sorting by molecular weight?")
        print("=" * 76)
        print(base.to_string(index=False))
        print(
            "\n  'delta' is the model's contribution beyond molecular weight on the same\n"
            "  active/decoy split, and 'auc_resolution' is the smallest change the metric\n"
            "  can express (one swapped pair). A delta inside that resolution is not a\n"
            "  measurement. The decoy set is confounded with size -- dietary decoys are\n"
            "  systematically smaller than drug-like actives -- which INVALIDATES the AUC\n"
            "  as evidence of target recognition. Property-matched decoys (DUD-E style,\n"
            "  matched on MW and logP) are required before these numbers mean anything."
        )

        print("\n" + "=" * 76)
        print("DIAGNOSTIC 4 - Where do each target's own actives rank?")
        print("=" * 76)
        print(ranks.to_string(index=False))

        if not aggsens.empty:
            print("\n" + "=" * 76)
            print("DIAGNOSTIC 5 - Does the ranking survive DeepPurpose's convert_y flag?")
            print("=" * 76)
            print(aggsens.to_string(index=False))
            print(
                "\n  DeepPurpose aggregates in nM when convert_y=True (oneliner's default,\n"
                "  an arithmetic mean dominated by the weakest-binding model) and in pKd\n"
                "  when convert_y=False (a geometric mean). These are different estimators,\n"
                "  not different units. Spearman near 1.0 means your conclusions are robust\n"
                "  to that choice; well below 1.0 means they are not."
            )

    out = {
        "cross_target_all": xt_all,
        "cross_target_controls": xt_ctrl,
        "size_correlation": size,
        "mw_baseline": base,
        "own_active_ranks": ranks,
        "aggregation_sensitivity": aggsens,
    }
    written = []
    for name, table in out.items():
        path = os.path.join(output_dir, f"diagnostic_{name}.csv")
        table.to_csv(path, index=False)
        written.append(path)
    if verbose:
        print("\nWrote:")
        for p in written:
            print(f"  {p}")
    return out
