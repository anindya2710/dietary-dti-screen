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

from .benchmark import pairwise_auc
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
        rows.append(
            {
                "target": target,
                "model_auc": round(model_auc, 4),
                "mw_only_auc": round(mw_auc, 4),
                "delta": round(delta, 4),
                "n_actives": len(act),
                "n_decoys": len(dec),
                "median_mw_active": round(float(act["MW"].median()), 1),
                "median_mw_decoy": round(float(dec["MW"].median()), 1),
                "verdict": (
                    "AUC explained by molecular weight; no evidence of target recognition"
                    if delta <= 0.05
                    else "model beats the size baseline"
                ),
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
) -> dict[str, pd.DataFrame]:
    """Run every diagnostic, print a report, and write the tables as CSVs."""
    output_dir = output_dir or predictions_dir
    os.makedirs(output_dir, exist_ok=True)
    df = load_predictions(os.path.join(predictions_dir, "all_targets_predictions.csv"))
    roles = control_roles()

    xt_all = cross_target_correlation(df, controls_only=False)
    xt_ctrl = cross_target_correlation(df, controls_only=True)
    size = size_correlation(df)
    base = mw_baseline_comparison(df, roles)
    ranks = own_active_ranks(df, roles)

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
            "  active/decoy split. delta <= 0.05 means that target's AUC is the size\n"
            "  confound, not target recognition. This is a limitation of the decoy set:\n"
            "  dietary decoys are smaller than drug-like actives. Property-matched\n"
            "  decoys (matched MW/logP) would remove it."
        )

        print("\n" + "=" * 76)
        print("DIAGNOSTIC 4 - Where do each target's own actives rank?")
        print("=" * 76)
        print(ranks.to_string(index=False))

    out = {
        "cross_target_all": xt_all,
        "cross_target_controls": xt_ctrl,
        "size_correlation": size,
        "mw_baseline": base,
        "own_active_ranks": ranks,
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
