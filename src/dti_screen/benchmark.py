"""
Control benchmark: does the ensemble separate known actives from dietary decoys?

This is the module that makes the screen interpretable. For each target it compares the
predicted affinity of that target's known actives against the dietary decoys, and
reports the probability that a randomly chosen active outranks a randomly chosen decoy.

That statistic is the area under the ROC curve. It is computed here by direct pairwise
comparison with a half-credit tie correction (the Mann-Whitney U formulation), which
needs no extra dependency and is exact for these sample sizes.

How to read the result
----------------------
==========  ==================================================================
AUC         Interpretation
==========  ==================================================================
>= 0.8      Actives clearly separated. The ranking carries signal.
0.6 - 0.8   Weak separation. Treat the ranking as a soft prior at best.
< 0.6       No usable separation. This target's scores are noise.
==========  ==================================================================

An AUC near 0.5 is a real and reportable finding, not a bug. Sequence-based DTI models
applied to dietary chemistry are extrapolating, and documenting where that fails is
stronger evidence of competence than publishing an unvalidated hit list.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pairwise_auc(actives: np.ndarray, decoys: np.ndarray) -> float:
    """
    P(a randomly chosen active scores above a randomly chosen decoy), ties at half
    credit. Returns ``nan`` if either group is empty.
    """
    a = np.asarray(actives, dtype=float)
    d = np.asarray(decoys, dtype=float)
    a = a[np.isfinite(a)]
    d = d[np.isfinite(d)]
    if a.size == 0 or d.size == 0:
        return float("nan")
    wins = float((a[:, None] > d[None, :]).sum())
    ties = float((a[:, None] == d[None, :]).sum())
    return (wins + 0.5 * ties) / (a.size * d.size)


def interpret_auc(auc: float) -> str:
    """One-line verdict for an AUC value."""
    if not np.isfinite(auc):
        return "not computable (missing actives or decoys)"
    if auc >= 0.8:
        return "actives clearly separated from decoys - ranking has signal"
    if auc >= 0.6:
        return "weak separation - treat the ranking with caution"
    return "NO useful separation - treat this target's scores as noise"


def benchmark_target(
    results: pd.DataFrame,
    target_label: str,
    roles: dict[str, str],
    score_col: str = "pKd_aggregate",
    verbose: bool = True,
) -> dict:
    """
    Compare this target's known actives against the dietary decoys.

    ``roles`` maps compound name -> target label or ``"decoy"``
    (see :func:`dti_screen.controls.control_roles`).
    """
    sub = results[results["compound_name"].isin(roles)].copy()
    if sub.empty:
        if verbose:
            print(f"\n=== {target_label}: control benchmark - no controls present ===")
        return {"target": target_label, "auc": float("nan"), "n_actives": 0, "n_decoys": 0}

    mapped = sub["compound_name"].map(roles)
    sub["role"] = np.where(
        mapped == target_label,
        "ACTIVE",
        np.where(mapped == "decoy", "decoy", "active (other target)"),
    )

    actives = sub.loc[sub["role"] == "ACTIVE", score_col].to_numpy(float)
    decoys = sub.loc[sub["role"] == "decoy", score_col].to_numpy(float)
    auc = pairwise_auc(actives, decoys)

    if verbose:
        print(f"\n=== {target_label}: control benchmark ===")
        cols = [c for c in ("rank", "compound_name", "role", "predicted_Kd_nM", score_col)
                if c in sub.columns]
        print(
            sub.sort_values("rank")[cols].to_string(
                index=False, float_format=lambda v: f"{v:,.3f}"
            )
        )
        if actives.size:
            print(f"\n  known actives : n={actives.size}  median pKd={np.nanmedian(actives):.2f}")
        if decoys.size:
            print(f"  dietary decoys: n={decoys.size}  median pKd={np.nanmedian(decoys):.2f}")
        if np.isfinite(auc):
            print(f"  AUC(active > decoy) = {auc:.3f}  ->  {interpret_auc(auc)}")

    return {
        "target": target_label,
        "auc": auc,
        "n_actives": int(actives.size),
        "n_decoys": int(decoys.size),
        "median_pkd_active": float(np.nanmedian(actives)) if actives.size else float("nan"),
        "median_pkd_decoy": float(np.nanmedian(decoys)) if decoys.size else float("nan"),
        "verdict": interpret_auc(auc),
    }


def benchmark_summary(rows: list[dict], verbose: bool = True) -> pd.DataFrame:
    """Collect per-target benchmark dicts into a summary table."""
    df = pd.DataFrame(rows)
    if verbose and not df.empty:
        print("\n" + "=" * 72)
        print("CONTROL BENCHMARK SUMMARY")
        print("=" * 72)
        for _, r in df.iterrows():
            auc = r["auc"]
            shown = f"{auc:.3f}" if np.isfinite(auc) else "n/a"
            print(f"  {r['target']:10s} AUC = {shown:>5s}   {r['verdict']}")
        print(
            "\nTargets near 0.5 produced no usable ranking. Report that result; a\n"
            "documented negative control is a stronger portfolio artifact than a\n"
            "ranked list nobody validated."
        )
    return df
