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

from itertools import combinations

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


def auc_resolution(n_actives: int, n_decoys: int) -> float:
    """
    The smallest possible non-zero change in AUC: one swapped pair.

    With ``n_actives * n_decoys`` pairwise comparisons, AUC is quantised to multiples of
    ``1 / (n_actives * n_decoys)``. Reporting or thresholding differences finer than this
    is reporting noise. For 4 actives vs 6 decoys the step is 1/24 = 0.042, so a "delta"
    of 0.042 is literally one pair changing places.
    """
    if n_actives <= 0 or n_decoys <= 0:
        return float("nan")
    return 1.0 / (n_actives * n_decoys)


def exact_permutation_p(actives, decoys) -> float:
    """
    Exact one-sided p-value for AUC > 0.5 by complete enumeration.

    Pools the scores and enumerates every way of labelling ``n_actives`` of them as
    active, giving the exact null distribution rather than a sampled approximation.
    Returns P(AUC_null >= AUC_observed).

    Feasible precisely because the control set is small: 4 actives vs 6 decoys gives
    C(10,4) = 210 labellings, 5 vs 6 gives C(11,5) = 462. Falls back to ``nan`` above
    ~200k labellings, where enumeration stops being cheap.
    """
    a = np.asarray(actives, dtype=float)
    d = np.asarray(decoys, dtype=float)
    a = a[np.isfinite(a)]
    d = d[np.isfinite(d)]
    if a.size == 0 or d.size == 0:
        return float("nan")

    pooled = np.concatenate([a, d])
    n, k = pooled.size, a.size
    from math import comb

    if comb(n, k) > 200_000:
        return float("nan")

    observed = pairwise_auc(a, d)
    idx = set(range(n))
    hits = total = 0
    for combo in combinations(range(n), k):
        rest = list(idx - set(combo))
        if pairwise_auc(pooled[list(combo)], pooled[rest]) >= observed:
            hits += 1
        total += 1
    return hits / total


def bootstrap_auc_ci(
    actives,
    decoys,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Percentile bootstrap CI for AUC, resampling actives and decoys independently.

    At n=4-6 per group this is coarse -- a resample can draw the same molecule four
    times -- so treat the interval as an honest width indicator, not a precise bound.
    It is reported alongside :func:`exact_permutation_p`, which makes no distributional
    assumption, for exactly that reason.
    """
    a = np.asarray(actives, dtype=float)
    d = np.asarray(decoys, dtype=float)
    a = a[np.isfinite(a)]
    d = d[np.isfinite(d)]
    if a.size == 0 or d.size == 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        stats[i] = pairwise_auc(
            a[rng.integers(0, a.size, a.size)],
            d[rng.integers(0, d.size, d.size)],
        )
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def interpret_benchmark(auc: float, p_exact: float, n_actives: int, n_decoys: int) -> str:
    """
    Verdict that respects significance, not just the AUC's magnitude.

    :func:`interpret_auc` describes how large the separation looks. At 4-6 molecules per
    group that is not enough: an AUC of 1.000 on 24 pairs is still only p = 0.005, and an
    AUC of 0.767 on 30 pairs is not significant at all. This function is what the reported
    verdict uses, so a large-but-unsupported AUC is never described as signal.
    """
    if not np.isfinite(auc):
        return "not computable (missing actives or decoys)"
    n = f"{n_actives}x{n_decoys}"
    if not np.isfinite(p_exact):
        return f"{interpret_auc(auc)} (significance not computed)"
    if p_exact >= 0.05:
        return (
            f"NOT significant (exact p={p_exact:.3f}, {n}); separation is within what "
            f"chance produces at this sample size"
        )
    if auc >= 0.8:
        return f"significant separation (exact p={p_exact:.3f}, {n}) - check the MW baseline next"
    return f"significant but modest (exact p={p_exact:.3f}, {n}) - check the MW baseline next"


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
    ci_lo, ci_hi = bootstrap_auc_ci(actives, decoys)
    p_exact = exact_permutation_p(actives, decoys)
    step = auc_resolution(actives.size, decoys.size)

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
            print(
                f"  AUC(active > decoy) = {auc:.3f}  ->  "
                f"{interpret_benchmark(auc, p_exact, actives.size, decoys.size)}"
            )
            print(
                f"  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]   exact permutation p = {p_exact:.3f}"
            )
            print(
                f"  NOTE: {actives.size} x {decoys.size} = {actives.size * decoys.size} "
                f"pairs, so AUC moves in steps of {step:.3f}. Differences smaller than\n"
                f"        that are single swapped pairs, not measurable effects."
            )

    return {
        "target": target_label,
        "auc": auc,
        "auc_ci_lo": ci_lo,
        "auc_ci_hi": ci_hi,
        "exact_p": p_exact,
        "auc_resolution": step,
        "n_actives": int(actives.size),
        "n_decoys": int(decoys.size),
        "median_pkd_active": float(np.nanmedian(actives)) if actives.size else float("nan"),
        "median_pkd_decoy": float(np.nanmedian(decoys)) if decoys.size else float("nan"),
        "verdict": interpret_benchmark(auc, p_exact, actives.size, decoys.size),
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
            "\nA high AUC here is necessary but not sufficient. Two further checks decide\n"
            "whether it means anything, both in `dti-screen analyze`:\n"
            "  - is it significant at this sample size? (exact p, above)\n"
            "  - does it beat sorting by molecular weight? (diagnostic 3)\n"
            "A separation that fails either check is not evidence of target recognition.\n"
            "Report that result; a documented negative control is a stronger portfolio\n"
            "artifact than a ranked list nobody validated."
        )
    return df
