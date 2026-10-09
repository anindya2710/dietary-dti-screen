"""
Affinity prediction and ensemble aggregation.

Units, stated explicitly because DeepPurpose leaves them implicit
----------------------------------------------------------------
The ``*_bindingdb`` checkpoints are trained on log-transformed BindingDB Kd, so raw
model output is **pKd**, where *higher means stronger binding*.

``DTI.repurpose(convert_y=True)`` silently applies
``convert_y_unit(y, 'p', 'nM') == 10**(-y) / 1e-9``, converting to **nM**, where *lower
means stronger binding*. That flag is also what flips the direction of DeepPurpose's own
sort, which is why the same library can appear to rank in opposite directions depending
on a keyword argument.

This module calls ``convert_y=False`` to keep native pKd and converts explicitly, so
both columns are present and the sort direction is declared rather than inferred.
"""

from __future__ import annotations

import contextlib
import io
import os
import time
import warnings

import numpy as np
import pandas as pd

#: Aggregation strategies, mirroring DeepPurpose ``oneliner``'s ``agg`` parameter,
#: expressed in pKd space (higher = stronger).
AGGREGATIONS = ("agg_mean_max", "mean", "max_effect")


def pkd_to_nM(pkd) -> np.ndarray:
    """pKd -> nM. Identical to DeepPurpose ``convert_y_unit(y, 'p', 'nM')``."""
    return np.power(10.0, -np.asarray(pkd, dtype=float)) / 1e-9


def nM_to_pkd(nm) -> np.ndarray:
    """nM -> pKd, the inverse of :func:`pkd_to_nM`."""
    return -np.log10(np.asarray(nm, dtype=float) * 1e-9)


def aggregate_pkd(matrix, how: str = "agg_mean_max") -> np.ndarray:
    """
    Aggregate per-model pKd predictions. ``matrix`` is ``(n_models, n_molecules)``.

    ``agg_mean_max`` reproduces DeepPurpose ``oneliner``'s default: the midpoint of the
    most optimistic model and the ensemble mean. In nM space (``convert_y=True``)
    oneliner computes ``(min + mean) / 2``; the pKd-space equivalent is
    ``(max + mean) / 2``, since the two scales are inversely ordered.
    """
    m = np.asarray(matrix, dtype=float)
    if how == "mean":
        return np.nanmean(m, axis=0)
    if how == "max_effect":
        return np.nanmax(m, axis=0)
    if how == "agg_mean_max":
        return (np.nanmax(m, axis=0) + np.nanmean(m, axis=0)) / 2.0
    raise ValueError(f"unknown aggregation {how!r}; expected one of {AGGREGATIONS}")


def screen_target(
    target_label: str,
    target_sequence: str,
    library: pd.DataFrame,
    ensemble: dict,
    aggregation: str = "agg_mean_max",
    work_dir: str = "dp_work",
    quiet: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Score every molecule in ``library`` against one target with every ensemble member.

    Returns ``(results, failures)`` with results sorted strongest-predicted-affinity
    first (ascending nM, equivalently descending pKd). A failing model is recorded and
    skipped; the run only raises if every model fails for this target.
    """
    from DeepPurpose import DTI as dp_models

    smiles = library["canonical_smiles"].tolist()
    names = library["compound_name"].tolist()
    result_dir = os.path.join(work_dir, f"res_{target_label}")
    os.makedirs(result_dir, exist_ok=True)

    per_model: dict[str, np.ndarray] = {}
    failures: dict[str, str] = {}

    for model_name, model in ensemble.items():
        start = time.time()
        try:
            buf = io.StringIO()
            ctx = contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()
            with ctx:
                y = dp_models.repurpose(
                    X_repurpose=np.array(smiles),
                    target=target_sequence,
                    model=model,
                    drug_names=names,
                    target_name=target_label,
                    result_folder=result_dir,
                    convert_y=False,  # keep native pKd; we convert explicitly
                    verbose=False,
                )
            y = np.asarray(y, dtype=float).ravel()

            if y.shape[0] != len(smiles):
                raise ValueError(f"expected {len(smiles)} predictions, got {y.shape[0]}")
            if not np.isfinite(y).all():
                n_bad = int((~np.isfinite(y)).sum())
                if verbose:
                    print(f"       {model_name}: {n_bad} non-finite prediction(s) -> NaN")
                y = np.where(np.isfinite(y), y, np.nan)

            per_model[f"pKd_{model_name}"] = y
            if verbose:
                print(
                    f"  [ok]   {model_name:26s} {time.time() - start:6.1f}s  "
                    f"pKd {np.nanmin(y):.2f} to {np.nanmax(y):.2f}"
                )
        except Exception as exc:  # noqa: BLE001
            failures[model_name] = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  [FAIL] {model_name:26s} {type(exc).__name__}: {exc}")

    if not per_model:
        raise RuntimeError(f"{target_label}: every model failed - {failures}")

    carry = [
        c
        for c in (
            "compound_name",
            "canonical_smiles",
            "formula",
            "MW",
            "cnn_smiles_truncated",
            "is_control",
        )
        if c in library.columns
    ]
    out = library[carry].copy()
    for col, values in per_model.items():
        out[col] = values

    matrix = np.vstack(list(per_model.values()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # all-NaN column -> NaN, not a crash
        out["pKd_mean"] = np.nanmean(matrix, axis=0)
        out["pKd_std"] = np.nanstd(matrix, axis=0)
        out["pKd_aggregate"] = aggregate_pkd(matrix, aggregation)

    out["predicted_Kd_nM"] = pkd_to_nM(out["pKd_aggregate"])
    out["n_models"] = int(matrix.shape[0])
    out["aggregation"] = aggregation
    out["target"] = target_label

    out = out.sort_values("predicted_Kd_nM", ascending=True).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out, failures
