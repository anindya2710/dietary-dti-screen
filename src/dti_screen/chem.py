"""
Structural QC for the compound library.

This module exists because of one specific DeepPurpose behaviour. Its featurizers
swallow invalid input: ``smiles2morgan`` and ``smiles2daylight`` wrap everything in
``try/except`` and return an **all-zero fingerprint** when RDKit cannot parse a SMILES
string. Nothing raises. The molecule still receives a numeric, plausible-looking
affinity score.

Verified directly against DeepPurpose 0.1.5::

    >>> from DeepPurpose.utils import smiles2morgan
    >>> smiles2morgan("not_a_smiles_at_all").sum()
    0.0

So filtering invalid structures up front is not hygiene. It is the only thing standing
between the pipeline and fabricated rows in the output.
"""

from __future__ import annotations

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

# Mirrors DeepPurpose.utils constants. Duplicated rather than imported so that this
# module can be used (and tested) without importing torch.
#: The CNN *protein* encoder pads/truncates sequences to this length.
MAX_SEQ_PROTEIN = 1000
#: The CNN *drug* encoder pads/truncates the SMILES *string* to this many characters.
MAX_SEQ_DRUG = 100

RDLogger.DisableLog("rdApp.*")  # we report parse failures ourselves


def canonicalize(smiles: str) -> str | None:
    """Return the canonical SMILES, or ``None`` if RDKit cannot parse it."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol)


def validate_library(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    name_col: str | None = None,
    dedupe: bool = True,
    label: str = "library",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """
    Validate and canonicalize a SMILES library, reporting attrition at every stage.

    Returns ``(clean, rejected, report)``. Molecules whose canonical SMILES exceed
    ``MAX_SEQ_DRUG`` are **flagged, not dropped** -- the CNN drug encoder truncates the
    string, so those rows' CNN-derived scores are unreliable while their
    fingerprint-derived scores remain valid.
    """
    if smiles_col not in df.columns:
        raise KeyError(
            f"Column {smiles_col!r} not found. Available columns: {list(df.columns)}"
        )

    report: dict[str, int] = {"input_rows": len(df)}
    work = df.copy()
    work["_raw"] = work[smiles_col].astype("string")

    # 1. missing / blank
    blank = work["_raw"].isna() | (work["_raw"].str.strip() == "")
    report["dropped_blank"] = int(blank.sum())
    rejected_parts = [work.loc[blank].assign(_reject_reason="missing or blank SMILES")]
    work = work.loc[~blank].copy()
    work["_raw"] = work["_raw"].str.strip()

    # 2. RDKit parse
    canon, reasons = [], []
    for smi in work["_raw"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            canon.append(None)
            reasons.append("RDKit could not parse SMILES")
        elif mol.GetNumHeavyAtoms() == 0:
            canon.append(None)
            reasons.append("zero heavy atoms")
        else:
            canon.append(mol)
            reasons.append(None)

    work["_mol"] = canon
    work["_reject_reason"] = reasons
    bad = work["_mol"].isna()
    report["dropped_unparseable"] = int(bad.sum())
    rejected_parts.append(work.loc[bad].drop(columns=["_mol"]))
    work = work.loc[~bad].copy()

    # Bail out before the descriptor stage. On an empty frame the assignments below
    # produce float64 columns, and `.str.len()` on one raises an opaque pandas
    # AttributeError instead of the actionable message the caller needs.
    if work.empty:
        raise ValueError(
            f"No valid molecules survived validation of {label!r}: "
            f"{report['input_rows']} input row(s), "
            f"{report['dropped_blank']} blank, "
            f"{report['dropped_unparseable']} unparseable by RDKit. "
            f"Check that {smiles_col!r} is the right column and holds SMILES strings."
        )

    # 3. descriptors
    work["canonical_smiles"] = [Chem.MolToSmiles(m) for m in work["_mol"]]
    work["formula"] = [rdMolDescriptors.CalcMolFormula(m) for m in work["_mol"]]
    work["MW"] = [round(Descriptors.MolWt(m), 2) for m in work["_mol"]]
    work["n_heavy_atoms"] = [m.GetNumHeavyAtoms() for m in work["_mol"]]

    # 4. flag CNN drug-encoder truncation
    work["cnn_smiles_truncated"] = work["canonical_smiles"].str.len() > MAX_SEQ_DRUG
    report["flagged_cnn_truncated"] = int(work["cnn_smiles_truncated"].sum())

    # 5. de-duplicate on canonical structure
    if dedupe:
        before = len(work)
        work = work.drop_duplicates(subset="canonical_smiles", keep="first").copy()
        report["dropped_duplicates"] = before - len(work)
    else:
        report["dropped_duplicates"] = 0

    # 6. names
    if name_col and name_col in work.columns:
        work["compound_name"] = work[name_col].astype(str)
    else:
        work["compound_name"] = [f"CMPD_{i:05d}" for i in range(len(work))]

    work = work.drop(columns=["_mol", "_reject_reason", "_raw"], errors="ignore")
    report["valid_molecules"] = len(work)

    rejected = (
        pd.concat(rejected_parts, ignore_index=True)
        if any(len(p) for p in rejected_parts)
        else pd.DataFrame()
    )
    if "_raw" in rejected.columns:
        rejected = rejected.drop(columns=["_raw"])

    if verbose:
        print(f"SMILES validation - {label}")
        print("-" * 46)
        for key, value in report.items():
            print(f"  {key:26s} {value:>8,}")
        if report["flagged_cnn_truncated"]:
            print(
                f"\n  NOTE: {report['flagged_cnn_truncated']} molecule(s) have canonical "
                f"SMILES longer than {MAX_SEQ_DRUG} characters.\n"
                f"        The CNN drug encoder truncates the string, so their CNN-based\n"
                f"        scores are unreliable. Flagged as cnn_smiles_truncated."
            )

    if report["valid_molecules"] == 0:
        raise ValueError(f"No valid molecules survived validation of {label!r}.")

    return work.reset_index(drop=True), rejected, report
