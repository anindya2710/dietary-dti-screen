"""
Positive and negative controls.

The point of this module is to make the screen falsifiable. Predictions from
sequence-based DTI models on dietary chemistry are out-of-domain, so a ranked list on
its own says nothing about whether the method works. Scoring known actives and known
inactives through the identical code path turns "here are some hits" into a measurable
claim: either the ensemble separates harmine from glucose at MAO-A, or it does not.

Every structure is checked against its literature molecular formula at import time. A
silent typo in a control SMILES would invalidate the benchmark that everything else
depends on, so this module *asserts* rather than warns. Two genuine errors were caught
this way while the set was being assembled (an extra hydroxyl on CP-55940, and a
monosaccharide standing in for sucrose).

Potency annotations are order-of-magnitude literature values for orientation only; they
are never used in any computation.
"""

from __future__ import annotations

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

#: ``(name, smiles, expected_formula, control_for, literature_potency)``
#: ``control_for`` is a target label, or ``"decoy"`` for expected-inactive compounds.
CONTROL_COMPOUNDS: list[tuple[str, str, str, str, str]] = [
    # --- MAOA: established inhibitors -------------------------------------------------
    ("harmine", "COc1ccc2c(c1)[nH]c1c(C)nccc12", "C13H12N2O", "MAOA", "IC50 ~5 nM"),
    ("clorgyline", "C#CCN(C)CCCOc1ccc(Cl)cc1Cl", "C13H15Cl2NO", "MAOA", "IC50 ~1 nM, irreversible"),
    ("phenelzine", "NNCCc1ccccc1", "C8H12N2", "MAOA", "irreversible"),
    ("tranylcypromine", "NC1CC1c1ccccc1", "C9H11N", "MAOA", "irreversible"),
    ("moclobemide", "O=C(NCCN1CCOCC1)c1ccc(Cl)cc1", "C13H17ClN2O2", "MAOA", "reversible, ~uM"),
    # --- ADORA2A: antagonists and xanthines -------------------------------------------
    ("caffeine", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "C8H10N4O2", "ADORA2A", "Ki ~10-45 uM"),
    ("theophylline", "Cn1c(=O)c2[nH]cnc2n(C)c1=O", "C7H8N4O2", "ADORA2A", "Ki ~10-25 uM"),
    (
        "istradefylline",
        "CCn1c(=O)c2n(C)c(/C=C/c3ccc(OC)c(OC)c3)nc2n(CC)c1=O",
        "C20H24N4O4",
        "ADORA2A",
        "Ki ~12 nM",
    ),
    (
        "ZM-241385",
        "Oc1ccc(CCNc2nc(N)n3nc(-c4ccco4)nc3n2)cc1",
        "C16H15N7O2",
        "ADORA2A",
        "Ki ~1 nM",
    ),
    # --- CB1 (CNR1): agonists and antagonists -----------------------------------------
    (
        "rimonabant",
        "Cc1c(-c2ccc(Cl)cc2)n(-c2ccc(Cl)cc2Cl)nc1C(=O)NN1CCCCC1",
        "C22H21Cl3N4O",
        "CB1",
        "Ki ~2 nM (antagonist)",
    ),
    (
        "delta9-THC",
        "CCCCCc1cc(O)c2c(c1)OC(C)(C)[C@@H]1CCC(C)=C[C@H]21",
        "C21H30O2",
        "CB1",
        "Ki ~25-40 nM",
    ),
    (
        "anandamide",
        r"CCCCC/C=C\C/C=C\C/C=C\C/C=C\CCCC(=O)NCCO",
        "C22H37NO2",
        "CB1",
        "Ki ~60-250 nM",
    ),
    (
        "CP-55940",
        "CCCCCCC(C)(C)c1ccc([C@@H]2CCC(O)C[C@H]2CCCO)c(O)c1",
        "C24H40O3",
        "CB1",
        "Ki ~0.5-5 nM",
    ),
    # --- Dietary decoys: expected inactive at all three targets -----------------------
    ("D-glucose", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "C6H12O6", "decoy", "inactive"),
    (
        "sucrose",
        "OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O",
        "C12H22O11",
        "decoy",
        "inactive",
    ),
    ("glycine", "NCC(=O)O", "C2H5NO2", "decoy", "inactive"),
    ("citric acid", "OC(=O)CC(O)(CC(=O)O)C(=O)O", "C6H8O7", "decoy", "inactive"),
    ("palmitic acid", "CCCCCCCCCCCCCCCC(=O)O", "C16H32O2", "decoy", "inactive"),
    ("L-ascorbic acid", "OC[C@H](O)[C@H]1OC(=O)C(O)=C1O", "C6H8O6", "decoy", "inactive"),
]


def verify_control_structures(
    controls: list[tuple[str, str, str, str, str]] | None = None,
) -> pd.DataFrame:
    """
    Parse every control and assert its molecular formula matches the literature value.

    Raises ``AssertionError`` listing every problem found. This is deliberately fatal:
    an unverified control set cannot support the benchmark it is used for.
    """
    controls = controls if controls is not None else CONTROL_COMPOUNDS
    rows, problems = [], []

    for name, smiles, expected_formula, control_for, potency in controls:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            problems.append(f"{name}: unparseable SMILES")
            continue
        formula = rdMolDescriptors.CalcMolFormula(mol).replace("+", "").replace("-", "")
        if expected_formula and formula != expected_formula:
            problems.append(f"{name}: computed {formula} != expected {expected_formula}")
            continue
        rows.append(
            {
                "compound_name": name,
                "SMILES": Chem.MolToSmiles(mol),
                "formula": formula,
                "MW": round(Descriptors.MolWt(mol), 2),
                "control_for": control_for,
                "lit_potency": potency,
            }
        )

    if problems:
        raise AssertionError(
            "Control structure verification FAILED:\n  " + "\n  ".join(problems)
        )
    return pd.DataFrame(rows)


def control_roles() -> dict[str, str]:
    """``{compound_name: control_for}`` for every verified control."""
    df = verify_control_structures()
    return dict(zip(df["compound_name"], df["control_for"]))
