# dietary-dti-screen

Ensemble drug–target affinity screening of dietary small molecules against human protein
targets, built on [DeepPurpose](https://github.com/kexinhuang12345/DeepPurpose)'s
BindingDB-Kd pretrained models — **with a control benchmark that tells you whether the
predictions mean anything.**

That last part is the point of the project. A ranked CSV of predicted affinities is easy
to produce and almost impossible to interpret. This pipeline scores known actives and
known inactives through the identical code path, then asks two further questions that
most screening write-ups skip: *does the ranking depend on which protein was supplied?*
and *does the model beat sorting by molecular weight?*

On the bundled 61-molecule dietary library, the answers are informative:

```
                 model AUC   MW-only AUC    delta
MAOA               0.767        0.533      +0.233     model beats the size baseline
ADORA2A            0.708        0.750      -0.042     AUC explained by molecular weight
CB1                1.000        0.958      +0.042     AUC explained by molecular weight

cross-target Spearman rho (same 72 molecules, different proteins):  0.910 - 0.954
```

CB1's apparently perfect AUC is almost entirely reproducible by ranking on molecular
weight alone, and the three targets produce very nearly the **same** ranking despite
being structurally unrelated proteins. See [Results](#results) for what that means.

---

## What this does

1. **Ingest** a compound library from CSV and validate every structure with RDKit.
2. **Fetch** canonical target sequences from the UniProt REST API, with retries and real
   response validation.
3. **Predict** binding affinity with a four-model DeepPurpose ensemble.
4. **Aggregate** per-model pKd into a consensus score, reproducing DeepPurpose's
   `agg_mean_max` strategy.
5. **Benchmark** the ensemble against verified actives and decoys per target.
6. **Write** one ranked CSV per target, strongest predicted affinity first.

Default targets (accessions verified against UniProt):

| Label     | UniProt  | Protein                              | Gene    | Length |
|-----------|----------|--------------------------------------|---------|--------|
| `MAOA`    | `P21397` | Amine oxidase [flavin-containing] A  | MAOA    | 527 aa |
| `ADORA2A` | `P29274` | Adenosine receptor A2a               | ADORA2A | 412 aa |
| `CB1`     | `P21554` | Cannabinoid receptor 1               | CNR1    | 472 aa |

All three are shorter than the CNN encoder's 1000-residue limit, so none is truncated.

---

## Install

Install order matters, and the one-liner you'd expect does not work. See
[Known friction](#known-friction) for why.

```bash
git clone https://github.com/anindya2710/dietary-dti-screen.git
cd dietary-dti-screen
python -m venv .venv && source .venv/bin/activate
make setup          # or follow the three steps below
```

Without `make`:

```bash
pip install --no-deps DeepPurpose==0.1.5     # --no-deps is deliberate
pip install -r requirements.txt              # includes descriptastorus from git
pip install -e .
```

Verify:

```bash
python -c "import dti_screen, DeepPurpose; print('ok')"
make controls        # verifies all 19 control structures, no downloads
```

---

## Usage

```bash
# Screen the bundled 61-molecule example set
dti-screen run --config config/default.yaml

# Your own library
dti-screen run --input data/my_library.csv --smiles-col moldb_smiles --name-col name

# Alternative aggregation
dti-screen run --input data/my_library.csv --aggregation mean

# Post-screen diagnostics: cross-target correlation + MW-only baseline
dti-screen analyze --indir predictions

# Inspect pieces without downloading checkpoints
dti-screen controls
dti-screen targets --config config/default.yaml
```

`analyze` is the step that makes the screen interpretable — run it every time. It answers
whether the ranking actually depends on the protein, and whether the model beats sorting
by molecular weight.

As a library:

```python
from dti_screen import ScreenConfig, run_screen

out = run_screen(ScreenConfig(input_csv="data/example_ingredients.csv", name_col="name"))
out["results"]["MAOA"].head(10)
out["benchmark"]          # per-target AUC and verdict
```

There's also a Colab notebook at
[`notebooks/colab_quickstart.ipynb`](notebooks/colab_quickstart.ipynb) that installs the
package from GitHub and calls the same entry point.

### Input format

A CSV with one SMILES column. Everything else is optional and carried through.

```csv
name,SMILES
caffeine,CN1C=NC2=C1C(=O)N(C)C(=O)N2C
quercetin,O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12
```

Point `--smiles-col` at whatever your source actually calls it — FooDB exports, for
instance, use `moldb_smiles`, not `SMILES`. A wrong column name fails immediately and
lists the columns it did find.

### Output

| File | Contents |
|------|----------|
| `<TARGET>_predictions.csv` | Ranked predictions, strongest first |
| `all_targets_predictions.csv` | All targets concatenated |
| `control_benchmark.csv` | Per-target AUC, medians, verdict |
| `rejected_molecules.csv` | Every dropped row, with the reason |
| `diagnostic_*.csv` | From `dti-screen analyze`: cross-target correlation, size correlation, MW baseline, own-active ranks |

Key columns: `predicted_Kd_nM` (lower = stronger), `pKd_aggregate` (higher = stronger),
`pKd_std` (ensemble disagreement), one `pKd_<model>` per ensemble member,
`cnn_smiles_truncated`, and `is_control`.

---

## Results

Run on `data/example_ingredients.csv` (61 dietary molecules + 19 controls = 72 screened),
four BindingDB-Kd checkpoints, `agg_mean_max` aggregation. Reproduce with
`make run && dti-screen analyze`; the tables below are
`predictions/diagnostic_*.csv`.

### 1. The ensemble produces one ranking, not three

Spearman correlation of the same molecules' predicted affinity between targets:

| Comparison | rho (all 72) | rho (19 controls) |
|---|---|---|
| MAOA vs ADORA2A | 0.954 | 0.949 |
| MAOA vs CB1 | 0.910 | 0.956 |
| ADORA2A vs CB1 | 0.947 | 0.981 |

Monoamine oxidase A is a flavin-dependent mitochondrial enzyme; ADORA2A and CB1 are
class-A GPCRs with unrelated binding sites. The model orders molecules nearly
identically for all three. ZM-241385, an adenosine A2A-selective antagonist, is ranked
**#1 of 72 at MAOA and #1 at CB1**.

This subsumes a scrambled-sequence control. Scrambling asks whether a *fake* protein
changes the prediction; this shows that three *real, unrelated* proteins barely do.

### 2. Molecular size explains part of the score, and most of it inside the control set

Spearman of predicted pKd against molecular weight:

| Target | rho (all 72) | rho (19 controls) |
|---|---|---|
| MAOA | 0.454 | 0.714 |
| ADORA2A | 0.522 | 0.755 |
| CB1 | 0.419 | 0.691 |

Across the full dietary library the size dependence is **moderate, not dominant** —
rho ≈ 0.42–0.52 accounts for well under a third of the rank variance. So "it is just
predicting molecular weight" would be an overstatement.

But note the gap between the two columns. Size correlates far more strongly *within the
19 controls* (0.69–0.76) than across the library as a whole. The control set is the
size-confounded subset, which is exactly why the AUC computed on it is unreliable — and
why §3 is the diagnostic that matters rather than this one.

### 3. Against an MW-only baseline, two of three AUCs vanish

| Target | Model AUC | MW-only AUC | Delta |
|---|---|---|---|
| MAOA | 0.767 | 0.533 | **+0.233** |
| ADORA2A | 0.708 | 0.750 | **−0.042** |
| CB1 | 1.000 | 0.958 | **+0.042** |

This inverts the naive reading of the benchmark:

* **CB1** looks like the success (AUC 1.000) but is the clearest confound. Its actives
  are the four largest, most lipophilic molecules in the control set (rimonabant 464 Da,
  CP-55940 377, anandamide 348, Δ9-THC 314) against decoys with median MW 186. Sorting
  by weight alone scores 0.958.
* **ADORA2A** performs *below* the size baseline.
* **MAOA** looks mediocre but is the only target where the model adds real signal
  (+0.233) — despite tranylcypromine, a marketed irreversible MAO inhibitor, ranking
  65 of 72 because at 133 Da the size prior buries it.

### 4. The model cannot express "does not bind"

Predicted affinities for molecules with no credible interaction at these targets:
sucrose 6,423 nM (ADORA2A), citric acid 254 nM (CB1), D-glucose 51,537 nM (CB1).

The actives are not badly predicted in absolute terms — caffeine came out at 16,095 nM
at ADORA2A against a literature Ki of roughly 10–45 µM. It is the **decoys that are
wildly over-predicted**, and that is what destroys the separation. The likely cause is
the training distribution: BindingDB holds *measured* affinities, i.e. pairs someone
considered worth assaying, so it contains few true non-binders. The model never learned
what non-binding looks like.

### Conclusion

For these three targets and this chemical space, the ensemble is **not** doing
drug–target prediction in a useful sense. The strongest evidence is §1, not §2: three
unrelated proteins yield rank correlations of 0.91–0.95 across 72 molecules, so the
prediction is almost entirely a property of the ligand. Molecular weight is one
contributor (§2) but not the whole story — the honest statement is *target-independent
ligand scoring*, not *a molecular-weight lookup*.

The single defensible positive result is MAOA's +0.233 over the size baseline, and even
that arrives with tranylcypromine — a marketed irreversible MAO inhibitor — ranked 65 of
72.

**This also exposes a design limitation in this project's own benchmark.** Because
dietary decoys are systematically smaller than drug-like actives, AUC alone can be
passed by a size prior. The MW baseline in §3 is the diagnostic; the proper fix is
**property-matched decoys** (DUD-E style, matched on MW and logP), which is the first
item in [Next steps](#next-steps).

---

## Reading the results

**Units, because DeepPurpose leaves them implicit.** The `*_bindingdb` checkpoints are
trained on log-transformed BindingDB Kd, so raw output is **pKd** (higher = stronger).
`DTI.repurpose(convert_y=True)` silently converts to **nM** (lower = stronger) *and flips
the direction of its own sort*. This pipeline requests `convert_y=False`, keeps native
pKd, and converts explicitly, so both columns are present and the sort direction is
declared rather than inferred.

**The benchmark is the first thing to read.** `pKd_std` tells you whether the four models
agree; the AUC tells you whether agreement means anything.

| AUC | Interpretation |
|-----|----------------|
| ≥ 0.8 | Actives clearly separated. The ranking carries signal. |
| 0.6–0.8 | Weak separation. A soft prior at best. |
| < 0.6 | No usable separation. This target's scores are noise. |

An AUC near 0.5 is a result, not a bug — and reporting it is the most valuable thing this
repo produces.

### What this can and cannot support

**Supported:** a reproducible pipeline with per-stage error isolation; a ranking of a
library against three targets under one stated model and aggregation rule; a quantitative
statement about whether the method separates actives from decoys per target.

**Not supported:**

- *That any top-ranked molecule binds its target.* These are out-of-domain predictions
  from sequence-only models. Absolute pKd/nM values should not be quoted as affinities.
- *Any physiological or health claim.* Target engagement is not a dietary effect. Oral
  bioavailability, first-pass metabolism, achievable plasma concentration from realistic
  intake, blood–brain-barrier penetration and tissue selectivity all sit between these
  numbers and anything happening in a person. Nothing here addresses any of them.
- *Cross-target comparisons.* Each target's scores are separately calibrated; pKd 7 at
  MAOA and pKd 7 at CB1 are not commensurable.

**Why out-of-domain:** the models see the target only as an amino-acid string — no
structure, no pocket, no cofactor. MAOA's FAD cofactor is invisible to them. And dietary
chemistry (sugars, fatty acids, polyphenols, vitamins) is poorly represented in a
BindingDB training set dominated by drug-like medicinal chemistry.

---

## Design decisions

### Why not `oneliner.repurpose()`

DeepPurpose's headline convenience function is the obvious choice here, and it doesn't
work for this task. Both reasons were verified against the 0.1.5 source.

**1. It returns `None`.** It prints a `PrettyTable` and pickles a list of pre-formatted,
2-decimal-rounded *strings* with no SMILES attached. There is no numeric return value, so
predictions cannot be merged back onto the input library.

**2. It hardcodes MPNN.** Its model list is a function-body local —
`[['MPNN','CNN'], ['CNN','CNN'], ['Morgan','CNN'], ['Morgan','AAC'], ['Daylight','AAC']]`
— not a parameter. The MPNN encoder imports `dgl`. PyPI serves only `dgl` **0.1.3
(2018)**; real wheels live on DGL's own index and stop at **torch 2.4 / cp312**. On any
current environment there is no installable compatible DGL.

This project calls `DTI.model_pretrained()` and `DTI.repurpose()` — the functions
`oneliner` wraps internally — across the **four non-MPNN members of that same ensemble**,
and reproduces `agg_mean_max` in `screen.py`. Same checkpoints, same arithmetic, numeric
output, no DGL. Config validation rejects any `*mpnn*` model with an explanation.

### Why RDKit validation is load-bearing

DeepPurpose's featurizers swallow invalid input. `smiles2morgan` and `smiles2daylight`
wrap everything in `try/except` and return an **all-zero fingerprint** when RDKit cannot
parse a SMILES string. Nothing raises. The molecule still receives a numeric,
plausible-looking affinity score:

```python
>>> from DeepPurpose.utils import smiles2morgan
>>> smiles2morgan("not_a_smiles_at_all").sum()
0.0
```

Passing `"not_a_smiles_at_all"` end-to-end produces a score indistinguishable from
caffeine's. So the upfront filter isn't hygiene — it's the only thing preventing
fabricated rows. There's a regression test pinning this behaviour
(`test_chem.py::TestInvalidSmiles`).

### Why controls are verified by formula

Every control structure is checked against its literature molecular formula at load time,
and the check is **fatal**, not a warning — an unverified control set cannot support the
benchmark built on it. This caught two real errors while the set was assembled: an extra
hydroxyl on CP-55940 (`C24H40O4` vs `C24H40O3`) and a monosaccharide standing in for
sucrose (`C6H12O6` vs `C12H22O11`). Both are now regression tests.

### Long SMILES are flagged, not dropped

The CNN *drug* encoder truncates the SMILES **string** at 100 characters
(`MAX_SEQ_DRUG`). Dietary glycosides routinely exceed this — rutin's canonical SMILES is
122 characters. Those molecules keep valid fingerprint-based scores, so they are retained
and marked `cnn_smiles_truncated` rather than discarded.

---

## Known friction

Things that will cost you an evening if you meet them cold.

**`pip install DeepPurpose` produces a package that cannot be imported.**
`DeepPurpose/utils.py` does `raise ImportError` at module level if `descriptastorus` is
missing. It is **not on PyPI** and **not in DeepPurpose's own `requirements.txt`**. It
must come from git — `requirements.txt` here handles it.

**`--no-deps` is deliberate.** DeepPurpose's declared requirements pull `ax-platform`
(→ botorch, gpytorch) and `dgllife`, neither of which this code path uses. Skipping them
takes the environment from ~244 packages to ~82.

**pandas gets downgraded, and it's unavoidable.** `DTI.py` imports
`lifelines.utils.concordance_index` at module level, and `lifelines` pins
`pandas<3.0`. On Colab, whose preinstalled pandas may be 3.x, pip downgrades it
mid-session and already-imported modules go stale — **restart the runtime** after install
(the notebook says so at the right moment).

**Checkpoints come from Harvard Dataverse.** `DTI.model_pretrained(model=...)` downloads
from `https://dataverse.harvard.edu/api/access/datafile/<id>`. Each model is loaded under
its own guard, so one failure degrades the ensemble rather than killing the run. If *all*
four fail, check that endpoint in a browser before debugging code.

**A GPU barely helps.** Measured ~17 ms/molecule for all four models against one target
— roughly **1 minute per 1,000 molecules per target**. The cost is RDKit fingerprinting
and AAC descriptors, both CPU-bound. 10,000 molecules × 3 targets runs in ~10 minutes on
CPU.

---

## Development

```bash
make setup-dev
make test            # 114 tests
make test-fast       # skips tests needing torch/DeepPurpose
make lint
```

The end-to-end test swaps pretrained checkpoints for **randomly initialized models of the
same architectures**, so CI exercises the whole path — featurization, prediction,
aggregation, sorting, CSV writing, benchmarking — with no large download. Predicted
values are meaningless under random weights, so every assertion is about pipeline
behaviour (shape, ordering, merge integrity, arithmetic), never predicted affinity.

UniProt is replaced by a local mock server that reproduces the failure modes the real API
produces only rarely: 404, a 200 with an empty body (obsolete/demerged accession), a 200
carrying an HTML error page, illegal residue characters, and a flaky endpoint that 500s
twice before succeeding.

```
src/dti_screen/
├── chem.py        SMILES validation and canonicalization
├── targets.py     UniProt retrieval with response validation
├── controls.py    Verified actives/decoys, formula-asserted
├── models.py      Ensemble loading (and why not oneliner)
├── screen.py      Prediction, unit conversion, aggregation
├── benchmark.py   Active-vs-decoy AUC
├── crosstarget.py Diagnostics: cross-target correlation, MW-only baseline
├── config.py      Config dataclass + YAML loading
├── pipeline.py    Orchestration, per-stage error isolation
└── cli.py         Command-line interface
```

---

## Next steps

Ordered by value, and honestly: the first two are what would turn this from "I ran a
model" into "I validated a model."

1. **Property-matched decoys.** The current decoy set is confounded with molecular size
   (see [Results §3](#3-against-an-mw-only-baseline-two-of-three-aucs-vanish)). Sampling
   decoys matched to each target's actives on MW and logP would make the AUC measure
   target recognition rather than drug-likeness. This is the highest-value fix.
2. **Correlate against measured data.** Pull measured affinities for these three targets
   from BindingDB/ChEMBL and report the ensemble's actual correlation on held-out
   *measured* values — the difference between "I ran a model" and "I validated a model."
3. **Add an applicability-domain check.** Tanimoto similarity of each library molecule to
   the model's training chemotypes; flag low-similarity rows as extrapolation.
4. **Dock the survivors** against real structures (MAOA `2Z5X`, ADORA2A `3EML`, CB1
   `5TGZ`/`5XRA`) for a structure-based second opinion.

*(A scrambled-sequence control was on this list and has been removed: the cross-target
correlation in [Results §1](#1-the-ensemble-produces-one-ranking-not-three) is a
stronger version of the same test, using real proteins instead of fake ones.)*

---

## Citation

If you use this, cite DeepPurpose and UniProt:

> Huang K, Fu T, Glass LM, Zitnik M, Xiao C, Sun J. *DeepPurpose: a deep learning library
> for drug–target interaction prediction.* Bioinformatics, 2020.

> The UniProt Consortium. *UniProt: the Universal Protein Knowledgebase.* Nucleic Acids
> Research.

## License

MIT — see [LICENSE](LICENSE). Third-party components retain their own licenses:
DeepPurpose (BSD-3-Clause), RDKit (BSD-3-Clause), PyTorch (BSD-style), UniProt data
(CC BY 4.0).
