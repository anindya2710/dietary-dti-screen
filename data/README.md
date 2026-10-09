# data/

## `example_ingredients.csv`

61 real dietary small molecules, spanning the chemotypes a food-constituent library
actually contains: methylxanthines, flavonols, flavones, flavanones, isoflavones, a
flavonol glycoside, stilbenes, curcuminoids, phenolic acids, capsaicinoids, terpenoids,
organosulfur compounds, vitamins, a carotenoid, biogenic amines, amino-acid derivatives,
fatty acids, sugars and organic acids.

Every structure was parsed with RDKit and **checked against its literature molecular
formula** before being committed; all 61 passed. Two tests keep it that way:

- `test_chem.py::test_example_csv_is_fully_valid` — every row survives QC, no duplicates
- `test_chem.py::test_example_csv_formulas_match_rdkit` — the `formula` column agrees
  with RDKit's recomputation

Columns: `name`, `compound_class`, `SMILES`, `formula`, `MW`.

This set exists so the pipeline can be run and tested immediately. It is **not** a
research-grade screening library — it is far too small and was assembled by hand.

## Bringing your own library

Any CSV with a SMILES column works; point `--smiles-col` at it.

```bash
dti-screen run --input data/my_library.csv --smiles-col moldb_smiles --name-col name
```

Real sources for dietary compound structures:

| Source | Scope | Note |
|--------|-------|------|
| [FooDB](https://foodb.ca) | ~70k food constituents | The canonical choice. SMILES column is `moldb_smiles`, not `SMILES`. |
| [Phenol-Explorer](http://phenol-explorer.eu) | Polyphenols | Rich metadata on food content. |
| [COCONUT](https://coconut.naturalproducts.net) | Natural products | Much broader than diet alone. |

Everything in `data/` other than `example_ingredients.csv` and this file is gitignored,
so a large downloaded library won't accidentally land in version control.

Two things to check on any new library before trusting a run:

1. **Scale.** ~1 minute per 1,000 molecules per target. Subsample first if you're
   iterating.
2. **Applicability domain.** A larger library does not make the predictions more
   reliable. Read `control_benchmark.csv` before reading your hits.
