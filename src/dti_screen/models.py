"""
Pretrained ensemble loading.

Why this module does not use ``DeepPurpose.oneliner.repurpose()``
----------------------------------------------------------------
Two blocking reasons, both verified against DeepPurpose 0.1.5 source:

1. **It returns ``None``.** ``oneliner.repurpose()`` prints a ``PrettyTable`` and
   pickles a list of pre-formatted, 2-decimal-rounded *strings* with no SMILES
   attached. There is no numeric return value, so predictions cannot be merged back
   onto the input library.

2. **It hardcodes MPNN.** Its model list is a function-body local,
   ``[['MPNN','CNN'], ['CNN','CNN'], ['Morgan','CNN'], ['Morgan','AAC'],
   ['Daylight','AAC']]``, and is not exposed as a parameter. The MPNN encoder imports
   ``dgl``. PyPI serves only dgl 0.1.3 (2018); real wheels live on DGL's own index and
   stop at torch 2.4 / cp312, so on any current environment there is no installable
   compatible DGL.

This module therefore calls ``DTI.model_pretrained()`` and ``DTI.repurpose()`` -- the
functions ``oneliner`` wraps internally -- across the four non-MPNN members of that same
ensemble, and reproduces its ``agg_mean_max`` aggregation in :mod:`dti_screen.screen`.
Same checkpoints, same arithmetic, numeric output, no DGL.
"""

from __future__ import annotations

import contextlib
import io
import os
import time

#: The four non-MPNN members of oneliner's ensemble. All are BindingDB-Kd
#: checkpoints, so raw model output is pKd (higher = stronger binding).
DEFAULT_ENSEMBLE: tuple[str, ...] = (
    "cnn_cnn_bindingdb",
    "morgan_cnn_bindingdb",
    "morgan_aac_bindingdb",
    "daylight_aac_bindingdb",
)


def load_ensemble(
    model_names: list[str] | tuple[str, ...] = DEFAULT_ENSEMBLE,
    work_dir: str = "dp_work",
    quiet: bool = True,
    verbose: bool = True,
) -> tuple[dict[str, object], dict[str, str]]:
    """
    Load pretrained DTI checkpoints by DeepPurpose model name.

    Each load is isolated: one unavailable checkpoint degrades the ensemble instead of
    killing the run. Returns ``(loaded, failures)``.

    Checkpoints are downloaded from Harvard Dataverse on first use
    (``https://dataverse.harvard.edu/api/access/datafile/<id>``) and cached under
    ``work_dir``. If *every* model fails, check that endpoint in a browser before
    debugging anything here.
    """
    from DeepPurpose import DTI as dp_models  # imported lazily; pulls in torch

    loaded: dict[str, object] = {}
    failures: dict[str, str] = {}

    os.makedirs(work_dir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(work_dir)  # DeepPurpose writes ./save_folder relative to the cwd
    try:
        for name in model_names:
            start = time.time()
            try:
                buf = io.StringIO()
                ctx = contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()
                with ctx:
                    model = dp_models.model_pretrained(model=name)
                loaded[name] = model
                if verbose:
                    print(
                        f"[ok]   {name:26s} drug={model.drug_encoding:9s} "
                        f"target={model.target_encoding:5s} "
                        f"binary={model.binary} ({time.time() - start:.1f}s)"
                    )
            except Exception as exc:  # noqa: BLE001 - per-model isolation is the point
                failures[name] = f"{type(exc).__name__}: {exc}"
                if verbose:
                    print(f"[FAIL] {name:26s} {type(exc).__name__}: {exc}")
    finally:
        os.chdir(cwd)

    return loaded, failures
