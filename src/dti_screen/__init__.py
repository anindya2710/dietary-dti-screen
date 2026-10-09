"""
dti_screen -- ensemble drug-target interaction screening with DeepPurpose.

Importing this package does **not** import torch or DeepPurpose. Those are pulled in
lazily by :mod:`dti_screen.models` and :mod:`dti_screen.screen` when a model is actually
loaded, which keeps ``--help``, the control verification and the structural QC fast and
usable in environments where the deep-learning stack is unavailable.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .benchmark import benchmark_summary, benchmark_target, interpret_auc, pairwise_auc
from .chem import MAX_SEQ_DRUG, MAX_SEQ_PROTEIN, canonicalize, validate_library
from .config import DEFAULT_TARGETS, ScreenConfig, load_config
from .controls import CONTROL_COMPOUNDS, control_roles, verify_control_structures
from .models import DEFAULT_ENSEMBLE
from .screen import AGGREGATIONS, aggregate_pkd, nM_to_pkd, pkd_to_nM
from .targets import TargetRecord, fetch_target, fetch_targets, parse_fasta

__all__ = [
    "__version__",
    # config
    "ScreenConfig",
    "load_config",
    "DEFAULT_TARGETS",
    "DEFAULT_ENSEMBLE",
    "AGGREGATIONS",
    # chem
    "canonicalize",
    "validate_library",
    "MAX_SEQ_DRUG",
    "MAX_SEQ_PROTEIN",
    # targets
    "TargetRecord",
    "fetch_target",
    "fetch_targets",
    "parse_fasta",
    # controls
    "CONTROL_COMPOUNDS",
    "verify_control_structures",
    "control_roles",
    # scoring
    "pkd_to_nM",
    "nM_to_pkd",
    "aggregate_pkd",
    # benchmark
    "pairwise_auc",
    "interpret_auc",
    "benchmark_target",
    "benchmark_summary",
]


def run_screen(*args, **kwargs):
    """Lazy re-export of :func:`dti_screen.pipeline.run_screen`."""
    from .pipeline import run_screen as _run

    return _run(*args, **kwargs)
