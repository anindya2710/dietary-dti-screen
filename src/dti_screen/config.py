"""Configuration loading and validation."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

from .models import DEFAULT_ENSEMBLE
from .screen import AGGREGATIONS

#: Accessions verified against UniProt (October 2026):
#:   P21397 -> sp|P21397|AOFA_HUMAN  Amine oxidase [flavin-containing] A  GN=MAOA    527 aa
#:   P29274 -> sp|P29274|AA2AR_HUMAN Adenosine receptor A2a               GN=ADORA2A 412 aa
#:   P21554 -> sp|P21554|CNR1_HUMAN  Cannabinoid receptor 1               GN=CNR1    472 aa
#: All three are shorter than MAX_SEQ_PROTEIN (1000), so none is truncated.
DEFAULT_TARGETS = {
    "MAOA": "P21397",
    "ADORA2A": "P29274",
    "CB1": "P21554",
}


@dataclass
class ScreenConfig:
    """Everything one screening run needs."""

    input_csv: str = "data/example_ingredients.csv"
    smiles_col: str = "SMILES"
    name_col: str | None = "name"

    output_dir: str = "predictions"
    work_dir: str = "dp_work"
    cache_dir: str = "dp_work/uniprot_cache"

    targets: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_TARGETS))
    ensemble: list[str] = field(default_factory=lambda: list(DEFAULT_ENSEMBLE))
    aggregation: str = "agg_mean_max"

    include_controls: bool = True
    dedupe: bool = True
    quiet_deeppurpose: bool = True

    def validate(self) -> ScreenConfig:
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {AGGREGATIONS}, got {self.aggregation!r}"
            )
        if not self.targets:
            raise ValueError("at least one target is required")
        if not self.ensemble:
            raise ValueError("at least one ensemble model is required")
        bad = [m for m in self.ensemble if "mpnn" in m.lower()]
        if bad:
            raise ValueError(
                f"MPNN models require dgl, which has no build compatible with current "
                f"torch; remove {bad} from the ensemble. See dti_screen/models.py."
            )
        return self

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | None = None, **overrides) -> ScreenConfig:
    """
    Build a :class:`ScreenConfig` from an optional YAML file plus keyword overrides.

    Keyword overrides (typically from the CLI) win over file values. ``None`` overrides
    are ignored so that unset CLI flags do not clobber file settings.
    """
    data: dict = {}
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"config file not found: {path}")
        import yaml

        with open(path) as fh:
            data = yaml.safe_load(fh) or {}

    known = {f for f in ScreenConfig.__dataclass_fields__}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")

    data.update({k: v for k, v in overrides.items() if v is not None})
    return ScreenConfig(**data).validate()
