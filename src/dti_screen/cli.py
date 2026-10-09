"""
Command-line interface.

    dti-screen run --input data/example_ingredients.csv
    dti-screen run --config config/default.yaml --aggregation mean
    dti-screen controls          # verify the control set, no model download
    dti-screen targets           # fetch and report target sequences only
"""

from __future__ import annotations

import argparse
import sys


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="YAML config file (CLI flags override its values)")
    p.add_argument("--input", dest="input_csv", help="library CSV with a SMILES column")
    p.add_argument("--smiles-col", dest="smiles_col", help="name of the SMILES column")
    p.add_argument("--name-col", dest="name_col", help="name of the compound-name column")
    p.add_argument("--outdir", dest="output_dir", help="directory for output CSVs")
    p.add_argument("--workdir", dest="work_dir", help="scratch dir for checkpoints/cache")
    p.add_argument(
        "--aggregation",
        choices=["agg_mean_max", "mean", "max_effect"],
        help="ensemble aggregation strategy (default: agg_mean_max)",
    )
    p.add_argument(
        "--no-controls",
        dest="include_controls",
        action="store_false",
        default=None,
        help="skip the control benchmark (not recommended)",
    )
    p.add_argument(
        "--verbose-deeppurpose",
        dest="quiet_deeppurpose",
        action="store_false",
        default=None,
        help="show DeepPurpose's own stdout",
    )
    p.add_argument("-q", "--quiet", action="store_true", help="suppress progress output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dti-screen",
        description=(
            "Ensemble drug-target affinity screening of small molecules against protein "
            "targets, with a built-in active/decoy control benchmark."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the full screening pipeline")
    _add_run_args(run)

    sub.add_parser("controls", help="verify the control structure set and exit")

    an = sub.add_parser(
        "analyze",
        help="post-screen diagnostics: cross-target correlation and MW baseline",
    )
    an.add_argument("--indir", default="predictions", help="directory holding the screen output")
    an.add_argument("--outdir", default=None, help="where to write diagnostic CSVs")

    tg = sub.add_parser("targets", help="fetch target sequences and exit")
    tg.add_argument("--config", help="YAML config file")
    tg.add_argument("--workdir", dest="work_dir", help="scratch dir for the UniProt cache")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "controls":
        from .controls import verify_control_structures

        df = verify_control_structures()
        print(df.to_string(index=False))
        print(f"\nAll {len(df)} control structures verified against expected formulas.")
        return 0

    if args.command == "analyze":
        from .crosstarget import run_analysis

        try:
            run_analysis(args.indir, args.outdir)
        except (FileNotFoundError, ValueError) as exc:
            print(f"analysis error: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.command == "targets":
        from .config import load_config
        from .targets import fetch_targets

        cfg = load_config(args.config, work_dir=getattr(args, "work_dir", None))
        records, failures = fetch_targets(cfg.targets, cache_dir=cfg.cache_dir)
        print(f"\n{len(records)}/{len(cfg.targets)} targets retrieved.")
        return 1 if failures else 0

    # run
    from .config import load_config
    from .pipeline import run_screen

    verbose = not args.quiet
    overrides = {
        k: v
        for k, v in vars(args).items()
        if k not in {"command", "config", "quiet"} and v is not None
    }

    try:
        cfg = load_config(args.config, **overrides)
    except (ValueError, FileNotFoundError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        out = run_screen(cfg, verbose=verbose)
    except FileNotFoundError as exc:
        print(f"\ninput error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"\npipeline error: {exc}", file=sys.stderr)
        return 1

    if verbose:
        print(f"\nWrote {len(out['written'])} file(s) to {cfg.output_dir}/")
        if out["failures"]:
            print(f"Completed with degraded stages: {list(out['failures'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
