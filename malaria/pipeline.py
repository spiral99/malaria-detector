"""Run the whole project with one command.

    python -m malaria.pipeline                  # everything, on the real dataset
    python -m malaria.pipeline --smoke          # everything, on tiny synthetic data (a few CPU minutes)
    python -m malaria.pipeline --from prepare   # resume from a stage
    python -m malaria.pipeline --only evaluate report
    python -m malaria.pipeline --fresh          # delete everything generated, keep the downloads, run it all again

Stages are idempotent: downloads, extractions, trained weights and predictions that already
exist are kept, so an interrupted run can simply be started again.
"""
from __future__ import annotations

import argparse
import shutil
import time

from malaria import config

STAGES = ["download", "unpack", "prepare", "shift", "baseline", "train", "evaluate", "adapt", "evaluate_adapted", "export", "report"]


def _stage(name: str, cfg: config.Cfg, smoke: bool, force: bool) -> None:
    if name == "download":
        if smoke:
            from malaria.data import synthetic
            synthetic.run(cfg)
        else:
            from malaria.data import download
            download.run(cfg)
    elif name == "unpack":
        from malaria.data import unpack
        unpack.run(cfg)
    elif name == "prepare":
        from malaria.data import prepare
        prepare.run(cfg)
    elif name == "shift":
        from malaria import shift
        shift.run(cfg)
    elif name == "baseline":
        from malaria.models import baseline
        baseline.run(cfg)
    elif name == "train":
        from malaria.models import train
        train.run(cfg, force)
    elif name in ("evaluate", "evaluate_adapted"):
        from malaria.models import evaluate
        evaluate.run(cfg)
    elif name == "adapt":
        from malaria.models import adapt
        adapt.run(cfg, force)
    elif name == "export":
        from malaria.models import export
        export.run(cfg)
    elif name == "report":
        from malaria import report
        report.run(cfg)


def wipe_generated(cfg: config.Cfg) -> None:
    """Delete everything the pipeline produced, so the next run starts from the downloaded files.

    Kept: the downloads (raw) and the extracted archives (interim), which are slow to recreate and are
    never changed by later stages. Deleted: prepared data, trained weights, training runs, reports,
    demo samples, and the MLflow store.
    """
    targets = [cfg.path("processed"), cfg.path("runs"), cfg.path("reports"), cfg.path("models")]
    if cfg.path("models") == config.ROOT / "models":
        targets += [config.ROOT / "samples", config.ROOT / "mlflow.db", config.ROOT / "mlruns"]
    else:
        targets += [cfg.path("models").parent / "samples"]
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
            print(f"[fresh] deleted {target}")
        elif target.exists():
            target.unlink()
            print(f"[fresh] deleted {target}")
    cfg.path("models", make=True)
    (cfg.path("models") / ".gitkeep").touch()
    cfg.path("reports", make=True)
    (cfg.path("reports") / ".gitkeep").touch()


def main() -> None:
    parser = config.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--smoke", action="store_true", help="use synthetic data and params.smoke.yaml")
    parser.add_argument("--from", dest="start", choices=STAGES, default=STAGES[0])
    parser.add_argument("--to", dest="stop", choices=STAGES, default=STAGES[-1])
    parser.add_argument("--only", nargs="+", choices=STAGES, help="run just these stages, in pipeline order")
    parser.add_argument("--force", action="store_true", help="retrain even if weights exist")
    parser.add_argument("--fresh", action="store_true",
                        help="delete prepared data, models, runs, reports and MLflow history first (downloads are kept)")
    args = parser.parse_args()
    if args.smoke:
        args.override = ["params.smoke.yaml", *args.override]
    cfg = config.from_args(args)
    if args.fresh:
        if args.only or args.start != STAGES[0]:
            raise SystemExit("--fresh deletes everything generated, so it cannot be combined with --only or --from.")
        wipe_generated(cfg)
    stages = [s for s in STAGES if s in args.only] if args.only else STAGES[STAGES.index(args.start): STAGES.index(args.stop) + 1]
    for name in stages:
        started = time.time()
        print(f"\n=== {name} ===")
        _stage(name, cfg, args.smoke, args.force)
        print(f"=== {name} done in {time.time() - started:.0f}s ===")


if __name__ == "__main__":
    main()
