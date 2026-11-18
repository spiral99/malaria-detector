"""Load params.yaml (plus optional override files) and resolve project paths."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _merge(base: dict, extra: dict) -> dict:
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


class Cfg(dict):
    """A dict of parameters with a helper that turns configured paths into absolute ones."""

    def path(self, name: str, *parts: str, make: bool = False) -> Path:
        p = Path(self["paths"][name])
        if not p.is_absolute():
            p = ROOT / p
        p = p.joinpath(*parts)
        if make:
            (p if not p.suffix else p.parent).mkdir(parents=True, exist_ok=True)
        return p

    @property
    def class_names(self) -> list[str]:
        return list(self["classes"]["names"])


def load(params: str | None = None, overrides: list[str] | None = None) -> Cfg:
    params_path = Path(params or os.environ.get("MALARIA_PARAMS") or ROOT / "params.yaml")
    cfg = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    env_over = os.environ.get("MALARIA_OVERRIDE")
    for extra in list(overrides or []) + ([env_over] if env_over else []):
        extra_path = Path(extra)
        if not extra_path.is_absolute() and not extra_path.exists():
            extra_path = ROOT / extra_path
        _merge(cfg, yaml.safe_load(extra_path.read_text(encoding="utf-8")))
    return Cfg(cfg)


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--params", default=None, help="path to params.yaml (default: repo root)")
    parser.add_argument("--override", action="append", default=[], help="extra YAML merged on top; repeatable")
    return parser


def from_args(args: argparse.Namespace) -> Cfg:
    return load(args.params, args.override)
