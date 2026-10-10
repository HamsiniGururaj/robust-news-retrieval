"""Load experiment settings from configs/.

Settings are layered, later layers winning:
  1. configs/default.yaml
  2. configs/<dataset>.yaml (when a dataset is given)
  3. the IR_DATA_ROOT environment variable (for data_root only)
  4. --set key.path=value overrides from the command line

Print the resolved config for a dataset:
  python -m src.config --dataset scifact
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
DATA_ROOT_ENV = "IR_DATA_ROOT"


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_override(text: str) -> tuple[str, Any]:
    """Split 'bm25.k1=1.2' into ('bm25.k1', 1.2). Values are parsed as YAML."""
    key, sep, raw = text.partition("=")
    key = key.strip()
    if not sep or not key:
        raise ValueError(f"Override must look like key.path=value, got {text!r}")
    return key, yaml.safe_load(raw)


def set_by_path(cfg: dict, dotted_key: str, value: Any) -> None:
    """Set an existing nested key. Unknown keys raise, so typos can't pass silently."""
    keys = dotted_key.split(".")
    node = cfg
    for i, key in enumerate(keys):
        if not isinstance(node, dict) or key not in node:
            known = ", ".join(sorted(node)) if isinstance(node, dict) else "(not a section)"
            raise KeyError(f"Unknown config key {'.'.join(keys[:i + 1])!r}; known here: {known}")
        if i == len(keys) - 1:
            node[key] = value
        else:
            node = node[key]


def load_config(
    dataset: str | None = None,
    overrides: Iterable[str] = (),
    config_dir: Path = CONFIG_DIR,
) -> dict:
    cfg = _read_yaml(config_dir / "default.yaml")
    if dataset:
        path = config_dir / f"{dataset}.yaml"
        if not path.exists():
            available = sorted(p.stem for p in config_dir.glob("*.yaml") if p.stem != "default")
            raise FileNotFoundError(f"No config for dataset {dataset!r}; available: {available}")
        cfg = _deep_merge(cfg, _read_yaml(path))

    env_root = os.environ.get(DATA_ROOT_ENV)
    if env_root:
        cfg["data_root"] = env_root

    for text in overrides:
        set_by_path(cfg, *parse_override(text))
    return cfg


def resolve_path(path: str | Path) -> Path:
    """Absolute paths are kept; relative paths are taken from the repo root."""
    p = Path(path).expanduser()
    # A POSIX path such as /content/drive is not "absolute" on Windows, but it is meant to be.
    if p.is_absolute() or p.as_posix().startswith("/"):
        return p
    return REPO_ROOT / p


def data_dir(cfg: dict, *parts: str) -> Path:
    """Folder for this dataset's large files, e.g. data_dir(cfg, "bm25_index")."""
    name = (cfg.get("dataset") or {}).get("name")
    if not name:
        raise ValueError("No dataset selected; pass --dataset")
    return resolve_path(cfg["data_root"]).joinpath(name, *parts)


def repo_dir(cfg: dict, key: str) -> Path:
    """Folder for one of the small repo outputs in cfg['paths'], e.g. repo_dir(cfg, "runs_dir")."""
    return resolve_path(cfg["paths"][key])


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--dataset", help="dataset config in configs/, e.g. scifact or core18")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="override a config value, e.g. --set bm25.threads=4 (repeatable)",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> dict:
    return load_config(args.dataset, args.overrides)


def main(argv: list[str] | None = None) -> None:
    parser = add_config_args(argparse.ArgumentParser(description="Print the resolved config."))
    cfg = config_from_args(parser.parse_args(argv))
    print(yaml.safe_dump(cfg, sort_keys=False), end="")


if __name__ == "__main__":
    main()
