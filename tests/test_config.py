import argparse
from pathlib import Path

import pytest

from src.config import (
    DATA_ROOT_ENV, REPO_ROOT, add_config_args, config_from_args, data_dir,
    load_config, parse_override, repo_dir, resolve_path,
)


@pytest.fixture(autouse=True)
def no_data_root_env(monkeypatch):
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)


def test_default_has_preregistered_settings():
    cfg = load_config()
    assert cfg["bm25"]["k1"] == 0.9
    assert cfg["bm25"]["b"] == 0.4
    assert cfg["rrf"]["k"] == 60
    assert cfg["dense"]["model"] == "intfloat/e5-base-v2"
    assert cfg["dense"]["query_prefix"] == "query: "


@pytest.mark.parametrize("name", ["scifact", "core18"])
def test_dataset_config_merges_onto_default(name):
    cfg = load_config(name)
    assert cfg["dataset"]["name"] == name
    assert cfg["bm25"]["k1"] == 0.9  # defaults survive the merge


def test_unknown_dataset_lists_available():
    with pytest.raises(FileNotFoundError, match="scifact"):
        load_config("nope")


def test_overrides_are_typed():
    cfg = load_config("scifact", ["bm25.threads=4", "dense.fp16=false", "dense.revision=abc123"])
    assert cfg["bm25"]["threads"] == 4
    assert cfg["dense"]["fp16"] is False
    assert cfg["dense"]["revision"] == "abc123"


def test_unknown_override_key_raises():
    with pytest.raises(KeyError, match="bm25.k"):
        load_config(None, ["bm25.k=1.2"])


@pytest.mark.parametrize("bad", ["bm25.k1", "=3"])
def test_malformed_override_raises(bad):
    with pytest.raises(ValueError):
        parse_override(bad)


def test_env_sets_data_root_and_cli_wins(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    assert load_config()["data_root"] == str(tmp_path)
    assert load_config(None, ["data_root=/elsewhere"])["data_root"] == "/elsewhere"


def test_resolve_path():
    assert resolve_path("runs") == REPO_ROOT / "runs"
    assert resolve_path("/content/drive").as_posix() == "/content/drive"


def test_data_and_repo_dirs(tmp_path):
    cfg = load_config("scifact", [f"data_root={tmp_path.as_posix()}"])
    assert data_dir(cfg, "bm25_index") == Path(tmp_path) / "scifact" / "bm25_index"
    assert repo_dir(cfg, "runs_dir") == REPO_ROOT / "runs"


def test_data_dir_needs_dataset():
    with pytest.raises(ValueError, match="--dataset"):
        data_dir(load_config())


def test_cli_args():
    parser = add_config_args(argparse.ArgumentParser())
    args = parser.parse_args(["--dataset", "core18", "--set", "rrf.depth=100"])
    cfg = config_from_args(args)
    assert cfg["dataset"]["name"] == "core18"
    assert cfg["rrf"]["depth"] == 100
