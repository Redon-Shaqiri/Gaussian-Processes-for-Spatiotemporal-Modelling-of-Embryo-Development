"""Tests for gpembryos.utils."""

from pathlib import Path

import pytest

from gpembryos.utils import get_artifact_dir, update_latest_symlink


def test_get_artifact_dir_creates_fresh_run(tmp_path: Path):
    artifact_dir, is_resume = get_artifact_dir(tmp_path, "exp999", resume=None)
    assert artifact_dir.exists()
    assert artifact_dir.parent == tmp_path / "artifacts" / "exp999"
    assert is_resume is False


def test_update_latest_symlink_points_to_run(tmp_path: Path):
    artifact_dir, _ = get_artifact_dir(tmp_path, "exp999", resume=None)
    update_latest_symlink(artifact_dir)
    latest = artifact_dir.parent / "latest"
    assert latest.is_symlink()
    assert latest.resolve() == artifact_dir.resolve()


def test_resume_latest_resolves_to_most_recent(tmp_path: Path):
    artifact_dir, _ = get_artifact_dir(tmp_path, "exp999", resume=None)
    update_latest_symlink(artifact_dir)
    resumed, is_resume = get_artifact_dir(tmp_path, "exp999", resume="latest")
    assert resumed.resolve() == artifact_dir.resolve()
    assert is_resume is True


def test_resume_specific_timestamp(tmp_path: Path):
    artifact_dir, _ = get_artifact_dir(tmp_path, "exp999", resume=None)
    ts = artifact_dir.name
    resumed, is_resume = get_artifact_dir(tmp_path, "exp999", resume=ts)
    assert resumed == artifact_dir
    assert is_resume is True


def test_resume_missing_run_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        get_artifact_dir(tmp_path, "exp999", resume="2099-01-01_000000")


def test_resume_latest_without_symlink_raises(tmp_path: Path):
    (tmp_path / "artifacts" / "exp999").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        get_artifact_dir(tmp_path, "exp999", resume="latest")
