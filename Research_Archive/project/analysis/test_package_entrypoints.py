"""Regression tests for package entry points, output paths and source integrity."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_scripts.reproduction_support import fresh_output, verify_recorded_sources


@pytest.mark.parametrize(
    "script",
    [
        "analysis/run_fair_selector_verification.py",
        "analysis/run_corruption_reachability_audit.py",
        "paper_csd_toy_revision5/scripts/run_reference_corruption.py",
        "run_scripts/smoke_train_matched.py",
    ],
)
def test_entrypoint_imports_from_unrelated_working_directory(script, tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(ROOT / script), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--output" in result.stdout


def test_output_cannot_overwrite_saved_evidence(tmp_path):
    with pytest.raises(FileExistsError):
        fresh_output(tmp_path)
    for name in [
        "results",
        "new_verification",
        "paper_csd_toy_revision5/assets/source_data",
    ]:
        with pytest.raises(ValueError):
            fresh_output(ROOT / name / "never_write_here")
    assert fresh_output(tmp_path / "fresh") == tmp_path / "fresh"


def test_changed_recorded_source_is_rejected(tmp_path):
    source = tmp_path / "code.py"
    source.write_text("original", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_hashes": {
                    "code.py": hashlib.sha256(source.read_bytes()).hexdigest()
                }
            }
        ),
        encoding="utf-8",
    )
    assert verify_recorded_sources(tmp_path, manifest)
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_recorded_sources(tmp_path, manifest)
