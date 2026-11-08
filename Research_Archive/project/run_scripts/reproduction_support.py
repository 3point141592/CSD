"""Small package-relative safety and integrity checks for reproducibility jobs."""

import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fresh_output(path):
    path = Path(path).resolve()
    root = Path(__file__).resolve().parents[1]
    # Reject output paths within scientific inputs or the manuscript to prevent
    # reproduction runs from overwriting the evidence they are meant to check.
    for name in [
        "results",
        "original_results",
        "checkpoints",
        "new_verification",
        "manuscript",
        "paper_reproduction",
        "experiment_code",
        "paper_csd_toy_revision5/assets/source_data",
        "creativity_toy",
        "gen3_gpu",
        "analysis",
        "run_scripts",
        "environment",
        "development_archive",
    ]:
        protected = root / name
        if path == protected or protected in path.parents:
            raise ValueError(f"output targets preserved evidence: {path}")
    if path.exists():
        raise FileExistsError(f"choose a new, nonexistent output directory: {path}")
    return path


def verify_recorded_sources(root, manifest):
    root = Path(root).resolve()
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    expected = {
        **payload.get("source_hashes", {}),
        **payload.get("checkpoint_hashes_before", {}),
    }
    checked = {}
    for name, wanted in expected.items():
        path = (root / Path(name.replace("\\", "/"))).resolve()
        if path != root and root not in path.parents:
            raise ValueError(f"input escapes package: {name}")
        actual = sha256(path)
        if actual.lower() != wanted.lower():
            raise ValueError(f"recorded input hash mismatch: {name}")
        checked[name] = actual
    return checked
