"""Run a bounded training and evaluation check through the matched pipeline.

This is an implementation check, not a reproduction of the 30-seed result.
It trains both reported architectures at d=2 for one canonical seed and 20
optimizer updates, then evaluates IID and exactly rotated tasks.  Outputs must
go to a fresh directory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper_csd_toy_revision5" / "scripts" / "run_matched_scaling_study.py"
sys.path.insert(0, str(ROOT))


def load_driver():
    spec = importlib.util.spec_from_file_location("csd_matched_driver", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {output}")

    driver = load_driver()
    driver.STUDY_SEEDS = [7]
    original = driver.matched_config

    def smoke_config(dimension: int):
        config = original(dimension)
        config.seeds = [7]
        config.supervised_steps = 20
        config.curve_steps = [0, 1, 5, 10, 20]
        return config

    driver.matched_config = smoke_config
    device = driver.configure_runtime(args.device)
    runtime = driver.runtime_info(device, driver.torch.float32)
    completion = driver.run_dimension(
        2, output_root=output, device=device, runtime=runtime
    )
    status = {
        "purpose": "bounded training/evaluation implementation smoke test",
        "not_a_claim_reproduction": True,
        "dimension": 2,
        "seed": 7,
        "conditions": list(driver.CONDITIONS),
        "optimizer_updates": 20,
        "completion": completion,
    }
    (output / "SMOKE_STATUS.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
