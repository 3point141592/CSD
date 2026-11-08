"""Retrain Direct-Vector models with wider output heads.

All model weights are optimized. The architecture intervention changes
proposal_hidden_dimension from 16 to d (32 or 64). The protocol includes
60 wider models, two narrow seed-7 checks, and evaluations of saved baselines.
This retrospective capacity ablation is not a parameter-matched architecture
comparison. Results go to a fresh output directory; supplied source files and
archived experiment records are preserved.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "frozen_ablation", ROOT / "analysis/run_direct_head_ablation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run(args.output, module.driver.configure_runtime(args.device))


if __name__ == "__main__":
    main()
