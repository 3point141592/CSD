"""CPU checks of untrained models and data before GPU training."""

import gc
import time

import torch
from experiment import *

check_lock()
results = []
device = torch.device("cpu")
for d in (32, 64):
    start = time.perf_counter()
    c, b, tr, h, eq, direct, stats, splits, feature_hashes, prep = data(
        d, 7, device, save=False
    )
    checks = verify_features(c, tr, h, eq, direct, stats, 7)
    checks["preflight_wall_seconds"] = time.perf_counter() - start
    checks["training_updates"] = 0
    results.append({"dimension": d, **checks})
    print(d, checks, flush=True)
    del tr, h, eq, direct, stats, splits
    gc.collect()
write_json(
    OUT / "verification/cpu_untrained_preflight.json",
    {
        "passed": True,
        "utc": now(),
        "results": results,
        "note": "No training; CPU normalization/features not reused in GPU study.",
    },
)
