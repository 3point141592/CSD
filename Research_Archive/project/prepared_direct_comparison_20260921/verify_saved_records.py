"""Independent CPU replay from saved NPZ records and completed pilot checkpoints."""

import dataclasses

from experiment import *
from gen3_gpu.study_symmetry import _benchmark
from gen3_gpu.task_batch import TorchTaskBatch

check_lock()
records = []
for d in (32, 64):
    folder = OUT / "data" / f"d{d}_s7"
    prov = read_json(folder / "provenance.json")
    assert sha(folder / "records.npz") == prov["records_sha256"]
    with np.load(folder / "records.npz") as arrays:
        values = {}
        for f in dataclasses.fields(TorchTaskBatch):
            arr = arrays["test_" + f.name]
            values[f.name] = (
                float(arr)
                if f.name == "region_threshold"
                else torch.from_numpy(arr.copy())
            )
        unique = TorchTaskBatch(**values)
        tasks = unique.repeat_interleave(2)
        h = torch.from_numpy(arrays["test_starts"].copy()).reshape(-1, d)
        ht = torch.from_numpy(arrays["train_starts"].copy()).reshape(-1, d)
        assert tensor_hash(h) == prov["feature_hashes"]["test_starts"]
        assert tensor_hash(ht) == prov["feature_hashes"]["train_starts"]
    c = matched_config(d)
    b = _benchmark(c)
    for kind in MODELS:
        job = OUT / f"jobs/d{d}/seed_7/{kind}"
        payload = torch.load(
            job / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        state = payload["state_dict"]
        prefix = "descriptor" if kind == "DirectPrepared" else "invariant"
        stats = state[prefix + "_mean"], state[prefix + "_scale"]
        assert tensor_hash(stats[0]) == prov["feature_hashes"][kind + "_0"]
        assert tensor_hash(stats[1]) == prov["feature_hashes"][kind + "_1"]
        model = factory(kind, c, 7, torch.device("cpu"), stats)
        model.load_state_dict(state)
        model.eval()
        summary, _ = evaluate(kind, model, tasks, h, b, c)
        primary = next(
            r for r in read_json(job / "metrics.json") if r["split"] == "iid"
        )
        assert abs(summary["mode_set_recall"] - primary["mode_set_recall"]) < 1e-6
        records.append(
            {
                "dimension": d,
                "model": kind,
                "saved_start_hashes_match": True,
                "checkpoint_normalization_matches_training_only_feature_hashes": True,
                "npz_only_cpu_coverage": summary["mode_set_recall"],
                "primary_gpu_coverage": primary["mode_set_recall"],
                "all_metric_differences": {
                    k: summary[k] - primary[k] for k in summary if k in primary
                },
            }
        )
write_json(
    OUT / "verification/saved_npz_cpu_replay.json",
    {
        "passed": True,
        "utc": now(),
        "training_updates": 0,
        "records": records,
        "note": "Independent saved-record replay. Primary results remain the contemporaneous GPU evaluations.",
    },
)
print([(r["dimension"], r["model"], r["npz_only_cpu_coverage"]) for r in records])
