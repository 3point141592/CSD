"""Compare Direct and Relational normalization statistics against a float64 reference."""

from experiment import *

check_lock()
device = configure_runtime("cuda")
verify_runtime_identity(runtime(device))
c, b, tr, h, eq, direct, stats, splits, hashes, seconds = data(
    64, 197, device, save=False
)
a = stats["DirectPrepared"][0][64:]
b = stats["Relational"][0]
sa = stats["DirectPrepared"][1][64:]
sb = stats["Relational"][1]
selected = eq.invariant[eq.mask]
reference = selected.double().mean(0)
sdref = selected.double().std(0, correction=0).clamp_min(1e-4)
rows = []
for i in range(len(a)):
    rows.append(
        {
            "index": i,
            "direct_mean": float(a[i]),
            "relational_mean": float(b[i]),
            "absolute_difference": float(abs(a[i] - b[i])),
            "double_precision_reference_mean": float(reference[i]),
            "direct_error_to_double": float(abs(a[i].double() - reference[i])),
            "relational_error_to_double": float(abs(b[i].double() - reference[i])),
            "direct_std": float(sa[i]),
            "relational_std": float(sb[i]),
            "double_precision_reference_std": float(sdref[i]),
            "difference_in_std_units": float(abs(a[i] - b[i]) / sb[i]),
        }
    )
result = {
    "dimension": 64,
    "seed": 197,
    "shared_phi_bitwise_identical": bool(
        torch.equal(direct.values[:, :, 64:], eq.invariant)
    ),
    "same_mask": bool(torch.equal(direct.mask, eq.mask)),
    "same_relative_vectors": bool(torch.equal(direct.values[:, :, :64], eq.relative)),
    "max_absolute_mean_difference": float(abs(a - b).max()),
    "max_relative_mean_difference": float((abs(a - b) / b.abs().clamp_min(1e-4)).max()),
    "max_difference_in_std_units": float((abs(a - b) / sb).max()),
    "max_std_absolute_difference": float(abs(sa - sb).max()),
    "rows": rows,
    "runtime": runtime(device),
}
write_json(OUT / "verification/normalization_assertion_d64_s197.json", result)
print({k: v for k, v in result.items() if k not in ["rows", "runtime"]})
print(
    "Largest mean differences:",
    sorted(rows, key=lambda x: x["absolute_difference"], reverse=True)[:5],
)
