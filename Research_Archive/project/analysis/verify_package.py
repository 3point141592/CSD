"""Verify a relocated package, optional ZIP, and checkpoint hashes without imports.

The file manifest excludes its own CSV and subsequent reviewer outputs. Choose
an output directory outside the package when checking a frozen delivery.
"""

import argparse
import collections
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--package-root", type=Path, default=Path(__file__).resolve().parents[1]
)
parser.add_argument("--output-dir", type=Path)
parser.add_argument(
    "--archive", type=Path, help="Optional archive to check against the same manifest"
)
parser.add_argument(
    "--checkpoint-manifest",
    type=Path,
    help="Relative to package root or an absolute CSV path",
)
args = parser.parse_args()
ROOT = args.package_root.resolve(strict=True)
OUT = (args.output_dir or ROOT / "new_verification/audit_20260909").resolve()
OUT.mkdir(parents=True, exist_ok=True)
dest = OUT / "integrity_results.json"
if dest.exists():
    raise FileExistsError(f"Refusing to replace {dest}; use a fresh --output-dir")


def sha(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()


def hash_file(path):
    with path.open("rb") as stream:
        return sha(stream)


def safe_member(raw):
    path = PurePosixPath(raw.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or ":" in raw:
        raise ValueError(f"Unsafe manifest path: {raw}")
    real = (ROOT / str(path)).resolve()
    if not real.is_relative_to(ROOT):
        raise ValueError(f"Path escapes package: {raw}")
    return path.as_posix(), real


manifest = ROOT / "FILE_MANIFEST.csv"
with manifest.open(encoding="utf-8-sig", newline="") as stream:
    rows = list(csv.DictReader(stream))
files = {p.relative_to(ROOT).as_posix(): p for p in ROOT.rglob("*") if p.is_file()}
missing = []
hashes = []
sizes = []
duplicates = []
seen = set()
for row in rows:
    name, path = safe_member(row["relative_path"])
    if name in seen:
        duplicates.append(name)
    seen.add(name)
    if not path.is_file():
        missing.append(name)
        continue
    if path.stat().st_size != int(row["bytes"]):
        sizes.append(name)
    if hash_file(path).lower() != row["sha256"].lower():
        hashes.append(name)
unlisted = sorted(set(files) - seen - {"FILE_MANIFEST.csv"})
result = {
    "manifest_rows": len(rows),
    "actual_files": len(files),
    "missing": missing,
    "hash_mismatches": hashes,
    "size_mismatches": sizes,
    "duplicate_manifest_paths": duplicates,
    "unlisted": unlisted,
    "manifest_excludes_itself": True,
    "scope": "File identity and declared completeness only; not scientific validity or experiment execution.",
}

checkpoint_path = args.checkpoint_manifest
if checkpoint_path is None:
    for candidate in [
        "CHECKPOINT_MANIFEST.csv",
        "original_results/recovered_attachment/manifests/checkpoints.csv",
    ]:
        if (ROOT / candidate).is_file():
            checkpoint_path = ROOT / candidate
            break
elif not checkpoint_path.is_absolute():
    checkpoint_path = ROOT / checkpoint_path
if checkpoint_path is not None:
    with checkpoint_path.open(encoding="utf-8-sig", newline="") as stream:
        checkpoints = list(csv.DictReader(stream))
    cp_missing = []
    cp_hashes = []
    cp_sizes = []
    for row in checkpoints:
        raw = row.get("relative_path") or row.get("repository_path")
        if not raw:
            raise ValueError("Checkpoint CSV requires relative_path or repository_path")
        name, path = safe_member(raw)
        if not path.is_file():
            cp_missing.append(name)
            continue
        if row.get("bytes") and path.stat().st_size != int(row["bytes"]):
            cp_sizes.append(name)
        if hash_file(path).lower() != row["sha256"].lower():
            cp_hashes.append(name)
    result["checkpoints"] = {
        "rows": len(checkpoints),
        "missing": cp_missing,
        "hash_mismatches": cp_hashes,
        "size_mismatches": cp_sizes,
        "scope": "One canonical location per checkpoint; no duplicate-collection requirement",
    }
else:
    result["checkpoints"] = {
        "status": "No checkpoint manifest found; file manifest still checks included checkpoint bytes"
    }

if args.archive:
    with zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        candidates = [
            n
            for n in names
            if n == "FILE_MANIFEST.csv" or n.endswith("/FILE_MANIFEST.csv")
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Archive must contain exactly one top-level FILE_MANIFEST.csv"
            )
        manifest_member = candidates[0]
        prefix = manifest_member[: -len("FILE_MANIFEST.csv")]
        zm = []
        zh = []
        zs = []
        for row in rows:
            name, _ = safe_member(row["relative_path"])
            member = prefix + name
            if member not in names:
                zm.append(member)
                continue
            if archive.getinfo(member).file_size != int(row["bytes"]):
                zs.append(member)
            with archive.open(member) as stream:
                if sha(stream).lower() != row["sha256"].lower():
                    zh.append(member)
        zextra = sorted(
            set(n for n in names if not n.endswith("/"))
            - {prefix + n for n in seen}
            - {manifest_member}
        )
        result["archive"] = {
            "bytes": args.archive.stat().st_size,
            "members": len(names),
            "duplicates": [
                n for n, count in collections.Counter(names).items() if count > 1
            ],
            "missing": zm,
            "hash_mismatches": zh,
            "size_mismatches": zs,
            "unlisted": zextra,
            "manifest_equal": archive.read(manifest_member) == manifest.read_bytes(),
            "member_prefix": prefix,
        }
else:
    result["archive"] = {"status": "Not requested; no sibling ZIP filename is assumed"}

issues = any(
    result[key]
    for key in [
        "missing",
        "hash_mismatches",
        "size_mismatches",
        "duplicate_manifest_paths",
        "unlisted",
    ]
)
issues |= any(
    result["checkpoints"].get(key, [])
    for key in ["missing", "hash_mismatches", "size_mismatches"]
)
issues |= any(
    result["archive"].get(key, [])
    for key in [
        "duplicates",
        "missing",
        "hash_mismatches",
        "size_mismatches",
        "unlisted",
    ]
)
issues |= result["archive"].get("manifest_equal", True) is False
result["passed"] = not issues
dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
sys.exit(1 if issues else 0)
