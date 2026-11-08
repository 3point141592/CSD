"""Read-only source/result discovery; excludes manuscript bodies."""

import json
import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
skip = {".venv", "node_modules", ".git", "__pycache__", "site-packages"}
hits = []
archives = []
count = 0
errors = []
terms = [
    "bottleneck",
    "prepared_feature",
    "prepared feature",
    "prepared_direct",
    "direct_prepared",
    "shared_feature",
    "shared feature",
    "0.8333",
]
for base, dirs, files in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d not in skip and Path(base, d) != OUT]
    for name in files:
        p = Path(base, name)
        rel = str(p.relative_to(ROOT))
        count += 1
        if p.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(p) as z:
                    relevant = [
                        n
                        for n in z.namelist()
                        if any(t in n.lower() for t in terms[:7])
                    ]
                    archives.append(
                        {
                            "path": rel,
                            "members": len(z.namelist()),
                            "relevant": relevant,
                        }
                    )
                    for n in z.namelist():
                        if n.endswith(".py") and z.getinfo(n).file_size < 2_000_000:
                            s = z.read(n).decode("utf-8", errors="replace")
                            if any(t in s.lower() for t in terms):
                                hits.append(
                                    {
                                        "path": rel + "::" + n,
                                        "terms": [t for t in terms if t in s.lower()],
                                    }
                                )
            except Exception as e:
                errors.append([rel, str(e)])
        elif (
            p.suffix.lower() in {".py", ".json", ".csv"}
            and p.stat().st_size < 2_000_000
            and "research_audit" not in rel
            and "checkpoints\\llm" not in rel
        ):
            try:
                s = p.read_text(encoding="utf-8", errors="replace")
                found = [t for t in terms if t in s.lower()]
                if found:
                    hits.append({"path": rel, "terms": found})
            except Exception as e:
                errors.append([rel, str(e)])
result = {
    "files_enumerated": count,
    "hits": hits,
    "archives": archives,
    "errors": errors,
    "disk": dict(zip(["total", "used", "free"], shutil.disk_usage(ROOT))),
}
try:
    import psutil

    result["memory"] = dict(psutil.virtual_memory()._asdict())
    result["python_processes"] = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        if "python" in (p.info["name"] or "").lower():
            result["python_processes"].append(p.info)
except Exception as e:
    result["process_error"] = str(e)
(OUT / "discovery.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(
    json.dumps(
        {k: v for k, v in result.items() if k not in ("hits", "archives")}, indent=2
    )
)
print(
    "candidate source hits:",
    json.dumps([h for h in hits if h["path"].endswith(".py")], indent=2),
)
print("archives", len(archives), "all hits", len(hits))
