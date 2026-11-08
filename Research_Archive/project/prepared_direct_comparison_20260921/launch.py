"""Detached Windows launcher; normalize duplicate inherited Path/PATH keys."""

import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
env = {}
for key, value in os.environ.items():
    env[key.upper()] = value
stdout = OUT / f"worker_{stamp}.stdout.log"
stderr = OUT / f"worker_{stamp}.stderr.log"
command = [sys.executable, "-u", str(OUT / "experiment.py"), "run"]
startup = subprocess.STARTUPINFO()
startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
startup.wShowWindow = subprocess.SW_HIDE
with stdout.open("ab", buffering=0) as out, stderr.open("ab", buffering=0) as err:
    process = subprocess.Popen(
        command,
        cwd=OUT.parent,
        stdin=subprocess.DEVNULL,
        stdout=out,
        stderr=err,
        env=env,
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        startupinfo=startup,
        close_fds=True,
    )
record = {
    "launched_at": datetime.datetime.now().isoformat(),
    "launcher_pid": process.pid,
    "command": command,
    "command_line": subprocess.list2cmdline(command),
    "stdout": str(stdout),
    "stderr": str(stderr),
    "detached": True,
    "hidden": True,
    "environment_note": "Inherited duplicate Path/PATH casing normalized in child environment only.",
}
(OUT / f"launch_{stamp}.json").write_text(
    json.dumps(record, indent=2), encoding="utf-8"
)
print(json.dumps(record, indent=2))
