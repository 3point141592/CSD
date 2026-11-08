"""Start the resumable fixed study as a detached hidden Windows process."""

import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

outdir = Path(__file__).resolve().parent
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
environment = {key.upper(): value for key, value in os.environ.items()}
stdout = outdir / f"worker_{stamp}.stdout.log"
stderr = outdir / f"worker_{stamp}.stderr.log"
command = [sys.executable, "-u", str(outdir / "run.py"), "run"]
startup = subprocess.STARTUPINFO()
startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
startup.wShowWindow = subprocess.SW_HIDE
with stdout.open("ab", buffering=0) as out, stderr.open("ab", buffering=0) as err:
    process = subprocess.Popen(
        command,
        cwd=outdir.parent,
        stdin=subprocess.DEVNULL,
        stdout=out,
        stderr=err,
        env=environment,
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        startupinfo=startup,
        close_fds=True,
    )
record = {
    "launched_at": datetime.datetime.now().isoformat(),
    "pid": process.pid,
    "command": command,
    "command_line": subprocess.list2cmdline(command),
    "stdout": str(stdout),
    "stderr": str(stderr),
    "detached": True,
    "hidden": True,
}
(outdir / f"launch_{stamp}.json").write_text(
    json.dumps(record, indent=2), encoding="utf-8"
)
print(json.dumps(record, indent=2))
