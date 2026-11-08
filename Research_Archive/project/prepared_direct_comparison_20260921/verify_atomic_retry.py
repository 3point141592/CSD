"""Reproduce a real Windows file-sharing lock in an isolated fixture."""

import ctypes
import threading
import time
import uuid
from ctypes import wintypes

from experiment import OUT, atomic_replace, write_json

folder = OUT / "verification" / ("atomic_retry_fixture_" + uuid.uuid4().hex[:8])
folder.mkdir(parents=True)
source = folder / "replacement.tmp"
destination = folder / "current.txt"
source.write_bytes(b"new complete contents")
destination.write_bytes(b"old complete contents")
kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.CreateFileW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.c_void_p,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.HANDLE,
]
kernel.CreateFileW.restype = wintypes.HANDLE
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
kernel.CloseHandle.restype = wintypes.BOOL
handle = kernel.CreateFileW(str(destination), 0x80000000, 1, None, 3, 0x80, None)
assert handle not in (None, ctypes.c_void_p(-1).value), ctypes.get_last_error()


def release():
    time.sleep(0.3)
    assert kernel.CloseHandle(handle)


thread = threading.Thread(target=release)
thread.start()
start = time.perf_counter()
atomic_replace(source, destination)
elapsed = time.perf_counter() - start
thread.join()
assert destination.read_bytes() == b"new complete contents" and not source.exists()
assert elapsed >= 0.25
write_json(
    OUT / "verification/atomic_retry.json",
    {
        "passed": True,
        "real_windows_no_delete_sharing_handle": True,
        "replacement_waited_seconds": elapsed,
        "complete_content_after_atomic_replace": True,
        "fixture": str(folder.relative_to(OUT)),
        "study_files_modified_by_test": False,
    },
)
print("Real Windows file-lock retry passed, seconds:", elapsed)
