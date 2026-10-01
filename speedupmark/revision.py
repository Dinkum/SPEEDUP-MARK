"""Content identity for the files that define a task and its grading contract."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib


# These files affect direct grading, managed grading, or the agent handoff.
SHARED_FILES = (
    "__init__.py", "__main__.py", "harness.py", "revision.py",
    "run_manager.py", "run_prompt.txt", "suites.py", "task.py",
)


def task_revision(task_dir: pathlib.Path, package_dir: pathlib.Path) -> tuple[str, dict[str, str]]:
    """Return a stable digest and its path-to-content-hash input manifest.

    The candidate is a submission, so its changing bytes do not redefine the
    benchmark. Python caches are likewise excluded. Paths are relative to the
    benchmark root, making the digest independent of checkout and run paths.
    """
    task_dir = pathlib.Path(task_dir)
    package_dir = pathlib.Path(package_dir)
    if not (task_dir / "task_spec.py").is_file():
        raise ValueError(f"{task_dir.name}: task_spec.py is required for a revision")
    files = {}
    for path in sorted(task_dir.rglob("*")):
        relative = path.relative_to(task_dir)
        if (relative.parts[0] == "__pycache__" or "__pycache__" in relative.parts
                or path.suffix == ".pyc" or relative.as_posix() == "candidate.py"):
            continue
        if path.is_symlink():
            digest = "symlink:" + os.readlink(path)
        elif path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            continue
        files[f"tasks/{task_dir.name}/{relative.as_posix()}"] = digest
    for name in SHARED_FILES:
        path = package_dir / name
        files[f"speedupmark/{name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    # A domain tag prevents this identifier from being confused with a hash of
    # raw concatenated files or a different future manifest format.
    payload = json.dumps({"format": "speedupmark-task-revision-v1", "files": files},
                         sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest(), files
