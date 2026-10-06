"""Prepare source-only archives or check staged starter candidates; never publish."""

import argparse
import ast
import hashlib
import json
import pathlib
import subprocess
import sys
import zipfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from speedupmark.task import DEFAULT_CANDIDATE


ROOT_FILES = (".gitignore", "LICENSE", "README.md", "GUIDE.md", "version.json",
              "THIRD_PARTY.json", "requirements-numerical.txt",
              "diagrams/luna-hermes-progress.png",
              "diagrams/luna-hermes-progress.svg",
              "diagrams/luna-hermes-progress.json",
              "diagrams/smoke-codex-luna-hermes-deepseek-high-20261002.png",
              "diagrams/smoke-codex-luna-hermes-deepseek-high-20261002.svg",
              "diagrams/smoke-codex-luna-hermes-deepseek-high-20261002.json")
PACKAGE_FILES = ("__init__.py", "__main__.py", "catalog.py", "harness.py", "revision.py",
                 "run_manager.py", "run_prompt.txt", "suites.py", "task.py")
TASK_FILES = ("README.md", "candidate.py", "task_spec.py", "VALIDATION.md")


def public_payloads(root=ROOT):
    """Read only approved source paths; exclude local/private trees by construction."""
    root = pathlib.Path(root)
    names = list(ROOT_FILES) + [f"speedupmark/{name}" for name in PACKAGE_FILES]
    names += ["tasks/SHORTLIST.md", "scripts/prepare_public.py", "scripts/sync_task_names.py"]
    names += [path.relative_to(root).as_posix() for path in sorted((root / "tests").glob("*.py"))]
    names += [f"examples/example_gzip/{name}" for name in ("README.md", "candidate.py", "task_spec.py")]
    candidates = {"examples/example_gzip/candidate.py"}
    for directory in sorted((root / "tasks").iterdir()):
        if not directory.is_dir() or not (directory / "task_spec.py").is_file():
            continue
        if directory.is_symlink():
            raise ValueError(f"public source cannot be a symlink: {directory}")
        tree = ast.parse((directory / "task_spec.py").read_text())
        # This inventory deliberately exports the shared starter. A new custom
        # submission interface needs explicit review instead of a guessed starter.
        if any(isinstance(node, (ast.FunctionDef, ast.Assign, ast.AnnAssign))
               and (getattr(node, "name", None) == "fresh_candidate"
                    or any(isinstance(target, ast.Name) and target.id == "fresh_candidate"
                           for target in getattr(node, "targets", ()))
                    or getattr(getattr(node, "target", None), "id", None) == "fresh_candidate")
               for node in ast.walk(tree)):
            raise ValueError(f"review custom starter before exporting {directory.name}")
        for name in TASK_FILES:
            path = directory / name
            if name != "VALIDATION.md" or path.is_file():
                names.append(path.relative_to(root).as_posix())
        candidates.add(f"tasks/{directory.name}/candidate.py")
    payloads = {}
    for name in sorted(names):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or linked public source: {name}")
        payloads[name] = (DEFAULT_CANDIDATE.encode() if name in candidates else path.read_bytes())
    return payloads


def check_staged_candidates(root=ROOT):
    """Reject non-starter candidate bytes in Git's index; never alter local files."""
    root = pathlib.Path(root)

    def git_output(*arguments):
        try:
            return subprocess.run(["git", *arguments], cwd=root, check=True,
                                  capture_output=True).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError("could not inspect the Git index for staged candidates") from error

    entries = git_output("ls-files", "--stage", "-z")
    count = 0
    invalid = []
    for entry in entries.decode("utf-8", "surrogateescape").split("\0"):
        if not entry:
            continue
        metadata, name = entry.split("\t", 1)
        parts = pathlib.PurePosixPath(name).parts
        if len(parts) < 3 or parts[0] not in ("tasks", "examples") or parts[-1] != "candidate.py":
            continue
        mode, object_id, stage = metadata.split()
        count += 1
        if (stage != "0" or mode not in ("100644", "100755")
                or git_output("cat-file", "blob", object_id) != DEFAULT_CANDIDATE.encode()):
            invalid.append(name)
    if invalid:
        raise ValueError("staged candidates must be fresh reference-delegating starters: "
                         + ", ".join(sorted(set(invalid))))
    if not count:
        raise ValueError("no staged candidate files found in the Git index")
    return count


def build_archive(destination, root=ROOT):
    payloads = public_payloads(root)
    manifest = {
        "format": "speedupmark-public-source-v1",
        "candidate_policy": "fresh reference-delegating starter",
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
    }
    payloads["PUBLIC_MANIFEST.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    destination = pathlib.Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents replacing an existing reviewed artifact.
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in payloads.items():
            item = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o100644 << 16
            archive.writestr(item, data)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=pathlib.Path, nargs="?")
    parser.add_argument("--check-index", action="store_true",
                        help="reject non-starter candidates staged for a Git release")
    args = parser.parse_args()
    if args.destination is None and not args.check_index:
        parser.error("supply an archive destination or --check-index")
    if args.check_index:
        try:
            count = check_staged_candidates()
        except ValueError as error:
            parser.error(str(error))
        print(f"Checked Git index: {count} fresh starter candidates.")
    if args.destination is not None:
        manifest = build_archive(args.destination)
        print(f"Prepared {args.destination}: {len(manifest['files'])} source files; publication is a separate action.")
