"""Build a source-only archive from an explicit inventory; never publish it."""

import argparse
import ast
import hashlib
import json
import pathlib
import sys
import zipfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from speedupmark.task import DEFAULT_CANDIDATE


ROOT_FILES = (".gitignore", "LICENSE", "README.md", "GUIDE.md",
              "THIRD_PARTY.json", "requirements-numerical.txt",
              "diagrams/luna-hermes-progress.png",
              "diagrams/luna-hermes-progress.svg",
              "diagrams/luna-hermes-progress.json")
PACKAGE_FILES = ("__init__.py", "__main__.py", "harness.py", "revision.py",
                 "run_manager.py", "run_prompt.txt", "suites.py", "task.py")
TASK_FILES = ("README.md", "candidate.py", "task_spec.py", "VALIDATION.md")


def public_payloads(root=ROOT):
    """Read only approved source paths; exclude local/private trees by construction."""
    root = pathlib.Path(root)
    names = list(ROOT_FILES) + [f"speedupmark/{name}" for name in PACKAGE_FILES]
    names += ["tasks/SHORTLIST.md", "scripts/prepare_public.py"]
    names += [path.relative_to(root).as_posix() for path in sorted((root / "tests").glob("*.py"))]
    candidates = set()
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
    parser.add_argument("destination", type=pathlib.Path)
    args = parser.parse_args()
    manifest = build_archive(args.destination)
    print(f"Prepared {args.destination}: {len(manifest['files'])} source files; publication is a separate action.")
