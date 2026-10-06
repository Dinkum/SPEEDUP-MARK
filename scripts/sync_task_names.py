"""Check or synchronize task README titles and the root README task catalog."""

import argparse
import ast
import difflib
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from speedupmark.catalog import TASK_CATALOG
from speedupmark.suites import EXTENDED_TASKS, SMOKE_TASKS


def validate_task_names(root=ROOT):
    """Validate identifiers without executing task or candidate code."""
    root = pathlib.Path(root)
    directories = {path.parent.name for path in (root / "tasks").glob("*/task_spec.py")}
    if directories != set(TASK_CATALOG):
        raise ValueError(f"task directories differ from catalog: {sorted(directories ^ set(TASK_CATALOG))}")
    if not set(SMOKE_TASKS) <= set(EXTENDED_TASKS) <= directories:
        raise ValueError("suite identifiers must belong to the task catalog")
    for name in TASK_CATALOG:
        source = root / "tasks" / name / "task_spec.py"
        tree = ast.parse(source.read_text(), filename=str(source))
        declared = [ast.literal_eval(node.value) for cls in tree.body
                    if isinstance(cls, ast.ClassDef) for node in cls.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "name"
                            for target in node.targets)]
        if declared != [name]:
            raise ValueError(f"{name}: spec must reference its catalog identifier")


def task_table():
    lines = ["| Task | Workload | Included in |", "| --- | --- | --- |"]
    for name, metadata in TASK_CATALOG.items():
        suites = (["smoke"] if name in SMOKE_TASKS else [])
        suites += (["extended"] if name in EXTENDED_TASKS else []) + ["all"]
        lines.append(f"| [{metadata.display_name}](tasks/{name}/README.md) "
                     f"| {metadata.workload} | {', '.join(suites)} |")
    return "\n".join(lines)


def documentation_updates(root=ROOT):
    """Return proposed content; callers decide which files to write."""
    root = pathlib.Path(root)
    validate_task_names(root)
    updates = {}
    for name, metadata in TASK_CATALOG.items():
        path = root / "tasks" / name / "README.md"
        source = path.read_text()
        heading, separator, body = source.partition("\n")
        if not heading.startswith("# ") or not separator:
            raise ValueError(f"{path}: expected a Markdown title")
        updated = f"# {metadata.display_name}\n{body}"
        if source != updated:
            updates[path] = updated
    path = root / "README.md"
    source = path.read_text()
    before, separator, tail = source.partition("| Task | Workload | Included in |\n")
    if not separator or not tail.startswith("| --- | --- | --- |\n"):
        raise ValueError("README.md: expected the task catalog table")
    _, separator, after = tail.partition("\n\n")
    if not separator:
        raise ValueError("README.md: expected content after the task catalog")
    updated = before + task_table() + "\n\n" + after
    if source != updated:
        updates[path] = updated
    return updates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="update task README titles")
    parser.add_argument("--write-root-readme", action="store_true",
                        help="also apply the root README diff after explicit user approval")
    args = parser.parse_args(argv)
    updates = documentation_updates()
    for path, updated in updates.items():
        relative = path.relative_to(ROOT).as_posix()
        if args.write_root_readme or (args.write and path != ROOT / "README.md"):
            path.write_text(updated)
            print(f"Updated {relative}")
        else:
            print("".join(difflib.unified_diff(path.read_text().splitlines(True),
                                             updated.splitlines(True),
                                             fromfile=relative, tofile=relative)), end="")
    remaining = documentation_updates()
    return 1 if remaining else 0


if __name__ == "__main__":
    raise SystemExit(main())
