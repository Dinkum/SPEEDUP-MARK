"""Run manager: one agent attempt per task, with snapshots and evaluation history.

Run workspaces and snapshots are ordinary files, not a security sandbox.
Agents can be any noninteractive command; the prompt is supplied on stdin,
through environment paths, and optionally through explicit argv placeholders.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import datetime
import fcntl
import hashlib
import json
import math
import os
import pathlib
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import time
import uuid
from types import SimpleNamespace

from .harness import TASK_ROOT, discover_tasks, load_task, resolve_seed
from .revision import task_revision
from .suites import selected_tasks
from .task import declared_task_version, grading_cases


ROOT = TASK_ROOT.parent


def _write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _manifest(run):
    return json.loads((run / "manifest.json").read_text())


def _utc(timestamp):
    return datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc).isoformat()


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy(source, destination):
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _inventory(root, *, exclude=()):
    """Hash file contents without following symlinks or including Python caches."""
    found = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        name = relative.as_posix()
        if ("__pycache__" in relative.parts or path.suffix == ".pyc"
                or any(name == item or name.startswith(item + "/") for item in exclude)):
            continue
        if path.is_symlink():
            found[name] = "symlink:" + os.readlink(path)
        elif path.is_file():
            found[name] = _hash(path)
    return found


def _differences(expected, actual, label):
    return [f"{label}: changed, missing, or unexpected file {name}"
            for name in sorted(expected.keys() | actual.keys())
            if expected.get(name) != actual.get(name)]


def _controller_log(name):
    return name.startswith("grader-") and name.endswith(".stderr.txt")


def _integrity_errors(run, manifest):
    if manifest.get("version") != 3:
        raise ValueError("this run predates separate agent workspaces; create a fresh run")
    candidate = f"tasks/{manifest['task']}/candidate.py"
    expected = manifest["baseline_files"]
    errors = _differences(expected, _inventory(run / "baseline"), "baseline")
    controller_hash = manifest.get("task_revision_files", {}).get("speedupmark/run_manager.py")
    if controller_hash and _hash(pathlib.Path(__file__)) != controller_hash:
        errors.append("controller: run_manager.py changed since run creation")
    errors += _differences(
        manifest["workspace_files"],
        _inventory(run / "workspace", exclude=(candidate, "scratch")), "workspace",
    )
    submitted = run / "workspace" / candidate
    if not submitted.is_file() or submitted.is_symlink():
        errors.append("candidate.py must be a regular file")
    return errors


@contextlib.contextmanager
def _locked(run):
    # Serialize evaluations so two grading commands cannot overlap CPU work or
    # overwrite a checkpoint. Locking is coordination, not tamper protection.
    with (run / "evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def history(run):
    path = run / "evaluations.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _verified_development(rows):
    return [row for row in rows if row.get("phase") == "development"
            and row.get("correct") is True and row.get("integrity") == "passed"
            and type(row.get("speedup")) in (int, float)
            and math.isfinite(row["speedup"]) and row["speedup"] > 0]


def _elapsed(manifest):
    if manifest["started_at"] is None:
        return 0.0
    return max(0.0, time.time() - manifest["started_at"])


def _optimizer(harness, model, effort, runtime_notes):
    fields = {"harness": harness, "model": model, "effort": effort}
    for name, value in fields.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"run requires a nonempty optimizer {name}")
        fields[name] = value.strip()
    if runtime_notes is not None:
        if not isinstance(runtime_notes, str) or not runtime_notes.strip():
            raise ValueError("optimizer runtime notes must be nonempty when supplied")
        fields["runtime_notes"] = runtime_notes.strip()
    return fields


def _reported_optimizer(manifest):
    if isinstance(manifest.get("optimizer"), dict):
        return manifest["optimizer"]
    return {
        "harness": "legacy-unrecorded",
        "model": "unknown",
        "effort": "unknown",
        "runtime_notes": f"legacy agent label: {manifest.get('agent', 'unknown')}",
    }


def _task_versions(tasks):
    """Read literal version metadata without importing task implementations."""
    versions = {}
    for name in tasks:
        source = TASK_ROOT / name / "task_spec.py"
        if not source.is_file():
            continue
        tree = ast.parse(source.read_text(), filename=str(source))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            attributes = {}
            for statement in node.body:
                if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Constant):
                    for target in statement.targets:
                        if isinstance(target, ast.Name):
                            attributes[target.id] = statement.value.value
            if "name" not in attributes and "task_version" not in attributes:
                continue
            if attributes.get("name") != name:
                raise ValueError(f"{name}: task name must match its directory identifier")
            versions[name] = declared_task_version(SimpleNamespace(
                name=name, task_version=attributes.get("task_version")
            ))
            break
        if name not in versions:
            raise ValueError(f"{name}: task spec does not declare a versioned task")
    return versions


def _task_revisions(tasks):
    """Identify each selected task before a suite starts, including unrun tasks."""
    return {name: task_revision(TASK_ROOT / name, ROOT / "speedupmark")[0]
            for name in tasks}


def create_run(task, *, harness=None, model=None, effort=None, runtime_notes=None,
               root=None, seed=None, samples=3, n=None,
               grade_timeout=60.0, safety_timeout=0.0, start=True):
    optimizer = _optimizer(harness, model, effort, runtime_notes)
    if seed is not None:
        resolve_seed(seed)
    if samples < 1 or (n is not None and n < 1):
        raise ValueError("samples and n must be positive")
    if not math.isfinite(grade_timeout) or grade_timeout <= 0:
        raise ValueError("grade timeout must be positive and finite")
    if not math.isfinite(safety_timeout) or safety_timeout < 0:
        raise ValueError("safety timeout must be nonnegative and finite; 0 disables it")
    source = TASK_ROOT / task
    if source not in discover_tasks():
        raise ValueError(f"{task}: select an implemented task name")
    task_definition = load_task(source, with_candidate=False)
    task_version = declared_task_version(task_definition)
    run_root = pathlib.Path(root or ROOT / "runs").resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    run = run_root / f"{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%SZ}-{task}-{uuid.uuid4().hex[:8]}"
    workspace = run / "workspace"
    destination = workspace / "tasks" / task
    destination.mkdir(parents=True)

    # Copy the single maintained reference; never inherit a source-checkout
    # submission or expose the controller's specification in the workspace.
    candidate = destination / "candidate.py"
    shutil.copyfile(source / "reference.py", candidate)
    readme = source / "README.md"
    if readme.exists():
        # Source-only generator links and direct-grading commands do not apply
        # inside the standalone agent workspace. Preserve the full task contract.
        sections = readme.read_text().split("\n## ")
        task_readme = "\n## ".join(section for section in sections
                                   if not section.startswith("Grading\n"))
        task_readme = "\n\n".join(paragraph for paragraph in task_readme.split("\n\n")
                                    if not paragraph.startswith("The executable definition is "))
    else:
        task_readme = f"# {task}\n\nOptimize `tasks/{task}/candidate.py` and check it with `grade.py`.\n"
    task_readme += (
        f"\nTask contract version: `{task_version}`.\n"
        f"\n## Managed run\n\nEdit only `tasks/{task}/candidate.py`. "
        "Use the standard library, plus libraries the task README lists as already installed. Do not install dependencies. "
        "Keep temporary files under `scratch/`. Do not change `grade.py` or run records; "
        "benchmark file changes invalidate the run.\n\n"
        "Run `python3 grade.py` from this workspace to grade and automatically save progress.\n"
    )
    (workspace / "README.md").write_text(task_readme)
    notice = (
        f"The launcher has a {safety_timeout:g}-second wall-clock safety timeout; "
        "it may stop a stuck or unfinished process."
        if safety_timeout else "The launcher has no wall-clock safety timeout."
    )
    if start:
        notice = "This is a manually managed run; its owner controls completion and stopping."
    prompt = (ROOT / "speedupmark" / "run_prompt.txt").read_text().format(
        candidate=f"tasks/{task}/candidate.py", python=sys.executable,
        safety_timeout=notice,
    )
    (workspace / "prompt.md").write_text(prompt)
    (workspace / "AGENTS.md").write_text(prompt)
    # Only this launcher enters the workspace. The frozen benchmark and
    # measurement runner stay outside it; this is not a permissions boundary.
    (workspace / "grade.py").write_text(
        "import subprocess\nimport sys\n\n"
        f"raise SystemExit(subprocess.call([{sys.executable!r}, '-m', 'speedupmark', "
        f"'run', 'evaluate', {str(run)!r}], cwd={str(ROOT)!r}))\n"
    )
    # Freeze the benchmark directly from controller-owned source rather than
    # deriving it from agent-visible files.
    baseline = run / "baseline"
    baseline_task = baseline / "tasks" / task
    baseline_task.parent.mkdir(parents=True)
    _copy(source, baseline_task)
    shutil.copyfile(source / "reference.py", baseline_task / "candidate.py")
    _copy(ROOT / "speedupmark", baseline / "speedupmark")
    revision, revision_files = task_revision(baseline / "tasks" / task,
                                             baseline / "speedupmark")
    now = time.time()
    manifest = {
        "version": 3, "run_id": run.name, "task": task,
        "task_version": task_version, "task_revision": revision,
        "task_revision_files": revision_files, "optimizer": optimizer,
        "created_at": now, "created_utc": _utc(now),
        "started_at": now if start else None, "finished_at": None,
        "agent_elapsed_seconds": None, "status": "running" if start else "prepared",
        "seed": seed, "samples": samples, "n": n,
        "grade_timeout_seconds": grade_timeout,
        "safety_timeout_seconds": safety_timeout,
        "starter_sha256": _hash(candidate),
        "reference_sha256": _hash(baseline_task / "reference.py"),
        "prompt_sha256": _hash(workspace / "prompt.md"),
        "submission_policy": "best_verified",
        "harness_sha256": _hash(baseline / "speedupmark" / "harness.py"),
        "python": platform.python_version(), "platform": platform.platform(),
        "command": None, "usage": None,
        "baseline_files": _inventory(baseline),
        "workspace_files": _inventory(workspace, exclude=(f"tasks/{task}/candidate.py", "scratch")),
        "integrity_errors": [],
    }
    _write_json(run / "manifest.json", manifest)
    return run


def _grading_sizes(snapshot, task_name, override):
    """Sizes for one evaluation. An explicit ``n`` replaces the spec's cases."""
    if override is not None:
        return (override,)
    # Metadata must not execute candidate imports in the controller. All task
    # execution belongs to the isolated worker, after the seed plan is saved.
    source = snapshot / "tasks" / task_name / "task_spec.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        attributes = {}
        for statement in node.body:
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name) and target.id in ("name", "grading_cases"):
                        attributes[target.id] = ast.literal_eval(statement.value)
        if attributes.get("name") == task_name:
            return grading_cases(SimpleNamespace(name=task_name, **{
                "grading_cases": attributes.get("grading_cases")
            }))
    raise ValueError(f"{task_name}: task spec must declare literal grading_cases")


def _evaluate_locked(run, manifest, phase):
    if manifest["status"] != "running":
        raise ValueError("only a running run can be evaluated")
    # A killed agent may leave an unfinished snapshot without a history row.
    # Preserve it and allocate a new number when final grading resumes.
    snapshot_root = run / "snapshots"
    numbers = [int(path.name.split("-")[0]) for path in snapshot_root.glob("[0-9]*-*")
               if path.name.split("-")[0].isdigit()]
    index = max(numbers, default=0) + 1
    snapshot = run / "snapshots" / f"{index:04d}-{phase}"
    workspace = run / "workspace"
    task = manifest["task"]
    candidate = snapshot / "tasks" / task / "candidate.py"
    started = _elapsed(manifest)
    reports = []
    integrity_errors = list(dict.fromkeys(manifest["integrity_errors"] + _integrity_errors(run, manifest)))
    selected = (max(_verified_development(history(run)), key=lambda row: row["speedup"], default=None)
                if phase == "final" else None)
    source = workspace / "tasks" / task / "candidate.py"
    if selected is not None:
        source = run / selected["snapshot"] / "tasks" / task / "candidate.py"
        # Regrade the recorded bytes, not a later workspace edit or a changed
        # checkpoint. Final measurements must independently validate selection.
        if (not source.is_file() or source.is_symlink()
                or not source.resolve().is_relative_to((run / "snapshots").resolve())
                or _hash(source) != selected["candidate_sha256"]):
            integrity_errors.append("selected candidate snapshot changed or missing")
    errors = list(integrity_errors)
    snapshot_files = None
    if not errors:
        snapshot.parent.mkdir(exist_ok=True)
        _copy(run / "baseline", snapshot)
        candidate.write_bytes(source.read_bytes())
        snapshot_files = _inventory(snapshot)
    else:
        snapshot.mkdir(parents=True)
    sizes = []
    if not errors:
        try:
            sizes = list(_grading_sizes(snapshot, task, manifest["n"]))
        except Exception as exc:
            # A spec that cannot be imported or does not declare cases fails
            # this evaluation. It must not fall back to a controller-chosen size.
            errors.append(f"grading cases: {exc}")
    # Use the same sampling rule in both phases. An explicit seed is a replay
    # override; otherwise every evaluation draws a new base, including development.
    seed_base = resolve_seed(manifest["seed"])
    seed_source = "random" if manifest["seed"] is None else "explicit"
    seed_plan = {
        "seed": seed_base, "seed_source": seed_source,
        "samples": manifest["samples"],
        "cases": [{"n": size, "seed": seed_base + case * manifest["samples"]}
                  for case, size in enumerate(sizes)],
    }
    # Persist before candidate execution so timeouts and crashes can be replayed.
    plans = run / "grading-plans"
    plans.mkdir(exist_ok=True)
    plan_path = plans / f"{index:04d}-{phase}.json"
    _write_json(plan_path, seed_plan)
    verification_logs = []
    for case, size in enumerate(sizes):
        seed = seed_base + case * manifest["samples"]
        verification_log = plans / f"{index:04d}-{phase}-{case}-verification.jsonl"
        verification_logs.append(str(verification_log.relative_to(run)))
        command = [sys.executable, "-B", "-m", "speedupmark", task, "--json",
                   "--seed", str(seed), "--samples", str(manifest["samples"]),
                   "--timeout", str(manifest["grade_timeout_seconds"]),
                   "--verification-log", str(verification_log.resolve()),
                   "--n", str(size)]
        try:
            # The inner task timeout includes validation. The outer grace also
            # bounds a broken command, while leaving time to serialize its result.
            completed = subprocess.run(
                command, cwd=snapshot, capture_output=True, text=True,
                timeout=manifest["grade_timeout_seconds"] + 5,
            )
            if completed.stderr:
                (snapshot / f"grader-{case}.stderr.txt").write_text(completed.stderr)
            report = json.loads(completed.stdout)
            if (
                completed.returncode not in (0, 1)
                or not isinstance(report, dict)
                or not isinstance(report.get("results"), list)
                or len(report["results"]) > 1
                or (report["results"] and (
                    not isinstance(report["results"][0], dict)
                    or report["results"][0].get("task") != task
                    or report["results"][0].get("task_version") != manifest.get("task_version")
                    or report["results"][0].get("task_revision") != manifest.get("task_revision")
                ))
                or (not report["results"] and not report.get("errors"))
                or type(report.get("correct")) is not bool
                or (report["correct"] and (
                    not isinstance(report.get("geomean_speedup"), (int, float))
                    or not math.isfinite(report["geomean_speedup"])
                    or report["geomean_speedup"] <= 0
                ))
            ):
                raise ValueError("grader returned an invalid report")
            reports.append(report)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            errors.append(str(exc))
            break
    # Check again after execution: candidate code can write files too. Persist
    # violations so restoring a file cannot rehabilitate an already-invalid run.
    integrity_errors += _integrity_errors(run, manifest)
    if snapshot_files is not None:
        actual = _inventory(snapshot)
        # Grader diagnostics are controller artifacts, not submitted code.
        actual = {name: digest for name, digest in actual.items() if not _controller_log(name)}
        integrity_errors += _differences(snapshot_files, actual, "grading snapshot")
    integrity_errors = sorted(set(integrity_errors))
    if integrity_errors:
        manifest["integrity_errors"] = integrity_errors
        _write_json(run / "manifest.json", manifest)
        errors = list(dict.fromkeys(errors + integrity_errors))
    correct = (not errors and bool(sizes) and len(reports) == len(sizes)
               and all(report["correct"] for report in reports))
    speedup = (
        math.exp(statistics.fmean(math.log(report["geomean_speedup"]) for report in reports))
        if correct else 0.0
    )
    row = {
        "task": task, "task_version": manifest.get("task_version"),
        "task_revision": manifest.get("task_revision"),
        "evaluation": index, "phase": phase, "started_elapsed_seconds": started,
        "seed": seed_base, "seed_source": seed_source,
        "seed_plan": str(plan_path.relative_to(run)),
        "verification_logs": verification_logs,
        "selected_development_evaluation": selected["evaluation"] if selected else None,
        "elapsed_seconds": _elapsed(manifest),
        "candidate_sha256": _hash(candidate) if candidate.is_file() else None,
        "harness_sha256": (_hash(snapshot / "speedupmark" / "harness.py")
                           if (snapshot / "speedupmark" / "harness.py").is_file() else None),
        "integrity": "failed" if integrity_errors else "passed",
        "integrity_errors": integrity_errors,
        "snapshot": str(snapshot.relative_to(run)), "correct": bool(correct),
        "speedup": speedup, "reports": reports, "errors": errors,
    }
    with (run / "evaluations.jsonl").open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
    return row


def evaluate_run(run):
    run = pathlib.Path(run).resolve()
    with _locked(run):
        return _evaluate_locked(run, _manifest(run), "development")


def finish_run(run, *, status="completed", agent_exit_code=None, usage=None):
    run = pathlib.Path(run).resolve()
    with _locked(run):
        manifest = _manifest(run)
        if manifest["status"] != "running":
            raise ValueError("run has already finished or has not started")
        if manifest["agent_elapsed_seconds"] is None:
            manifest["agent_elapsed_seconds"] = _elapsed(manifest)
        result = _evaluate_locked(run, manifest, "final")
        manifest.update(status="integrity_failed" if manifest["integrity_errors"] else status,
                        agent_completion_status=status, finished_at=time.time(), agent_exit_code=agent_exit_code,
                        final_evaluation=result["evaluation"], usage=usage)
        _write_json(run / "manifest.json", manifest)
    return report_run(run)


def report_run(run):
    run = pathlib.Path(run).resolve()
    manifest = _manifest(run)
    rows = history(run)
    development = [row for row in rows if row["phase"] == "development"]
    valid = _verified_development(development)
    best = max(valid, key=lambda row: row["speedup"], default=None)
    improvements = [row for row in valid if row["speedup"] > 1.0]
    # Carry forward only results actually available by the checkpoint. Final
    # evaluation happens after agent completion and never fills earlier points.
    checkpoints = {
        str(seconds): max((row["speedup"] for row in valid if row["elapsed_seconds"] <= seconds), default=None)
        for seconds in (60, 300, 600)
    }
    final = next((row for row in reversed(rows) if row["phase"] == "final"), None)
    return {
        "run_id": manifest["run_id"], "task": manifest["task"],
        "task_version": manifest.get("task_version"),
        "task_revision": manifest.get("task_revision"),
        "optimizer": _reported_optimizer(manifest),
        "submission_policy": manifest.get("submission_policy"),
        "status": manifest["status"], "agent_elapsed_seconds": manifest["agent_elapsed_seconds"],
        "integrity": "failed" if manifest["integrity_errors"] else "passed",
        "integrity_errors": manifest["integrity_errors"],
        "elapsed_seconds": (manifest["finished_at"] - manifest["started_at"]
                            if manifest["finished_at"] else _elapsed(manifest)),
        "evaluations": len(rows), "best_development": best,
        "first_observed_improvement_seconds": min((row["elapsed_seconds"] for row in improvements), default=None),
        "best_development_speedup_at_seconds": checkpoints,
        "final": final, "usage": manifest["usage"],
    }


def summarize_suite(tasks, items, optimizer=None, task_versions=None, task_revisions=None):
    """Credit each expected task's final measured speedup, or 1.0x on failure.

    The suite number measures improvement credited over the reference, not the
    runtime of an invalid submission. Missing, incorrect, and interrupted runs
    receive a neutral 1.0x factor while ``passed`` and ``correct`` retain the
    strict verdict. A correct slowdown still contributes its measured ratio
    below 1.0x; an agent can submit the reference to obtain about 1.0x instead.
    Final grading independently measures the selected best verified candidate.
    Historical reports retain the submission policy under which they ran.
    """
    versions = dict(task_versions or {})
    revisions = dict(task_revisions or {})
    rows = []
    for item in items:
        summary = item["summary"]
        if summary.get("task_version") is not None:
            versions[summary["task"]] = summary["task_version"]
        if summary.get("task_revision") is not None:
            revisions[summary["task"]] = summary["task_revision"]
        final = summary["final"] or {}
        score = final.get("speedup", 0)
        passed = (summary["status"] == "completed" and summary["integrity"] == "passed"
                  and final.get("correct") is True and type(score) in (int, float)
                  and math.isfinite(score) and score > 0)
        rows.append({"task": summary["task"],
                     "task_version": summary.get("task_version") or versions.get(summary["task"]),
                     "task_revision": summary.get("task_revision") or revisions.get(summary["task"]),
                     "run": item["run"],
                     "status": summary["status"], "passed": passed,
                     "speedup": score if passed else None,
                     "credited_speedup": score if passed else 1.0,
                     "agent_elapsed_seconds": summary["agent_elapsed_seconds"]})
    complete = [row["task"] for row in rows] == list(tasks)
    correct = bool(tasks) and complete and all(row["passed"] for row in rows)
    # Neutral credit for a failed attempt keeps a score for every expected task.
    # Do not clamp a correct slowdown: the submitted program's measured ratio
    # remains visible in both the task row and the suite aggregate.
    scored = {row["task"]: row["credited_speedup"] for row in rows}
    score_factors = [{"task": task, "task_version": versions.get(task),
                      "task_revision": revisions.get(task),
                      "credited_speedup": scored.get(task, 1.0)}
                     for task in tasks]
    factors = [row["credited_speedup"] for row in score_factors]
    return {"expected_tasks": list(tasks), "task_versions": versions,
            "task_revisions": revisions,
            "optimizer": optimizer,
            "complete": complete, "correct": correct,
            "passed": sum(row["passed"] for row in rows), "results": rows,
            "score_factors": score_factors,
            "geomean_speedup": (math.exp(statistics.fmean(math.log(factor) for factor in factors))
                                if factors else None), "runs": items}


def _print_suite(report):
    optimizer = report.get("optimizer")
    if optimizer:
        print(f"Harness: {optimizer['harness']}  Model: {optimizer['model']}  Effort: {optimizer['effort']}")
    print(f"{'Task':40} {'Version':26} {'Status':18} {'Speedup':>10} {'Credit':>10} {'Agent time':>12}")
    for row in report["results"]:
        score = f"{row['speedup']:.3f}x" if row["speedup"] is not None else "—"
        credit = f"{row.get('credited_speedup', row['speedup'] if row['passed'] else 1.0):.3f}x"
        seconds = row["agent_elapsed_seconds"]
        elapsed = f"{seconds:.1f}s" if seconds is not None else "—"
        status = "PASS" if row["passed"] else row["status"] if row["status"] != "completed" else "FAIL"
        version = row.get("task_version") or "—"
        if row.get("task_revision"):
            version += "+" + row["task_revision"][:12]
        print(f"{row['task']:40} {version:26} {status:18} {score:>10} {credit:>10} {elapsed:>12}")
    reported = {row["task"] for row in report["results"]}
    versions = report.get("task_versions", {})
    for task in report["expected_tasks"]:
        if task not in reported:
            version = versions.get(task) or "—"
            revision = report.get("task_revisions", {}).get(task)
            if revision:
                version += "+" + revision[:12]
            print(f"{task:40} {version:26} {'NOT RUN':18} {'—':>10} {'1.000x':>10} {'—':>12}")
    score = (f"{report['geomean_speedup']:.3f}x geometric mean"
             if report["geomean_speedup"] is not None else "no suite score")
    counted = len(report["expected_tasks"]) - report["passed"]
    note = f" ({counted} counted as 1.000x)" if counted else ""
    print(f"\n{report['passed']}/{len(report['expected_tasks'])} passed; {score}{note}.")


def _stop_process_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    # Stop descendants even when their parent has already exited on SIGTERM.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def launch_run(task, command=None, **options):
    if not command:
        raise ValueError("run launch requires an agent command after --command")
    run = create_run(task, start=False, **options)
    workspace = run / "workspace"
    prompt_file = workspace / "prompt.md"
    argv = [part.replace("{prompt_file}", str(prompt_file))
            .replace("{workspace}", str(workspace)) for part in command]
    env = os.environ.copy()
    env.update(SPEEDUPMARK_RUN_DIR=str(run), SPEEDUPMARK_PROMPT_FILE=str(prompt_file))
    manifest = _manifest(run)
    manifest.update(started_at=time.time(), status="running", command=argv)
    _write_json(run / "manifest.json", manifest)
    print(f"Run: {run}", file=sys.stderr, flush=True)
    status = "completed"
    returncode = None
    with (run / "agent.stdout.log").open("w") as stdout, (run / "agent.stderr.log").open("w") as stderr:
        try:
            process = subprocess.Popen(
                argv, cwd=workspace, env=env, stdin=subprocess.PIPE,
                stdout=stdout, stderr=stderr, text=True, start_new_session=True,
            )
        except OSError as exc:
            stderr.write(str(exc))
            status = "agent_failed"
        else:
            try:
                process.communicate(prompt_file.read_text(), timeout=manifest["safety_timeout_seconds"] or None)
                returncode = process.returncode
                if returncode:
                    status = "agent_failed"
            except subprocess.TimeoutExpired:
                status = "timed_out"
                _stop_process_group(process)
            except KeyboardInterrupt:
                status = "interrupted"
                _stop_process_group(process)
            returncode = process.returncode
    summary = finish_run(run, status=status, agent_exit_code=returncode)
    _write_json(run / "summary.json", summary)
    return run, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create, launch, grade, and inspect fresh agent runs")
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("create", "launch"):
        command = commands.add_parser(action)
        if action == "launch":
            command.add_argument("task", nargs="?", default="smoke", help="default/smoke, extended, all, or an implemented task name")
        else:
            command.add_argument("task", help="implemented task name")
        command.add_argument("--harness", required=True, help="required agent harness name")
        command.add_argument("--model", required=True, help="required exact model identifier")
        command.add_argument("--effort", required=True, help="required model effort")
        command.add_argument("--runtime-notes", help="optional runtime and configuration details")
        command.add_argument("--root", type=pathlib.Path)
        command.add_argument("--seed", type=int, help="fixed replay seed for every evaluation (default: fresh each time)")
        command.add_argument("--samples", type=int, default=3)
        command.add_argument("--n", type=int)
        command.add_argument("--grade-timeout", type=float, default=60)
        if action == "launch":
            command.add_argument("--safety-timeout", type=float, default=3600,
                                 help="agent process backstop seconds; 0 disables")
            command.add_argument("--json", action="store_true", help="emit full suite JSON instead of a compact table")
            command.add_argument("--command", nargs=argparse.REMAINDER,
                                 help="agent command; prompt is passed on stdin")
    for action in ("evaluate", "finish", "report"):
        command = commands.add_parser(action)
        command.add_argument("run", type=pathlib.Path)
        if action == "report":
            command.add_argument("--json", action="store_true", help="emit full JSON for a suite report")
        if action == "finish":
            command.add_argument("--usage", type=pathlib.Path, help="optional JSON of externally collected token/cost metrics")
    args = parser.parse_args(argv)
    try:
        if args.action in ("create", "launch"):
            optimizer = _optimizer(args.harness, args.model, args.effort, args.runtime_notes)
            options = dict(harness=args.harness, model=args.model, effort=args.effort,
                           runtime_notes=args.runtime_notes, root=args.root, seed=args.seed,
                           samples=args.samples, n=args.n, grade_timeout=args.grade_timeout,
                           safety_timeout=getattr(args, "safety_timeout", 0))
            if args.action == "create":
                run = create_run(args.task, **options)
                print(json.dumps({"run": str(run), "workspace": str(run / "workspace"),
                                  "task_version": _manifest(run)["task_version"],
                                  "task_revision": _manifest(run)["task_revision"],
                                  "prompt": str(run / "workspace" / "prompt.md")}, indent=2))
            else:
                if not args.command:
                    raise ValueError("run launch requires an agent command after --command")
                tasks = selected_tasks(args.task, tuple(path.name for path in discover_tasks()))
                task_versions = _task_versions(tasks)
                task_revisions = _task_revisions(tasks)
                suite_root = pathlib.Path(args.root or ROOT / "runs").resolve() / f"suite-{uuid.uuid4().hex[:12]}"
                suite_root.mkdir(parents=True)
                options["root"] = suite_root
                summaries = []
                _write_json(suite_root / "suite.json", summarize_suite(
                    tasks, summaries, optimizer, task_versions, task_revisions))
                for task in tasks:
                    run, summary = launch_run(task, args.command, **options)
                    summaries.append({"run": str(run), "summary": summary})
                    _write_json(suite_root / "suite.json", summarize_suite(
                        tasks, summaries, optimizer, task_versions, task_revisions))
                    if summary["status"] == "interrupted":
                        break
                report = summarize_suite(tasks, summaries, optimizer, task_versions, task_revisions)
                print(f"Suite report: {suite_root / 'suite.json'}", file=sys.stderr)
                if args.json:
                    print(json.dumps(report, indent=2, allow_nan=False))
                else:
                    _print_suite(report)
                if not report["correct"]:
                    raise SystemExit(1)
        elif args.action == "evaluate":
            result = evaluate_run(args.run)
            print(json.dumps(result, indent=2, allow_nan=False))
            if not result["correct"]:
                raise SystemExit(1)
        elif args.action == "finish":
            usage = json.loads(args.usage.read_text()) if args.usage else None
            result = finish_run(args.run, usage=usage)
            _write_json(args.run / "summary.json", result)
            print(json.dumps(result, indent=2, allow_nan=False))
            if not result["final"]["correct"]:
                raise SystemExit(1)
        else:
            if (args.run / "suite.json").is_file():
                report = json.loads((args.run / "suite.json").read_text())
                if args.json:
                    print(json.dumps(report, indent=2, allow_nan=False))
                else:
                    _print_suite(report)
            else:
                print(json.dumps(report_run(args.run), indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Run error: {exc}\n")
