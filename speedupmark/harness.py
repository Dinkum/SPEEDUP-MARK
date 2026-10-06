"""Verified paired measurements and a dependency-free, isolated suite runner."""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.util
import json
import math
import pathlib
import platform
import secrets
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field

from speedupmark.catalog import TASK_CATALOG
from speedupmark.suites import selected_tasks
from speedupmark.task import SolutionEvaluation, declared_task_version, freeze_output
from speedupmark.revision import task_revision


TASK_ROOT = pathlib.Path(__file__).resolve().parents[1] / "tasks"


@dataclass
class SampleResult:
    seed: int
    reference_score: float
    candidate_score: float
    speedup: float
    correct: bool
    verification: dict | None = None


@dataclass
class RunResult:
    task: str
    task_version: str
    task_revision: str
    reference_score: float
    candidate_score: float
    metric_unit: str
    speedup: float
    correct: bool
    samples: list[SampleResult] = field(default_factory=list)
    candidate_is_reference: bool = False
    problem_size: int = 0


def load_task(task_dir: pathlib.Path):
    task_dir = pathlib.Path(task_dir).resolve()
    module_name = f"speedupmark_task_{task_dir.name}"
    spec = importlib.util.spec_from_file_location(module_name, task_dir / "task_spec.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load task from {task_dir}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    if getattr(mod.TASK, "PORT_TODO", False):
        raise ValueError(f"{task_dir.name}: task is a placeholder, not implemented")
    declared_task_version(mod.TASK)
    if task_dir.parent == TASK_ROOT and getattr(mod.TASK, "name", None) != task_dir.name:
        raise ValueError(
            f"{task_dir.name}: task name must match its directory identifier"
        )
    if task_dir.parent == TASK_ROOT and task_dir.name in TASK_CATALOG:
        metadata = TASK_CATALOG[task_dir.name]
        if getattr(mod.TASK, "display_name", None) != metadata.display_name:
            raise ValueError(f"{task_dir.name}: display name must match the task catalog")
    return mod.TASK


def _positive_score(score: float) -> bool:
    return (
        isinstance(score, (int, float))
        and not isinstance(score, bool)
        and math.isfinite(score)
        and score > 0
    )


def _geomean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def resolve_seed(seed: int | None) -> int:
    """Draw a replayable base seed unless the caller explicitly supplied one."""
    if seed is None:
        return secrets.randbits(63)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed


def _grading_functions(task):
    """Track direct grading entrypoints; this is a sanity check, not a sandbox."""
    functions = {
        name: getattr(task, name, None)
        for name in ("solve", "is_solution", "evaluate_solution", "evaluate_pair")
    }
    functions["timer"] = time.perf_counter_ns
    # Bound method objects are recreated on access; compare their functions.
    return {name: getattr(function, "__func__", function)
            for name, function in functions.items()}


def _check_grading_functions(task, expected):
    for name, function in _grading_functions(task).items():
        if function is not expected[name]:
            raise ValueError(f"{task.name}: grading function changed: {name}")


def run_task(
    task_dir: pathlib.Path, n: int | None = None, seed: int | None = None, samples: int = 3,
    *, verification_replay: list[dict] | None = None, verification_record=None,
) -> RunResult:
    """Measure calls and output freezing; verification/input copying are untimed.

    Each sample uses a different seed and alternating solver order. Solvers get
    detached input copies and the verifier sees pristine data. This prevents
    ordinary accidental state sharing; it is not a sandbox against hostile code.
    """
    if samples < 1 or (n is not None and n < 1):
        raise ValueError("samples and problem size must be positive")
    seed = resolve_seed(seed)
    task_dir = pathlib.Path(task_dir)
    revision = task_revision(task_dir, pathlib.Path(__file__).resolve().parent)[0]
    candidate_path = task_dir / "candidate.py"
    candidate_hash = (hashlib.sha256(candidate_path.read_bytes()).hexdigest()
                      if candidate_path.is_file() else None)
    if verification_replay is not None:
        if len(verification_replay) != samples:
            raise ValueError("verification replay must contain one record per sample")
        # Bind private-data replay to the original source before candidate import.
        for index, receipt in enumerate(verification_replay):
            if (receipt.get("task_revision") != revision
                    or receipt.get("candidate_sha256") != candidate_hash
                    or receipt.get("seed") != seed + index):
                raise ValueError("verification replay source revision, candidate, or seed changed")
    task = load_task(task_dir)
    problem_size = n if n is not None else getattr(task, "default_n", 1000)
    candidate_is_reference = not hasattr(task, "candidate_solve")
    candidate = getattr(task, "candidate_solve", task.solve)
    evaluate = getattr(task, "evaluate_solution", None)
    evaluate_pair = getattr(task, "evaluate_pair", None)
    uses_reference_output = getattr(task, "uses_reference_output", False)
    metric_unit = getattr(task, "metric_unit", "score") if evaluate else "ms"
    measurements = []
    grading_functions = _grading_functions(task)

    for index in range(samples):
        case_seed = seed + index
        problem = task.generate_problem(problem_size, case_seed)
        solvers = {"reference": task.solve, "candidate": candidate}
        # Copy both before either solver runs: mutation cannot affect its peer.
        inputs = {role: copy.deepcopy(problem) for role in solvers}
        outputs = {}
        scores = {}
        order = (
            ("reference", "candidate")
            if index % 2 == 0
            else ("candidate", "reference")
        )
        for role in order:
            _check_grading_functions(task, grading_functions)
            start = time.perf_counter_ns()
            try:
                outputs[role] = freeze_output(solvers[role](inputs[role]))
            except ValueError as exc:
                raise ValueError(f"{task.name}: {role} output: {exc}") from None
            scores[role] = (time.perf_counter_ns() - start) / 1_000_000
            _check_grading_functions(task, grading_functions)

        verification = None
        paired = None
        if evaluate_pair:
            replay = verification_replay[index] if verification_replay is not None else None
            if replay is not None and replay.get("problem_size") != problem_size:
                raise ValueError("verification replay problem size changed")

            def record(context):
                nonlocal verification
                verification = {"task_revision": revision, "candidate_sha256": candidate_hash,
                                "seed": case_seed, "problem_size": problem_size,
                                "context": context}
                if verification_record is not None:
                    verification_record(verification)

            paired = evaluate_pair(copy.deepcopy(problem), outputs,
                                   replay=replay["context"] if replay is not None else None,
                                   record=record)
            if type(paired) is not dict or set(paired) != {"reference", "candidate"}:
                raise TypeError("evaluate_pair must return reference and candidate evaluations")
        elif verification_replay is not None:
            raise ValueError("task does not support verification replay")
        correctness = {}
        for role in ("reference", "candidate"):
            if evaluate or paired is not None:
                evaluation = (paired[role] if paired is not None else
                              evaluate(copy.deepcopy(problem), outputs[role]))
                if not isinstance(evaluation, SolutionEvaluation):
                    raise TypeError("evaluate_solution must return SolutionEvaluation")
                scores[role] = evaluation.score
                correctness[role] = bool(evaluation.correct)
            else:
                # Canonical-output tasks reuse the measured answer, outside
                # timing, instead of supplying another solution algorithm.
                verification_kwargs = ({"reference_output": copy.deepcopy(outputs["reference"])}
                                       if uses_reference_output else {})
                correctness[role] = bool(
                    task.is_solution(copy.deepcopy(problem), outputs[role], **verification_kwargs)
                )
        if not correctness["reference"] or not _positive_score(scores["reference"]):
            raise ValueError(
                f"{task.name}: reference failed verification or returned "
                f"an invalid score (seed {case_seed})"
            )
        correct = correctness["candidate"] and _positive_score(scores["candidate"])
        # Invalid candidates never receive a performance reward. Normalize invalid
        # costs to zero so JSON remains strict (no NaN or Infinity extensions).
        candidate_score = (
            float(scores["candidate"])
            if _positive_score(scores["candidate"])
            else 0.0
        )
        ratio = scores["reference"] / candidate_score if correct else 0.0
        measurements.append(
            SampleResult(
                case_seed, float(scores["reference"]), candidate_score, ratio, correct,
                verification,
            )
        )

    correct = all(sample.correct for sample in measurements)
    return RunResult(
        task=task.name,
        task_version=declared_task_version(task),
        task_revision=revision,
        reference_score=statistics.median(sample.reference_score for sample in measurements),
        candidate_score=statistics.median(sample.candidate_score for sample in measurements),
        metric_unit=metric_unit,
        speedup=_geomean([sample.speedup for sample in measurements]) if correct else 0.0,
        correct=correct,
        samples=measurements,
        candidate_is_reference=candidate_is_reference,
        problem_size=problem_size,
    )


def discover_tasks(
    root: pathlib.Path | None = None, include_todo: bool = False
) -> list[pathlib.Path]:
    """List task files without importing candidates or their dependencies.

    PORT_TODO must be a literal flag in task_spec.py. Syntax errors are reported,
    not silently disguised as missing tasks.
    """
    root = root or TASK_ROOT
    out = []
    for path in sorted(root.iterdir()):
        source = path / "task_spec.py"
        if (
            not path.is_dir()
            or path.name.startswith(("_", "."))
            or not source.is_file()
        ):
            continue
        tree = ast.parse(source.read_text(), filename=str(source))
        todo = any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "PORT_TODO"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and node.value.value is True
            for node in ast.walk(tree)
        )
        if include_todo or not todo:
            out.append(path)
    return out


def _run_isolated(
    task_dir: pathlib.Path, n: int | None, seed: int, samples: int, timeout: float,
    *, verification_log: pathlib.Path | None = None,
) -> RunResult:
    """One fresh interpreter per task, sequentially to avoid CPU contention."""
    command = [
        sys.executable, "-m", "speedupmark", str(task_dir.resolve()),
        "--_worker", "--seed", str(seed), "--samples", str(samples),
    ]
    if n is not None:
        command.extend(["--n", str(n)])
    if verification_log is not None:
        command.extend(["--verification-log", str(verification_log.resolve())])
    completed = subprocess.run(
        command, cwd=TASK_ROOT.parent, capture_output=True, text=True, timeout=timeout
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(detail[-2000:] or f"worker exited {completed.returncode}")
    payload = json.loads(completed.stdout)
    payload["samples"] = [SampleResult(**sample) for sample in payload["samples"]]
    return RunResult(**payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SPEEDUP-MARK's smoke 10 (default), extended 25, all implemented tasks, or one task.")
    parser.add_argument("target", nargs="?", default="smoke", help="default/smoke, extended, all, task name, or task directory")
    parser.add_argument("--list", action="store_true", help="list the selected tasks without loading them")
    parser.add_argument("--n", type=int, default=None, help="override each task's default problem size")
    parser.add_argument("--seed", type=int, help="replay a nonnegative first sample seed (default: fresh random seed)")
    parser.add_argument("--samples", type=int, default=3, help="fresh paired inputs per task (default: 3)")
    parser.add_argument("--timeout", type=float, default=60, help="wall-clock limit per task, including verification (seconds)")
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    parser.add_argument("--verification-log", type=pathlib.Path,
                        help="append private verifier replay records before simulation")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.samples < 1 or (args.n is not None and args.n < 1) or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("samples, n, and timeout must be positive and finite")
    path = pathlib.Path(args.target)
    if path.is_dir():
        paths = [path]
    else:
        try:
            names = selected_tasks(args.target, tuple(path.name for path in discover_tasks()))
        except ValueError as exc:
            parser.error(str(exc))
        paths = [TASK_ROOT / name for name in names]
    if args.list:
        names = [path.name for path in paths]
        print(json.dumps(names) if args.json else "\n".join(names))
        return
    seed_source = "random" if args.seed is None else "explicit"
    try:
        args.seed = resolve_seed(args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    if args._worker:
        if len(paths) != 1:
            parser.error("worker requires one task")
        # Task print statements must not corrupt the worker's result protocol.
        with contextlib.redirect_stdout(sys.stderr):
            def record(receipt):
                if args.verification_log is not None:
                    with args.verification_log.open("a") as stream:
                        stream.write(json.dumps(receipt, allow_nan=False) + "\n")
            result = run_task(paths[0], n=args.n, seed=args.seed, samples=args.samples,
                              verification_record=record)
        print(json.dumps(asdict(result), allow_nan=False))
        return

    results = []
    errors = []
    if not args.json:
        print(f"Seed: {args.seed} ({seed_source}; replay with --seed {args.seed})", flush=True)
    for path in paths:
        try:
            result = _run_isolated(path, args.n, args.seed, args.samples, args.timeout,
                                   verification_log=args.verification_log)
            results.append(result)
            if not args.json:
                marker = "PASS" if result.correct else "FAIL"
                baseline = " (reference fallback)" if result.candidate_is_reference else ""
                print(f"{marker} {result.task}@{result.task_version}+{result.task_revision[:12]}: {result.speedup:.3f}x | reference {result.reference_score:.3f}, candidate {result.candidate_score:.3f} {result.metric_unit}{baseline}", flush=True)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            message = f"exceeded {args.timeout:g}s task timeout" if isinstance(exc, subprocess.TimeoutExpired) else str(exc)
            errors.append({"task": path.name, "error": message})
            if not args.json:
                print(f"ERROR {path.name}: {message}", flush=True)
    correct = bool(results) and not errors and all(result.correct for result in results)
    # A partial suite must not look like a successful score on easier survivors.
    aggregate = _geomean([result.speedup for result in results]) if correct else None
    report = {
        "suite": args.target,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "config": {
            "seed": args.seed,
            "seed_source": seed_source,
            "samples": args.samples,
            "n": args.n,
            "timeout_seconds": args.timeout,
        },
        "results": [asdict(result) for result in results],
        "errors": errors,
        "correct": correct,
        "geomean_speedup": aggregate,
    }
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        passed = sum(result.correct for result in results)
        summary = f"{aggregate:.3f}x geometric mean" if aggregate is not None else "no suite score"
        print(f"\n{passed}/{len(paths)} passed; {summary}.")
    if not correct:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
