"""Verified paired measurements and a dependency-free, isolated suite runner."""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
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
from speedupmark.dependencies import (CandidateSolver, DependencyViolation, GradingEnvironmentError,
                                     grading_command)
from speedupmark.suites import selected_tasks
from speedupmark.task import (SolutionEvaluation, covering_samples, declared_task_version,
                             freeze_output, load_candidate, read_task_metadata, timing_batch)
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
    problem_size: int = 0


class CandidateNotReady(ValueError):
    """A submission is absent or still contains the unchanged reference source."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def candidate_sha256(task_dir: pathlib.Path) -> str:
    """Reject absent/unchanged submissions before importing any task code."""
    candidate = task_dir / "candidate.py"
    if not candidate.is_file():
        raise CandidateNotReady("missing_candidate", "candidate.py is missing; nothing to grade")
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    reference = task_dir / "reference.py"
    if reference.is_file() and digest == hashlib.sha256(reference.read_bytes()).hexdigest():
        raise CandidateNotReady("unchanged_candidate",
                                "candidate.py matches reference.py; nothing to grade")
    return digest


def load_task(task_dir: pathlib.Path, *, with_candidate: bool = True):
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
    if with_candidate and (task_dir / "candidate.py").is_file():
        candidate = load_candidate(str(task_dir / "task_spec.py"),
                                   forbidden_roots=getattr(mod.TASK, "forbidden_import_roots", ()))
        mod.TASK.candidate_solve = candidate.solve
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
        for name in ("solve", "is_solution", "evaluate_solution", "evaluate_pair", "correctness_problem")
    }
    functions["timer"] = time.perf_counter_ns
    # Bound method objects are recreated on access; compare their functions.
    return {name: getattr(function, "__func__", function)
            for name, function in functions.items()}


def _check_grading_functions(task, expected):
    for name, function in _grading_functions(task).items():
        if function is not expected[name]:
            raise ValueError(f"{task.name}: grading function changed: {name}")


def _check_small_problem(task, problem, seed, solvers, grading_functions, uses_reference_output):
    answers, correct = {}, {}
    for role, solve in solvers.items():
        _check_grading_functions(task, grading_functions)
        try:
            answers[role] = freeze_output(solve(copy.deepcopy(problem)))
        except (DependencyViolation, GradingEnvironmentError):
            raise
        except Exception as exc:
            raise ValueError(
                f"{task.name}: {role} failed the untimed correctness check (seed {seed}): {exc}"
            ) from exc
        _check_grading_functions(task, grading_functions)
        expected = ({"reference_output": copy.deepcopy(answers["reference"])}
                    if role == "candidate" and uses_reference_output else {})
        correct[role] = task.is_solution(copy.deepcopy(problem), answers[role], **expected)
    return correct


def run_task(
    task_dir: pathlib.Path, n: int | None = None, seed: int | None = None, samples: int = 3,
    *, verification_replay: list[dict] | None = None, verification_record=None,
) -> RunResult:
    """Measure calls and output freezing; verification/input copying are untimed.

    Each sample uses a different seed. The sample count is rounded up to cover
    every seed-selected workload family, and a task declaring ``timing_batch``
    sums that many distinct same-family problems per sample. Solver order
    alternates per problem. Solvers get detached input copies and the verifier
    sees pristine data. This prevents ordinary accidental state sharing; it is
    not a sandbox against hostile code.
    """
    if samples < 1 or (n is not None and n < 1):
        raise ValueError("samples and problem size must be positive")
    task_dir = pathlib.Path(task_dir)
    candidate_hash = candidate_sha256(task_dir)
    seed = resolve_seed(seed)
    revision = task_revision(task_dir, pathlib.Path(__file__).resolve().parent)[0]
    if verification_replay is not None:
        # Bind private-data replay to the original source before candidate import.
        for index, receipt in enumerate(verification_replay):
            if (receipt.get("task_revision") != revision
                    or receipt.get("candidate_sha256") != candidate_hash
                    or receipt.get("seed") != seed + index):
                raise ValueError("verification replay source revision, candidate, or seed changed")
    task = load_task(task_dir)
    samples = covering_samples(task, samples)
    batch = timing_batch(task)
    if verification_replay is not None and len(verification_replay) != samples:
        raise ValueError("verification replay must contain one record per sample")
    problem_size = n if n is not None else getattr(task, "default_n", 1000)
    candidate = task.candidate_solve
    correctness_problem = getattr(task, "correctness_problem", None)
    evaluate = getattr(task, "evaluate_solution", None)
    evaluate_pair = getattr(task, "evaluate_pair", None)
    if batch > 1 and (evaluate or evaluate_pair):
        raise ValueError(f"{task.name}: timing_batch applies only to wall-clock tasks")
    uses_reference_output = getattr(task, "uses_reference_output", False)
    metric_unit = getattr(task, "metric_unit", "score") if (evaluate or evaluate_pair) else "ms"
    measurements = []
    grading_functions = _grading_functions(task)
    solvers = {"reference": task.solve, "candidate": candidate}
    timed_solvers = solvers.copy()
    import_scopes = dict.fromkeys(solvers, contextlib.nullcontext())
    if isinstance(candidate, CandidateSolver):
        timed_solvers["candidate"] = candidate.solve
        import_scopes["candidate"] = candidate.imports

    for index in range(samples):
        case_seed = seed + index
        # The small check also warms both solvers on input separate from timing.
        small_correct = {}
        if correctness_problem is not None:
            small_correct = _check_small_problem(task, correctness_problem(case_seed), case_seed,
                                                 solvers, grading_functions, uses_reference_output)
        # Batch members step by the sample count, a multiple of the family
        # period: they share this sample's family and no two samples share a seed.
        problems = [task.generate_problem(problem_size, case_seed + member * samples)
                    for member in range(batch)]
        # Copy both before either solver runs: mutation cannot affect its peer.
        batch_inputs = [{role: copy.deepcopy(item) for role in solvers} for item in problems]
        batch_outputs = [{} for _ in problems]
        scores = dict.fromkeys(solvers, 0.0)
        for member, inputs in enumerate(batch_inputs):
            order = (
                ("reference", "candidate")
                if (index + member) % 2 == 0
                else ("candidate", "reference")
            )
            for role in order:
                _check_grading_functions(task, grading_functions)
                solve = timed_solvers[role]
                problem_input = inputs[role]
                # Cyclic collection cost scales with every live object, and the
                # peer's input copy and earlier output are alive, so whichever
                # solver runs second would pay more. Start both from a clean
                # heap and keep collection out of the timed call.
                gc.collect()
                gc_was_enabled = gc.isenabled()
                gc.disable()
                try:
                    with import_scopes[role]:
                        start = time.perf_counter_ns()
                        output = solve(problem_input)
                        try:
                            frozen = freeze_output(output)
                        except ValueError as exc:
                            raise ValueError(f"{task.name}: {role} output: {exc}") from None
                        elapsed = time.perf_counter_ns() - start
                    del output
                    batch_outputs[member][role] = frozen
                    scores[role] += elapsed / 1_000_000
                finally:
                    if gc_was_enabled:
                        gc.enable()
                _check_grading_functions(task, grading_functions)
        # Deterministic-metric tasks never batch, so these are the whole sample.
        problem, outputs = problems[0], batch_outputs[0]

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
                correctness[role] = all(
                    task.is_solution(
                        copy.deepcopy(item), answers[role],
                        **({"reference_output": copy.deepcopy(answers["reference"])}
                           if uses_reference_output else {}))
                    for item, answers in zip(problems, batch_outputs)
                )
            correctness[role] = correctness[role] and small_correct.get(role, True)
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
        metadata = read_task_metadata(source)
        if include_todo or not metadata.port_todo:
            out.append(path)
    return out


def _run_isolated(
    task_dir: pathlib.Path, n: int | None, seed: int, samples: int, timeout: float,
    *, verification_log: pathlib.Path | None = None,
) -> RunResult:
    """One fresh interpreter per task, sequentially to avoid CPU contention."""
    candidate_sha256(task_dir)
    command = grading_command([
        str(task_dir.resolve()),
        "--_worker", "--seed", str(seed), "--samples", str(samples),
    ], TASK_ROOT.parent)
    if n is not None:
        command.extend(["--n", str(n)])
    if verification_log is not None:
        command.extend(["--verification-log", str(verification_log.resolve())])
    completed = subprocess.run(
        command, cwd=TASK_ROOT.parent, capture_output=True, text=True, timeout=timeout
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        try:
            failure = json.loads(completed.stdout)
        except (ValueError, TypeError):
            failure = {}
        if failure.get("code") == "dependency_violation":
            raise DependencyViolation(task_dir.name, failure["module"])
        if failure.get("code") == "environment_error":
            raise GradingEnvironmentError(failure["error"])
        raise RuntimeError(detail[-2000:] or f"worker exited {completed.returncode}")
    payload = json.loads(completed.stdout)
    payload["samples"] = [SampleResult(**sample) for sample in payload["samples"]]
    return RunResult(**payload)


def _run_worker(path, args):
    # Task print statements must not corrupt the worker's result protocol.
    with contextlib.redirect_stdout(sys.stderr):
        def record(receipt):
            if args.verification_log is not None:
                with args.verification_log.open("a") as stream:
                    stream.write(json.dumps(receipt, allow_nan=False) + "\n")

        return run_task(path, n=args.n, seed=args.seed, samples=args.samples,
                        verification_record=record)


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
    if args._worker and len(paths) != 1:
        parser.error("worker requires one task")
    if args._worker and not args.json:
        try:
            result = _run_worker(paths[0], args)
        except (DependencyViolation, GradingEnvironmentError) as exc:
            failure = {"code": exc.code, "error": str(exc)}
            if isinstance(exc, DependencyViolation):
                failure["module"] = exc.module
            print(json.dumps(failure))
            raise SystemExit(1)
        print(json.dumps(asdict(result), allow_nan=False))
        return

    results = []
    errors = []
    if not args.json:
        print(f"Seed: {args.seed} ({seed_source}; replay with --seed {args.seed})", flush=True)
    for path in paths:
        try:
            result = (_run_worker(path, args) if args._worker else
                      _run_isolated(path, args.n, args.seed, args.samples, args.timeout,
                                    verification_log=args.verification_log))
            results.append(result)
            if not args.json:
                marker = "PASS" if result.correct else "FAIL"
                print(f"{marker} {result.task}@{result.task_version}+{result.task_revision[:12]}: {result.speedup:.3f}x | reference {result.reference_score:.3f}, candidate {result.candidate_score:.3f} {result.metric_unit}", flush=True)
        except (CandidateNotReady, DependencyViolation, GradingEnvironmentError) as exc:
            errors.append({"task": path.name, "error": str(exc), "code": exc.code})
            if not args.json:
                marker = "INFO" if exc.code == "unchanged_candidate" else "ERROR"
                print(f"{marker} {path.name}: {exc}", flush=True)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            message = f"exceeded {args.timeout:g}s task timeout" if isinstance(exc, subprocess.TimeoutExpired) else str(exc)
            errors.append({"task": path.name, "error": message})
            if not args.json:
                print(f"ERROR {path.name}: {message}", flush=True)
    correct = bool(results) and not errors and all(result.correct for result in results)
    # A partial suite must not look like a successful score on easier survivors.
    aggregate = _geomean([result.speedup for result in results]) if correct else None
    report = {
        "suite": paths[0].name if args._worker else args.target,
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
