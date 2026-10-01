"""Verify and measure fixed hand-written schedules; never invoke a model."""

import hashlib
import json
import math
import pathlib
import secrets
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from speedupmark.harness import load_task
from speedupmark.revision import task_revision
from simd_traversal_kernel_schedules import compile_workload


def calibrate():
    task_dir = ROOT / "tasks/simd_traversal_kernel"
    task = load_task(task_dir)
    spec = sys.modules[type(task).__module__]
    seed = secrets.token_hex(32)
    variants = {
        "scalar": None,
        "vector": {},
        "scheduled": {"scheduled": True},
        "two_streams": {"scheduled": True, "streams": 2},
        "four_streams": {"scheduled": True, "streams": 4},
        "cached_table": {"scheduled": True, "streams": 4, "cached": True},
        "compact_five_streams": {"streams": 5, "compact": True},
        "compact_cached_table": {"streams": 4, "compact": True, "cached": True},
    }
    rows = []
    for size in (32, 64, 128):
        for public_seed in (17, 701, 100003):
            for index, workload in enumerate(task.generate_problem(size, public_seed)["workloads"]):
                memories = [spec._runtime_memory(workload, seed, index, trial) for trial in range(3)]
                expected = [spec._oracle(workload, memory) for memory in memories]
                for name, options in variants.items():
                    program = spec._compile(workload) if options is None else compile_workload(workload, **options)
                    # Exercise the production plain-data boundary as well.
                    program = spec._freeze((program,), (workload,))[0]
                    stats = []
                    for memory, answer in zip(memories, expected):
                        machine = spec.Machine(workload, memory)
                        machine.run(program)
                        actual = tuple(tuple(machine.memory[workload[key]:workload[key] + workload["batch"]])
                                       for key in ("out_positions", "out_values"))
                        if actual != answer:
                            raise AssertionError((size, public_seed, workload["family"], name))
                        stats.append(machine.stats)
                    rows.append(dict(size=size, public_seed=public_seed, family=workload["family"],
                                     variant=name, descriptor=workload,
                                     program_sha256=spec._digest(program), memory_sha256=spec._digest(memories),
                                     mean_cycles=sum(s["cycles"] for s in stats) / 3, trials=stats))
    summary = {}
    for family in spec.FAMILIES:
        summary[family] = {}
        for name in variants:
            ratios = []
            for size in (32, 64, 128):
                for public_seed in (17, 701, 100003):
                    cases = {r["variant"]: r["mean_cycles"] for r in rows
                             if (r["family"], r["size"], r["public_seed"]) == (family, size, public_seed)}
                    ratios.append(cases["scalar"] / cases[name])
            summary[family][name] = math.exp(sum(map(math.log, ratios)) / len(ratios))
    revision, source_files = task_revision(task_dir, ROOT / "speedupmark")
    return dict(task=task.name, task_version=task.task_version, task_revision=revision,
                source_files=source_files,
                calibration_source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                schedules_sha256=hashlib.sha256((ROOT / "tests/simd_traversal_kernel_schedules.py").read_bytes()).hexdigest(),
                generator_version=spec.GENERATOR, private_seed=seed,
                description="Hand-written schedules; per-family geometric mean speedups over nine descriptor cases.",
                summary=summary, rows=rows)


if __name__ == "__main__":
    print(json.dumps(calibrate(), indent=2))
