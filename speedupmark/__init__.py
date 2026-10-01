"""SPEEDUP-MARK — drop-in speed benchmark in the style of AlgoTune."""

from .task import SpeedupMarkTask

__all__ = ["SpeedupMarkTask", "run_task", "discover_tasks", "RunResult"]


def __getattr__(name: str):
    """Keep the executable harness module lazy for ``python -m`` runs."""

    if name in {"run_task", "discover_tasks", "RunResult"}:
        from . import harness

        return getattr(harness, name)
    raise AttributeError(name)
