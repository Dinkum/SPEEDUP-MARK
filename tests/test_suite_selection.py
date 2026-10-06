"""The public selectors resolve the same complete task sets in both runners."""

import contextlib
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from speedupmark import run_manager
from speedupmark.harness import discover_tasks
from speedupmark.suites import EXTENDED_TASKS, SMOKE_TASKS, selected_tasks


ROOT = pathlib.Path(__file__).resolve().parents[1]


class SuiteSelectionTests(unittest.TestCase):
    def test_membership_and_every_selected_task_is_implemented(self):
        implemented = {path.name for path in discover_tasks()}
        self.assertEqual(len(SMOKE_TASKS), 10)
        self.assertEqual(len(EXTENDED_TASKS), 25)
        self.assertEqual(len(set(EXTENDED_TASKS)), 25)
        self.assertTrue(set(SMOKE_TASKS) <= set(EXTENDED_TASKS) <= implemented)
        self.assertTrue(set(EXTENDED_TASKS) <= implemented)
        self.assertEqual(len(implemented), 50)
        self.assertTrue({
            "job_shop_scheduling", "multi_dim_knapsack", "pagerank", "asymmetric_tsp", "affine_gap_sequence_alignment"
        } <= implemented)
        self.assertNotIn("example_gzip", implemented)
        self.assertIn("labeled_graph_isomorphism", implemented)
        self.assertNotIn("labeled_graph_isomorphism", EXTENDED_TASKS)
        self.assertIn("multi_pattern_matching", EXTENDED_TASKS)
        self.assertIn("temporal_asof_join", EXTENDED_TASKS)
        self.assertIn("live_fleet_dispatch", SMOKE_TASKS)
        self.assertNotIn("near_duplicate_document_clustering", SMOKE_TASKS)
        self.assertIn("near_duplicate_document_clustering", EXTENDED_TASKS)

    def test_direct_cli_lists_selectors_without_running_candidates(self):
        implemented = tuple(path.name for path in discover_tasks())
        for name in ("default", "smoke", "extended", "all", "affine_gap_sequence_alignment"):
            completed = subprocess.run([sys.executable, "-B", "-m", "speedupmark", name, "--list", "--json"],
                                       cwd=ROOT, text=True, capture_output=True, check=True)
            self.assertEqual(json.loads(completed.stdout), list(selected_tasks(name, implemented)))

    def test_all_is_discovered_and_unknown_names_fail(self):
        self.assertEqual(selected_tasks("all", ("new_task", "example_gzip")), ("new_task", "example_gzip"))
        for name in ("light", "lite", "lightweight", "not_a_task"):
            with self.assertRaises(ValueError):
                selected_tasks(name, ("example_gzip",))

    def test_managed_launch_selectors_and_omitted_default(self):
        implemented = tuple(path.name for path in discover_tasks())
        for selector in (None, "default", "smoke", "extended", "all"):
            expected = selected_tasks(selector or "smoke", implemented)
            seen = []

            def launch(task, command, **options):
                seen.append(task)
                return options["root"] / task, {
                    "task": task, "task_version": "1.0.0",
                    "status": "completed", "integrity": "passed",
                    "final": {"correct": True, "speedup": 1.0}, "agent_elapsed_seconds": 0,
                }

            with tempfile.TemporaryDirectory() as directory:
                arguments = ["launch"] + ([selector] if selector else []) + [
                    "--root", directory, "--harness", "test-harness",
                    "--model", "test-model", "--effort", "medium", "--command", "unused"]
                with patch.object(run_manager, "launch_run", side_effect=launch), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    run_manager.main(arguments)
                report = json.loads(next(pathlib.Path(directory).glob("suite-*/suite.json")).read_text())
                self.assertTrue(report["correct"])
                self.assertEqual(report["expected_tasks"], list(expected))
                self.assertEqual(set(report["task_versions"]), set(expected))
                self.assertTrue(all(row["task_version"] == "1.0.0"
                                    for row in report["score_factors"]))
            self.assertEqual(seen, list(expected))


if __name__ == "__main__":
    unittest.main()
