"""End-to-end lifecycle checks with local fake agents, never paid model calls."""

import ast
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
from speedupmark.revision import task_revision


REFERENCE = 'def solve(problem): return sum(problem)\n'


FIXTURE = '''from speedupmark.task import load_reference
_reference = load_reference(__file__)
class Task:
    name = 'fixture'
    task_version = '1.0.0'
    default_n = 4
    grading_cases = (4, 8)
    def generate_problem(self, n, random_seed=0): return list(range(n)) + [random_seed]
    solve = staticmethod(_reference.solve)
    def is_solution(self, problem, proposed): return type(proposed) is int and proposed == sum(problem)
TASK = Task()
'''
def _sizes(row):
    return [report['results'][0]['problem_size'] for report in row['reports']]


PROVENANCE = {
    'harness': 'test-harness',
    'model': 'test-model',
    'effort': 'test-effort',
    'runtime_notes': 'local unit test',
}


class RunFlowTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = pathlib.Path(directory.name)
        self.source = self.root / 'tasks' / 'fixture'
        self.source.mkdir(parents=True)
        (self.source / 'task_spec.py').write_text(FIXTURE)
        (self.source / 'reference.py').write_text(REFERENCE)
        (self.source / 'candidate.py').write_text('def solve(problem): return -999\n')
        (self.source / 'README.md').write_text('# Fixture\nReturn the integer sum.\n')
        for patcher in (patch.object(run_manager, 'TASK_ROOT', self.source.parent),
                        patch.object(run_manager, 'discover_tasks', return_value=[self.source])):
            patcher.start()
            self.addCleanup(patcher.stop)

    def create(self, *, edited=True, **kwargs):
        run = run_manager.create_run('fixture', root=self.root / 'runs', samples=1,
                                      **(PROVENANCE | kwargs))
        if edited:
            # A distinct submission for lifecycle tests; starter generation is
            # checked separately with edited=False.
            (run / 'workspace/tasks/fixture/candidate.py').write_text(
                'def solve(problem): return sum(reversed(problem))\n')
        return run

    def test_optimizer_identity_is_required_and_reported(self):
        run = self.create(**PROVENANCE)
        self.assertEqual(run_manager._manifest(run)['optimizer'], PROVENANCE)
        self.assertEqual(run_manager.report_run(run)['optimizer'], PROVENANCE)
        self.assertEqual(run_manager._manifest(run)['task_version'], '1.0.0')
        self.assertEqual(run_manager.report_run(run)['task_version'], '1.0.0')
        self.assertEqual(run_manager._manifest(run)['task_revision'],
                         run_manager.report_run(run)['task_revision'])
        self.assertTrue((run / 'workspace/prompt.md').is_file())
        self.assertEqual((run / 'workspace/AGENTS.md').read_bytes(),
                         (run / 'workspace/prompt.md').read_bytes())
        self.assertEqual(run_manager._manifest(run)['workspace_files']['AGENTS.md'],
                         run_manager._manifest(run)['prompt_sha256'])
        for field in ('harness', 'model', 'effort'):
            invalid = dict(PROVENANCE)
            invalid[field] = ' '
            with self.subTest(field=field), self.assertRaises(ValueError):
                run_manager.create_run('fixture', root=self.root / 'runs', samples=1, **invalid)
            invalid.pop(field)
            with self.subTest(missing=field), self.assertRaises(ValueError):
                run_manager.create_run('fixture', root=self.root / 'runs', samples=1, **invalid)
        optional = dict(PROVENANCE)
        optional.pop('runtime_notes')
        without_notes = run_manager.create_run('fixture', root=self.root / 'runs', samples=1, **optional)
        self.assertEqual(run_manager._manifest(without_notes)['optimizer'], optional)
        with self.assertRaises(ValueError):
            self.create(runtime_notes=' ')
        legacy = run_manager._manifest(run)
        legacy['optimizer'] = None
        self.assertEqual(run_manager._reported_optimizer(legacy)['model'], 'unknown')

    def test_workspace_contains_only_submission_instructions_and_launcher(self):
        run = self.create()
        files = set(run_manager._inventory(run / 'workspace'))
        self.assertEqual(files, {'README.md', 'prompt.md', 'AGENTS.md', 'grade.py',
                                 'tasks/fixture/candidate.py'})
        self.assertTrue((run / 'baseline/tasks/fixture/task_spec.py').is_file())
        self.assertTrue((run / 'baseline/tasks/fixture/reference.py').is_file())
        self.assertTrue((run / 'baseline/speedupmark/harness.py').is_file())
        self.assertNotIn('reference_solve', (run / 'workspace/tasks/fixture/candidate.py').read_text())

    def test_fresh_runs_ignore_optimized_source_candidates(self):
        left, right = self.create(edited=False), self.create(edited=False)
        self.assertNotEqual(left, right)
        for run in (left, right):
            candidate = run / 'workspace/tasks/fixture/candidate.py'
            self.assertEqual(candidate.read_text(), REFERENCE)
            self.assertEqual(run_manager._manifest(run)['starter_sha256'], run_manager._hash(candidate))
        (left / 'workspace/tasks/fixture/candidate.py').write_text('changed')
        self.assertEqual((right / 'workspace/tasks/fixture/candidate.py').read_text(), REFERENCE)
        self.assertIn('-999', (self.source / 'candidate.py').read_text())

    def test_run_creation_does_not_execute_source_submission(self):
        (self.source / 'candidate.py').write_text('raise RuntimeError("must not import")\n')
        run = self.create(edited=False)
        self.assertEqual((run / 'workspace/tasks/fixture/candidate.py').read_text(), REFERENCE)

    def test_revision_links_task_grader_and_prompt_bytes(self):
        package = self.root / 'speedupmark'
        package.mkdir()
        for name in ('__init__.py', '__main__.py', 'catalog.py', 'harness.py', 'revision.py',
                     'run_manager.py',
                     'run_prompt.txt', 'suites.py', 'task.py'):
            (package / name).write_bytes((run_manager.ROOT / 'speedupmark' / name).read_bytes())
        original, files = task_revision(self.source, package)
        self.assertIn('tasks/fixture/task_spec.py', files)
        self.assertIn('speedupmark/harness.py', files)
        self.assertIn('speedupmark/run_prompt.txt', files)
        (self.source / 'candidate.py').write_text('changed candidate')
        self.assertEqual(task_revision(self.source, package)[0], original)
        for path in (self.source / 'task_spec.py', self.source / 'reference.py', package / 'harness.py',
                     package / 'run_manager.py',
                     package / 'run_prompt.txt'):
            before = path.read_bytes()
            path.write_bytes(before + b'\n')
            self.assertNotEqual(task_revision(self.source, package)[0], original)
            path.write_bytes(before)
        self.assertEqual(task_revision(self.source, package)[0], original)

    def test_managed_run_rejects_controller_revision_drift(self):
        run = self.create()
        manifest = run_manager._manifest(run)
        manifest['task_revision_files']['speedupmark/run_manager.py'] = '0' * 64
        run_manager._write_json(run / 'manifest.json', manifest)
        row = run_manager.evaluate_run(run)
        self.assertFalse(row['correct'])
        self.assertIn('controller: run_manager.py changed since run creation',
                      row['integrity_errors'])

    def test_unchanged_starter_has_no_measurements_or_verified_checkpoint(self):
        run = self.create(edited=False)
        development = run_manager.evaluate_run(run)
        self.assertFalse(development['correct'])
        self.assertEqual(development['speedup'], 0)
        for report in development['reports']:
            self.assertEqual(report['results'], [])
            self.assertEqual(report['errors'][0]['code'], 'unchanged_candidate')
            self.assertIsNone(report['geomean_speedup'])
        final = run_manager.finish_run(run)
        self.assertIsNone(final['best_development'])
        self.assertFalse(final['final']['correct'])
        self.assertIn('unchanged_candidate', json.dumps(final['final']))

    def test_shared_grading_command_logs_snapshot_and_result(self):
        run = self.create()
        completed = subprocess.run([sys.executable, 'grade.py'], cwd=run / 'workspace', capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        row = run_manager.history(run)[0]
        self.assertTrue(row['correct'])
        self.assertEqual(row['task_version'], '1.0.0')
        self.assertEqual(row['reports'][0]['results'][0]['task_version'], '1.0.0')
        self.assertEqual(row['task_revision'], row['reports'][0]['results'][0]['task_revision'])
        self.assertEqual(row['task_revision'], run_manager._manifest(run)['task_revision'])
        self.assertGreaterEqual(row['elapsed_seconds'], row['started_elapsed_seconds'])
        snapshot_candidate = run / row['snapshot'] / 'tasks/fixture/candidate.py'
        before = snapshot_candidate.read_bytes()
        (run / 'workspace/tasks/fixture/candidate.py').write_text('def solve(problem): return -1\n')
        bad = run_manager.evaluate_run(run)
        self.assertFalse(bad['correct'])
        self.assertEqual(bad['speedup'], 0)
        self.assertEqual(snapshot_candidate.read_bytes(), before)
        self.assertNotEqual(row['candidate_sha256'], bad['candidate_sha256'])

    def test_cli_create_requires_optimizer_identity(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run_manager.main(['create', 'fixture', '--root', str(self.root / 'runs')])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            run_manager.main(['create', 'fixture', '--root', str(self.root / 'runs'), '--samples', '1',
                              '--harness', 'test-harness', '--model', 'test-model', '--effort', 'test-effort'])
        paths = json.loads(output.getvalue())
        self.assertEqual(paths['prompt'], str(pathlib.Path(paths['workspace']) / 'prompt.md'))
        self.assertTrue(pathlib.Path(paths['prompt']).is_file())
        self.assertEqual(run_manager._manifest(pathlib.Path(paths['run']))['optimizer'],
                         {'harness': 'test-harness', 'model': 'test-model', 'effort': 'test-effort'})

    def test_explicit_seed_replays_declared_cases_in_both_phases(self):
        run = self.create(seed=41)
        development = run_manager.evaluate_run(run)
        summary = run_manager.finish_run(run, usage={'output_tokens': 123, 'cost_usd': 0.02})
        final = summary['final']
        self.assertTrue(development['correct'])
        self.assertTrue(final['correct'])
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(_sizes(development), [4, 8])
        self.assertEqual(_sizes(final), [4, 8])
        self.assertEqual([report['config']['seed'] for report in development['reports']], [41, 42])
        self.assertEqual([report['config']['seed'] for report in final['reports']], [41, 42])
        self.assertEqual(final['seed_source'], 'explicit')
        self.assertEqual(summary['usage']['output_tokens'], 123)
        with self.assertRaises(ValueError):
            run_manager.evaluate_run(run)
        with self.assertRaises(ValueError):
            run_manager.finish_run(run)

    def test_single_declared_case_is_not_doubled(self):
        (self.source / 'task_spec.py').write_text(FIXTURE.replace('grading_cases = (4, 8)', 'grading_cases = (5,)'))
        run = self.create(seed=7)
        development = run_manager.evaluate_run(run)
        final = run_manager.finish_run(run)['final']
        self.assertEqual(_sizes(development), [5])
        self.assertEqual(_sizes(final), [5])
        self.assertEqual(final['reports'][0]['config']['seed'], 7)
        self.assertEqual(development['reports'][0]['config']['seed'], 7)

    def test_default_draws_and_records_fresh_seeds_for_every_evaluation(self):
        run = self.create()
        self.assertIsNone(run_manager._manifest(run)['seed'])
        with patch('speedupmark.harness.secrets.randbits', side_effect=[101, 901, 501]) as entropy:
            rows = [run_manager.evaluate_run(run), run_manager.evaluate_run(run),
                    run_manager.finish_run(run)['final']]
        self.assertEqual(entropy.call_count, 3)
        for row, base in zip(rows, [101, 901, 501]):
            self.assertTrue(row['correct'])
            self.assertEqual(row['seed'], base)
            self.assertEqual(row['seed_source'], 'random')
            self.assertEqual([r['config']['seed'] for r in row['reports']], [base, base + 1])
            plan = json.loads((run / row['seed_plan']).read_text())
            self.assertEqual(plan['cases'], [{'n': 4, 'seed': base}, {'n': 8, 'seed': base + 1}])
            self.assertEqual(plan['seed'], row['seed'])
            self.assertEqual(plan['samples'], 1)

    def test_size_override_replaces_declared_cases_for_both_phases(self):
        run = self.create(n=6)
        development = run_manager.evaluate_run(run)
        final = run_manager.finish_run(run)['final']
        self.assertEqual(_sizes(development), [6])
        self.assertEqual(_sizes(final), [6])

    def test_missing_grading_cases_fail_without_inventing_a_size(self):
        (self.source / 'task_spec.py').write_text(FIXTURE.replace('    grading_cases = (4, 8)\n', ''))
        row = run_manager.evaluate_run(self.create())
        self.assertFalse(row['correct'])
        self.assertEqual(row['reports'], [])
        self.assertEqual(row['speedup'], 0)
        self.assertIn('grading_cases', json.dumps(row))

    def test_workspace_omits_the_entire_benchmark_package(self):
        run = self.create()
        self.assertFalse((run / 'workspace/speedupmark').exists())
        self.assertFalse((run / 'workspace/tasks/fixture/task_spec.py').exists())
        self.assertTrue((run / 'baseline/speedupmark/harness.py').is_file())
        self.assertTrue((run / 'baseline/speedupmark/run_manager.py').is_file())
        self.assertFalse((run / 'workspace/speedupmark/harness.py').exists())

    def test_final_grades_even_without_agent_evaluations(self):
        summary = run_manager.finish_run(self.create())
        self.assertIsNone(summary['best_development'])
        self.assertTrue(summary['final']['correct'])
        self.assertEqual(summary['evaluations'], 1)
        self.assertTrue(all(value is None for value in summary['best_development_speedup_at_seconds'].values()))

    def test_final_selects_best_recorded_candidate_despite_later_regressions(self):
        run = self.create()
        candidate = run / 'workspace/tasks/fixture/candidate.py'
        candidate.write_text('def solve(problem): return sum(value for value in problem)\n')
        best = run_manager.evaluate_run(run)
        candidate.write_text('import time\ndef solve(problem):\n'
                             '    time.sleep(0.01)\n    return sum(problem)\n')
        slower = run_manager.evaluate_run(run)
        self.assertGreater(best['speedup'], slower['speedup'])
        candidate.write_text('def solve(problem): return -1\n')
        self.assertFalse(run_manager.evaluate_run(run)['correct'])
        final = run_manager.finish_run(run)['final']
        self.assertTrue(final['correct'])
        self.assertEqual(final['candidate_sha256'], best['candidate_sha256'])
        self.assertEqual(final['selected_development_evaluation'], best['evaluation'])
        self.assertNotEqual(run_manager._hash(candidate), final['candidate_sha256'])

    def test_selected_best_must_pass_independent_final_grading(self):
        run = self.create()
        (run / 'workspace/tasks/fixture/candidate.py').write_text(
            'def solve(problem):\n'
            '    return sum(problem) if problem[-1] < 1000 else -1\n')
        with patch('speedupmark.harness.secrets.randbits', side_effect=[10, 2000]):
            best = run_manager.evaluate_run(run)
            self.assertTrue(best['correct'])
            final = run_manager.finish_run(run)['final']
        self.assertEqual(final['selected_development_evaluation'], best['evaluation'])
        self.assertFalse(final['correct'])

    def test_seed_plan_exists_before_a_grader_crash(self):
        run = self.create(seed=73)

        def fail_grader(*args, **kwargs):
            plans = list((run / 'grading-plans').glob('*.json'))
            self.assertEqual(len(plans), 1)
            plan = json.loads(plans[0].read_text())
            self.assertEqual(plan['cases'], [{'n': 4, 'seed': 73}, {'n': 8, 'seed': 74}])
            raise OSError('simulated grader crash')

        with patch.object(run_manager.subprocess, 'run', side_effect=fail_grader):
            row = run_manager.evaluate_run(run)
        self.assertFalse(row['correct'])
        self.assertIn('simulated grader crash', row['errors'])
        self.assertTrue((run / row['seed_plan']).is_file())

    def test_modified_selected_snapshot_invalidates_run(self):
        run = self.create()
        best = run_manager.evaluate_run(run)
        (run / best['snapshot'] / 'tasks/fixture/candidate.py').write_text(
            'def solve(problem): return sum(value for value in problem)\n')
        summary = run_manager.finish_run(run)
        self.assertEqual(summary['status'], 'integrity_failed')
        self.assertFalse(summary['final']['correct'])
        self.assertIn('selected candidate snapshot changed or missing', summary['integrity_errors'])

    def test_best_selection_rejects_invalid_measurements(self):
        valid = {'phase': 'development', 'correct': True, 'integrity': 'passed', 'speedup': 2}
        rows = [valid] + [valid | change for change in (
            {'phase': 'final'}, {'correct': False}, {'integrity': 'failed'},
            {'speedup': float('inf')}, {'speedup': float('nan')}, {'speedup': -1}, {'speedup': True})]
        self.assertEqual(run_manager._verified_development(rows), [valid])

    def test_interrupted_snapshot_does_not_block_finalization(self):
        run = self.create()
        (run / 'snapshots/0001-development').mkdir(parents=True)
        summary = run_manager.finish_run(run)
        self.assertEqual(summary['final']['evaluation'], 2)
        self.assertTrue(summary['final']['correct'])

    def test_grading_timeout_is_logged(self):
        run = self.create(grade_timeout=0.2)
        (run / 'workspace/tasks/fixture/candidate.py').write_text(
            'import time\ndef solve(problem):\n    time.sleep(10)\n    return sum(problem)\n'
        )
        row = run_manager.evaluate_run(run)
        self.assertFalse(row['correct'])
        self.assertIn('timeout', json.dumps(row).lower())
        self.assertEqual(len(run_manager.history(run)), 1)

    def test_launch_command_gets_prompt_logs_progress_and_finishes(self):
        agent = self.root / 'fake_agent.py'
        agent.write_text('''import os, subprocess, sys
from pathlib import Path
Path('tasks/fixture/candidate.py').write_text('def solve(problem): return sum(reversed(problem))\\n')
assert 'Read README.md' in sys.stdin.read()
assert os.path.isfile(os.environ['SPEEDUPMARK_PROMPT_FILE'])
assert os.path.isfile(sys.argv[1])
subprocess.run([sys.executable, 'grade.py'], check=True)
print('agent finished')
''')
        run, summary = run_manager.launch_run('fixture', [sys.executable, str(agent), '{prompt_file}'],
                                      root=self.root / 'runs', samples=1, safety_timeout=10,
                                      **PROVENANCE)
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(summary['evaluations'], 2)
        self.assertTrue(summary['final']['correct'])
        self.assertIn('agent finished', (run / 'agent.stdout.log').read_text())
        self.assertTrue((run / 'summary.json').exists())

    def test_agent_timeout_and_failure_still_trigger_final_grading(self):
        for command, expected in (([sys.executable, '-c', 'import time; time.sleep(10)'], 'timed_out'),
                                  ([sys.executable, '-c', 'raise SystemExit(7)'], 'agent_failed')):
            with self.subTest(expected=expected):
                _, summary = run_manager.launch_run(
                    'fixture', command, root=self.root / 'runs', samples=1,
                    safety_timeout=0.2, **PROVENANCE,
                )
                self.assertEqual(summary['status'], expected)
                self.assertFalse(summary['final']['correct'])
                self.assertIn('unchanged_candidate', json.dumps(summary['final']))

    def test_checkpoint_curve_never_uses_future_or_final_results(self):
        run = self.create()
        rows = [
            {'phase': 'development', 'correct': True, 'elapsed_seconds': 20, 'speedup': 1.5},
            {'phase': 'development', 'correct': True, 'elapsed_seconds': 80, 'speedup': 2},
            {'phase': 'development', 'correct': False, 'elapsed_seconds': 350, 'speedup': 1000},
            {'phase': 'development', 'correct': True, 'elapsed_seconds': 610, 'speedup': 3},
            {'phase': 'final', 'correct': True, 'elapsed_seconds': 700, 'speedup': 1000},
        ]
        (run / 'evaluations.jsonl').write_text(''.join(json.dumps(row | {'integrity': 'passed'}) + '\n' for row in rows))
        report = run_manager.report_run(run)
        self.assertEqual(report['best_development_speedup_at_seconds'], {'60': 1.5, '300': 2, '600': 2})
        self.assertEqual(report['first_observed_improvement_seconds'], 20)
        self.assertEqual(report['best_development']['speedup'], 3)

    def test_run_controller_does_not_embed_task_specs(self):
        source = pathlib.Path(run_manager.__file__).read_text()
        self.assertNotIn('simd_traversal_kernel', source.lower())
        self.assertNotIn('SIMDTraversalKernelTask', source)
        self.assertNotIn('problem_size', source)
        # The shared protocol module is named once. Per-task specs are not.
        self.assertNotIn('AGENT_SPEEDUPMARK', source)

    def test_reference_is_the_only_starter_source(self):
        reference = self.source / 'reference.py'
        starter = 'def solve(problem): return sum(value for value in problem)\n'
        reference.write_text(starter)
        run = self.create(edited=False)
        self.assertEqual((run / 'workspace/tasks/fixture/candidate.py').read_bytes(),
                         reference.read_bytes())
        self.assertEqual(run_manager._manifest(run)['reference_sha256'],
                         run_manager._manifest(run)['starter_sha256'])

    def test_simd_traversal_kernel_managed_run_grades_declared_sizes_and_finishes(self):
        source = run_manager.ROOT / 'tasks/simd_traversal_kernel'
        with patch.object(run_manager, 'TASK_ROOT', source.parent), patch.object(run_manager, 'discover_tasks', return_value=[source]):
            run = run_manager.create_run(source.name, root=self.root / 'runs', samples=1, **PROVENANCE)
        self.assertEqual((run / 'workspace/tasks/simd_traversal_kernel/candidate.py').read_bytes(),
                         (source / 'reference.py').read_bytes())
        candidate = run / 'workspace/tasks/simd_traversal_kernel/candidate.py'
        candidate.write_bytes(candidate.read_bytes() + b'\n# Test submission.\n')
        row = run_manager.evaluate_run(run)
        self.assertTrue(row['correct'])
        self.assertEqual(_sizes(row), [32, 128])
        self.assertEqual(row['integrity'], 'passed')
        summary = run_manager.finish_run(run)
        self.assertTrue(summary['final']['correct'])
        self.assertEqual(summary['final']['speedup'], 1.0)


    def test_modified_benchmark_files_are_rejected_before_execution(self):
        targets = ('workspace/speedupmark/task.py', 'workspace/speedupmark/harness.py',
                   'workspace/tasks/fixture/task_spec.py', 'workspace/tasks/fixture/reference.py',
                   'workspace/grade.py', 'baseline/speedupmark/harness.py')
        for target in targets:
            with self.subTest(target=target):
                run = self.create()
                marker = run / 'untrusted-code-ran'
                (run / target).parent.mkdir(parents=True, exist_ok=True)
                (run / target).write_text(
                    f'from pathlib import Path\nPath({str(marker)!r}).touch()\n'
                    'print(\'{"correct": true, "geomean_speedup": 999999}\')\n'
                )
                row = run_manager.evaluate_run(run)
                self.assertEqual(row['integrity'], 'failed')
                self.assertFalse(row['correct'])
                self.assertEqual(row['speedup'], 0)
                self.assertEqual(row['reports'], [])
                self.assertFalse(marker.exists())

    def test_verifier_bypass_stays_invalid_after_file_restoration(self):
        run = self.create()
        task = run / 'baseline/tasks/fixture/task_spec.py'
        original = task.read_bytes()
        (run / 'workspace/tasks/fixture/candidate.py').write_text('def solve(problem): return -1\n')
        self.assertFalse(run_manager.evaluate_run(run)['correct'])
        with task.open('a') as stream:
            stream.write('\nTASK.is_solution = lambda problem, proposed: True\n')
        self.assertEqual(run_manager.evaluate_run(run)['integrity'], 'failed')
        task.write_bytes(original)
        summary = run_manager.finish_run(run)
        self.assertEqual(summary['status'], 'integrity_failed')
        self.assertFalse(summary['final']['correct'])
        self.assertTrue(summary['integrity_errors'])

    def test_missing_added_and_symlinked_submission_files_fail(self):
        for kind in ('missing', 'added', 'symlink'):
            with self.subTest(kind=kind):
                run = self.create()
                if kind == 'missing':
                    (run / 'baseline/speedupmark/harness.py').rename(run / 'removed-harness.py')
                elif kind == 'added':
                    (run / 'workspace/json.py').write_text('# shadow module\n')
                else:
                    candidate = run / 'workspace/tasks/fixture/candidate.py'
                    candidate.rename(run / 'external-candidate.py')
                    candidate.symlink_to(run / 'external-candidate.py')
                self.assertEqual(run_manager.evaluate_run(run)['integrity'], 'failed')

    def test_scratch_files_are_allowed_but_never_submitted(self):
        run = self.create()
        scratch = run / 'workspace/scratch'
        scratch.mkdir()
        (scratch / 'experiment.py').write_text('raise RuntimeError("not submitted")\n')
        row = run_manager.evaluate_run(run)
        self.assertTrue(row['correct'])
        self.assertFalse((run / row['snapshot'] / 'scratch').exists())

    def test_source_changes_do_not_change_frozen_grading(self):
        run = self.create()
        (self.source / 'task_spec.py').write_text('raise RuntimeError("source changed")\n')
        row = run_manager.evaluate_run(run)
        self.assertTrue(row['correct'])
        self.assertEqual((run / row['snapshot'] / 'tasks/fixture/task_spec.py').read_text(), FIXTURE)

    def test_candidate_printing_fake_results_cannot_pass(self):
        run = self.create()
        (run / 'workspace/tasks/fixture/candidate.py').write_text(
            'def solve(problem):\n'
            '    print(\'{"correct": true, "geomean_speedup": 999999}\')\n'
            '    return -1\n'
        )
        row = run_manager.evaluate_run(run)
        self.assertFalse(row['correct'])
        self.assertEqual(row['speedup'], 0)

    def test_writes_during_candidate_execution_invalidate_result(self):
        run = self.create()
        (run / 'workspace/tasks/fixture/candidate.py').write_text(
            'from pathlib import Path\n'
            'def solve(problem):\n'
            '    path = Path(__file__).with_name("task_spec.py")\n'
            '    path.write_text(path.read_text() + "\\n# modified during execution\\n")\n'
            '    return sum(problem)\n'
        )
        row = run_manager.evaluate_run(run)
        self.assertEqual(row['integrity'], 'failed')
        self.assertFalse(row['correct'])
        self.assertEqual(row['speedup'], 0)
        self.assertTrue(any('grading snapshot' in error for error in row['integrity_errors']))


class GradingCaseContractTests(unittest.TestCase):
    def test_size_metadata_never_executes_candidate_or_task_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            task = root / 'tasks/fixture'
            task.mkdir(parents=True)
            (task / 'task_spec.py').write_text(
                'raise RuntimeError("must not import")\n' + FIXTURE)
            (task / 'candidate.py').write_text('while True: pass\n')
            self.assertEqual(run_manager._grading_sizes(root, 'fixture', None), (4, 8))

    def test_task_versions_require_canonical_semver(self):
        from speedupmark.task import declared_task_version

        class Spec:
            name = 'spec'

        with self.assertRaisesRegex(ValueError, 'task_version'):
            declared_task_version(Spec())
        for invalid in (None, 1, '1', '1.0', '01.0.0', '1.0.0-beta', '1.0.0\n'):
            Spec.task_version = invalid
            with self.subTest(version=invalid), self.assertRaisesRegex(ValueError, 'task_version'):
                declared_task_version(Spec())
        Spec.task_version = '1.0.0'
        self.assertEqual(declared_task_version(Spec()), '1.0.0')

    def test_grading_cases_reject_missing_and_invalid_declarations(self):
        from speedupmark.task import grading_cases

        class Spec:
            name = 'spec'

        with self.assertRaises(ValueError):
            grading_cases(Spec())
        Spec.grading_cases = (4, True)
        with self.assertRaises(ValueError):
            grading_cases(Spec())
        Spec.grading_cases = ()
        with self.assertRaises(ValueError):
            grading_cases(Spec())
        Spec.grading_cases = (4, 8)
        self.assertEqual(grading_cases(Spec()), (4, 8))

    def test_every_implemented_spec_declares_grading_cases(self):
        from speedupmark.harness import discover_tasks, load_task
        from speedupmark.task import grading_cases

        found = discover_tasks()
        self.assertTrue(found)
        for path in found:
            with self.subTest(task=path.name):
                tree = ast.parse((path / 'task_spec.py').read_text())
                declarations = []
                for node in ast.walk(tree):
                    if not isinstance(node, ast.ClassDef):
                        continue
                    values = {}
                    for statement in node.body:
                        if isinstance(statement, ast.Assign):
                            for target in statement.targets:
                                if isinstance(target, ast.Name):
                                    try:
                                        values[target.id] = ast.literal_eval(statement.value)
                                    except (ValueError, TypeError):
                                        pass
                    if {'name', 'task_version', 'grading_cases'} <= values.keys():
                        declarations.append(values)
                self.assertEqual(len(declarations), 1)
                declared = declarations[0]
                self.assertEqual(declared['name'], path.name)
                self.assertRegex(
                    declared['task_version'],
                    r'^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$',
                )
                sizes = declared['grading_cases']
                self.assertIsInstance(sizes, (tuple, list))
                self.assertTrue(sizes)
                self.assertTrue(all(type(size) is int and size > 0 for size in sizes))
                try:
                    task = load_task(path)
                except ImportError as exc:
                    if (
                        'requires optional dependenc' in str(exc)
                        and 'requirements-numerical.txt' in str(exc)
                    ):
                        continue
                    raise
                sizes = grading_cases(task)
                self.assertGreaterEqual(len(sizes), 1)
                self.assertEqual(task.name, path.name)
                self.assertRegex(task.task_version, r'^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$')


class SuiteSummaryTests(unittest.TestCase):
    def item(self, task, speedup=2, status='completed', correct=True, integrity='passed'):
        return {'run': '/runs/' + task, 'summary': {
            'task': task, 'task_version': '1.0.0', 'status': status,
            'integrity': integrity, 'agent_elapsed_seconds': 12,
            'final': {'correct': correct, 'speedup': speedup},
            'best_development': {'speedup': 999},
        }}

    def test_aggregate_uses_all_final_scores_with_equal_weight(self):
        report = run_manager.summarize_suite(('a', 'b'), [self.item('a', 2), self.item('b', 8)])
        self.assertTrue(report['correct'])
        self.assertEqual([row['task_version'] for row in report['results']], ['1.0.0', '1.0.0'])
        self.assertAlmostEqual(report['geomean_speedup'], 4)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            run_manager._print_suite(report)
        self.assertIn('2/2 passed; 4.000x geometric mean', output.getvalue())
        self.assertIn('1.0.0', output.getvalue())

    def test_unrun_expected_tasks_keep_their_contract_versions(self):
        tasks = ('affine_gap_sequence_alignment', 'incremental_multiway_join')
        versions = run_manager._task_versions(tasks)
        report = run_manager.summarize_suite(tasks, [], task_versions=versions)
        self.assertEqual(report['score_factors'], [
            {'task': task, 'task_version': versions[task], 'task_revision': None,
             'credited_speedup': 1.0}
            for task in tasks
        ])

    def test_missing_duplicate_failed_invalid_and_interrupted_runs_count_as_one_x(self):
        cases = ([self.item('a')], [self.item('a'), self.item('a')],
                 [self.item('a'), self.item('b', correct=False)],
                 [self.item('a'), self.item('b', integrity='failed')],
                 [self.item('a'), self.item('b', status='timed_out')],
                 [self.item('a'), self.item('b', status='interrupted')],
                 [self.item('a'), self.item('b', speedup=float('nan'))])
        for items in cases:
            with self.subTest(items=items):
                report = run_manager.summarize_suite(('a', 'b'), items)
                self.assertFalse(report['correct'])
                self.assertAlmostEqual(report['geomean_speedup'], 2 ** 0.5)
                self.assertEqual(report['results'][-1]['credited_speedup'],
                                 2 if report['results'][-1]['passed'] else 1)
                self.assertEqual(len(report['score_factors']), 2)
                self.assertEqual(report['score_factors'][1]['credited_speedup'], 1)

    def test_correct_slowdown_keeps_measured_credit(self):
        report = run_manager.summarize_suite(('a',), [self.item('a', speedup=0.5)])
        self.assertTrue(report['correct'])
        self.assertEqual(report['results'][0]['speedup'], 0.5)
        self.assertEqual(report['results'][0]['credited_speedup'], 0.5)
        self.assertEqual(report['geomean_speedup'], 0.5)

    def test_empty_suite_has_no_score(self):
        report = run_manager.summarize_suite((), [])
        self.assertIsNone(report['geomean_speedup'])
        self.assertFalse(report['correct'])

    def test_print_suite_marks_failed_tasks_as_counted_at_one_x(self):
        report = run_manager.summarize_suite(
            ('a', 'b'), [self.item('a', 2), self.item('b', status='timed_out')])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            run_manager._print_suite(report)
        self.assertIn('1/2 passed; 1.414x geometric mean (1 counted as 1.000x)',
                      output.getvalue())

    def test_partial_suite_shows_unrun_task_and_its_credit(self):
        report = run_manager.summarize_suite(('a', 'b'), [self.item('a', 2)])
        self.assertEqual(report['score_factors'], [
            {'task': 'a', 'task_version': '1.0.0', 'task_revision': None,
             'credited_speedup': 2},
            {'task': 'b', 'task_version': None, 'task_revision': None,
             'credited_speedup': 1.0},
        ])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            run_manager._print_suite(report)
        self.assertIn('NOT RUN', output.getvalue())

    def test_cli_saves_partial_suite_and_reports_completed_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            def launch(task, command, **options):
                saved = json.loads((options['root'] / 'suite.json').read_text())
                self.assertEqual(len(saved['runs']), 0 if task == 'a' else 1)
                self.assertAlmostEqual(saved['geomean_speedup'], 1.0 if task == 'a' else 2 ** 0.5)
                item = self.item(task)
                return pathlib.Path(item['run']), item['summary']
            with patch.object(run_manager, 'selected_tasks', return_value=('a', 'b')), \
                 patch.object(run_manager, '_task_revisions', return_value={'a': 'a' * 64, 'b': 'b' * 64}), \
                 patch.object(run_manager, 'launch_run', side_effect=launch):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    run_manager.main([
                        'launch', 'smoke', '--root', str(root),
                        '--harness', 'test-harness', '--model', 'test-model', '--effort', 'test-effort',
                        '--command', 'fake-agent',
                    ])
            suite = next(root.glob('suite-*/suite.json'))
            self.assertTrue(json.loads(suite.read_text())['correct'])
            self.assertIn('2/2 passed', output.getvalue())
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                run_manager.main(['report', str(suite.parent), '--json'])
            self.assertTrue(json.loads(output.getvalue())['correct'])

    def test_launch_requires_an_explicit_agent_command(self):
        with self.assertRaises(ValueError):
            run_manager.launch_run('missing', None)


if __name__ == '__main__':
    unittest.main()
