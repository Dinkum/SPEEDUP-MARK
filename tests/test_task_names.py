"""Names stay consistent across discovery, specs, documentation, and exports."""

import ast
import pathlib
import tempfile
import unittest

from scripts.sync_task_names import documentation_updates, task_table, validate_task_names
from speedupmark.catalog import TASK_CATALOG
from speedupmark.harness import discover_tasks, load_task
from speedupmark.suites import EXTENDED_TASKS


ROOT = pathlib.Path(__file__).resolve().parents[1]


class TaskNameTests(unittest.TestCase):
    def test_catalog_matches_task_directories_and_spec_references(self):
        validate_task_names()
        self.assertEqual(set(TASK_CATALOG), {path.name for path in discover_tasks()})
        for path in discover_tasks():
            with self.subTest(task=path.name):
                if path.name in EXTENDED_TASKS:
                    task = load_task(path)
                    self.assertEqual(task.name, path.name)
                    self.assertEqual(task.display_name, TASK_CATALOG[path.name].display_name)
                tree = ast.parse((path / 'task_spec.py').read_text())
                displays = [node.value for node in ast.walk(tree)
                            if isinstance(node, ast.Assign)
                            and any(isinstance(target, ast.Name) and target.id == 'display_name'
                                    for target in node.targets)]
                self.assertEqual(len(displays), 1)
                self.assertEqual(ast.unparse(displays[0]), 'TASK_CATALOG[name].display_name')

    def test_generated_documentation_is_current(self):
        self.assertEqual(list(documentation_updates()), [])

    def test_synchronization_preserves_other_readme_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            for name in TASK_CATALOG:
                directory = root / 'tasks' / name
                directory.mkdir(parents=True)
                (directory / 'task_spec.py').write_text(
                    'raise RuntimeError("do not execute")\n'
                    f'class Spec:\n    name = {name!r}\n')
                (directory / 'README.md').write_text('# stale title\n\nBody stays verbatim.\n')
            before = 'Intro stays verbatim.\n\n'
            after = '\n\n## Following section\n\nAlso stays verbatim.\n'
            (root / 'README.md').write_text(before +
                '| Task | Workload | Included in |\n| --- | --- | --- |\n| stale | row | here |' + after)
            updates = documentation_updates(root)
            self.assertEqual(updates[root / 'README.md'], before + task_table() + after)
            for name, metadata in TASK_CATALOG.items():
                self.assertEqual(updates[root / 'tasks' / name / 'README.md'],
                                 f'# {metadata.display_name}\n\nBody stays verbatim.\n')
            path = root / 'tasks' / next(iter(TASK_CATALOG)) / 'task_spec.py'
            path.write_text('class Spec:\n    name = "wrong_identifier"\n')
            with self.assertRaisesRegex(ValueError, 'catalog identifier'):
                validate_task_names(root)


if __name__ == '__main__':
    unittest.main()
