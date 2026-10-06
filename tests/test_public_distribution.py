"""The reviewed source artifact excludes private content and remains self-contained."""

import hashlib
import importlib.util
import json
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest
import zipfile

from speedupmark.task import DEFAULT_CANDIDATE


ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_public", ROOT / "scripts/prepare_public.py")
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


class PublicDistributionTests(unittest.TestCase):
    def test_inventory_excludes_private_material_and_uses_fresh_starters(self):
        files = publication.public_payloads()
        for name in files:
            self.assertNotIn(name.split('/')[0], ('wiki', 'runs', 'harness', 'watcher', '.git', '.github'))
            self.assertNotIn('__pycache__', name)
            self.assertNotIn(name, ('AGENTS.md', 'ROADMAP.MD'))
        candidates = [name for name in files if name.startswith('tasks/') and name.endswith('/candidate.py')]
        self.assertEqual(len(candidates), 50)
        self.assertIn('examples/example_gzip/README.md', files)
        self.assertNotIn('tasks/example_gzip/candidate.py', files)
        self.assertEqual(files['examples/example_gzip/candidate.py'], DEFAULT_CANDIDATE.encode())
        self.assertTrue(all(files[name] == DEFAULT_CANDIDATE.encode() for name in candidates))

    def test_public_markdown_links_resolve_inside_distribution(self):
        files = publication.public_payloads()
        for name, data in files.items():
            if not name.endswith('.md'):
                continue
            for target in re.findall(r'\]\(([^\s)]+)\)', data.decode()):
                if '://' in target or target.startswith('#'):
                    continue
                path = (ROOT / name).parent / target.split('#', 1)[0]
                with self.subTest(source=name, target=target):
                    self.assertTrue(path.resolve().is_relative_to(ROOT))
                    self.assertIn(path.resolve().relative_to(ROOT).as_posix(), files)

    def test_archive_hashes_match_and_existing_archive_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / 'source.zip'
            publication.build_archive(path)
            before = path.read_bytes()
            with zipfile.ZipFile(path) as archive:
                manifest = json.loads(archive.read('PUBLIC_MANIFEST.json'))
                self.assertEqual(set(archive.namelist()), set(manifest['files']) | {'PUBLIC_MANIFEST.json'})
                for name, digest in manifest['files'].items():
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest)
            with self.assertRaises(FileExistsError):
                publication.build_archive(path)
            self.assertEqual(path.read_bytes(), before)


@unittest.skipUnless(shutil.which("git"), "staged publication checks require Git")
class StagedCandidateTests(unittest.TestCase):
    def repository(self, candidates):
        # Isolated, persistent scratch repos keep tests out of the real index.
        root = pathlib.Path(tempfile.mkdtemp(prefix="speedupmark-public-index-test-"))
        subprocess.run(["git", "init", "--quiet"], cwd=root, check=True, capture_output=True)
        for name, data in candidates.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        if candidates:
            subprocess.run(["git", "add", "--", *candidates], cwd=root,
                           check=True, capture_output=True)
        return root

    def test_staged_starters_pass_without_touching_local_solutions(self):
        starter = DEFAULT_CANDIDATE.encode()
        root = self.repository({"tasks/example/candidate.py": starter,
                                "examples/example_gzip/candidate.py": starter})
        candidate = root / "tasks/example/candidate.py"
        solution = b"def solve(problem, reference_solve):\n    return 42\n"
        candidate.write_bytes(solution)
        index_before = (root / ".git/index").read_bytes()
        self.assertEqual(publication.check_staged_candidates(root), 2)
        self.assertEqual(candidate.read_bytes(), solution)
        self.assertEqual((root / ".git/index").read_bytes(), index_before)

    def test_staged_solution_fails_even_when_working_copy_is_a_starter(self):
        name = "tasks/example/candidate.py"
        solution = b"def solve(problem, reference_solve):\n    return 42\n"
        root = self.repository({name: solution})
        candidate = root / name
        candidate.write_bytes(DEFAULT_CANDIDATE.encode())
        index_before = (root / ".git/index").read_bytes()
        with self.assertRaisesRegex(ValueError, name):
            publication.check_staged_candidates(root)
        self.assertEqual(candidate.read_bytes(), DEFAULT_CANDIDATE.encode())
        self.assertEqual((root / ".git/index").read_bytes(), index_before)

    def test_new_staged_candidates_and_examples_are_checked(self):
        for name in ("tasks/new_task/candidate.py", "examples/new_example/candidate.py"):
            with self.subTest(path=name):
                root = self.repository({"tasks/existing/candidate.py": DEFAULT_CANDIDATE.encode(),
                                        name: b"optimized local solution\n"})
                with self.assertRaisesRegex(ValueError, name):
                    publication.check_staged_candidates(root)

    def test_missing_git_or_candidate_entries_fail_clearly(self):
        root = pathlib.Path(tempfile.mkdtemp(prefix="speedupmark-public-no-git-test-"))
        with self.assertRaisesRegex(ValueError, "could not inspect the Git index"):
            publication.check_staged_candidates(root)
        root = self.repository({})
        with self.assertRaisesRegex(ValueError, "no staged candidate files"):
            publication.check_staged_candidates(root)


if __name__ == '__main__':
    unittest.main()
