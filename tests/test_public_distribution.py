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



ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_public", ROOT / "scripts/prepare_public.py")
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


class PublicDistributionTests(unittest.TestCase):
    def test_inventory_excludes_private_material_and_publishes_single_references(self):
        files = publication.public_payloads()
        for name in files:
            self.assertNotIn(name.split('/')[0], ('wiki', 'runs', 'harness', 'watcher', '.git', '.github'))
            self.assertNotIn('__pycache__', name)
            self.assertNotIn(name, ('AGENTS.md', 'ROADMAP.MD'))
        references = [name for name in files if name.startswith('tasks/') and name.endswith('/reference.py')]
        self.assertEqual(len(references), 50)
        self.assertFalse(any(name.endswith('/candidate.py') for name in files))
        self.assertIn('examples/example_gzip/README.md', files)
        self.assertEqual(files['examples/example_gzip/reference.py'],
                         (ROOT / 'examples/example_gzip/reference.py').read_bytes())
        for name in references:
            self.assertEqual(files[name], (ROOT / name).read_bytes())

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
class StagedSourceTests(unittest.TestCase):
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

    def test_staged_references_pass_without_touching_local_submissions(self):
        reference = b"def solve(problem): return sum(problem)\n"
        root = self.repository({"tasks/example/reference.py": reference,
                                "examples/example_gzip/reference.py": reference})
        candidate = root / "tasks/example/candidate.py"
        solution = b"def solve(problem): return 42\n"
        candidate.write_bytes(solution)
        index_before = (root / ".git/index").read_bytes()
        self.assertEqual(publication.check_staged_sources(root), 2)
        self.assertEqual(candidate.read_bytes(), solution)
        self.assertEqual((root / ".git/index").read_bytes(), index_before)

    def test_any_staged_candidate_is_rejected(self):
        for name in ("tasks/example/candidate.py", "examples/example_gzip/candidate.py"):
            with self.subTest(path=name):
                root = self.repository({"tasks/example/reference.py": b"def solve(problem): return 0\n",
                                        name: b"def solve(problem): return 0\n"})
                index_before = (root / ".git/index").read_bytes()
                with self.assertRaisesRegex(ValueError, name):
                    publication.check_staged_sources(root)
                self.assertEqual((root / ".git/index").read_bytes(), index_before)

    def test_symlinked_reference_is_rejected(self):
        name = "tasks/example/reference.py"
        root = self.repository({name: b"def solve(problem): return 0\n"})
        reference = root / name
        reference.rename(root / 'external-reference.py')
        reference.symlink_to(root / 'external-reference.py')
        subprocess.run(['git', 'add', '--', name], cwd=root, check=True, capture_output=True)
        with self.assertRaisesRegex(ValueError, name):
            publication.check_staged_sources(root)

    def test_missing_git_or_reference_entries_fail_clearly(self):
        root = pathlib.Path(tempfile.mkdtemp(prefix="speedupmark-public-no-git-test-"))
        with self.assertRaisesRegex(ValueError, "could not inspect the Git index"):
            publication.check_staged_sources(root)
        root = self.repository({})
        with self.assertRaisesRegex(ValueError, "no staged reference files"):
            publication.check_staged_sources(root)


if __name__ == '__main__':
    unittest.main()
