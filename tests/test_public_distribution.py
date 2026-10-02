"""The reviewed source artifact excludes private content and remains self-contained."""

import hashlib
import importlib.util
import json
import pathlib
import re
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


if __name__ == '__main__':
    unittest.main()
