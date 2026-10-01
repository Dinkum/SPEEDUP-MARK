"""Exact set-similarity boundaries and independent component verification."""

import pathlib
import random
import unittest
from unittest.mock import patch

from speedupmark.harness import load_task


ROOT = pathlib.Path(__file__).resolve().parents[1]


class DocumentClusteringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = load_task(ROOT / "tasks/near_duplicate_document_clustering")

    def check(self, documents, threshold, expected, width=1):
        problem = {"documents": documents, "threshold": threshold,
                   "shingle_width": width}
        self.assertEqual(self.task.solve(problem), expected)
        self.assertTrue(self.task.is_solution(problem, expected))
        return problem

    def test_transitive_chain_relabels_shuffled_indices(self):
        # Adjacent sets have Jaccard 1/2; endpoints only 1/5.
        self.check(("a b c", "b c d", "c d e", "x"), (1, 2), (0, 0, 0, 3))
        self.check(("x", "c d e", "a b c", "b c d"), (1, 2), (0, 1, 1, 1))

    def test_exact_threshold_and_one_less_shared_token(self):
        self.check(("a b c", "b c d"), (1, 2), (0, 0))
        self.check(("a b c", "c d e"), (1, 2), (0, 1))
        self.check(("a a b b", "a b", "", ""), (1, 1), (0, 0, 2, 3))
        self.check(("a", "a", "b", ""), (1, 1), (0, 0, 2, 3), width=3)

    def test_verifier_does_not_call_reference(self):
        problem = self.check(("a b c", "b c d", "x"), (1, 2), (0, 0, 2))
        with patch.dict(self.task.solve.__globals__, {"_clusters": lambda _: (0, 1, 2)}):
            self.assertTrue(self.task.is_solution(problem, (0, 0, 2)))
            self.assertFalse(self.task.is_solution(problem, (0, 1, 2)))

    def test_random_small_graphs_match_pairwise_reference(self):
        rng = random.Random(927)
        for _ in range(80):
            documents = tuple(" ".join(rng.choices("abcdef", k=rng.randrange(12)))
                              for _ in range(rng.randrange(1, 15)))
            problem = {"documents": documents, "shingle_width": rng.randrange(1, 5),
                       "threshold": rng.choice(((1, 2), (2, 3), (4, 5), (1, 1)))}
            self.assertTrue(self.task.is_solution(problem, self.task.solve(problem)))

    def test_generated_profiles_are_deterministic_and_nontrivial(self):
        for seed in range(3):
            problem = self.task.generate_problem(90, seed)
            self.assertEqual(problem, self.task.generate_problem(90, seed))
            self.assertGreater(len({len(doc.split()) for doc in problem["documents"]}), 3)
            labels = self.task.solve(problem)
            self.assertGreater(len(set(labels)), 1)
            self.assertLess(len(set(labels)), len(labels))
            self.assertTrue(self.task.is_solution(problem, labels))

    def test_generated_components_require_transitive_closure(self):
        problem = self.task.generate_problem(360, 0)
        labels = self.task.solve(problem)
        width = problem["shingle_width"]
        numerator, denominator = problem["threshold"]
        shingles = []
        for document in problem["documents"]:
            words = document.split()
            shingles.append({tuple(words[start:start + width])
                             for start in range(len(words) - width + 1)})
        self.assertTrue(any(
            labels[left] == labels[right]
            and len(shingles[left] & shingles[right]) * denominator
            < len(shingles[left] | shingles[right]) * numerator
            for left in range(len(labels)) for right in range(left + 1, len(labels))
        ), "Generated clustering must include connected but dissimilar endpoints")


if __name__ == "__main__":
    unittest.main()
