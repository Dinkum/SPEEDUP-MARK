import copy
import pathlib
import pickle
import unittest
import zlib

from speedupmark.harness import load_task


ROOT = pathlib.Path(__file__).resolve().parents[1]
SLUGS = (
    "near_duplicate_document_clustering",
    "labeled_graph_isomorphism",
    "durable_log_recovery",
    "incremental_spreadsheet_recalculation",
    "ranked_bpe_tokenization",
    "sqlite_analytics_reports",
    "dynamic_document_search",
    "streaming_literal_replacement",
    "temporal_asof_join",
)


def task(slug):
    return load_task(ROOT / "tasks" / slug)


def checksum(lsn, payload):
    return zlib.crc32(lsn.to_bytes(8, "little") + payload) & 0xFFFFFFFF


class NewTaskSemanticsTests(unittest.TestCase):
    def test_bpe_duplicate_pairs_keep_the_lowest_rank_and_token_id(self):
        benchmark = task("ranked_bpe_tokenization")
        problem = {
            "data": b"abababc",
            "merges": ((97, 98), (97, 98), (256, 99)),
        }
        expected = (256, 256, 258)
        self.assertEqual(benchmark.solve(problem), expected)
        self.assertTrue(benchmark.is_solution(problem, expected))
        self.assertFalse(benchmark.is_solution(problem, (257, 257, 257, 99)))

    def test_durable_log_stops_at_out_of_range_lsn_and_keeps_other_segments(self):
        benchmark = task("durable_log_recovery")
        for invalid_lsn in (-1, 2**64):
            with self.subTest(lsn=invalid_lsn):
                problem = {
                    "start_lsn": 1,
                    "segments": (
                        {"segment_id": 0, "records": (
                            (1, b"first", checksum(1, b"first")),
                            (invalid_lsn, b"malformed", 0),
                            (2, b"ignored", checksum(2, b"ignored")),
                        )},
                        {"segment_id": 1, "records": (
                            (2, b"second", checksum(2, b"second")),
                        )},
                    ),
                }
                expected = ((1, b"first", 0), (2, b"second", 1))
                self.assertEqual(benchmark.solve(problem), expected)
                self.assertTrue(benchmark.is_solution(problem, expected))

        maximum = 2**64 - 1
        problem = {"start_lsn": maximum, "segments": (
            {"segment_id": 0, "records": ((maximum, b"last", checksum(maximum, b"last")),)},
        )}
        self.assertEqual(benchmark.solve(problem), ((maximum, b"last", 0),))

    def test_near_duplicate_components_short_and_empty_documents(self):
        benchmark = task("near_duplicate_document_clustering")
        problem = {
            "documents": ("a b c d", "a b c x", "zz", "", ""),
            "shingle_width": 2,
            "threshold": (1, 2),
        }
        self.assertEqual(benchmark.solve(problem), (0, 0, 2, 3, 4))
        self.assertFalse(benchmark.is_solution(problem, (0, 0, 2, 3, 3)))
        self.assertFalse(benchmark.is_solution(problem, (False, 0, 2, 3, 4)))

    def test_labeled_graph_positive_and_hard_negative(self):
        benchmark = task("labeled_graph_isomorphism")
        positive = {
            "left": {"labels": (0, 1, 0), "edges": ((0, 1), (1, 2))},
            "right": {"labels": (1, 0, 0), "edges": ((0, 1), (0, 2))},
        }
        mapping = benchmark.solve(positive)
        self.assertTrue(benchmark.is_solution(positive, mapping))
        self.assertFalse(benchmark.is_solution(positive, None))
        self.assertFalse(benchmark.is_solution(positive, (True, 0, 2)))

        negative = benchmark.generate_problem(9, 2)
        self.assertIsNone(benchmark.solve(negative))
        self.assertTrue(benchmark.is_solution(negative, None))

    def test_durable_prefix_duplicate_priority_and_first_gap(self):
        benchmark = task("durable_log_recovery")
        p0, p1, p2 = b"low", b"second", b"high"
        problem = {
            "start_lsn": 1,
            "segments": (
                {"segment_id": 2, "records": ((1, p2, checksum(1, p2)),)},
                {
                    "segment_id": 0,
                    "records": (
                        (1, p0, checksum(1, p0)),
                        (2, b"torn", 0),
                        (2, b"ignored", checksum(2, b"ignored")),
                    ),
                },
                {
                    "segment_id": 1,
                    "records": (
                        (2, p1, checksum(2, p1)),
                        (4, b"later", checksum(4, b"later")),
                    ),
                },
            ),
        }
        expected = ((1, p0, 0), (2, p1, 1))
        self.assertEqual(benchmark.solve(problem), expected)
        self.assertFalse(benchmark.is_solution(problem, ((True, p0, 0), (2, p1, 1))))

    def test_spreadsheet_exact_edit_query_stream(self):
        benchmark = task("incremental_spreadsheet_recalculation")
        problem = {
            "input_count": 2,
            "initial_inputs": (2, 3),
            "formulas": (None, None, ("add", 0, 1), ("scale", 2, 2), ("sub", 3, 0)),
            "operations": (("query", 4), ("set", 0, 4), ("query", 2), ("query", 4)),
        }
        self.assertEqual(benchmark.solve(problem), (8, 7, 10))
        self.assertFalse(benchmark.is_solution(problem, (8, 7, True)))

    def test_ranked_bpe_overlap_and_created_token_ids(self):
        benchmark = task("ranked_bpe_tokenization")
        problem = {"data": b"abaaba", "merges": ((97, 98), (256, 97))}
        self.assertEqual(benchmark.solve(problem), (257, 257))
        overlap = {"data": b"aaa", "merges": ((97, 97),)}
        self.assertEqual(benchmark.solve(overlap), (256, 97))
        self.assertFalse(benchmark.is_solution(problem, (True, True)))

    def test_sqlite_null_duplicate_and_order_semantics(self):
        benchmark = task("sqlite_analytics_reports")
        problem = {
            "rows": (
                (1, None, None, None),
                (1, None, None, None),
                (1, "a", 5, "x"),
                (1, "a", 5, "x"),
                (2, "b", -2, None),
            ),
            "minimum_total": 0,
        }
        expected = (
            (("a", 2, 2, 10, 1), ("b", 1, 1, -2, 0), (None, 2, 0, None, 0)),
            ((1, None, 2), (1, "a", 2)),
            ((1, 10),),
        )
        self.assertEqual(benchmark.solve(problem), expected)
        self.assertTrue(benchmark.is_solution(problem, expected))
        self.assertFalse(benchmark.is_solution(problem, (iter(expected[0]), expected[1], expected[2])))

    def test_dynamic_search_replacement_delete_repeated_query_terms(self):
        benchmark = task("dynamic_document_search")
        problem = {
            "operations": (
                ("add", 1, "a a b"),
                ("add", 2, "a b b"),
                ("query", ("a", "b", "b"), 5),
                ("add", 1, "b"),
                ("query", ("a",), 5),
                ("delete", 2),
                ("delete", 99),
                ("query", ("b",), 5),
            )
        }
        expected = (((2, 5), (1, 4)), ((2, 1),), ((1, 1),))
        self.assertEqual(benchmark.solve(problem), expected)
        self.assertFalse(benchmark.is_solution(problem, (iter(expected[0]), expected[1], expected[2])))

    def test_streaming_replacement_crosses_chunks_and_is_nonrecursive(self):
        benchmark = task("streaming_literal_replacement")
        problem = {
            "chunks": (b"a", b"ba", b"b", b"a"),
            "replacements": ((b"aba", b"ab"), (b"ab", b"Y"), (b"ba", b"Z")),
        }
        self.assertEqual(benchmark.solve(problem), b"abZ")
        self.assertFalse(benchmark.is_solution(problem, bytearray(b"abZ")))

    def test_temporal_join_duplicate_tie_and_no_match(self):
        benchmark = task("temporal_asof_join")
        problem = {
            "dimensions": (
                ("a", 5, 1, "old"),
                ("a", 5, 2, "sequence"),
                ("a", 5, 2, "later-position"),
                ("a", 7, 0, "new"),
            ),
            "events": ((0, "a", 4), (1, "a", 5), (2, "a", 8)),
        }
        expected = ((0, None), (1, "later-position"), (2, "new"))
        self.assertEqual(benchmark.solve(problem), expected)
        self.assertFalse(benchmark.is_solution(problem, ((False, None), expected[1], expected[2])))


class NewTaskContractTests(unittest.TestCase):
    def test_generation_pickle_determinism_candidate_parity_and_no_mutation(self):
        small_sizes = {
            "labeled_graph_isomorphism": 9,
            "incremental_spreadsheet_recalculation": 40,
        }
        for slug in SLUGS:
            with self.subTest(task=slug):
                benchmark = task(slug)
                size = small_sizes.get(slug, 60)
                first = benchmark.generate_problem(size, 7)
                second = benchmark.generate_problem(size, 7)
                self.assertEqual(first, second)
                self.assertEqual(pickle.loads(pickle.dumps(first)), first)
                pristine = copy.deepcopy(first)
                reference = benchmark.solve(first)
                candidate = benchmark.candidate_solve(first)
                self.assertEqual(first, pristine)
                self.assertTrue(benchmark.is_solution(first, reference))
                self.assertTrue(benchmark.is_solution(first, candidate))

    def test_default_workloads_finish_as_a_light_suite(self):
        for slug in SLUGS:
            with self.subTest(task=slug):
                benchmark = task(slug)
                problem = benchmark.generate_problem(benchmark.default_n, 0)
                reference = benchmark.solve(copy.deepcopy(problem))
                candidate = benchmark.candidate_solve(copy.deepcopy(problem))
                self.assertTrue(benchmark.is_solution(problem, reference))
                self.assertTrue(benchmark.is_solution(problem, candidate))


if __name__ == "__main__":
    unittest.main()
