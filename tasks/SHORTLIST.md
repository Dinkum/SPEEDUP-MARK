# SPEEDUP-MARK — Suite Selection

The selectors are shared by direct grading and managed launches:

| Selector | Selection |
| --- | --- |
| default invocation, `default`, `smoke` | 10 tasks selected for varied optimization decisions and local practicality |
| `extended` | 25 tasks including the entire smoke set |
| `all` | All 50 implemented runnable task directories |

`speedupmark/suites.py` owns the ordered selections. `--list` lists the direct-grading selection without importing candidates. The `all` selector includes the selected 25 and 25 additional benchmark tasks.

## Smoke ten

| Task | Why it earns a default slot |
| --- | --- |
| `incremental_spreadsheet_recalculation` | Dependency tracking, selective evaluation, invalidation, and symbolic reuse |
| `live_fleet_dispatch` | Resource-constrained paths, exact multi-stop tours, and disjoint fleet allocation |
| `incremental_multiway_join` | Indexed weighted triangle maintenance; eager deltas versus deferred batches under skew and update bursts |
| `ranked_bpe_tokenization` | Priority-driven rewriting and efficient sequence updates |
| `dynamic_exact_ray_queries` | Exact rational comparisons, spatial hierarchies, and refit-versus-rebuild under geometry updates |
| `dynamic_document_search` | Mutable inverted indexes, exact ranking and top-k selection |
| `dynamic_shortest_paths` | Repeated and cold queries through graph mutations, closures and reopenings |
| `adaptive_query_engine` | Predicate placement, join order, bounded selection and cross-query reuse for changing plans |
| `layout_aware_pipeline_compiler` | Fusion boundaries, intermediate layouts, recomputation and shuffling under simulated resource limits |
| `simd_traversal_kernel` | Dependent SIMD traversals with banked gathers, scratch pressure, and private runtime data |

Multiway join and dynamic paths include three distinct workload families in every generated input; the ray task, the query engine, the compiler and the pattern matcher include four, five, four and five. Their baselines respectively implement indexed first-order deltas, heap Dijkstra, brute-force exact intersection, a fixed left-deep plan, a materialize-everything schedule, and per-pattern automata. Independent verification does not rely on candidate-reported timing or correctness.

## Extended additions

These 15 tasks complete the extended set of 25:

| Task | Selection rationale |
| --- | --- |
| `temporal_asof_join` | Useful sorting/indexing control; exact timestamp and sequence ties |
| `streaming_literal_replacement` | Byte matching, overlaps and simultaneous replacement semantics |
| `sqlite_analytics_reports` | Exact SQL NULL, grouping, duplicate and ordering behavior |
| `durable_log_recovery` | Checksums, torn segments, duplicate precedence and durable prefixes |
| `integer_factorization` | Balanced and unbalanced semiprimes; exact prime-factor witnesses |
| `minimum_spanning_tree` | Sparse graph processing and exact minimum-weight tree witnesses |
| `articulation_points` | Graph connectivity under vertex deletion; traversal reuse |
| `min_weight_assignment` | Signed cost matrices, ties, and exact optimal assignments |
| `kd_tree` | Exact nearest neighbors across dimensions, ties and clustered points |
| `max_flow_min_cost` | Residual networks, augmentation, and exact optimal flow witnesses |
| `gzip_compression` | Lossless compression with an enforced size ceiling |
| `matrix_multiplication` | Exact signed integer dense/sparse matrix kernels |
| `queens_with_obstacles` | Bounded exact combinatorial search with attack-blocking obstacles |
| `compiled_streaming_pattern_matching` | Representation choice per pattern family; literal scanning, shared automata and a compile-versus-scan budget |
| `near_duplicate_document_clustering` | Exact set-similarity search and component closure under size variation, common shingles, and threshold-adjacent overlaps |

As-of join and literal replacement remain useful controls even though their first large improvements often come from familiar algorithms. Graph isomorphism is also bounded and often fast; its small timings are less useful for fine performance ranking. Queens has data-dependent exact-search cost, so its 10/12 board sizes are intentionally bounded. Dispatch measures whether improvements to paths, tours, and allocation survive their integration into one timed pipeline.

The catalog adaptations are independently authored standard-library tasks with bounded generators. Each README defines its SPEEDUP-MARK-specific representation, oracle, metric, and provenance.

## All and the starter template

`examples/example_gzip` is an interface example outside scored suites. Its verifier checks a round trip without limiting compressed size. Affine gap sequence alignment is included in `all`. Labeled graph isomorphism remains in `all` and is excluded from `extended`. `gzip_compression` instead requires one complete gzip member, an exact round trip, no trailing data, and at most `ceil(1.001 * reference_bytes)` bytes relative to zlib level 9.

Every retained runnable task declares `task_version` in `task_spec.py`; [Task versions](../GUIDE.md#task-versions) defines when to bump it.

## Measurement and interpretation

Every task has an exact correctness contract and a declared set of managed grading sizes. Direct grading uses one `default_n` per task; development and final managed grading use `grading_cases`, with fresh final seeds. All preprocessing in wall-clock tasks is timed, including the query engine's planning and index construction and the ray task's hierarchy construction and updates. Three tasks score simulated cycles — SIMD Traversal Kernel, the layout-aware compiler, and the scratchpad dataflow compiler — with compilation bounded by the overall worker timeout rather than included in the cycle score. The layout-aware compiler enforces simulated memory limits.

Suite scores equally weight dimensionless speedup ratios. Direct grading withholds an aggregate unless all selected tasks pass. Managed suites credit failed or missing tasks at 1.000x, retain measured slowdowns for correct submissions, and report the pass count and strict all-pass verdict separately. A credited factor is not evidence that a failed task was solved. This remains a good-faith local benchmark, not a hostile-code sandbox. See [Scoring](../GUIDE.md#scoring).

Selection is based on breadth, verified implementation, and plausible optimization depth. It does not establish months of frontier-model separation. That requires repeated model runs under comparable budgets, workload-family analysis, improvement curves, and checks that further gains exceed timing noise.
