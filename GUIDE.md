# Grading guide

SPEEDUP-MARK supplies a task contract, a correct reference, an editable candidate,
and deterministic correctness checks. The optional runner prepares a workspace
or invokes a command you supply. Direct grading evaluates local code; model
performance requires a recorded agent run with its harness/model provenance.

## Scoring

For each sample, SPEEDUP-MARK verifies both solvers on copies of the same problem.
The speedup is reference cost divided by candidate cost. A correct result below
1.000x is a slowdown and remains below 1.000x. A task's speedup is the geometric
mean of its sample ratios; printed reference/candidate costs are separate medians
and their ratio may differ from that mean.

Most tasks measure host milliseconds. Candidate preparation and copying the
completed output are inside the timer. Generation, input copying, imports, and
verification are outside the timer but inside the worker timeout. Both solvers
follow the same timing rule. Return completed data in the forms accepted by the
task; custom objects, subclasses, iterators, object arrays, and cycles are invalid.

`layout_aware_pipeline_compiler`, `simd_traversal_kernel`, and
`scratchpad_dataflow_compiler` measure simulated cycles. Host compilation time is
bounded by the worker timeout and excluded from their cycle ratios. Their READMEs
define the machines, instruction budgets, and correctness challenges.

Direct suite grading reports a geometric mean only when every selected task
passes. A managed suite credits failed or missing tasks at 1.000x, preserves
correct slowdowns, and reports pass count and strict all-pass status separately.
The neutral factor is a reporting rule; it does not mean the task was solved.

## Sampling

Without `--seed`, grading draws a fresh nonnegative 63-bit base seed. Samples use
consecutive seeds beginning at that base; three samples are the default. Direct
suites share the base across tasks. Managed grading advances by the sample count
between sizes. Each task README describes its generated workload support.

Some tasks select families with `seed % k`; k consecutive seeds cover that family
cycle. Three samples do not cover four-family tasks or every combination in longer
cycles. Other tasks include all their families in every problem. The default
sample count makes iteration practical; it is not a precision guarantee. Fresh
inputs do not establish equal difficulty or stable rankings.

`--seed` fixes the batch for replay. In managed runs it fixes development and final
grading to the same seeds; omit it when final grading should draw fresh inputs.
Direct JSON reports record seeds, sizes, task versions, content revisions, and
runtime information. Managed runs save seed plans before candidate execution.

The layout compiler and SIMD task draw separate runtime values after programs
are frozen. Replaying their public seed reproduces compilation descriptors;
exact correctness/cycle replay also needs the verification receipt and the same
candidate bytes, programs, and task revision.

## Agent runs

From the repository root:

```console
python3 -m speedupmark run create temporal_asof_join \
  --harness my-agent --model exact-model-id --effort medium
```

The printed path contains a fresh reference-delegating candidate, task README,
shared prompt, and `grade.py`. Give the workspace to your chosen runner on the
same machine. It edits only `candidate.py` and runs `python3 grade.py` inside that
workspace to record candidate snapshots and measurements. Temporary work belongs
under `scratch/`. Installing dependencies or changing benchmark files invalidates
the contract. Task-specific library restrictions are stated in each README.

After the agent stops, from the repository root:

```console
python3 -m speedupmark run finish runs/<id>
python3 -m speedupmark run report runs/<id>
```

Final grading selects the fastest correct recorded candidate that passed integrity
checks and grades those exact bytes again. If none qualifies, it grades the current
workspace candidate. Development and final grading use the spec's same
`grading_cases`; direct grading uses `default_n`. An explicit `--n` overrides sizes.
The default worker limit is 60 seconds for the entire task-size batch, including
generation and verification. Increasing `--samples` may require a larger timeout.

`run launch` accepts an explicitly supplied command; it does not choose a model
or download a harness. See `python3 -m speedupmark run launch --help` and
`python3 -m speedupmark run launch extended --help` for command and suite options.

## Validation

Run from the repository root:

```console
python3 -B -m unittest discover -s tests -v
python3 -m speedupmark all --list
```

Numerical checks skip when their optional packages are unavailable. To validate
the full pool, install the exact versions and wheel hashes in
`requirements-numerical.txt` in a separate environment on its supported platform:

```console
python3 -m pip install --require-hashes --only-binary :all: -r requirements-numerical.txt
```

The file targets macOS arm64 CPython 3.14 wheels. Python 3.10+ can run the
standard-library suites; another numerical platform needs reviewed hashes for
the same pinned versions. Numerical tasks set default thread caps before imports.
Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` explicitly to 1 for comparable
measurements.

Test each task at every declared grading size and every family-selection residue.
Record version, revision, candidate hash, size, seeds, Python/platform, package
versions, timeout, and result. Regression tests and a finite seed sweep establish
the inspected contracts and tested inputs, rather than proving every possible
instance or model optimization depth. Small timing differences need comparable
runtime conditions and enough samples to distinguish them from noise.

## Task versions

Every spec declares canonical `MAJOR.MINOR.PATCH`. Use major for entrypoint or valid
answer-contract changes, minor for workload, size, reference, runtime, metric, or
sampling changes, and patch for corrections to the documented contract.
Documentation-only changes need no semantic bump. The content revision hashes
task files, shared grading code, and the shared prompt. Candidate bytes have a
separate identity. A changed content revision requires checking validation again.

## Task distributions

The task README defines size, family selection, random inputs, and fixed structure.
The generator is the executable distribution definition. Legal inputs may be
broader than generated performance inputs. `examples/example_gzip` is an interface template outside scored suites
and is excluded from curated suites; its simple round-trip contract is not a deep
compression challenge. The selected tasks include both algorithmic controls and
problems with interacting optimization decisions. Selection does not establish a
model ranking or guaranteed months of optimization runway.

| Task distribution | Managed sizes | Cost |
| --- | --- | --- |
| [Adaptive Query Engine](tasks/adaptive_query_engine/README.md#workload-distribution) | 2400, 3600 | ms |
| [Articulation Points](tasks/articulation_points/README.md#workload-distribution) | 300, 600 | ms |
| [Capacitated Facility Location](tasks/capacitated_facility_location/README.md#workload-distribution) | 8, 11 | ms |
| [Compiled Streaming Pattern Matching](tasks/compiled_streaming_pattern_matching/README.md#workload-distribution) | 2048, 3072 | ms |
| [Delaunay Triangulation](tasks/delaunay/README.md#workload-distribution) | 128, 256 | ms |
| [Prime-Field Discrete Logarithm](tasks/discrete_log/README.md#workload-distribution) | 300000, 600000 | ms |
| [Checksummed Durable Log Recovery](tasks/durable_log_recovery/README.md#workload-distribution) | 12000, 24000 | ms |
| [Dynamic Ranked Document Search](tasks/dynamic_document_search/README.md#workload-distribution) | 800, 1600 | ms |
| [Dynamic Exact Ray Queries](tasks/dynamic_exact_ray_queries/README.md#workload-distribution) | 192, 288 | ms |
| [Dynamic Shortest-Path Query Engine](tasks/dynamic_shortest_paths/README.md#workload-distribution) | 400, 800 | ms |
| [Earth Mover's Distance](tasks/earth_movers_distance/README.md#workload-distribution) | 48, 90 | ms |
| [Affine gap sequence alignment](tasks/affine_gap_sequence_alignment/README.md#workload-distribution) | 256, 384 | ms |
| [Exact Integer Signal Convolution](tasks/fft_convolution/README.md#workload-distribution) | 420, 840 | ms |
| [Graph Coloring](tasks/graph_coloring_assign/README.md#workload-distribution) | 14, 20 | ms |
| [Group Lasso](tasks/group_lasso/README.md#workload-distribution) | 24, 40 | ms |
| [Gzip Compression](tasks/gzip_compression/README.md#workload-distribution) | 100000, 250000 | ms |
| [Incremental Weighted Multiway Join](tasks/incremental_multiway_join/README.md#workload-distribution) | 1200, 2400 | ms |
| [Incremental Spreadsheet Recalculation](tasks/incremental_spreadsheet_recalculation/README.md#workload-distribution) | 650, 1300 | ms |
| [Integer Factorization](tasks/integer_factorization/README.md#workload-distribution) | 400000, 800000 | ms |
| [Fixed-Route Job-Shop Makespan](tasks/job_shop_scheduling/README.md#workload-distribution) | 3, 4 | ms |
| [Kd Tree](tasks/kd_tree/README.md#workload-distribution) | 700, 1400 | ms |
| [Vertex-Labeled Graph Isomorphism](tasks/labeled_graph_isomorphism/README.md#workload-distribution) | 10, 20 | ms |
| [Layout-Aware Pipeline Compiler](tasks/layout_aware_pipeline_compiler/README.md#workload-distribution) | 32, 48 | cycles |
| [Battery-Limited Fleet Tours](tasks/live_fleet_dispatch/README.md#workload-distribution) | 60, 96 | ms |
| [Spectral-Radius Matrix Completion](tasks/matrix_completion/README.md#workload-distribution) | 4, 6 | ms |
| [Matrix Multiplication](tasks/matrix_multiplication/README.md#workload-distribution) | 45, 75 | ms |
| [Max Flow Min Cost](tasks/max_flow_min_cost/README.md#workload-distribution) | 100, 180 | ms |
| [Min Weight Assignment](tasks/min_weight_assignment/README.md#workload-distribution) | 70, 110 | ms |
| [Minimum Spanning Tree](tasks/minimum_spanning_tree/README.md#workload-distribution) | 500, 1000 | ms |
| [Three-Resource 0/1 Knapsack](tasks/multi_dim_knapsack/README.md#workload-distribution) | 24, 30 | ms |
| [Exact Near-Duplicate Document Clustering](tasks/near_duplicate_document_clustering/README.md#workload-distribution) | 360, 720 | ms |
| [Robertson Kinetics](tasks/ode_stiff_robertson/README.md#workload-distribution) | 4, 10 | ms |
| [Out-of-Order Session Window Trace](tasks/out_of_order_session_windows/README.md#workload-distribution) | 2400, 4800 | ms |
| [Sparse Weighted PageRank](tasks/pagerank/README.md#workload-distribution) | 256, 384 | ms |
| [Burgers Equation](tasks/pde_burgers1d/README.md#workload-distribution) | 120, 200 | ms |
| [Queens With Obstacles](tasks/queens_with_obstacles/README.md#workload-distribution) | 10, 12 | ms |
| [Randomized SVD](tasks/randomized_svd/README.md#workload-distribution) | 200, 320 | ms |
| [Ranked Byte-Pair Tokenization](tasks/ranked_bpe_tokenization/README.md#workload-distribution) | 3000, 6000 | ms |
| [RBF Interpolation](tasks/rbf_interpolation/README.md#workload-distribution) | 250, 500 | ms |
| [Robust Kalman Filter](tasks/robust_kalman_filter/README.md#workload-distribution) | 16, 30 | ms |
| [Rocket Landing](tasks/rocket_landing_optimization/README.md#workload-distribution) | 30, 60 | ms |
| [Scratchpad Dataflow Compiler](tasks/scratchpad_dataflow_compiler/README.md#workload-distribution) | 240, 480 | cycles |
| [SIMD Traversal Kernel](tasks/simd_traversal_kernel/README.md#workload-distribution) | 32, 128 | cycles |
| [Sinkhorn](tasks/sinkhorn/README.md#workload-distribution) | 32, 64 | ms |
| [Smallest Sparse Eigenvalues](tasks/sparse_lowest_eigenvalues_posdef/README.md#workload-distribution) | 60, 120 | ms |
| [SQLite Analytics Report Workload](tasks/sqlite_analytics_reports/README.md#workload-distribution) | 5000, 10000 | ms |
| [Chunked Simultaneous Literal Replacement](tasks/streaming_literal_replacement/README.md#workload-distribution) | 8000, 16000 | ms |
| [Temporal Per-Entity As-Of Join](tasks/temporal_asof_join/README.md#workload-distribution) | 700, 1400 | ms |
| [Asymmetric Traveling Salesperson](tasks/tsp/README.md#workload-distribution) | 10, 12 | ms |
| [Vehicle Routing](tasks/vehicle_routing/README.md#workload-distribution) | 8, 11 | ms |

## Source distributions

Prepare a reviewed archive without publishing it:

```console
python3 scripts/prepare_public.py runs/speedupmark-public.zip
```

The builder includes an explicit source inventory and fresh reference-delegating
candidates. It excludes private notes, local harnesses, runs, monitoring, caches,
and Git metadata, and records file hashes in `PUBLIC_MANIFEST.json`. It refuses to
overwrite an archive. Validate the extracted artifact before distributing it.
The archive contains the root MIT license and `THIRD_PARTY.json` source inventory.
Publishing a Git repository separately requires checking its index and history;
archive exclusions cannot remove earlier commits.
