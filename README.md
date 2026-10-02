# SPEEDUP-MARK

**The problem:** Most pass/fail benchmarks become stale and saturate as AI models become smarter. They measure a binary outcome.

**The goal:** Use hard optimization tasks to measure the “percent speed up” a model can achieve. These tasks are designed to be attempted by any model but provide lots of runway for even smarter models to optimize.

Models tweak and submit their `candidate.py` to a central grader, so we can track their progress over the course of a run. Each submission must preserve verified correctness.

Speedup is reference cost divided by candidate cost: **2.000x means twice as fast, a 100% speed increase**.

**Task Suites:**

- **Smoke**: 10 core tasks to quickly test agent setups.
- **Extended**: 25 tasks.
- **All**: 50 tasks.

![GPT-6 Luna medium with Hermes: verified speedup over time on the layout aware pipeline compiler](diagrams/luna-hermes-progress.png)

## Prerequisites

- **Python 3.10+** and a local copy of this repository. Run commands from the repository root.
- **Standard library only** for the `smoke` and `extended` suites. Some additional tasks need the optional packages in [requirements-numerical.txt](requirements-numerical.txt).
- **Your chosen agent harness**: this repo is agnostic to the specific agent setup used.

## Usage

### Create a task run

```console
python3 -m speedupmark run create temporal_asof_join \
  --harness my-agent --model exact-model-id --effort medium
```

Give your agent the printed workspace path on the same machine. It reads `prompt.md` and the task README, edits `candidate.py`, and submits each attempt by running this command inside the workspace:

```console
python3 grade.py
```

When the agent stops, finish the run and view its results from the repository root:

```console
python3 -m speedupmark run finish runs/<id>
```

### Run a suite with your agent

Supply your harness’s noninteractive command, configured for the model and effort you record:

```console
python3 -m speedupmark run launch smoke \
  --harness my-agent --model exact-model-id --effort medium \
  --command my-agent
```

Replace `my-agent` with your harness command. The prompt is passed on stdin; each task gets a fresh workspace. Use a suite name or task name to select the tasks you would like to run.

View the suite results using the printed suite path:

```console
python3 -m speedupmark run report runs/suite-<id>
```

Reports include progress measurements and final scores. Final grading selects the fastest correct recorded candidate and evaluates it on fresh seeds by default. Timing comparisons need the same machine and Python version.

### Commands

| Command | Purpose |
| --- | --- |
| `python3 -m speedupmark all --list` | List all 50 tasks |
| `python3 grade.py` | Submit the candidate and record progress from inside a run workspace |
| `python3 -m speedupmark run evaluate runs/<id>` | Submit the workspace candidate from the repository root |
| `python3 -m speedupmark run finish runs/<id>` | Finish a manually handed off run and grade its best verified candidate |
| `python3 -m speedupmark run report runs/<id>` | View a task run’s progress and results |
| `python3 -m speedupmark run report runs/suite-<id>` | View suite results; add `--json` for the full report |
| `python3 -m speedupmark temporal_asof_join` | Grade the local candidate directly |
| `python3 -m speedupmark --json > results.json` | Save direct smoke suite grading results |
| `python3 -m speedupmark run --help` | Show run commands and options |

Grading uses three samples and fresh seeds by default. Use `--seed` to replay a batch and `--samples` to change the sample count. Direct grading evaluates local candidates without launching a model.

## Task List

`smoke` contains 10 tasks. `extended` includes those 10 plus 15 more. `all` includes all 50 benchmark tasks below. Each task links to its full contract; use its directory name to run it individually.

| Task | Workload | Included in |
| --- | --- | --- |
| [Incremental spreadsheet recalculation](tasks/incremental_spreadsheet_recalculation/README.md) | Update inputs and answer spreadsheet cell queries | smoke, extended, all |
| [Live fleet dispatch](tasks/live_fleet_dispatch/README.md) | Minimize battery limited tour time and penalties for unserved jobs on changing roads | smoke, extended, all |
| [Incremental multiway join](tasks/incremental_multiway_join/README.md) | Weighted triangle aggregates under relation updates | smoke, extended, all |
| [Ranked BPE tokenization](tasks/ranked_bpe_tokenization/README.md) | Apply ranked byte pair merges to arbitrary bytes | smoke, extended, all |
| [Dynamic exact ray queries](tasks/dynamic_exact_ray_queries/README.md) | Find the nearest ray hit triangle as geometry changes | smoke, extended, all |
| [Dynamic document search](tasks/dynamic_document_search/README.md) | Rank matching documents after additions, replacements, and deletions | smoke, extended, all |
| [Dynamic shortest paths](tasks/dynamic_shortest_paths/README.md) | Distance queries under graph updates | smoke, extended, all |
| [Adaptive query engine](tasks/adaptive_query_engine/README.md) | Execute relational query plans over in memory integer tables | smoke, extended, all |
| [Layout aware pipeline compiler](tasks/layout_aware_pipeline_compiler/README.md) | Tensor layouts, fusion, and scheduling; simulated cycles | smoke, extended, all |
| [SIMD Traversal Kernel](tasks/simd_traversal_kernel/README.md) | Schedule dependent SIMD table traversals; simulated cycles | smoke, extended, all |
| [Temporal as of join](tasks/temporal_asof_join/README.md) | Match events to the latest eligible row for each entity | extended, all |
| [Streaming literal replacement](tasks/streaming_literal_replacement/README.md) | Replace byte patterns across chunks using longest match precedence | extended, all |
| [SQLite analytics reports](tasks/sqlite_analytics_reports/README.md) | Category aggregates, duplicate groups, and user totals with SQL NULL rules | extended, all |
| [Durable log recovery](tasks/durable_log_recovery/README.md) | Checksummed prefix recovery and duplicate resolution | extended, all |
| [Labeled graph isomorphism](tasks/labeled_graph_isomorphism/README.md) | Find a label preserving vertex mapping, or report no match | all |
| [Integer factorization](tasks/integer_factorization/README.md) | Prime factors of bounded semiprimes | extended, all |
| [Minimum spanning tree](tasks/minimum_spanning_tree/README.md) | Connect all vertices using a tree of minimum total weight | extended, all |
| [Articulation points](tasks/articulation_points/README.md) | Vertices whose removal increases the number of connected components | extended, all |
| [Minimum weight assignment](tasks/min_weight_assignment/README.md) | Find a minimum cost one to one assignment from a cost matrix | extended, all |
| [Nearest neighbors](tasks/kd_tree/README.md) | Exact k nearest neighbors with deterministic ties | extended, all |
| [Minimum cost maximum flow](tasks/max_flow_min_cost/README.md) | Maximum flow with minimum total cost | extended, all |
| [Gzip compression](tasks/gzip_compression/README.md) | Exact round trip within a compressed size bound | extended, all |
| [Matrix multiplication](tasks/matrix_multiplication/README.md) | Exact signed integer products | extended, all |
| [Queens with obstacles](tasks/queens_with_obstacles/README.md) | Place as many nonattacking queens as possible, with obstacles blocking attacks | extended, all |
| [Compiled streaming pattern matching](tasks/compiled_streaming_pattern_matching/README.md) | Compile patterns resembling regular expressions and detect matches across byte chunks | extended, all |
| [Near duplicate document clustering](tasks/near_duplicate_document_clustering/README.md) | Group documents into components using exact word shingle Jaccard similarity | extended, all |
| [Capacitated facility location](tasks/capacitated_facility_location/README.md) | Minimum cost facility opening and customer assignment | all |
| [Delaunay triangulation](tasks/delaunay/README.md) | Triangulate planar integer points with exact Delaunay validity checks | all |
| [Discrete logarithm](tasks/discrete_log/README.md) | Recover x from g^x ≡ h (mod p), with prime p | all |
| [Earth mover's distance](tasks/earth_movers_distance/README.md) | Minimum cost transport between mass distributions | all |
| [Integer signal convolution](tasks/fft_convolution/README.md) | Exact signed integer signal convolution | all |
| [Graph coloring](tasks/graph_coloring_assign/README.md) | Proper coloring with the fewest colors | all |
| [Logistic group lasso](tasks/group_lasso/README.md) | Fit logistic regression with a group sparsity penalty | all |
| [Job shop scheduling](tasks/job_shop_scheduling/README.md) | Schedule fixed job operations to minimize the final completion time | all |
| [Spectral radius matrix completion](tasks/matrix_completion/README.md) | Fill missing positive entries to minimize spectral radius, with product one | all |
| [Three resource 0/1 knapsack](tasks/multi_dim_knapsack/README.md) | Maximum profit under three resource limits | all |
| [Robertson chemical kinetics](tasks/ode_stiff_robertson/README.md) | Integrate a stiff three species chemical reaction system | all |
| [Out of order session windows](tasks/out_of_order_session_windows/README.md) | Session merging, watermarks, and late events | all |
| [PageRank](tasks/pagerank/README.md) | Compute PageRank scores on weighted graphs with dangling vertices | all |
| [1D viscous Burgers equation](tasks/pde_burgers1d/README.md) | Integrate a fixed advection and diffusion system on a one dimensional grid | all |
| [Low rank SVD](tasks/randomized_svd/README.md) | Low rank approximation within a quality bound | all |
| [RBF interpolation](tasks/rbf_interpolation/README.md) | Radial basis fitting and prediction | all |
| [Robust state estimation](tasks/robust_kalman_filter/README.md) | Estimate a state trajectory using a Huber penalty on measurement errors | all |
| [Rocket landing](tasks/rocket_landing_optimization/README.md) | Minimum fuel trajectories under thrust constraints | all |
| [Scratchpad dataflow compiler](tasks/scratchpad_dataflow_compiler/README.md) | Schedule integer computation graphs with limited scratch memory; simulated cycles | all |
| [Sinkhorn scaling](tasks/sinkhorn/README.md) | Optimal transport with entropy regularization | all |
| [Smallest sparse eigenvalues](tasks/sparse_lowest_eigenvalues_posdef/README.md) | Lowest eigenvalues of sparse symmetric positive definite matrices | all |
| [Asymmetric traveling salesperson](tasks/tsp/README.md) | Minimum cost directed tour | all |
| [Vehicle routing](tasks/vehicle_routing/README.md) | Minimize distance over depot returning routes, without capacity constraints | all |
| [Affine gap sequence alignment](tasks/affine_gap_sequence_alignment/README.md) | Minimum cost global byte alignment with gap opening and extension costs | all |

## Inspiration

- [AlgoTune](https://github.com/oripress/AlgoTune)
- [Anthropic’s original performance take-home](https://github.com/anthropics/original_performance_takehome)
