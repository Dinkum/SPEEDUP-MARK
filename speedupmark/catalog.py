"""Canonical task identifiers, display names, and public catalog descriptions.

Task specs and suites reference these keys. README headings and the public task
list are generated with scripts/sync_task_names.py, without importing candidates.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskMetadata:
    display_name: str
    workload: str


TASK_CATALOG = {
    'incremental_spreadsheet_recalculation': TaskMetadata(
        'Incremental Spreadsheet Recalculation',
        'Dependency invalidation and selective recomputation: answer cell queries without recalculating unchanged spreadsheet regions',
    ),
    'live_fleet_dispatch': TaskMetadata(
        'Battery-Limited Fleet Tours',
        'Resource constrained routing and fleet allocation: minimize battery limited tour time plus unserved job penalties on changing roads',
    ),
    'incremental_multiway_join': TaskMetadata(
        'Incremental Weighted Multiway Join',
        'Incremental join maintenance: balance weighted triangle deltas and cached partial joins under skew and update bursts',
    ),
    'ranked_bpe_tokenization': TaskMetadata(
        'Ranked Byte-Pair Tokenization',
        'Dynamic sequence maintenance: track the next ranked byte pair merge without repeatedly scanning and rebuilding the token list',
    ),
    'dynamic_exact_ray_queries': TaskMetadata(
        'Dynamic Exact Ray Queries',
        'Dynamic spatial indexing: balance hierarchy construction, refits, and exact nearest hit search as triangles change',
    ),
    'dynamic_document_search': TaskMetadata(
        'Dynamic Ranked Document Search',
        'Mutable inverted indexing and exact top k selection: balance posting updates against ranked document query cost',
    ),
    'dynamic_shortest_paths': TaskMetadata(
        'Dynamic Shortest-Path Query Engine',
        'Dynamic shortest path maintenance: balance search, distance cache reuse, invalidation, and repair under graph updates',
    ),
    'adaptive_query_engine': TaskMetadata(
        'Adaptive Query Engine',
        'Relational query optimization: choose predicate placement, join order, indexes, and shared computation across query plans',
    ),
    'layout_aware_pipeline_compiler': TaskMetadata(
        'Layout-Aware Pipeline Compiler',
        'Joint tensor fusion, layout, and scheduling: minimize simulated cycles under scratch memory, bank, and issue limits',
    ),
    'simd_traversal_kernel': TaskMetadata(
        'SIMD Traversal Kernel',
        'Latency hiding SIMD scheduling: balance traversal interleaving, table caching, banked gathers, and limited scratch; simulated cycles',
    ),
    'temporal_asof_join': TaskMetadata(
        'Temporal Per-Entity As-Of Join',
        'Temporal indexing and batch joins: match each event to its latest eligible entity row while preserving duplicate tie rules',
    ),
    'multi_literal_replacement': TaskMetadata(
        'Multi-Literal Replacement',
        'Overlapping multi pattern search: balance index construction and scanning while preserving longest matches across byte chunks',
    ),
    'grouped_analytics_reports': TaskMetadata(
        'Grouped Analytics Reports',
        'Shared aggregation planning: fuse grouping, distinct tracking, and sorting across three reports while preserving SQL NULL rules',
    ),
    'durable_log_recovery': TaskMetadata(
        'Checksummed Durable Log Recovery',
        'Validation constrained recovery: reduce checksum copying and duplicate bookkeeping while preserving the exact durable log prefix',
    ),
    'labeled_graph_isomorphism': TaskMetadata(
        'Vertex-Labeled Graph Isomorphism',
        'Exact graph matching: combine label refinement, search ordering, and symmetry pruning to find a mapping or prove none exists',
    ),
    'integer_factorization': TaskMetadata(
        'Integer Factorization',
        'Semiprime factor search: choose factoring methods and batch modular arithmetic for bounded factors with differing structure',
    ),
    'minimum_spanning_tree': TaskMetadata(
        'Minimum Spanning Tree',
        'Minimum weight graph connectivity: choose edge ordering, heaps, and disjoint set structures for graphs of differing density',
    ),
    'articulation_points': TaskMetadata(
        'Articulation Points',
        'Graph separation analysis: find cut vertices with low link traversal instead of repeating connectivity checks after each removal',
    ),
    'min_weight_assignment': TaskMetadata(
        'Exact Minimum Weight Assignment',
        'Exact bipartite assignment: minimize total cost through augmenting paths, dual potentials, and efficient slack updates',
    ),
    'exact_k_nearest_neighbors': TaskMetadata(
        'Exact k-Nearest Neighbors',
        'Exact geometric search: balance spatial index construction, distance pruning, and selection without losing tied neighbors',
    ),
    'min_cost_max_flow': TaskMetadata(
        'Exact Minimum Cost Maximum Flow',
        'Residual network optimization: maximize flow, then minimize cost through augmentations, reverse arcs, and shortest path potentials',
    ),
    'gzip_compression': TaskMetadata(
        'Gzip Compression',
        'Compression speed versus size: choose compression policies that preserve an exact round trip within the compressed size ceiling',
    ),
    'matrix_multiplication': TaskMetadata(
        'Exact Integer Matrix Multiplication',
        'Exact matrix product algorithm selection: balance sparse expansion, blocking, integer packing, and conversion across matrix shapes',
    ),
    'queens_with_obstacles': TaskMetadata(
        'Queens With Obstacles',
        'Maximum independent set on queen attack graphs: find the largest nonattacking placement with obstacle blocked attacks',
    ),
    'multi_pattern_matching': TaskMetadata(
        'Multi-Pattern Matching',
        'Automaton compilation versus scanning: choose shared states, literal filters, and representations for exact matches across byte chunks',
    ),
    'near_duplicate_document_clustering': TaskMetadata(
        'Exact Near-Duplicate Document Clustering',
        'Exact set similarity joins: prune document pairs without missing Jaccard threshold edges, then compute connected components',
    ),
    'capacitated_facility_location': TaskMetadata(
        'Capacitated Facility Location',
        'Coupled facility opening and assignment: minimize fixed and service costs while fitting indivisible customers into shared capacities',
    ),
    'delaunay': TaskMetadata(
        'Delaunay Triangulation',
        'Robust geometric construction: balance point location, local triangulation updates, and exact orientation and incircle checks',
    ),
    'discrete_log': TaskMetadata(
        'Prime-Field Discrete Logarithm',
        'Finite group exponent recovery: exploit subgroup orders and balance modular search time against lookup table memory',
    ),
    'earth_movers_distance': TaskMetadata(
        "Earth Mover's Distance",
        'Unregularized optimal transport: minimize mass moving cost while exploiting transport constraints, sparsity, and reduced cost structure',
    ),
    'exact_integer_convolution': TaskMetadata(
        'Exact Integer Convolution',
        'Exact polynomial multiplication: choose sparse, packed integer, or transform methods while reconstructing every signed coefficient exactly',
    ),
    'graph_coloring': TaskMetadata(
        'Graph Coloring',
        'Minimum graph coloring: combine clique bounds, vertex ordering, and symmetry pruning to prove the fewest colors',
    ),
    'group_lasso': TaskMetadata(
        'Logistic Group Lasso',
        'Nonsmooth convex optimization: balance logistic loss steps, group sparsity, active set screening, and convergence',
    ),
    'job_shop_scheduling': TaskMetadata(
        'Fixed-Route Job-Shop Makespan',
        'Precedence constrained resource scheduling: minimize makespan while jointly choosing exclusive machine orders and respecting job routes',
    ),
    'spectral_radius_matrix_completion': TaskMetadata(
        'Spectral-Radius Matrix Completion',
        'Log domain convex completion: minimize the Perron root under fixed observations and a product one constraint on missing entries',
    ),
    'multi_dim_knapsack': TaskMetadata(
        'Three-Resource 0/1 Knapsack',
        'Multidimensional subset optimization: maximize profit under three resource limits using bounds, dominance pruning, and state selection',
    ),
    'ode_stiff_robertson': TaskMetadata(
        'Robertson Chemical Kinetics',
        'Stiff chemical integration: balance implicit solves, Jacobian work, and error control across widely separated reaction timescales',
    ),
    'out_of_order_session_windows': TaskMetadata(
        'Out-of-Order Session Window Trace',
        'Dynamic interval merging and expiry: maintain exact session traces under out of order events, watermarks, and late event rules',
    ),
    'pagerank': TaskMetadata(
        'Sparse Weighted PageRank',
        'Sparse fixed point convergence: balance graph representation, iteration cost, and convergence while handling dangling vertices',
    ),
    'pde_burgers1d': TaskMetadata(
        '1D Viscous Burgers Equation',
        'Stiff nonlinear PDE integration: exploit the sparse spatial Jacobian while balancing advection, diffusion, and time step accuracy',
    ),
    'low_rank_approximation': TaskMetadata(
        'Low-Rank Matrix Approximation',
        'Accuracy constrained low rank factorization: balance matrix passes, subspace iteration, and orthogonalization within the reconstruction bound',
    ),
    'rbf_interpolation': TaskMetadata(
        'RBF Interpolation',
        'Structured kernel system solving: balance kernel assembly, constrained factorization, and batched prediction while preserving interpolation accuracy',
    ),
    'robust_kalman_filter': TaskMetadata(
        'Robust State Estimation',
        'Robust convex trajectory estimation: exploit temporal structure while minimizing process noise and Huber measurement loss',
    ),
    'rocket_landing_optimization': TaskMetadata(
        'Minimum Fuel Rocket Landing',
        'Constrained optimal control: minimize fuel with coupled trajectory dynamics, landing conditions, altitude, and thrust limits',
    ),
    'scratchpad_dataflow_compiler': TaskMetadata(
        'Scratchpad Dataflow Compiler',
        'Joint instruction scheduling and storage allocation: balance spilling, recomputation, fusion, and bank conflicts; simulated cycles',
    ),
    'sinkhorn': TaskMetadata(
        'Entropy-Regularized Transport',
        'Entropy regularized matrix scaling: balance convergence and numerical stability while satisfying transport marginals and the Gibbs condition',
    ),
    'smallest_eigenvalues_sparse_spd': TaskMetadata(
        'Smallest Eigenvalues of Sparse SPD Matrices',
        'Sparse extremal eigensolving: balance Krylov iteration, preconditioning, and shift invert factorization for clustered or ill conditioned spectra',
    ),
    'asymmetric_tsp': TaskMetadata(
        'Asymmetric Traveling Salesperson',
        'Exact directed tour optimization: combine bounds and search for small cases, and recover assignment tight Hamiltonian tours for larger cases',
    ),
    'vehicle_routing': TaskMetadata(
        'Vehicle Routing',
        'Joint customer partitioning and tour optimization: minimize total distance over exactly K nonempty depot returning routes',
    ),
    'affine_gap_sequence_alignment': TaskMetadata(
        'Exact Affine Gap Sequence Alignment',
        'Exact edit path optimization: balance full dynamic programming, wavefront search, and pruning across sparse edits, long gaps, and dense differences',
    ),
}
