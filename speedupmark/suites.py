"""Curated runnable suites; directory names are stable task identifiers."""

SMOKE_TASKS = (
    "incremental_spreadsheet_recalculation",
    "live_fleet_dispatch",
    "incremental_multiway_join",
    "ranked_bpe_tokenization",
    "dynamic_exact_ray_queries",
    "dynamic_document_search",
    "dynamic_shortest_paths",
    "adaptive_query_engine",
    "layout_aware_pipeline_compiler",
    "simd_traversal_kernel",
)

EXTENDED_TASKS = SMOKE_TASKS + (
    "temporal_asof_join",
    "streaming_literal_replacement",
    "sqlite_analytics_reports",
    "durable_log_recovery",
    "integer_factorization",
    "minimum_spanning_tree",
    "articulation_points",
    "min_weight_assignment",
    "kd_tree",
    "max_flow_min_cost",
    "gzip_compression",
    "matrix_multiplication",
    "queens_with_obstacles",
    "compiled_streaming_pattern_matching",
    "near_duplicate_document_clustering",
)

def selected_tasks(name, implemented):
    """Resolve public suite names, or validate one runnable task identifier."""
    if name in ("default", "smoke"):
        return SMOKE_TASKS
    if name == "extended":
        return EXTENDED_TASKS
    if name == "all":
        return tuple(implemented)
    if name in implemented:
        return (name,)
    raise ValueError(f"{name}: select smoke, extended, all, or an implemented task name")
