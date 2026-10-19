# Shared definitions for RL transition comparison figures and aggregation.
# Include after `@quickactivate :DRL_RDE` (uses no package symbols).
#
# Each entry is (experiment_name, filter_function, nickname):
# - `filter_function` selects rows in `datadir("studies", experiment_name, "results")`.
# - Aggregated matrices are saved by `aggregate_transition_matrices` as
#   `datadir("studies", experiment_name, "transition_analyses", experiment_name * nickname * ".jld2")`.
#   Load with `load_matrices(experiment_name, nickname; mode = :best_transition)`.
#
# Prerequisite: each results row must already have a per-row analysis file at
# `datadir("studies", experiment_name, "transition_analyses", basename(row.path))`
# (e.g. from `analyze_all_transitions`).

const TRANSITION_RL_DATASETS = [
    ("SAU3", r -> r.movingframe == false, "SUS"),
    ("SAU3", r -> r.movingframe == true, "SUM"),
    ("MA3", r -> r.movingframe == false, "MSS"),
    ("MA3", r -> r.movingframe == true, "MSM"),
    ("SAS3", r -> r.movingframe == false, "SSS"),
    ("SAS3", r -> r.movingframe == true, "SSM"),
]

# Segmented paper comparison (no uniform configs).
const TRANSITION_SEGMENTED_DATASETS = [
    ("SAS3", r -> r.movingframe == false, "SSS"),
    ("SAS3", r -> r.movingframe == true, "SSM"),
    ("MA3", r -> r.movingframe == false, "MSS"),
    ("MA3", r -> r.movingframe == true, "MSM"),
]
