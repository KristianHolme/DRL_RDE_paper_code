# Aggregate best transition matrices for all RL dataset slices in TRANSITION_SEGMENTED_DATASETS.
#
# Prerequisite: for each study, every row in `datadir("studies", <name>, "results")` that
# matches the filter must already have a file under `transition_analyses/` (same basename as
# the result row). Typically produced via `DRL_RDE.analyze_all_transitions` (or equivalent).
#
# Usage (from project root):
#   julia --project scripts/process_drl_transition_matrices.jl

using DrWatson
@quickactivate :DRL_RDE
using DRL_RDE

include(joinpath(@__DIR__, "transition_dataset_specs.jl"))

foreach(TRANSITION_SEGMENTED_DATASETS) do (experiment_name, filter_function, nickname)
    out_path = datadir(
        "studies",
        experiment_name,
        "transition_analyses",
        experiment_name * nickname * ".jld2",
    )
    @info "Aggregating transition matrices" experiment_name nickname out_path
    DRL_RDE.aggregate_transition_matrices(
        experiment_name,
        experiment_name * nickname;
        filter_function = filter_function,
    )
    @info "Done" out_path
end
