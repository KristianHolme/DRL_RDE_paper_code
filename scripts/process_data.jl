# Process saved experiment outputs into analysis artifacts used by figure scripts.
#
# Pipeline (idempotent by default; use `--force` to recompute all stages):
#
# 1. PID transition: raw PID results under `datadir("studies", "transition", study)` →
#    analysis outputs (see `analyze_pid_transition_results`).
# 2. Two-step baseline: `datadir("two_step_transition_time")` → overview and per-target
#    `transition_time_matrices.jld2` files.
# 3. DRL per-row transitions: for each study listed in `TRANSITION_SEGMENTED_DATASETS`, run
#    `analyze_all_transitions` → `datadir("studies", <name>, "transition_analyses", ...)`.
#    Use `-p` to add worker processes so `pmap` inside `analyze_all_transitions` can use multiple CPUs.
# 4. Aggregation: combined matrices for each slice in `TRANSITION_SEGMENTED_DATASETS`
#    (same outputs as `scripts/process_drl_transition_matrices.jl`).
#
# Does not render figures (see `scripts/transition_time_analysis.jl` for plots).
#
# JLD2 can emit long `@warn` lines when reconstructing SciML types from `.jld2` files.
# `DRL_RDE.install_process_data_logging!()` (see `src/utils/process_data_logging.jl`) filters those
# on the master and on Distributed workers; other `@info` / `@warn` output is kept.
#
# Usage (from project root):
#   julia --project scripts/process_data.jl
#   julia --project scripts/process_data.jl --force
#   julia --project scripts/process_data.jl -p 4
#   julia --project scripts/process_data.jl --force -p 8 -t auto

using DrWatson
@quickactivate :DRL_RDE
using ArgParse
using DataFrames
using Distributed
using DRL_RDE

install_process_data_logging!()

function parse_commandline()
    s = ArgParseSettings(
        description = """
        Process saved experiment outputs into analysis artifacts (PID transition,
        two-step transition-time matrices, DRL transition analyses, aggregation).
        By default, skips stages whose output files already exist; use --force to recompute.
        Use -p to add Distributed workers for the DRL transition stage (pmap parallelism).
        """,
    )
    @add_arg_table! s begin
        "--force"
        help = "Recompute all stages even when output files already exist"
        action = :store_true
        "--n_procs", "-p"
        help = "Number of extra worker processes for DRL transition analysis (0 = master only)"
        arg_type = Int
        default = 0
        "--threads_per_proc", "-t"
        help = "Threads per worker ('auto' to omit --threads in worker Julia)"
        arg_type = String
        default = "4"
    end
    return parse_args(s)
end

"""
Return whether any experiment in `unique_experiments` would run `analyze_all_transitions`
(i.e. not skipped by missing dirs, empty results, or idempotent existing files).
"""
function any_drl_work_pending(unique_experiments, force::Bool)::Bool
    for experiment_name in unique_experiments
        results_dir = datadir("studies", experiment_name, "results")
        if !isdir(results_dir)
            continue
        end
        res = collect_results!(results_dir)
        if nrow(res) == 0
            continue
        end
        save_dir = datadir("studies", experiment_name, "transition_analyses")
        if !force && all(row -> isfile(joinpath(save_dir, basename(row.path))), eachrow(res))
            continue
        end
        return true
    end
    return false
end

parsed_args = parse_commandline()
force = parsed_args["force"]
n_procs = parsed_args["n_procs"]
threads_per_proc = parsed_args["threads_per_proc"]

include(scriptsdir("transition_dataset_specs.jl"))

# --- 1. PID transition ---
@info "Processing PID baseline..."
study = default_pid_transition_study()
pid_paths = (
    get_pid_transition_best_pids_path(study),
    get_pid_transition_time_matrix_path(study),
    get_pid_transition_time_matrix_per_target_path(study),
)
if !force && all(isfile, pid_paths)
    @info "Skipping PID transition analysis (outputs exist)" study
else
    @info "PID transition analysis" study
    analyze_pid_transition_results(study)
end

# --- 2. Two-step transition matrices ---
@info "Processing two-step baseline..."
overview_path = datadir("two_step_transition_time", "transition_time_matrices.jld2")
per_target_path = datadir("two_step_transition_time_per_target", "transition_time_matrices.jld2")
if !force && isfile(overview_path) && isfile(per_target_path)
    @info "Skipping two-step transition matrices (outputs exist)" overview_path per_target_path
else
    @info "Two-step transition: collect results and save matrices"
    res = collect_results!(
        datadir("two_step_transition_time");
        black_list = ["transition_success_matrix", "transition_time_matrix"],
    )

    print_transition_time_summary(res)

    success_matrix, time_matrix = transition_overview_matrices(res)
    safesave(
        overview_path,
        Dict(
            "transition_success_matrix" => success_matrix,
            "transition_time_matrix" => time_matrix,
        ),
    )
    @info "Saved overview matrices" path = overview_path

    best_configs = get_best_configs_per_target(res)
    time_matrix_per_target = transition_time_matrix_per_target(res, best_configs)

    per_target_dir = datadir("two_step_transition_time_per_target")
    if !isdir(per_target_dir)
        mkpath(per_target_dir)
    end
    safesave(per_target_path, Dict("transition_time_matrix" => time_matrix_per_target))
    @info "Saved per-target time matrix" per_target_path
end

# --- 3. DRL per-row transition analyses ---
@info "Processing DRL runs..."
unique_experiments = unique(map(x -> x[1], TRANSITION_SEGMENTED_DATASETS))

drl_workers_added = false
if n_procs > 0 && any_drl_work_pending(unique_experiments, force)
    exeflags = ["--startup-file=yes"]
    if threads_per_proc != "auto"
        exeflags = vcat(exeflags, "--threads=$(threads_per_proc)")
    end
    @info "Adding $n_procs worker processes for DRL transition analysis" threads_per_proc = threads_per_proc
    addprocs(n_procs; exeflags = exeflags)
    @everywhere using DrWatson
    @everywhere @quickactivate :DRL_RDE
    @everywhere DRL_RDE.install_process_data_logging!()
    drl_workers_added = true
end

try
    for experiment_name in unique_experiments
        results_dir = datadir("studies", experiment_name, "results")
        if !isdir(results_dir)
            @warn "Skipping DRL transition analysis: results directory missing" results_dir
            continue
        end
        res = collect_results!(results_dir)
        save_dir = datadir("studies", experiment_name, "transition_analyses")
        if nrow(res) == 0
            @warn "No collected results for experiment; skipping DRL transition analysis" experiment_name
            continue
        end
        if !force && all(row -> isfile(joinpath(save_dir, basename(row.path))), eachrow(res))
            @info "Skipping DRL transition analysis (all per-row files exist)" experiment_name
            continue
        end
        @info "DRL transition analysis" experiment_name
        analyze_all_transitions(experiment_name; force = force)
    end
finally
    if drl_workers_added
        rmprocs(workers())
    end
end

# --- 4. Aggregated RL transition matrices (per TRANSITION_SEGMENTED_DATASETS slice) ---
@info "Aggregating DRL run data..."
for (experiment_name, filter_function, nickname) in TRANSITION_SEGMENTED_DATASETS
    out_path = datadir(
        "studies",
        experiment_name,
        "transition_analyses",
        experiment_name * nickname * ".jld2",
    )
    if !force && isfile(out_path)
        @info "Skipping aggregated transition matrices (already exist)" out_path
        continue
    end
    @info "Aggregating transition matrices" experiment_name nickname out_path
    DRL_RDE.aggregate_transition_matrices(
        experiment_name,
        experiment_name * nickname;
        filter_function = filter_function,
    )
    @info "Done" out_path
end

@info "Done processing data."
