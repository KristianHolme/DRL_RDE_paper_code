# Purpose: Run PID transition grid search; analysis is in PID_transition_study_analysis.jl.
#
# Usage:
#   include(joinpath("scripts", "PID_transition_study.jl"))
#   in a julia session, or
#   julia --project scripts/PID_transition_study.jl
#   in the project root folder

using DrWatson
@quickactivate :DRL_RDE
using Random
using IntervalSets
using ProgressMeter
using Base.Threads
using Accessors

# ── Configuration ─────────────────────────────────────────────────────────
shock_pressures = RDE.SHOCK_PRESSURES
study = default_pid_transition_study()
n_runs = 512                  # number of random PID samples per target
seed = 0                      # RNG seed for reproducibility
Kp_range = 0.0f0 .. 5.0f0
Ki_range = 0.0f0 .. 1.0f0
Kd_range = 0.0f0 .. 3.0f0

Random.seed!(seed)

# ── Sample PID gains ──────────────────────────────────────────────────────
Kps = rand(Kp_range, n_runs)
Kis = rand(Ki_range, n_runs)
Kds = rand(Kd_range, n_runs)

# ── Build transition experiments ──────────────────────────────────────────
experiments = PIDTransitionExperiment[]
for target_shock_count in 1:length(shock_pressures)
    eval_config = get_PID_eval_config(target_shock_count)
    for i in 1:n_runs
        pid_policy = PIDControllerPolicy(;
            Kp = Kps[i], Ki = Kis[i], Kd = Kds[i],
        )
        for init_shock_count in setdiff(1:4, target_shock_count)
            push!(
                experiments, PIDTransitionExperiment(;
                    eval_config,
                    pid_policy,
                    target_shock_count,
                    init_shock_count,
                    study,
                ),
            )
        end
    end
end

@info "Built $(length(experiments)) experiments"

function precompile_experiments(experiments::Vector{PIDTransitionExperiment})
    mock_experiments = experiments[1:4]
    for i in 1:4
        @reset mock_experiments[i].study = "mock_experiment"
    end
    run_pid_transition_experiments_threaded(mock_experiments)
    return nothing
end

@info "Running precompile..."
@time precompile_experiments(experiments)
@time precompile_experiments(experiments)

# ── Run grid (threaded) ──────────────────────────────────────────────────
@info "Running experiments..."
run_pid_transition_experiments_threaded(experiments)
@info "Experiments completed."

include(scriptsdir("PID_transition_study_analysis.jl"))
run_pid_transition_study_analysis(; study)
