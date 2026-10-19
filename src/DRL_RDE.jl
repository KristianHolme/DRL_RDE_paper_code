# __precompile__(false)
module DRL_RDE
using Reexport: @reexport, Reexport
using DrWatson: @produce_or_load, DrWatson, datadir, plotsdir, projectdir, safesave,
    savename
using ArgParse: ArgParse
using Dates: Dates, TimeType, now
@reexport using Dates
using Drill: ActorCriticLayer, Agent, BroadcastedParallelEnv, Drill,
    MultiThreadedParallelEnv, PPO, SAC, SACLayer, evaluate_agent, get_hparams, steps_taken, train!
using DrillInterface: AbstractParallelEnv, DrillInterface, number_of_envs, reset!
using RDE_Env: AbstractActionStrategy, AbstractMultiAgentObservationStrategy,
    AbstractObservationStrategy, AbstractRDEPolicy, AbstractRewardStrategy,
    DirectScalarPressureAction, DirectVectorPressureAction, MovingFrameControlShift,
    MultiAgentRDEEnv, PolicyRunData, RDEEnv, RDE_Env, ScalarToVectorReward,
    _predict_action, _reset!, plot_shifted_history, run_policy
@reexport using RDE_Env

using Accessors: @reset, @set, Accessors
using DataFrames: AbstractDataFrame, DataFrame, DataFrameRow, DataFrames, Not, PrettyTables,
    combine, groupby, nrow, rename!, select!
using Distributed: @distributed, Distributed, RemoteChannel, myid, pmap, workers
using Enzyme: Enzyme
using JLD2: JLD2, jldsave, load
using Lux: Lux
using LinearAlgebra: LinearAlgebra, norm
using Logging: @logmsg, LogLevel, Logging
using NamedTupleTools: NamedTupleTools
using Makie: Axis, Button, Colorbar, Figure, GridLayout, IntervalsBetween, Label, Legend,
    Makie, Menu, Observable, PolyElement, Theme, Toggle, Top, VecTypes, axislegend,
    band!, barplot!, boxplot!, cgrad, density!, errorbars!, heatmap!, hexbin!,
    hidedecorations!, hidespines!, hideydecorations!, hist!, hlines!, lines!, linkxaxes!,
    linkyaxes!, mesh!, on, save, scatter!, scatterlines!, text!, theme_latexfonts,
    to_color, to_world, violin!, vlines!, cam3d!
using GeometryBasics: faces
using GLM: GLM, @formula, lm
using MultiProgressManagers: MultiProgressManagers, ProgressManager, ProgressTask,
    default_db_path, finish!, get_task, update!
using ProgressMeter: ProgressMeter, @showprogress, Progress, next!
using PrettyTables: pretty_table, TextHighlighter, @crayon_str
using Printf: Printf, @sprintf
using Random: Random
using RDE: RDE, AbstractReset, CycleShockReset, FiniteVolumeMethod, NShock, RDEParam,
    RDEProblem, RandomShock, RandomShockOrCombination, ShiftReset, ZeroControlShift
@reexport using RDE
import SciMLBase

include("state_and_injection_pressure_reset.jl")
export StateAndInjectionPressureReset
using Statistics: Statistics, median
using StatsBase: StatsBase, cor, corspearman, mean, predict, quantile, r2, std
# using TestItems
using Base.Threads
using HTTP: HTTP

import TensorBoardLogger as TB
import ValueHistories as VH

export single_run
export interactive_analysis, categorical_comparison, continuous_comparison, group_by_seed
export LinearSchedule, CosineSchedule, ExpDecaySchedule, AbstractSchedule
export detect_transition, run_transition
export find_nearest_shock
export run_experiments
export continue_experiments
export analyze_continued_training, compare_experiment_versions,
    cleanup_incomplete_runs, find_incomplete_runs
export get_jl_policy

export create_env, create_envs
export AbstractExperiment, RLExperiment, AbstractRLConfig, LayerKwargs,
    LearnConfig, EnvConfig, EvalConfig
export RDEEnvConfig, RDEEvalConfig
export BasicLearnConfig, PPO

export DrillLearnConfig, DrillConfig, DrillExperiment

export evaluate_transition, plot_transition_matrices, plot_and_save_transition,
    analyze_transitions, analyze_all_transitions
export transition_comparison
export plot_best_transitions, process_single_transition
export get_best_configs_per_target, get_best_config_for_target,
    transition_time_matrix_per_target, print_transition_time_summary,
    transition_overview_matrices

export PIDExperiment, run_pid_experiment, plot_pid_run
export PIDTransitionExperiment, run_pid_transition, run_pid_transition_experiments,
    run_pid_transition_experiments_threaded
export get_PID_eval_config, load_matrices
export default_pid_transition_study, get_pid_transition_results_dir,
    get_pid_transition_analysis_dir, get_pid_transition_best_pids_path,
    get_pid_transition_time_matrix_path, get_pid_transition_time_matrix_per_target_path
export load_pid_transition_results, aggregate_pid_transition_results,
    get_best_pid_configs_per_target, get_best_pid_for_target,
    pid_transition_time_matrix_per_target, pid_transition_best_time_matrix,
    analyze_pid_transition_results, save_pid_transition_analysis,
    load_pid_transition_best_pids
export reevaluate_results!
export create_migration_typemap
export create_dril_rename_typemap
export create_injection_migration_typemap
export get_study_dir, get_study_results_dir, get_study_models_dir
export calculate_saves_per_action

export get_linear_pressure_change_configs, LinearPressureChangeResults,
    LinearPressureChangeExperiment

export reportdir
export plot_error_analysis, error_analysis
export get_default_policy

export setup_dril_config
export EvalCallback, SaveCallback

export get_train_log_mvhist, plot_mvhist_key!, plot_mvhist_key, aggregate_histories_by_seed,
    plot_aggregated_histories!, plot_aggregated_histories, plot_aggregated_histories_grid

export send_ntfy

# Plotting recipes
export pie3d, pie3d!, plot_rde_data_3d!, plot_rde_data_3d

include("policies.jl")
export ConstantRDEPolicy, SinusoidalRDEPolicy, StepwiseRDEPolicy, RandomRDEPolicy,
    DelayedPolicy, HeldActionPolicy, PIDControllerPolicy, PIDPressurePolicy,
    reset_pid_pressure_policy!
export _predict_action

include("rde_env_strategies/rde_env_strategies.jl")
@reexport using .RDE_Env_Strategies

# Include all utility files
include("utils/schedules.jl")
include("utils/types.jl")
include("utils/setup.jl")
include("utils/sample_experiments.jl")
export setup_experiments, experiment_savename, get_strats, get_abbreviation, verify_cfg, get_or_default
include("utils/pid.jl")
include("utils/single_run.jl")
include("utils/analysis/reevaluation.jl")
include("utils/analysis/analyze_results.jl")
include("utils/analysis/transition_analysis.jl")
include("utils/utils.jl")
export get_best_policy
include("utils/process_data_logging.jl")
export install_process_data_logging!
include("utils/transition_utils.jl")
include("utils/experiment_running.jl")
include("utils/migration.jl")
include("utils/fast_and_slow_transitions/setup.jl")
include("utils/analysis/convergence_analysis.jl")
include("utils/analysis/train_stats_utils.jl")

# Optional: src/ml/ is left out of the public paper code
isfile(joinpath(@__DIR__, "ml", "ml.jl")) && include(joinpath("ml", "ml.jl"))

# Plotting recipes
include("plotting/pie3d.jl")
include("plotting/rde_3d_diagram.jl")

# NOTE: DrWatson.default_allowed is non-public; keep until public alternative exists.
DrWatson.default_allowed(c::Dict{String, Any}) = (Real, String, Symbol, TimeType, Nothing, Vector)

reportdir(args...) = projectdir("documents", "report", args...)

macro comment(vars...) end

export @comment

const inch = 96
const pt = 4 / 3
const cm = inch / 2.54
export cm, inch, pt
"""
    theme_drl_rde(; fontsize = 10pt) -> Theme

Custom theme for DRL_RDE project plots.
Combines `theme_latexfonts()` with a font size setting.

# Keywords
- `fontsize::Integer = 10pt`: Font size for the theme.
"""
function theme_drl_rde(; fontsize = 10pt)
    return merge(
        theme_latexfonts(), Theme(
            fontsize = fontsize,
            fonts = (
                regular = "TeX Gyre Termes",
                bold = "TeX Gyre Termes Bold",
                italic = "TeX Gyre Termes Italic",
                bold_italic = "TeX Gyre Termes Bold Italic",
                sans = "TeX Gyre Heros",
                sans_bold = "TeX Gyre Heros Bold",
            )
        )
    )
end

export theme_drl_rde

end # module DRL_RDE
