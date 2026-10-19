using DrWatson
@quickactivate :DRL_RDE
using ArgParse
using DataFrames
using Distributed

function parse_commandline()
    s = ArgParseSettings(
        description = """
        Analyze all transitions for a given experiment.

        This script runs transition analysis on all results for the specified
        experiment, generating transition matrices and related analyses.
        """,
        version = "1.0.0",
        add_version = true,
    )

    @add_arg_table! s begin
        "experiment_name"
        help = "Name of the experiment to analyze (e.g., SAS2, SAU2)"
        required = true
        arg_type = String
        "--n_procs", "-p"
        help = "Number of worker processes to add"
        arg_type = Int
        default = 4
        "--threads_per_proc", "-t"
        help = "Number of threads per worker process ('auto' to omit --threads); default 4"
        arg_type = String
        default = "4"
        "--reevaluate"
        help = "Re-evaluate results before analysis"
        action = :store_true
        "--force"
        help = "Recompute transition matrices even when output files already exist"
        action = :store_true
    end

    return parse_args(s)
end

parsed_args = parse_commandline()
experiment_name = parsed_args["experiment_name"]
reevaluate = parsed_args["reevaluate"]
force = parsed_args["force"]
n_procs = parsed_args["n_procs"]
threads_per_proc = parsed_args["threads_per_proc"]

println("Analyzing transitions for experiment: $experiment_name")
if reevaluate
    println("Re-evaluating results before analysis...")
end

# Setup distributed computing
if n_procs > 0
    exeflags = ["--startup-file=yes"]
    if threads_per_proc != "auto"
        exeflags = vcat(exeflags, "--threads=$(threads_per_proc)")
    end

    @info "Adding $n_procs processes with threads_per_proc=$threads_per_proc..."
    addprocs(n_procs, exeflags = exeflags)

    @everywhere using DrWatson
    @everywhere @quickactivate :DRL_RDE
end

try
    DRL_RDE.analyze_all_transitions(experiment_name; reevaluate = reevaluate, force = force)
finally
    if n_procs > 0
        rmprocs(workers())
    end
end

# Examples:
# julia --project scripts/compute_transition_matrices.jl SAS2
# julia --project scripts/compute_transition_matrices.jl SAS2 -p 4 -t 4
