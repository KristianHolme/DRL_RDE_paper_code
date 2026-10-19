#!/usr/bin/env julia

using Pkg
using ArgParse
using Dates
using DrWatson
using Random
# Parse command line arguments at top level
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--n_procs", "-p"
        help = "Number of processes to add"
        arg_type = Int
        default = 10
        "--study", "-s"
        help = "Study name (e.g., MA, SAU, SAUMF, SAS, SASMF, MAMF)"
        arg_type = String
        required = true
        "--threads_per_proc", "-t"
        help = "Number of threads per worker process ('auto' for system default)"
        arg_type = String
        default = "auto"
        "--n_samples", "-N"
        help = "Number of samples to generate"
        arg_type = Int
        default = 12
        "--max_steps"
        help = "Maximum number of steps to train for"
        arg_type = Int
        default = 10_000
        "--study_suffix", "-S"
        help = "Suffix to add to the study name"
        arg_type = String
        default = ""
        "--runs", "-r"
        help = "Number of runs (different seeds) to perform for each config"
        arg_type = Int
        default = 1
    end

    return parse_args(s)
end


args = parse_commandline()

@quickactivate :DRL_RDE

using Distributed
using Enzyme

# Setup distributed computing
threads_per_proc = args["threads_per_proc"]
exeflags = ["--startup-file=yes"]
if threads_per_proc != "auto"
    exeflags = vcat(exeflags, "--threads=$(threads_per_proc)")
end
n_procs = args["n_procs"]
@info "Adding $n_procs processes with threads_per_proc=$threads_per_proc..."
addprocs(n_procs, exeflags = exeflags)

@everywhere using DrWatson
@everywhere begin
    @quickactivate :DRL_RDE
    using Drill
    using Enzyme
end

# Include the specified study script (autoresearch/ first, then experiment_configs/)
study = args["study"]
study_candidates = (
    projectdir("autoresearch", "$(study).jl"),
    projectdir("experiment_configs", "$(study).jl"),
)
study_script = findfirst(isfile, study_candidates)
if study_script === nothing
    error("Study script not found for $(study). Looked in: $(join(study_candidates, ", "))")
end
study_script = study_candidates[study_script]
@info "Loading study script: $study_script"

include(study_script)

n_samples = args["n_samples"]
max_steps = args["max_steps"]
study_suffix = args["study_suffix"]
n_runs = args["runs"]
experiments = DRL_RDE.setup_experiments(sample_cfgs, n_samples, max_steps, study_suffix, n_runs)
@info "Running experiment $(study * study_suffix) with $(length(experiments) ÷ n_runs) configs, $max_steps steps, $n_runs runs"
shuffle!(experiments)
# Run experiment
@info "Starting $(study * study_suffix) at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))"
run_experiments(experiments)
rmprocs(workers())
@info "Finished $(study * study_suffix) at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))"

# ## Continue experiment (commented out for now)
# extra_steps = 500_000
# additional_learning = BasicLearnConfig(total_timesteps=extra_steps, progress_bar=false)
# continue_experiments(experiments, learn_config=additional_learning, distributed=true)

#template to run:
#julia --project scripts/train.jl -s SAU -S test -p 4 -t auto -N 12 --max_steps 10000 --runs 3
