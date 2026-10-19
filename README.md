# DRL_RDE_paper_code

This repository includes the necessary code to reproduce the figures from our paper "Timescale Separation Enables Deep Reinforcement Learning Control of Rotating Detonation Engine Mode Transitions" using data uploaded to zenodo.

preprint: <https://arxiv.org/abs/2604.14398>

## Setup

- [Install](https://docs.julialang.org/en/v1/manual/installation/) [Julia](https://julialang.org/). The project has used Julia 1.12, but other versions may also work.

run the setup script

```bash
julia scripts/setup.jl
```

This script does the following:

- add the registry KristianHolmeRegistry for the RDE and RDE_Env packages, which the projet depends on.
- add [DrWatson.jl](https://github.com/JuliaDynamics/DrWatson.jl) to your global Julia environment.
- instantiate project

## Reproducing results from scratch

### Figure 2

To generate the data used for Figure 2 in the paper, run the command:
`julia scripts/speed_analysis_ramp_sweep.jl`

This does many long simulations so will take a long time.

### Baselines

#### Two-step Baseline

Runt the scripts:
'transition_time_baseline.jl' to run the simulations and 'transition_time_analysis.jl' to process the data.

#### PID Baseline

To run the grid search for the PID baselines, do:
'julia scripts/PID_transition_study.jl'

### DRL runs

training script 'scripts/train.jl' launch: 'julia scripts/train.jl'

- run with '--help' to see possible arguments

Two "experiments" have to be run for the segmented configurations: "SAS3" (single agent, configurations SSS and SSM in the paper) and "MA3" (multi-agent, MSS and MSM). The uniform configuration SU is trained separately, see the section on the uniform comparison below.

- The important arguments to supply are '-N' , '--max_steps' and '--runs'.
- '-N': must be higher than or equal to 56 to run all configurations
- '--max_steps': must be 800000
- '--runs' must be 5
  If only testing, may use lower numbers.
  Do 'julia scripts/train.jl --help' for explanations of the arguments. The number of processes and threads per process should be adjusted to fit your hardware.

Example of commands for running all experiments done in the paper:

```bash
julia --project scripts/train.jl -s SAS3 -p 13 -t 8 -N 56 --max_steps 800000 --runs 5
julia --project scripts/train.jl -s MA3 -p 13 -t 8 -N 56 --max_steps 800000 --runs 5
```

Each of these commands will spawn 13 processes with 8 threads each to run.
Depending on hardware, running all configurations can take a long time (multiple days).
Easier to download the training data.
After either running experiments or downloading the data, you should have the data in the directories:

```
data/studies/SAS3
data/studies/MA3
```

To view live progress of the training, use the [MultiProgressManagers](https://github.com/KristianHolme/MultiProgressManagers.jl) julia app. The app is a terminal user interface (TUI) for monitoring parallel tasks.Install with the julia package manager:

```julia
using Pkg
Pkg.Apps.add("MultiProgressManagers")
```

Then run the app with:

```bash
mpm ./data/progresslogs
```

### Uniform (SU) comparison with fair-clock baselines

The uniform single-agent comparison against the two-step and PID baselines uses a separate training run, `ar14_all_target`: one SU policy per target wave count, six seeds each.

```bash
julia --project scripts/train.jl -s ar14_all_target -N 48 -p 12 -t 4 --max_steps 200000
```

The config `autoresearch/ar14_all_target.jl` also trains a second recipe (`ar4_tv`). Only the `us4` runs are used in the paper.

All three controllers are then evaluated on the same detection clock (step 0.5, 5 trials per transition, `tmax = 600`, 100 time units uncontrolled hold before control starts). Run, in this order:

```bash
# PID gains tuned under the fair-clock protocol -> data/pid_baseline/pid_retune_fair_clock.jld2
julia --project -t <ncores> scripts/studies/transition/retune_pid_fair_clock.jl
# SU (best seed per target), two-step and PID -> data/studies/ar14_all_target/fair_clock/uniform_fair_clock.jld2
julia --project -t <ncores> scripts/studies/transition/evaluate_uniform_fair_clock.jl
# SU 3 -> 2 example trajectory -> data/studies/ar14_all_target/fair_clock/uniform_3to2_trajectory_t380.jld2
julia --project scripts/studies/transition/uniform_3to2_trajectory.jl
```

Run the two-step baseline (`transition_time_baseline.jl`, see above) before `evaluate_uniform_fair_clock.jl`, because the evaluation reads the best two-step configuration per target from `data/two_step_transition_time`.

### Processing data

After baselines and DRL training is done, process data with the script 'scripts/process_data.jl'. To speed up processing time, multiple julia processes can be used by supplying the flag `-p`, optionally specifying the number of threads per process with the flag `-t`. Using 16 processes with 4 threads each:

```bash
julia --project scripts/process_data.jl -p 16 -t 4
```

After processing, the figures for the paper can be generated, see the section below.

### Reproducing results using downloaded data

To download the precomputed and preprocessed results, you can use the script 'scripts/download_data.jl'.

```bash
julia --project scripts/download_data.jl
```

Then, you can run any of the scripts in 'scripts/making_figures' to reproduce the figures.

## Reproducing figures

all scripts for reproducing figures are in 'scripts/making_figures'
each of these can be run, for example:

```
julia --project scripts/making_figures/pid_run.jl
```

this will save plots to 'plots/figures'.

Alternatively, use the script `scripts/generate_all_figures.jl` to generate all figures using simulation data.
