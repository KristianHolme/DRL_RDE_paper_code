# Generate the SU 3→2 trajectory shown in comparison_split/uniform_3to2.jl.
# Fair-clock protocol (see evaluate_uniform_fair_clock.jl): ar14 US4 policy for
# target 2, seed 84 (the fair-clock winner), started from 3 waves, 100-time-unit
# uncontrolled hold, action held for the trained control step on a 0.5 grid.
# The episode runs to t=380 so it matches the two-step example (Fig. 6).
#
# ShiftReset draws a random wave shift, so a rerun is not bit-identical to the
# cached file in the published data. Change SEED to get another realisation.
#
# Usage: julia --project scripts/studies/transition/uniform_3to2_trajectory.jl
using DrWatson
@quickactivate :DRL_RDE
using DRL_RDE
using Accessors
using JLD2
using Random
##
const START = 3
const TARGET = 2
const SU_SEED = 84
const TMAX = 380.0f0
const SEED = 1

study = "ar14_all_target"
results_path = datadir(
    "studies", study, "results",
    "experiment_seed=$(SU_SEED)_recipe=us4_run=1_target_shock_count=$(TARGET).jld2",
)
out_path = get(
    ENV, "UNIFORM_3TO2_OUT",
    datadir("studies", study, "fair_clock", "uniform_3to2_trajectory_t380.jld2"),
)
##
result = load(results_path)
experiment = result["experiment"]
model_path = datadir("studies", study, "models", basename(String(result["model_path"])))
control_dt = experiment.env_config.dt
@reset experiment.env_config.reward_strat = StabilityReward(1.0f0)
@reset experiment.env_config.context_strat = FixedTargetShockCountStrategy(TARGET)
@reset experiment.env_config.reset_strategy = ShiftReset(NShock(START))
@reset experiment.env_config.tmax = TMAX
@reset experiment.env_config.dt = 0.5f0
jl_policy = get_jl_policy(experiment, model_path)
##
Random.seed!(SEED)
env = create_env(experiment.env_config)
delayed = DelayedPolicy(jl_policy, DRL_RDE.DELAY_TIME, env)
policy = HeldActionPolicy(delayed, env, control_dt)
su_data = run_policy(policy, env)
su_x = env.prob.x
found, _, _ = detect_transition(su_data, env.prob, TARGET)
@info "SU 3→2" found su_data.state_ts[end]
##
mkpath(dirname(out_path))
jldsave(out_path; su_data, su_x)
println("saved SU 3→2 trajectory to ", out_path)
