# Generate the 3→2 two-step example run shown in making_figures/two-step-example.jl.
# Best two-step configuration for target 2, transition starting at t=50, tmax=380.
#
# ShiftReset draws the initial wave shift from the global RNG. SEED is chosen so
# the run matches the bracket annotations in the figure script; another seed
# gives the same transition at a different phase, and possibly different timing.
#
# Usage: julia --project scripts/two_step_example_data.jl
using DrWatson
@quickactivate :DRL_RDE
using JLD2
using Random
##
const SEED = 862
const INIT_SHOCK_COUNT = 3
const TARGET_SHOCK_COUNT = 2
const TRANSITION_START = 50.0f0

best_configs = get_best_configs_per_target()
row = get_best_config_for_target(best_configs, TARGET_SHOCK_COUNT)
target_shocks = row[:target_shocks]
intermediate_u_p = row[:intermediate_u_p]
intermediate_duration = row[:intermediate_duration]

u_p_start = RDE.SHOCK_PRESSURES[INIT_SHOCK_COUNT]
u_p_end = RDE.SHOCK_PRESSURES[TARGET_SHOCK_COUNT]
env = RDEEnv(
    RDEParam(tmax = 380.0f0), dt = 0.5f0, τ_smooth = 0.1f0,
    observation_strat = StateObservation(),
    reset_strategy = ShiftReset(NShock(INIT_SHOCK_COUNT)),
    reward_strat = StabilityReward(1.0f0),
    action_strat = DirectScalarPressureAction(),
    context_strat = FixedTargetShockCountStrategy(target_shocks)
)
policy = StepwiseRDEPolicy(
    env, [0.0f0, TRANSITION_START, TRANSITION_START + intermediate_duration],
    [u_p_start, intermediate_u_p, u_p_end]
)
##
Random.seed!(SEED)
data = run_policy(policy, env)
out_path = get(ENV, "TWO_STEP_EXAMPLE_OUT", datadir("two-step-example-data.jld2"))
jldsave(out_path; data)
println("saved two-step example run to ", out_path)
