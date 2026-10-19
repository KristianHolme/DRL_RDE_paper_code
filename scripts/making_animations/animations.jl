# Dual-frame MP4s for the paper YouTube playlist.
# SU uses ar14 us4 (selected fair-clock seeds); segmented configs use SAU3-era studies.
using DrWatson
@quickactivate :DRL_RDE
using Accessors
using CairoMakie
CairoMakie.activate!()
using JLD2
using ProgressMeter

# Makie 0.25 dropped FFMPEG_jll; use the system binary for record().
let ffmpeg = Sys.which("ffmpeg")
    ffmpeg === nothing && error("ffmpeg not found on PATH; install it or add FFMPEG_jll")
    Makie.ffmpeg_path!(ffmpeg)
end
##

function stationary_and_moving_frames(; data, env, controls = false, skip = 1)
    time_idx = Observable(1)
    time_steps = length(data.state_ts)
    iter = collect(1:skip:time_steps)
    fig = plot_policy_data(
        data, env; time_idx, player_controls = controls,
        show_mouse_vlines = false, rewards_and_shocks = false, control_history = false,
        moving_frame = true,
        live_control = true,
        fig_size = (1000, 450)
    )
    stationary_layout = fig[1, 0] = GridLayout()
    RDE_Env.plot_policy_data!(
        stationary_layout, data, env; time_idx, player_controls = controls,
        show_mouse_vlines = false, rewards_and_shocks = false, control_history = false,
        live_control = true,
        moving_frame = false,
    )
    return fig, iter, time_idx
end

function animate(; fig, iter, time_idx, fname, fps = 60, dpi = 300)
    path = plotsdir("animations", fname * ".mp4")
    if !isdir(dirname(path))
        mkpath(plotsdir("animations"))
    end
    p = Progress(length(iter), desc = "Recording animation...")
    record(fig, path, iter; framerate = fps, px_per_unit = dpi / inch) do i
        time_idx[] = i
        next!(p)
    end
    @info "Animation saved to $path"
    return path
end

function sample_run()
    dt = 1.0f0
    tmax = 700.0f0
    target_shock_count = 3
    N = 512
    env = RDEEnv(;
        dt = dt,
        params = RDEParam(; tmax, N),
        observation_strat = SectionedStateObservation(),
        action_strat = DirectScalarPressureAction(),
        reward_strat = PeriodMinimumReward(),
        context_strat = FixedTargetShockCountStrategy(target_shock_count)
    )
    _reset!(env)
    time_points = [50.0f0, 140.0f0, 300.0f0, 450.0f0, 530.0f0]
    policy = StepwiseRDEPolicy(
        env, time_points,
        [0.65f0, 0.85f0, 0.64f0, 0.98f0, 1.0f0]
    )
    @time run_data = run_policy(policy, env)
    return run_data, env, "sample-run"
end

function get_pid_data(;
        initial_waves::Int,
        target_waves::Int,
        delay = 50.0f0,
        tmax = 330.0f0,
    )
    best_pids = load_pid_transition_best_pids()
    row = get_best_pid_for_target(best_pids, target_waves)
    eval_config = get_PID_eval_config(target_waves)
    experiment = PIDTransitionExperiment(;
        eval_config,
        pid_policy = PIDControllerPolicy(; Kp = row.Kp, Ki = row.Ki, Kd = row.Kd),
        target_shock_count = target_waves,
        init_shock_count = initial_waves,
    )
    policy = experiment.pid_policy
    env_config = experiment.eval_config.env_config
    env_config = @set env_config.tmax = tmax
    env_config = @set env_config.reward_strat = StabilityReward(1.0f0)
    env_config = @set env_config.reset_strategy = ShiftReset(NShock(initial_waves))
    env = create_env(env_config)
    policy = DelayedPolicy(policy, delay, env)
    data = run_policy(policy, env)
    return data, env
end

function get_twostep_data(;
        initial_waves::Int,
        target_waves::Int,
        delay = 0.0f0,
        tmax = 380.0f0,
    )
    best_configs = get_best_configs_per_target()
    row = get_best_config_for_target(best_configs, target_waves)
    target_shocks = row[:target_shocks]
    intermediate_u_p = row[:intermediate_u_p]
    intermediate_duration = row[:intermediate_duration]
    u_p_start = RDE.SHOCK_PRESSURES[initial_waves]
    u_p_end = RDE.SHOCK_PRESSURES[target_waves]
    params = RDEParam(; tmax)
    env = RDEEnv(
        params,
        dt = 0.5f0,
        τ_smooth = 0.1f0,
        observation_strat = StateObservation(),
        reset_strategy = ShiftReset(NShock(initial_waves)),
        reward_strat = StabilityReward(1.0f0),
        action_strat = DirectScalarPressureAction(),
        context_strat = FixedTargetShockCountStrategy(target_shocks),
    )
    transition_start = delay
    policy = StepwiseRDEPolicy(
        env,
        [0.0f0, transition_start, transition_start + intermediate_duration],
        [u_p_start, intermediate_u_p, u_p_end],
    )
    data = run_policy(policy, env)
    return data, env
end

function get_drl_data(; dataname, moving_frame, initial_waves, target_waves, tmax, delay = 20.0f0)
    function filter_function(r)
        return !(:movingframe in propertynames(r)) || r.movingframe == moving_frame && r.target_shock_count == target_waves
    end
    policy, env = get_best_policy(
        dataname;
        filter_function,
        target_shock_count = target_waves,
        initial_shock = initial_waves,
        delay,
        tmax
    )
    data = run_policy(policy, env)
    return data, env
end

"""Load selected ar14 us4 policy and run one fair-clock-style transition."""
function get_su_data(;
        initial_waves::Int,
        target_waves::Int,
        seed::Int,
        tmax = 400.0f0,
        delay = DRL_RDE.DELAY_TIME,
        lock_dt = 0.5f0,
    )
    results_dir = datadir("studies", "ar14_all_target", "results")
    models_dir = datadir("studies", "ar14_all_target", "models")
    result_files = filter(
        p -> occursin("recipe=us4_run", p) &&
            occursin("experiment_seed=$(seed)_", p) &&
            occursin("target_shock_count=$(target_waves).", p),
        readdir(results_dir),
    )
    isempty(result_files) && error("No us4 result for seed=$seed target=$target_waves in $results_dir")
    length(result_files) > 1 && error("Ambiguous us4 results: $result_files")
    row = load(joinpath(results_dir, only(result_files)))
    experiment = row["experiment"]
    model_path = joinpath(models_dir, basename(String(row["model_path"])))
    control_dt = experiment.env_config.dt
    @reset experiment.env_config.reward_strat = StabilityReward(1.0f0)
    @reset experiment.env_config.context_strat = FixedTargetShockCountStrategy(target_waves)
    @reset experiment.env_config.tmax = Float32(tmax)
    @reset experiment.env_config.dt = Float32(lock_dt)
    @reset experiment.env_config.reset_strategy = ShiftReset(NShock(initial_waves))
    jl_policy = get_jl_policy(experiment, model_path)
    env = create_env(experiment.env_config)
    delayed = DelayedPolicy(jl_policy, Float32(delay), env)
    policy = HeldActionPolicy(delayed, env, control_dt)
    data = run_policy(policy, env)
    return data, env
end
##
# Fair-clock selected us4 seeds for targets 1–4
const SU_SEEDS = (44, 84, 44, 98)
const SU_TMAX = 400.0f0
const SU_DELAY = DRL_RDE.DELAY_TIME

segmented_cfgs = [
    ("MA3", false, "MSS", 200.0f0),
    ("MA3", true, "MSM", 200.0f0),
    ("SAS3", false, "SSS", 200.0f0),
    ("SAS3", true, "SSM", 200.0f0),
]
const SEGMENTED_DELAY = 20.0f0

wave_configurations = collect(zip((1, 2, 3, 4), (4, 3, 1, 2)))
##
# Smoke-test one SU transition (3→2, seed 84)
initial_waves = 3
target_waves = 2
name = "SU_initial=$(initial_waves)_target=$(target_waves)"
data, env = get_su_data(;
    initial_waves, target_waves,
    seed = SU_SEEDS[target_waves],
    tmax = SU_TMAX,
    delay = SU_DELAY,
)
fig, iter, time_idx = stationary_and_moving_frames(; data, env, controls = false, skip = 2)
animate(; fig, iter, time_idx, fname = name)
##
# Animate SU (replaces old SUS/SUM playlist videos)
foreach(wave_configurations) do (initial_waves, target_waves)
    data, env = get_su_data(;
        initial_waves, target_waves,
        seed = SU_SEEDS[target_waves],
        tmax = SU_TMAX,
        delay = SU_DELAY,
    )
    fname = "SU_initial=$(initial_waves)_target=$(target_waves)"
    fig, iter, time_idx = stationary_and_moving_frames(; data, env, controls = false, skip = 2)
    animate(; fig, iter, time_idx, fname)
end
##
# Animate segmented DRL (unchanged studies)
foreach(segmented_cfgs) do (dataname, moving_frame, short_name, tmax)
    foreach(wave_configurations) do (initial_waves, target_waves)
        data, env = get_drl_data(;
            dataname, moving_frame, initial_waves, target_waves,
            tmax, delay = SEGMENTED_DELAY,
        )
        fname = short_name * "_initial=$(initial_waves)_target=$(target_waves)"
        fig, iter, time_idx = stationary_and_moving_frames(; data, env, controls = false, skip = 2)
        animate(; fig, iter, time_idx, fname)
    end
end
##
# PID
initial_waves = 3
target_waves = 2
data, env = get_pid_data(; initial_waves, target_waves, delay = SEGMENTED_DELAY)
name = "PID_initial=$(initial_waves)_target=$(target_waves)"
fig, iter, time_idx = stationary_and_moving_frames(; data, env, controls = false, skip = 2)
animate(; fig, iter, time_idx, fname = name)
##
# Two-step open-loop baseline
data, env = get_twostep_data(; initial_waves = 3, target_waves = 2, delay = SEGMENTED_DELAY, tmax = 330.0f0)
name = "twostep_initial=$(initial_waves)_target=$(target_waves)"
fig, iter, time_idx = stationary_and_moving_frames(; data, env, controls = false, skip = 2)
animate(; fig, iter, time_idx, fname = name)
##
# sample-run
data, env, name = sample_run()
fig, iter, time_idx = stationary_and_moving_frames(; data, env)
animate(; fig, iter, time_idx, fname = name)
