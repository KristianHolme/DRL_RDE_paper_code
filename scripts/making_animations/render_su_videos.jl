# Batch-render the four SU playlist videos (ar14 us4, fair-clock seeds).
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
        fig_size = (1000, 450),
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
    mkpath(dirname(path))
    p = Progress(length(iter), desc = "Recording $fname...")
    record(fig, path, iter; framerate = fps, px_per_unit = dpi / inch) do i
        time_idx[] = i
        next!(p)
    end
    @info "Animation saved to $path"
    return path
end

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
const SU_SEEDS = (44, 84, 44, 98)
const SU_TMAX = 400.0f0
wave_configurations = collect(zip((1, 2, 3, 4), (4, 3, 1, 2)))
##
foreach(wave_configurations) do (initial_waves, target_waves)
    @info "SU run" initial_waves target_waves seed = SU_SEEDS[target_waves]
    data, env = get_su_data(;
        initial_waves, target_waves,
        seed = SU_SEEDS[target_waves],
        tmax = SU_TMAX,
    )
    fname = "SU_initial=$(initial_waves)_target=$(target_waves)"
    fig, iter, time_idx = stationary_and_moving_frames(; data, env, controls = false, skip = 2)
    animate(; fig, iter, time_idx, fname)
end
@info "SU videos done" plotsdir("animations")
