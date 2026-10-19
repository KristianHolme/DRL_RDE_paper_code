# Helper function to handle IO conversion
function _get_io_context(io_or_tty::Union{IO, Int})
    if io_or_tty isa Int
        ty = open("/dev/pts/$(io_or_tty)", "w")
        io = IOContext(ty, :color => true)
    else
        io = io_or_tty
    end
    return io
end

function iteration_string(step::Int, progress_bar::Progress)
    return "$(progress_bar.counter + step) / $(progress_bar.n)"
end

"""
    run_experiment(experiment_name, search_space, secondary_config; kwargs...)

Run an experiment either in distributed or single-process mode.

# Arguments
- `experiment_name::String`: Name of the experiment
- `search_space::Dict`: Dictionary of parameter ranges to search over
- `secondary_config::Dict`: Fixed configuration parameters
- `kwargs...`: Additional arguments passed to single_run, including:
  - `distributed::Bool=false`: Whether to run in distributed mode (requires workers to be added). Progress is stored in SQLite under `datadir("progresslogs")` and can be viewed with the MultiProgressManagers dashboard.
  - `algorithm::String="PPO"`: Algorithm to use (e.g., "PPO", "A2C", "SAC", "TD3", "DDPG")

# Example
```julia
# Single process mode:
run_experiment("my_exp", search_space, secondary_config)

# Distributed mode:
using Distributed
addprocs(4)
run_experiment("my_exp", search_space, secondary_config, distributed=true)

# Using a different algorithm:
run_experiment("my_exp", search_space, secondary_config, algorithm="SAC")
```
"""


function run_experiments(experiments::Vector{Tuple{E, Dict}}; mode::Symbol = :distributed, kwargs...) where {E <: AbstractExperiment}
    return if mode == :distributed
        @debug "Running experiments in distributed mode"
        _run_experiments_distributed(experiments; kwargs...)
    elseif mode == :sequential
        @debug "Running experiments in single process mode"
        _run_experiments(experiments; kwargs...)
    else
        error("Invalid mode: $mode")
    end
end

function _run_experiments(experiments::Vector{Tuple{E, Dict}}; io = nothing, kwargs...) where {E <: AbstractExperiment}
    return @showprogress for (experiment, cfg) in experiments
        single_run(cfg, experiment; kwargs...)
    end
end

function _run_experiments_distributed_old(
        experiments::Vector{Tuple{E, Dict}};
        io::Union{IO, Int} = stdout,
        kwargs...
    ) where {E <: AbstractExperiment}
    @debug "Establishing channels and progress bars"

    # Create progress channels
    ioc = _get_io_context(io)
    n_items = length(experiments)

    # Initialize main progress bar
    main_progress = Progress(
        n_items;
        desc = "Total Progress:",
        showspeed = true,
        output = ioc
    )
    @async begin
        sleep(0.1) #wait for progress bar
        ProgressMeter.update!(
            main_progress, 0, showvalues = [
                ("Runs completed", "0 / $(n_items)"),
            ]
        )
    end

    main_channel = RemoteChannel(() -> Channel{Bool}(1000), 1)
    worker_channel = RemoteChannel(() -> Channel{Tuple{Int, Int}}(1000), 1)
    worker_inds = Dict(worker_id => findfirst(==(worker_id), workers()) for worker_id in workers())

    # Initialize worker progress bars
    worker_progress_bars = [
        Progress(
                100;
                desc = "Worker $worker_id:",
                showspeed = true,
                offset = worker_inds[worker_id] * 2,
                output = ioc
            )
            for worker_id in workers()
    ]


    # Control flags
    stop = Ref(false)
    completed_items = Ref(0)
    @info "Starting parallel execution in distributed mode"

    try
        # Start the periodic update task OUTSIDE the @sync block
        t_periodic = @async while !stop[]
            try
                sleep(10)  # Update every 10 seconds
                if !stop[]
                    ProgressMeter.update!(
                        main_progress, completed_items[], showvalues = [
                            ("Runs", iteration_string(0, main_progress)),
                        ]
                    )
                end
            catch e
                if e isa InvalidStateException && stop[]
                    break
                else
                    rethrow()
                end
            end
        end
        errormonitor(t_periodic)

        @sync begin
            # Task to update main progress
            t_main = @async while !stop[]
                try
                    val = take!(main_channel)
                    @logmsg LogLevel(-10000) "Main channel taken: $val"
                    if val
                        @logmsg LogLevel(-10000) "Main progress update"
                        completed_items[] += 1
                        next!(
                            main_progress; showvalues = [
                                ("Runs", iteration_string(1, main_progress)),
                            ]
                        )
                    else
                        break
                    end
                catch e
                    if e isa InvalidStateException && stop[]
                        break
                    else
                        rethrow()
                    end
                end
            end
            errormonitor(t_main)

            # Task to update worker progress
            t_worker = @async while !stop[]
                try
                    id, signal = take!(worker_channel)
                    if signal == 999 || stop[]
                        @logmsg LogLevel(-500) "Worker $id signal: $signal, breaking"
                        break
                    end
                    if signal < 0
                        @logmsg LogLevel(-500) "Resetting worker progress, id: $id"
                        # New progress bar with different length
                        worker_progress_bars[worker_inds[id]] = Progress(
                            -signal;
                            desc = "Worker $(id):",
                            showspeed = true,
                            offset = worker_inds[id] * 2,
                            output = ioc
                        )
                        @async begin
                            sleep(0.1) #wait for progress bar
                            ProgressMeter.update!(
                                worker_progress_bars[worker_inds[id]], 0, showvalues = [
                                    ("Initializing...", "0 / $(-signal)"),
                                ]
                            )
                        end
                    else
                        @logmsg LogLevel(-500) "Worker $id signal: $signal, updating progress"
                        p = worker_progress_bars[worker_inds[id]]
                        if p.counter + signal > p.n
                            @info "Worker $id progress bar counter + signal > n, setting to n"
                            signal = p.n - p.counter
                        end
                        next!(
                            p,
                            step = signal,
                            showvalues = [
                                ("Steps", iteration_string(signal, p)),
                            ]
                        )
                    end
                catch e
                    if e isa InvalidStateException && stop[]
                        @info "intercepted error $e"
                        break
                    else
                        rethrow()
                    end
                end
            end
            errormonitor(t_worker)

            # Main distributed work
            t_distributed = @async begin
                @logmsg LogLevel(-10000) "Starting async runs"
                try
                    @logmsg LogLevel(-10000) "Starting distributed loop"
                    @distributed (+) for (experiment, cfg) in experiments
                        @logmsg LogLevel(-10000) "stop[] = $(stop[])"
                        if !stop[]  # Only process if not stopped
                            sleep(rand() * 10) #random sleep to avoid all workers starting at the same time
                            try
                                single_run(cfg, experiment; kwargs...)
                            catch e
                                @error "Run failed" exception = (e, catch_backtrace()) experiment = experiment
                                @info "Not doing anything to channels"
                            end
                        else
                            @info "stop[] is $(stop[])"
                        end
                        1  # Always return 1 for the reduction
                    end

                    @logmsg LogLevel(-500) "Done with distributed loop, putting false in main_channel"
                    put!(main_channel, false)
                    @logmsg LogLevel(-500) "Putting 999 in worker_channel"
                    put!(worker_channel, (myid(), 999))
                catch e
                    if e isa InterruptException
                        @info "Interrupted, cleaning up..."
                        stop[] = true
                        rethrow()
                    else
                        rethrow()
                    end
                end
            end
            errormonitor(t_distributed)
        end

        # After @sync, signal the periodic task to stop and wait for it to finish
        stop[] = true
        wait(t_periodic)
    catch e
        if e isa InterruptException
            @info "Interrupted, cleaning up..."
            # Clean up channels
            put!(main_channel, false)
            put!(worker_channel, (myid(), 999))
            close(main_channel)
            close(worker_channel)
            @info "Execution stopped by user"
            return
        else
            rethrow()
        end
    finally
        @logmsg LogLevel(-10000) "finally"
        stop[] = true
    end
    return @info "Distributed execution completed"
end

function error_handling(e::Exception)
    @error "Error in distributed training" exception = (e, catch_backtrace())
    return 0
end

function _run_experiments_distributed(
        experiments::Vector{Tuple{E, Dict}};
        kwargs...
    ) where {E <: AbstractExperiment}
    @debug "Establishing progress manager and task handles"

    n_items = length(experiments)
    experiment_name = experiments[1][1].study
    timestamp = Dates.format(now(), "yyyy-mm-ddTHH-MM-ss")
    db_dir = datadir("progresslogs")
    mkpath(db_dir)
    db_path = joinpath(db_dir, experiment_name * "-" * timestamp * ".db")
    task_descriptions = [experiment_savename(cfg) for (_, cfg) in experiments]
    progress_manager = ProgressManager(
        experiment_name,
        n_items;
        db_path = db_path,
        description = "$(n_items) runs, started $(timestamp)",
        task_descriptions = task_descriptions,
    )
    remote_tasks = [get_task(progress_manager, i, :remote) for i in 1:n_items]
    jobs = [(cfg, experiment, remote_tasks[i]) for (i, (experiment, cfg)) in enumerate(experiments)]

    @info "Inspect progress with: mpm $(db_dir)"

    try
        result = pmap(jobs, on_error = error_handling) do (cfg, experiment, task)
            sleep(rand() * 10) # to balance load
            single_run(cfg, experiment; progress_task = task)
            return 1
        end
        sucesses = sum(result)
        failures = length(result) - sucesses
        @info "pmap done"
        @info "Successes: $sucesses, Failures: $failures"
        ntfy_topic = NTFY_TOPIC
        if !isnothing(ntfy_topic)
            send_ntfy(
                ntfy_topic,
                """Experiment $(experiment_name)\n
                finished at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
                with successes: $sucesses and failures: $failures"""
            )
        end
    catch e
        if e isa InterruptException
            @info "Interrupted!"
        end
        @error "Error in distributed training" exception = (e, catch_backtrace())
    finally
        finish!(progress_manager)
    end

    @info "Distributed execution completed"
    return nothing
end
function continue_experiments(
        experiments::Vector{Tuple{E, Dict}};
        learn_config::Union{LearnConfig, Nothing} = nothing,
        distributed::Bool = false,
        kwargs...
    ) where {E <: AbstractExperiment}
    if learn_config !== nothing
        for (i, (experiment, cfg)) in enumerate(experiments)
            experiment = @set experiment.sb_config.learn_config = learn_config
            experiments[i] = (experiment, cfg)
        end
    end
    run_experiments(experiments; distributed, continue_training = true, kwargs...)
    return nothing
end
