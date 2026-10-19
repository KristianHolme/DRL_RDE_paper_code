function create_env(
        N;
        method = FiniteVolumeMethod(),
        tmax = 800.0f0,
        T = Float32
    )
    env = RDEEnv(;
        params = RDEParam{T}(; tmax, N),
        action_strat = ScalarPressureAction(),
        reward_strat = PeriodicityReward(),
        method = method,
    )
    return env
end

function get_data_N(data::PolicyRunData)
    N = length(data.states[1]) ÷ 2
    return N
end

function interpolate_data(run_data::PolicyRunData, N::Int)
    N_data = get_data_N(run_data)
    N_new = N
    N_ratio = N_data ÷ N_new
    @assert N_data % N_new == 0 "N_data ($N_data) must be a multiple of N_new ($N_new)"
    states = run_data.states
    us, λs = RDE.split_sol(states)
    new_us = [u[1:N_ratio:end] for u in us]
    new_λs = [λ[1:N_ratio:end] for λ in λs]
    new_states::Vector{<:AbstractVector} = [vcat(u, λ) for (u, λ) in zip(new_us, new_λs)]
    return new_states
end

function states_error(states_err::Vector{Vector{T}}, states_truth::Vector{Vector{T}}) where {T <: AbstractFloat}
    N = size(states_err, 1) ÷ 2
    err_states_matrix = stack(states_err)
    truth_states_matrix = stack(states_truth)
    err::T = norm(err_states_matrix - truth_states_matrix) / T(sqrt(N))
    return err
end

function get_error(data_err, data_truth)
    states_err = data_err.states
    if get_data_N(data_err) != get_data_N(data_truth)
        states_truth = interpolate_data(data_truth, get_data_N(data_err))
    else
        states_truth = data_truth.states
    end
    return states_error(states_err, states_truth)
end

function error_analysis(N_ref::Int, Ns::Vector{Int}, T::Type{<:AbstractFloat} = Float32)
    errors = Vector{Union{Missing, T}}(undef, length(Ns))

    # Combine all N values including reference for parallel execution
    all_Ns = [N_ref; Ns]
    all_data = Vector{PolicyRunData}(undef, length(all_Ns))
    all_times = Vector{Float64}(undef, length(all_Ns))

    # Run all simulations in parallel, including reference
    @info "Running all simulations in parallel"
    p = ProgressMeter.Progress(length(all_Ns), desc = "Running simulations")
    @threads for (i, N) in collect(enumerate(all_Ns))
        env = create_env(N, T = T)
        policy = get_default_policy(env)
        time = @elapsed data = run_policy(policy, env)
        all_data[i] = data
        all_times[i] = time
        next!(p)
    end

    # Extract reference data and time
    data_ref = all_data[1]
    ref_time = all_times[1]

    # Extract times for the test cases (excluding reference)
    times = all_times[2:end]
    data = all_data[2:end]

    # Calculate errors in separate loop (can also be parallelized if needed)
    @info "Calculating errors"
    for (i, N) in enumerate(Ns)
        if length(data[i].states) != length(data_ref.states)
            @info "N=$N: differs in number of states ($(length(data[i].states))) from reference ($(length(data_ref.states)))"
            errors[i] = missing
        else
            errors[i] = get_error(data[i], data_ref)
        end
    end

    return errors, times, ref_time
end

function plot_error_analysis(
        N_ref::Int,
        Ns::Vector{Int},
        errors::Vector{T},
        env_times::Vector{Float64},
        ref_time::Float64
    ) where {T}
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        yscale = log10,
        xscale = log2,
        ylabel = "Error",
        xlabel = "N",
        title = "Error Analysis for N_ref = $N_ref, T = $T"
    )
    lines!(ax, Ns, errors)
    ax2 = Axis(fig[1, 2], xscale = log2, xlabel = "N", ylabel = "Time")
    lines!(ax2, Ns, env_times)
    return fig
end

function get_default_policy(env::RDEEnv{T}) where {T}
    policy = StepwiseRDEPolicy(
        env, convert(Vector{T}, [80.0, 220.0, 400.0, 650.0f0]),
        convert(Vector{T}, [0.65, 0.86, 0.65, 0.97])
    )
    return policy
end
