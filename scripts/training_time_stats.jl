using DrWatson
@quickactivate :DRL_RDE

using DataFrames
using PrettyTables
using Statistics

const STUDY_TO_CONFIG = (
    ("SAU3", "SA"),
    ("MA3", "MA"),
    ("SAS3", "SAS"),
)

function empty_run_df()
    df = DataFrame(
        config = String[],
        study = String[],
        computer = String[],
        training_time_seconds = Float64[],
        training_time_hours = Float64[],
    )
    return df
end

function empty_host_df()
    df = DataFrame(
        computer = String[],
        ssh_target = String[],
        architecture = String[],
        model_name = String[],
        cpu_count = Union{Missing, Int}[],
        ssh_status = String[],
        ssh_error = String[],
    )
    return df
end

function row_get(row::DataFrameRow, key::String)
    col = Symbol(key)
    if col in propertynames(row)
        return row[col]
    end
    return missing
end

function safe_string(x)
    if ismissing(x) || isnothing(x)
        return nothing
    end
    value = strip(String(x))
    if isempty(value)
        return nothing
    end
    return value
end

function safe_float64(x)
    if ismissing(x) || isnothing(x)
        return nothing
    end
    if x isa Number
        return Float64(x)
    end
    parsed = tryparse(Float64, strip(String(x)))
    if isnothing(parsed)
        return nothing
    end
    return parsed
end

function load_study_results(study::String, config::String)
    results_dir = datadir("studies", study, "results")
    if !isdir(results_dir)
        @warn "Results directory not found, skipping study" study results_dir
        return empty_run_df(), 0, 0
    end

    results = DrWatson.collect_results!(results_dir)
    n_loaded = nrow(results)
    if n_loaded == 0
        return empty_run_df(), 0, 0
    end

    rows = NamedTuple[]
    dropped = 0
    for row in eachrow(results)
        training_time_seconds = safe_float64(row_get(row, "training_time"))
        computer = safe_string(row_get(row, "computer"))
        if isnothing(training_time_seconds) || isnothing(computer)
            dropped += 1
            continue
        end

        push!(
            rows,
            (
                config = config,
                study = study,
                computer = computer,
                training_time_seconds = training_time_seconds,
                training_time_hours = training_time_seconds / 3600.0,
            ),
        )
    end

    if isempty(rows)
        return empty_run_df(), n_loaded, dropped
    end

    return DataFrame(rows), n_loaded, dropped
end

function run_capture(cmd::Cmd)
    stdout_buffer = IOBuffer()
    stderr_buffer = IOBuffer()
    process = run(pipeline(ignorestatus(cmd), stdout = stdout_buffer, stderr = stderr_buffer))
    result = (
        success = process.exitcode == 0,
        exitcode = process.exitcode,
        stdout = String(take!(stdout_buffer)),
        stderr = String(take!(stderr_buffer)),
    )
    return result
end

function parse_lscpu_output(lscpu_text::String)
    parsed = Dict{String, String}()
    for line in split(lscpu_text, '\n')
        if !occursin(':', line)
            continue
        end
        key, value = split(line, ':'; limit = 2)
        parsed[strip(key)] = strip(value)
    end
    return parsed
end

function parse_cpu_count(value::String)
    match_result = match(r"^\d+", strip(value))
    if isnothing(match_result)
        return missing
    end
    parsed = tryparse(Int, match_result.match)
    if isnothing(parsed)
        return missing
    end
    return parsed
end

function first_error_line(stderr::String)
    stripped = strip(stderr)
    if isempty(stripped)
        return ""
    end
    return split(stripped, '\n'; limit = 2)[1]
end

function host_candidates(host::AbstractString)
    candidates = String[String(host)]
    if occursin('.', host)
        short_host = split(String(host), '.'; limit = 2)[1]
        if !isempty(short_host) && short_host != host
            push!(candidates, short_host)
        end
    end
    return candidates
end

function fetch_host_info(host::AbstractString)
    last_error = ""
    for ssh_target in host_candidates(host)
        ssh_base = `ssh -o BatchMode=yes -o ConnectTimeout=5 $ssh_target`
        lscpu_result = run_capture(`$ssh_base lscpu`)

        if lscpu_result.success
            parsed = parse_lscpu_output(lscpu_result.stdout)
            architecture = get(parsed, "Architecture", "unknown")
            model_name = get(parsed, "Model name", "unknown")
            cpu_count = haskey(parsed, "CPU(s)") ? parse_cpu_count(parsed["CPU(s)"]) : missing
            status = ssh_target == host ? "ok" : "ok_via_short_host"
            return (
                computer = String(host),
                ssh_target = String(ssh_target),
                architecture = architecture,
                model_name = model_name,
                cpu_count = cpu_count,
                ssh_status = status,
                ssh_error = "",
            )
        end

        last_error = first_error_line(lscpu_result.stderr)
        uname_result = run_capture(`$ssh_base uname -m`)
        if uname_result.success
            architecture = strip(uname_result.stdout)
            if isempty(architecture)
                architecture = "unknown"
            end
            status = ssh_target == host ? "uname_only" : "uname_only_via_short_host"
            return (
                computer = String(host),
                ssh_target = String(ssh_target),
                architecture = architecture,
                model_name = "unavailable",
                cpu_count = missing,
                ssh_status = status,
                ssh_error = first_error_line(uname_result.stderr),
            )
        end
        if isempty(last_error)
            last_error = first_error_line(uname_result.stderr)
        end
    end

    return (
        computer = String(host),
        ssh_target = String(host),
        architecture = "unreachable",
        model_name = "unreachable",
        cpu_count = missing,
        ssh_status = "unreachable",
        ssh_error = last_error,
    )
end

function collect_host_cpu_info(data::DataFrame)
    if nrow(data) == 0
        return empty_host_df()
    end

    hosts = sort(unique(data.computer))
    rows = map(fetch_host_info, hosts)
    return DataFrame(rows)
end

function summarize_by_config_host(data::DataFrame, host_info::DataFrame)
    grouped = combine(
        groupby(data, [:config, :computer]),
        :training_time_hours => length => :n_runs,
        :training_time_hours => mean => :mean_hours,
        :training_time_hours => (x -> std(x; corrected = false)) => :std_hours,
        :training_time_hours => median => :median_hours,
        :training_time_hours => minimum => :min_hours,
        :training_time_hours => maximum => :max_hours,
    )
    grouped = leftjoin(grouped, host_info; on = :computer)

    config_rank = Dict("SA" => 1, "MA" => 2, "SAS" => 3)
    grouped.config_rank = [get(config_rank, c, 99) for c in grouped.config]
    sort!(grouped, [:config_rank, :computer])
    select!(grouped, Not(:config_rank))
    return grouped
end

function summarize_by_config(data::DataFrame)
    grouped = combine(
        groupby(data, :config),
        :training_time_hours => length => :n_runs,
        :training_time_hours => mean => :mean_hours,
        :training_time_hours => (x -> std(x; corrected = false)) => :std_hours,
        :training_time_hours => median => :median_hours,
        :training_time_hours => minimum => :min_hours,
        :training_time_hours => maximum => :max_hours,
    )
    config_rank = Dict("SA" => 1, "MA" => 2, "SAS" => 3)
    grouped.config_rank = [get(config_rank, c, 99) for c in grouped.config]
    sort!(grouped, :config_rank)
    select!(grouped, Not(:config_rank))
    return grouped
end

function round_numeric_columns!(table::DataFrame, columns::Vector{Symbol}; digits::Int = 3)
    for col in columns
        if col in names(table)
            table[!, col] = round.(table[!, col]; digits = digits)
        end
    end
    return table
end

function print_tables(main_table::DataFrame, config_table::DataFrame)
    rounded_main = copy(main_table)
    round_numeric_columns!(
        rounded_main,
        [:mean_hours, :std_hours, :median_hours, :min_hours, :max_hours];
        digits = 3,
    )

    rounded_config = copy(config_table)
    round_numeric_columns!(
        rounded_config,
        [:mean_hours, :std_hours, :median_hours, :min_hours, :max_hours];
        digits = 3,
    )

    println()
    println("Training time stats by configuration and host")
    pretty_table(rounded_main)

    println()
    println("Training time stats by configuration (all hosts combined)")
    pretty_table(rounded_config)
    return nothing
end

function _escape_csv_field(x)
    s = x isa Missing ? "" : string(x)
    if occursin(r"[,\"\r\n]", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function _dataframe_to_csv(path::String, df::DataFrame)
    open(path, "w") do io
        println(io, join(_escape_csv_field.(string.(names(df))), ","))
        for row in eachrow(df)
            println(io, join(_escape_csv_field.(values(row)), ","))
        end
    end
    return path
end

function save_training_time_data(
    all_runs::DataFrame,
    grouped_host::DataFrame,
    grouped_config::DataFrame,
    host_info::DataFrame,
)
    save_dir = datadir("training_time_stats")
    mkpath(save_dir)
    paths = String[]
    for (name, df) in [
        ("all_runs", all_runs),
        ("by_config_host", grouped_host),
        ("by_config", grouped_config),
        ("host_info", host_info),
    ]
        p = joinpath(save_dir, "training_time_stats_$(name).csv")
        _dataframe_to_csv(p, df)
        push!(paths, p)
    end
    @info "Saved training time stats (CSV)" paths = paths
    return paths
end

function main()
    all_data = DataFrame[]
    total_loaded = 0
    total_dropped = 0

    for (study, config) in STUDY_TO_CONFIG
        data, loaded, dropped = load_study_results(study, config)
        total_loaded += loaded
        total_dropped += dropped
        if nrow(data) > 0
            push!(all_data, data)
        end
    end

    if isempty(all_data)
        println("No valid training runs found in SAU3, MA3, or SAS3.")
        return nothing
    end

    all_runs = vcat(all_data...)
    println("Loaded runs: $total_loaded")
    println("Usable runs: $(nrow(all_runs))")
    println("Dropped runs (missing/invalid training_time or computer): $total_dropped")

    host_info = collect_host_cpu_info(all_runs)
    grouped_host = summarize_by_config_host(all_runs, host_info)
    grouped_config = summarize_by_config(all_runs)

    save_training_time_data(all_runs, grouped_host, grouped_config, host_info)
    print_tables(grouped_host, grouped_config)
    return nothing
end

main()
