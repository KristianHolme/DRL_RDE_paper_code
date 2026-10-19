using LoggingExtras: ActiveFilteredLogger

# Stable substring from JLD2 (see JLD2/src/data/reconstructing_datatypes.jl).
const JLD2_RECONSTRUCT_WARN_SUBSTRING = "some parameters could not be resolved"

"""
    install_process_data_logging!()

Set the global logger to a console logger wrapped in `ActiveFilteredLogger`, dropping
`@warn` messages whose text contains [`JLD2_RECONSTRUCT_WARN_SUBSTRING`](@ref).

Used by `scripts/process_data.jl` on the master and on each Distributed worker after loading
the project.
"""
function install_process_data_logging!()
    function keep_log_message(log)
        if log.level == Logging.Warn &&
                occursin(JLD2_RECONSTRUCT_WARN_SUBSTRING, string(log.message))
            return false
        end
        return true
    end
    sink = Logging.ConsoleLogger(stderr, Logging.Info)
    Logging.global_logger(ActiveFilteredLogger(keep_log_message, sink))
    return nothing
end
