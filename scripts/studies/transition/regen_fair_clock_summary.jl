# Regenerate uniform_fair_clock.txt from the current bundle (PID leg at dt=0.5).
using DrWatson
@quickactivate :DRL_RDE
using DRL_RDE
using JLD2
using Statistics

function fmt_time(x)
    return isfinite(x) ? string(round(x; digits = 1)) : "fail"
end

function column_metrics(record)
    t = record.target
    starts = setdiff(1:4, t)
    rates = [record.pairs[(s, t)].success_rate for s in starts]
    times = [record.pairs[(s, t)].mean_time for s in starts]
    n_complete = count(s -> record.pairs[(s, t)].n_ok == record.pairs[(s, t)].n_trials, starts)
    mean_success = mean(rates)
    finite_times = filter(isfinite, times)
    mean_time = isempty(finite_times) ? Inf : mean(finite_times)
    return (; mean_success, mean_time, n_complete)
end

function write_fair_clock_summary(io, us4_records, winners, sus_success, sus_time, sus_seeds, two_step_success, two_step_time, pid_success, pid_time, summary)
    println(io, "Uniform fair-clock evaluation")
    println(io, "Protocol: 5 trials, tmax=600, delay=100, ShiftReset; SU/two-step/PID scored on 0.5 grid")
    println(io, "SUS selection: best-of-six per target, max mean success then min mean time")
    println(io, "Selected SUS seeds by target: ", sus_seeds)
    println(io)
    println(io, "Aggregate median successful transition time")
    println(io, "  SUS:      ", summary.sus_median)
    println(io, "  Two-step: ", summary.two_step_median)
    println(io, "  PID:      ", summary.pid_median)
    println(io)
    println(io, "Paired wins (faster time; success beats failure): W-L-T")
    println(io, "  SUS vs two-step: ", summary.vs_two_step.n_win, "-", summary.vs_two_step.n_lose, "-", summary.vs_two_step.n_tie)
    println(io, "  SUS vs PID:      ", summary.vs_pid.n_win, "-", summary.vs_pid.n_lose, "-", summary.vs_pid.n_tie)
    println(io)
    println(io, "Cells where SUS does not beat two-step:")
    for t in 1:4, s in setdiff(1:4, t)
        st = sus_time[s, t]
        bt = two_step_time[s, t]
        ss = sus_success[s, t]
        bs = two_step_success[s, t]
        sus_better = (isfinite(st) && !isfinite(bt)) || (isfinite(st) && isfinite(bt) && st < bt)
        if !sus_better
            println(io, "  ", s, "→", t, "  SUS ", fmt_time(st), " (", round(ss; digits = 1), "%)  two-step ", fmt_time(bt), " (", round(bs; digits = 1), "%)")
        end
    end
    println(io)
    println(io, "Cells where SUS does not beat PID:")
    for t in 1:4, s in setdiff(1:4, t)
        st = sus_time[s, t]
        bt = pid_time[s, t]
        ss = sus_success[s, t]
        bs = pid_success[s, t]
        sus_better = (isfinite(st) && !isfinite(bt)) || (isfinite(st) && isfinite(bt) && st < bt)
        if !sus_better
            println(io, "  ", s, "→", t, "  SUS ", fmt_time(st), " (", round(ss; digits = 1), "%)  PID ", fmt_time(bt), " (", round(bs; digits = 1), "%)")
        end
    end
    println(io)
    println(io, "Selected SUS cells (start → target): success%, mean time, trial times")
    for t in 1:4
        rec = winners[t]
        println(io, "  target ", t, " seed ", rec.seed)
        for s in setdiff(1:4, t)
            r = rec.pairs[(s, t)]
            println(io, "    ", s, "→", t, "  ", round(100 * r.success_rate; digits = 1), "%  ", fmt_time(r.mean_time), "  ", r.times)
        end
    end
    println(io)
    println(io, "All US4 seeds (robustness)")
    for rec in sort(us4_records; by = r -> (r.target, r.seed))
        m = column_metrics(rec)
        println(io, "  t=", rec.target, " seed=", rec.seed, " mean_success=", round(100 * m.mean_success; digits = 1), "% mean_time=", fmt_time(m.mean_time), " complete_starts=", m.n_complete)
        for s in setdiff(1:4, rec.target)
            r = rec.pairs[(s, rec.target)]
            println(io, "    ", s, "→", rec.target, "  ", round(100 * r.success_rate; digits = 1), "%  ", fmt_time(r.mean_time), "  ", r.times)
        end
    end
    return nothing
end
##
out_dir = datadir("studies", "ar14_all_target", "fair_clock")
fair = load(joinpath(out_dir, "uniform_fair_clock.jld2"))
open(joinpath(out_dir, "uniform_fair_clock.txt"), "w") do io
    write_fair_clock_summary(
        io, fair["us4_records"], fair["winners"], fair["sus_success"], fair["sus_time"],
        fair["sus_seeds"], fair["two_step_success"], fair["two_step_time"],
        fair["pid_success"], fair["pid_time"], fair["summary"],
    )
end
println(first(read(joinpath(out_dir, "uniform_fair_clock.txt"), String), 1200))
println("summary txt regenerated")
