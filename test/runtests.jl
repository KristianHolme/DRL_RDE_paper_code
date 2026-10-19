using TestItemRunner

@testitem "Code quality (Aqua.jl)" tags = [:quality] begin
    using Aqua, DRL_RDE
    stale_ignore = [
        :AlgebraOfGraphics, :CairoMakie, :CurveFit, :IntervalSets, :JSON,
        :MakieExtra, :WGLMakie, :ZipArchives,
    ]
    Aqua.test_all(DRL_RDE; stale_deps = false, piracies = false)
    Aqua.test_stale_deps(DRL_RDE; ignore = stale_ignore)
    Aqua.test_piracies(DRL_RDE; broken = true)
end

@testitem "Code linting (JET.jl)" tags = [:quality] begin
    include(joinpath(@__DIR__, "jet_utils.jl"))
    using JET, DRL_RDE
    report = JET.report_package(DRL_RDE; target_modules = (DRL_RDE, RDE_Env_Strategies), toplevel_logger = nothing)
    bad = filter(r -> !_jet_false_positive(r), JET.get_reports(report))
    @test isempty(bad)
end

@run_package_tests filter = ti -> startswith(ti.filename, @__DIR__)
