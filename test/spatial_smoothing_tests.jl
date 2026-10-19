using Test
using DRL_RDE

@testitem "config dict applies spatial kernel width" begin
    cfg = Dict("spatial_kernel_width" => 8)
    spatial_kernel_width = get(cfg, "spatial_kernel_width", 0)
    env_config = DRL_RDE.RDEEnvConfig()
    env = DRL_RDE.create_env(
        DRL_RDE.RDEEnvConfig(; action_strat = LinearVectorPressureAction(; n_sections = 8));
        spatial_kernel_width = spatial_kernel_width,
    )
    profile = RDE.inner_profile(env.prob.injection)
    @test profile isa SpatialMultiStepPressureProfile
    @test length(profile.kernel) == 9
end
