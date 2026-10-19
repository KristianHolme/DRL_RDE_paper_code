@testitem "making envs" begin
    env_config = RDEEnvConfig(
        reward_strat = TransitionBasedReward(
            PeriodicityReward(),
        )
    )

    num_envs = 4
    envs = create_envs(env_config, num_envs)

    _act!.(envs, zeros(Float32, num_envs))


    @testset "env reward cache are not shared between envs" begin
        pass = true
        for i in 1:num_envs, j in setdiff(1:num_envs, i)
            pass = pass && !(envs[i].cache.reward_cache.outer_cache === envs[j].cache.reward_cache.outer_cache)
        end
        @test pass
    end

    @testset "cache is not shared between envs" begin
        pass = true
        for i in 1:num_envs, j in setdiff(1:num_envs, i)
            pass = pass && !(envs[i].cache === envs[j].cache)
        end
        @test pass
    end

    @test all(map(env -> length(env.cache.reward_cache.outer_cache.past_rewards) == 1, envs))
    @test all(map(env -> length(env.cache.reward_cache.outer_cache.past_shock_counts) == 1, envs))

    _act!.(envs, zeros(Float32, 4))

    @test !(envs[1].cache.reward_cache.outer_cache === envs[2].cache.reward_cache.outer_cache)
    @test all(map(env -> length(env.cache.reward_cache.outer_cache.past_rewards) == 2, envs))
    @test all(map(env -> length(env.cache.reward_cache.outer_cache.past_shock_counts) == 2, envs))
end
