using RDE_Env

@testitem "Utils" begin
    @testset "sigmoid" begin
        # Test basic sigmoid properties
        @test sigmoid(0.0) ≈ 0.5
        @test sigmoid(-100.0) ≈ 0.0 atol = 1.0e-10
        @test sigmoid(100.0) ≈ 1.0 atol = 1.0e-10

        # Test symmetry around 0
        @test sigmoid(2.0) + sigmoid(-2.0) ≈ 1.0

        # Test type stability
        @test typeof(sigmoid(1.0)) == Float64
        @test typeof(sigmoid(1.0f0)) == Float32

        # Test that it errors on non-float types
        @test_throws MethodError sigmoid(1)
    end

    @testset "reward_sigmoid" begin
        # Test basic properties
        @test reward_sigmoid(0.5) ≈ 0.5  # Inflection point
        @test reward_sigmoid(0.0) ≈ 0.07585 atol = 1.0e-3
        @test reward_sigmoid(1.0) ≈ 0.92414 atol = 1.0e-3

        # Test monotonicity
        x1, x2 = 0.3, 0.7
        @test reward_sigmoid(x1) < reward_sigmoid(x2)

        # Test type stability
        @test typeof(reward_sigmoid(1.0)) == Float64
        @test typeof(reward_sigmoid(1.0f0)) == Float32

        # Test that it errors on non-float types
        @test_throws MethodError reward_sigmoid(1)
    end

    @testset "sigmoid_to_linear" begin
        # Test basic properties
        @test sigmoid_to_linear(0.1) ≈ reward_sigmoid(0.1)
        @test sigmoid_to_linear(0.7) == 0.7

        # Test type stability
        @test typeof(sigmoid_to_linear(1.0)) == Float64
        @test typeof(sigmoid_to_linear(1.0f0)) == Float32

        # Test that it errors on non-float types
        @test_throws MethodError sigmoid_to_linear(1)
    end
end
