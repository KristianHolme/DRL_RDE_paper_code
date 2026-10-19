@testitem "Learning Rate Schedules" tags = [:schedules] begin
    @testset "Linear Schedule" begin
        # Test basic linear schedule
        schedule = LinearSchedule(0.001, 0.0001)
        @test schedule(1.0) ≈ 0.001  # At progress = 1
        @test schedule(0.5) ≈ 0.00055  # Halfway
        @test schedule(0.0) ≈ 0.0001  # At progress = 0

        # Test extrapolation beyond [0,1]
        @test schedule(-0.5) ≈ 0.0001 - 0.00045  # Continues linear decrease
        @test schedule(1.5) ≈ 0.001 + 0.00045  # Continues linear increase
    end

    @testset "Cosine Schedule" begin
        # Test basic cosine schedule
        schedule = CosineSchedule(0.001, 0.0001, 3)
        @test schedule(1.0) ≈ 0.001  # At progress = 1
        @test schedule(0.0) ≈ 0.0001  # At progress = 0

        # Test midpoint
        mid_value = schedule(0.5)
        @test mid_value < 0.001
        @test mid_value > 0.0001

        # Test extrapolation - cosine is periodic
        @test schedule(2.0) ≈ schedule(0.0)  # Full period
        @test schedule(-0.5) ≈ schedule(0.5)  # Half period before
    end

    @testset "Exponential Decay Schedule" begin
        # Test basic exponential decay
        initial_value = 0.001
        decay_rate = 2.0  # default decay rate
        schedule = ExpDecaySchedule(initial_value, decay_rate)
        @test schedule(1.0) ≈ initial_value  # At progress = 1 (start)
        @test schedule(0.0) ≈ initial_value * exp(-decay_rate)  # At progress = 0 (end)

        # Test that values decrease monotonically
        values = [schedule(t) for t in 1.0:-0.2:0.0]
        @test all(diff(values) .< 0)

        # Test extrapolation
        @test schedule(1.5) > initial_value  # Continues exponential pattern
        @test schedule(-0.5) < initial_value * exp(-decay_rate)  # Continues decay
    end

    @testset "Schedule Properties" begin
        # Test monotonic schedules
        monotonic_schedules = [
            LinearSchedule(0.001, 0.0001),
            ExpDecaySchedule(0.001),  # Using default decay rate
        ]

        for schedule in monotonic_schedules
            # Test values at standard boundaries
            @test schedule(1.0) ≈ 0.001  # Start (100% remaining)
            @test schedule(0.0) ≈ (schedule isa ExpDecaySchedule ? 0.001 * exp(-2.0) : 0.0001)  # End (0% remaining)

            # Test that values follow mathematical progression in [1,0]
            values = [schedule(t) for t in 1.0:-0.1:0.0]
            @test issorted(values, rev = true)  # These schedules should decrease monotonically
        end

        # Test cosine schedule separately since it's periodic
        cosine_schedule = CosineSchedule(0.001, 0.0001, 3)
        @test cosine_schedule(1.0) ≈ 0.001  # Start (100% remaining)
        @test cosine_schedule(0.0) ≈ 0.0001  # End (0% remaining)
    end
end
