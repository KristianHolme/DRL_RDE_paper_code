using DrWatson
@quickactivate :DRL_RDE
using RDE
using RDE_Env
using GLMakie
GLMakie.activate!()
## Create environment

env = RDEEnv(;
    dt = DRL_RDE.get_reasonable_time_scale(2.0f0π),
    params = RDEParam(;
        tmax = 4000.0f0,
        # ν_1 = 0.0f0, ν_2 = 0.0f0
    ),
    # observation_strat = SectionedStateWithPressureHistoryObservation(),
    observation_strat = SectionedStateObservation(),
    # observation_strat=MultiSectionObservation(n_sections=8),
    action_strat = DirectScalarPressureAction(),
    # action_strat=DirectVectorPressureAction(n_sections=8),
    # reset_strategy = NShock(2),
    # reset_strategy = Default(),
    # reward_strat = PeriodMinimumReward(),
    reward_strat = StabilityReward(),
    # reward_strat = StabilityTargetReward(),
    context_strat = FixedTargetShockCountStrategy(4)
);
_reset!(env)
## Run interactive control
envi, figi = interactive_control(
    env;
    show_observations = false,
    dtmax = 20.0f0,
    hard_u_limit = true
)
display(figi)
## MultiAgent
env = RDEEnv(;
    dt = DRL_RDE.get_reasonable_time_scale(2.0f0π, 8),
    params = RDEParam(tmax = 4000.0f0),
    observation_strat = MultiCenteredMovingFrameObservation(n_sections = 8),
    # observation_strat = MultiCenteredMovingFrameObservation(n_sections = 8),
    control_shift_strategy = MovingFrameControlShift(),
    action_strat = DirectVectorPressureAction(n_sections = 8),
    reward_strat = StabilityTargetReward(),
    context_strat = FixedTargetShockCountStrategy(4)
);
envi, figi = interactive_control(
    env;
    show_observations = true,
    dtmax = 5.0f0,
    hard_u_limit = true
)
display(figi)
