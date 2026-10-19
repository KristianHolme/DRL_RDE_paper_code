module RDE_Env_Strategies

import RDE_Env
import RDE_Env: AbstractScalarActionStrategy, AbstractScalarRewardStrategy,
    AbstractVectorActionStrategy, AbstractVectorRewardStrategy, AbstractCache, NoCache,
    AbstractRDEEnv, AbstractRewardStrategy, RDEEnv

# using RDE_Env
using Drill: Drill
using FFTW: FFTW, fft, rfft
using Logging: Logging, @logmsg, LogLevel
using RDE: RDE, RDEParam, current_s, current_u_p, committed_pressure_delta, control_shift,
    mean_u_p, previous_u_p, update_injection!
using Statistics: Statistics, mean, std

include("contexts.jl")
export
    EvalCycleTargetShockCountStrategy, EvalCycleTargetShockCountContext,
    FixedTargetShockCountStrategy, RandomTargetShockCountStrategy, TargetShockCountContext,
    get_target_shock_count, set_target_shock_count!, AbstractShockCountContextStrategy

include("actions.jl")
export ScalarPressureAction, ScalarAreaScalarPressureAction, VectorPressureAction,
    VectorActionCache, PIDActionCache, LinearScalarPressureAction, LinearVectorPressureAction,
    LinearMultiStepPressureAction, TargetSafeLinearScalarPressureAction,
    TargetSafeLinearMultiStepPressureAction,
    LinearScalarPressureWithDtAction, PIDAction, linear_action_to_control, linear_control_target,
    target_safe_control_target, target_safe_action_to_control, control_to_target_safe_action,
    target_safe_pressure


include("observations.jl")
export CompositeObservation, SectionedStateObservation, SectionedStateWithPressureHistoryObservation,
    SampledStateObservation, StateObservation, MultiCenteredObservation, MultiCenteredWithIndexObservation,
    MeanInjectionPressureObservation, MultiCenteredWithPressureHistoryObservation,
    MultiCenteredMovingFrameObservation, MultiAgentObservationWithSinusoidalPE,
    SectionedStateMovingFrameObservation, FourierObservation, FourierMagnitudeObservation,
    FourierMagnitudePressureShockObservation, FourierMagnitudeActionHistoryObservation,
    PressureHistoryWaveRewardObservation


include("rewards.jl")
export RewardShiftBufferCache, CompositeRewardCache, WrappedRewardCache, ExponentialAverageCache,
    TransitionBasedCache, ShockSpanReward, ShockPreservingReward, ShockPreservingSymmetryReward,
    PeriodicityReward, StabilityReward, StabilityTargetReward, TargetGatedPressureTVReward,
    PressureStepTVCache,
    CompositeReward,
    MultiSectionReward, TimeAggCompositeReward, ConstantTargetReward, TimeAggMultiSectionReward,
    TimeDiffNormReward, PeriodMinimumReward, PeriodMinimumVariationReward,
    MultiSectionPeriodMinimumReward, TransitionBasedReward, DtWeightedReward, TimeAggregation,
    CachedCompositeReward, MultiAgentCachedCompositeReward,
    TimeMin, TimeMax, TimeAvg, TimeSum, TimeProd, TimeMin,
    MultiplicativeReward, StepCostReward, TargetMatchReward, DenseLockReward, DenseLockCache


include("utils.jl")
export sigmoid, reward_sigmoid, sigmoid_to_linear
end #module
