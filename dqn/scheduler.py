from abc import ABC, abstractmethod

from pydantic import BaseModel, PositiveFloat, PositiveInt


class SchedulerConfig(BaseModel):
    start: PositiveFloat = 1.0
    end: PositiveFloat = 0.1
    decay_period: PositiveInt = 10e6


class EpsilonScheduler(ABC):
    @abstractmethod
    def compute_epsilon(self, *args, **kwargs):
        pass


class ConstantScheduler(EpsilonScheduler):

    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def compute_epsilon(self, *args, **kwargs):
        return self.epsilon


class LinearEpsilonScheduler(EpsilonScheduler):
    def __init__(self, config: SchedulerConfig = SchedulerConfig()):
        self.start = config.start
        self.end = config.end
        self.decay_period = config.decay_period

    def compute_epsilon(self, count: int, *args, **kwargs) -> float:
        if count > self.decay_period:
            return self.end

        return self.start - (self.start - self.end) * count / self.decay_period
