from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class TaskDescriptor:
    weather_year: int
    start_day: int
    parameter_uncertainty: float
    economic_scenario: str
    climate_constraint_scenario: str

    @property
    def stable_key(self) -> str:
        return (
            f"{self.weather_year}:{self.start_day}:"
            f"{self.parameter_uncertainty:.6f}:"
            f"{self.economic_scenario}:{self.climate_constraint_scenario}"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "TaskDescriptor":
        return cls(**value)


@dataclass(frozen=True, slots=True)
class TaskInstance:
    task: TaskDescriptor
    environment_index: int
    episode_index: int

    @property
    def stable_key(self) -> str:
        return (
            f"{self.task.stable_key}:env{self.environment_index}:"
            f"episode{self.episode_index}"
        )
