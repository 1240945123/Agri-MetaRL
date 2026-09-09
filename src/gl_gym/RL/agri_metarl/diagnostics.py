from dataclasses import dataclass, field

import numpy as np


@dataclass
class MetaDiagnostics:
    contexts: list[np.ndarray] = field(default_factory=list)
    residuals: list[np.ndarray] = field(default_factory=list)
    meta_losses: list[float] = field(default_factory=list)
    query_count: int = 0
    transition_count: int = 0
    residual_alpha: float = 1.0
    mc_gae_abs_differences: list[float] = field(default_factory=list)
    target_clip_fractions: list[float] = field(default_factory=list)
    completed_episode_count: int = 0
    nonfinite_meta_batch_count: int = 0
    calibration_queue_size: int = 0
    last_summary: dict[str, float] = field(default_factory=dict)

    def record_group(self, context, residual) -> None:
        context_array = np.asarray(context, dtype=float)
        residual_array = np.asarray(residual, dtype=float)
        if np.isfinite(context_array).all() and np.isfinite(residual_array).all():
            self.contexts.append(context_array)
            self.residuals.append(residual_array)
            self.query_count += residual_array.size

    def summarize(self) -> dict[str, float]:
        contexts = np.stack(self.contexts) if self.contexts else np.zeros((1, 1))
        residuals = np.concatenate(self.residuals) if self.residuals else np.zeros(1)
        saturation_threshold = 0.95 * self.residual_alpha
        return {
            "train/meta_loss": float(np.mean(self.meta_losses)) if self.meta_losses else 0.0,
            "train/context_norm_mean": float(np.linalg.norm(contexts, axis=-1).mean()),
            "train/context_between_task_variance": float(contexts.var(axis=0).mean()),
            "train/residual_abs_mean": float(np.abs(residuals).mean()),
            "train/residual_saturation_rate": float(
                np.mean(np.abs(residuals) >= saturation_threshold)
            ),
            "train/query_correction_fraction": (
                float(self.query_count / self.transition_count)
                if self.transition_count
                else 0.0
            ),
            "train/calibration_queue_size": float(self.calibration_queue_size),
            "train/completed_episode_count": float(self.completed_episode_count),
            "train/mc_gae_abs_difference_mean": (
                float(np.mean(self.mc_gae_abs_differences))
                if self.mc_gae_abs_differences
                else 0.0
            ),
            "train/target_residual_clip_fraction": (
                float(np.mean(self.target_clip_fractions))
                if self.target_clip_fractions
                else 0.0
            ),
            "train/nonfinite_meta_batch_count": float(
                self.nonfinite_meta_batch_count
            ),
        }

    def reset(self) -> None:
        self.contexts.clear()
        self.residuals.clear()
        self.meta_losses.clear()
        self.query_count = 0
        self.transition_count = 0
        self.mc_gae_abs_differences.clear()
        self.target_clip_fractions.clear()
        self.completed_episode_count = 0
        self.nonfinite_meta_batch_count = 0
        self.calibration_queue_size = 0
