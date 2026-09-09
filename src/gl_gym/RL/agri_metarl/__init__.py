"""Agri-MetaRL public API with lazy training-stack imports."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gl_gym.RL.agri_metarl.agri_metarl import AgriMetaRL
    from gl_gym.RL.agri_metarl.calibration import (
        CalibrationSample,
        CompletedCalibrationEpisode,
        EpisodeCalibrationMemory,
    )
    from gl_gym.RL.agri_metarl.legacy_agri_metarl import LegacyAgriMetaRL
    from gl_gym.RL.agri_metarl.meta_advantage_head import (
        AdvantageResidualHead,
        MetaAdvantageHead,
        TransitionSetEncoder,
    )

__all__ = [
    "AdvantageResidualHead",
    "AgriMetaRL",
    "CalibrationSample",
    "CompletedCalibrationEpisode",
    "EpisodeCalibrationMemory",
    "LegacyAgriMetaRL",
    "MetaAdvantageHead",
    "TransitionSetEncoder",
]


def __getattr__(name: str):
    if name == "AgriMetaRL":
        from gl_gym.RL.agri_metarl.agri_metarl import AgriMetaRL

        return AgriMetaRL
    if name == "LegacyAgriMetaRL":
        from gl_gym.RL.agri_metarl.legacy_agri_metarl import LegacyAgriMetaRL

        return LegacyAgriMetaRL
    if name in {
        "CalibrationSample",
        "CompletedCalibrationEpisode",
        "EpisodeCalibrationMemory",
    }:
        from gl_gym.RL.agri_metarl import calibration

        return getattr(calibration, name)
    if name in {
        "AdvantageResidualHead",
        "MetaAdvantageHead",
        "TransitionSetEncoder",
    }:
        from gl_gym.RL.agri_metarl import meta_advantage_head

        return getattr(meta_advantage_head, name)
    raise AttributeError(name)
