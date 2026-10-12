"""INSIGHT: spatially resolved survival models for whole-slide image cohorts.

    from insight import Cohort, Pipeline

    cohort = Cohort.from_folder("/data/wsi")
    preds = Pipeline(output="runs", experiment="my_cohort", run=1).run(cohort)
"""
__version__ = "0.1.0"

from .cohort import Cohort, Slide  # noqa: E402
from .config import PipelineConfig  # noqa: E402
from .pipeline import STAGES, ConfigMismatch, Pipeline  # noqa: E402
from .registry import register_features, register_model  # noqa: E402

__all__ = ["Cohort", "Slide", "Pipeline", "PipelineConfig", "STAGES", "ConfigMismatch", "register_model",
           "register_features", "__version__"]
