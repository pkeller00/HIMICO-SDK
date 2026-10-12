"""Built-in prediction models (importing registers them)."""
from .base import Prediction, Predictor
from .insight_crc import InsightCRC

__all__ = ["Prediction", "Predictor", "InsightCRC"]
