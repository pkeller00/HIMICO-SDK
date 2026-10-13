"""Built-in prediction models (importing registers them)."""
from .base import Prediction, Predictor, VirtualStainer
from .histochrome import HistoChrome
from .insight_crc import InsightCRC

__all__ = ["Prediction", "Predictor", "VirtualStainer", "InsightCRC", "HistoChrome"]
