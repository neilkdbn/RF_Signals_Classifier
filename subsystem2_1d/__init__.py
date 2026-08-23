# subsystem2_1d/__init__.py
# =========================================================
# Subsystem 2 -- 1D Time-Wave Preprocessing & Model
# Person 2 | feature/1d-model-p2
# =========================================================
# Public API surface -- consumers can import cleanly as:
#   from subsystem2_1d import normalize_1d_iq, IQAugmentation
#   from subsystem2_1d import get_1d_dataloaders, RFSignalDataset1D
#   from subsystem2_1d import CNN1DClassifier, CNNTransformerHybrid
#   from subsystem2_1d import build_model

from subsystem2_1d.normalization_1d import normalize_1d_iq, IQAugmentation
from subsystem2_1d.dataloader_1d    import get_1d_dataloaders, RFSignalDataset1D
from subsystem2_1d.classifiers_1d  import (
    CNN1DClassifier,
    CNNTransformerHybrid,
    PositionalEncoding,
    build_model,
)

__all__ = [
    # Preprocessing & normalization
    "normalize_1d_iq",
    "IQAugmentation",
    # DataLoader pipeline
    "get_1d_dataloaders",
    "RFSignalDataset1D",
    # Model architectures
    "CNN1DClassifier",
    "CNNTransformerHybrid",
    "PositionalEncoding",
    "build_model",
]
