# subsystem2_1d/__init__.py
# =========================================================
# Subsystem 2 -- 1D Time-Wave Preprocessing, Models & Training
# Person 2 | feature/1d-model-p2
# =========================================================
# Public API -- consumers can import cleanly as:
#   from subsystem2_1d import normalize_1d_iq, IQAugmentation
#   from subsystem2_1d import get_1d_dataloaders, RFSignalDataset1D
#   from subsystem2_1d import CNN1DClassifier, CNNTransformerHybrid
#   from subsystem2_1d import build_model
#   from subsystem2_1d import EarlyStopping, TrainingConfig, train_model
#   from subsystem2_1d import evaluate_model, print_edge_suitability_report

from subsystem2_1d.normalization_1d import normalize_1d_iq, IQAugmentation
from subsystem2_1d.dataloader_1d    import get_1d_dataloaders, RFSignalDataset1D
from subsystem2_1d.classifiers_1d  import (
    CNN1DClassifier,
    CNNTransformerHybrid,
    PositionalEncoding,
    build_model,
)
from subsystem2_1d.train_1d import (
    EarlyStopping,
    TrainingConfig,
    train_model,
)
from subsystem2_1d.eval_1d import (
    evaluate_model,
    print_edge_suitability_report,
    load_model_from_checkpoint,
    measure_inference_latency,
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
    # Training pipeline
    "EarlyStopping",
    "TrainingConfig",
    "train_model",
    # Evaluation pipeline
    "evaluate_model",
    "print_edge_suitability_report",
    "load_model_from_checkpoint",
    "measure_inference_latency",
]

