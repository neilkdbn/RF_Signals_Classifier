# subsystem2_1d/classifiers_1d.py
# =========================================================
# Subsystem 2 -- 1D Time-Wave Model Architectures
# Person 2 | feature/1d-model-p2
#
# Shape contract (must be respected by ALL models here):
#   Input  : (Batch_Size, 2, 128)   -- raw IQ time-domain signal
#   Output : (Batch_Size, num_classes)  -- unnormalized logit scores
#
# Two architectures are provided:
#   1. CNN1DClassifier       -- deep 4-block 1D CNN + FC head
#   2. CNNTransformerHybrid  -- CNN frontend + Transformer encoder
# =========================================================

import math
import torch
import torch.nn as nn
from typing import Optional


# ---------------------------------------------------------
# Shared constants
# ---------------------------------------------------------
_INPUT_CHANNELS   = 2    # I and Q channels
_INPUT_TIME_STEPS = 128  # samples per frame
_DEFAULT_CLASSES  = 11   # RadioML 2016.10a modulation classes


# =========================================================
# Helper: Convolutional Block (Conv1d -> BN -> ReLU -> Pool)
# =========================================================

class _ConvBNReLUPool(nn.Module):
    """
    Reusable building block:
        Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d

    padding='same' keeps the time dimension constant after conv;
    MaxPool1d(2) then halves it.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        pool_size:    int = 2,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                padding="same",   # preserves sequence length
                bias=False,       # bias redundant before BN
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# =========================================================
# Architecture A: Standard 1D-CNN Classifier
# =========================================================

class CNN1DClassifier(nn.Module):
    """
    Deep 1D Convolutional Neural Network for RF modulation classification.

    Processes raw IQ time-series signals through four progressive convolutional
    blocks that learn local temporal patterns, followed by a fully-connected
    classifier head.

    Architecture:
        Input   (B, 2, 128)
        Block 1  Conv1d(2->64,  k=7, same) -> BN -> ReLU -> MaxPool2  => (B, 64,  64)
        Block 2  Conv1d(64->128,k=5, same) -> BN -> ReLU -> MaxPool2  => (B, 128, 32)
        Block 3  Conv1d(128->256,k=3,same) -> BN -> ReLU -> MaxPool2  => (B, 256, 16)
        Block 4  Conv1d(256->256,k=3,same) -> BN -> ReLU -> MaxPool2  => (B, 256,  8)
        Flatten                                                         => (B, 2048)
        Linear(2048->256) -> BN -> ReLU -> Dropout(0.5)               => (B, 256)
        Linear(256->num_classes)                                        => (B, C)

    Args:
        num_classes (int): Number of output classification logits.
                           Default: 11 (RadioML 2016.10a).

    Example:
        >>> model = CNN1DClassifier(num_classes=11)
        >>> x = torch.randn(32, 2, 128)
        >>> logits = model(x)           # (32, 11)
        >>> model.get_param_count()     # prints & returns total trainable params
    """

    def __init__(self, num_classes: int = _DEFAULT_CLASSES):
        super().__init__()
        self.num_classes = num_classes

        # --------------------------------------------------
        # 4-Block Convolutional Feature Extractor
        # --------------------------------------------------
        self.conv_blocks = nn.Sequential(
            _ConvBNReLUPool(  2,  64, kernel_size=7),   # (B, 64,  64)
            _ConvBNReLUPool( 64, 128, kernel_size=5),   # (B, 128, 32)
            _ConvBNReLUPool(128, 256, kernel_size=3),   # (B, 256, 16)
            _ConvBNReLUPool(256, 256, kernel_size=3),   # (B, 256,  8)
        )

        # Flat size after 4 halvings: 256 channels * 8 time steps = 2048
        _flat_size = 256 * (_INPUT_TIME_STEPS // (2 ** 4))  # = 2048

        # --------------------------------------------------
        # Fully Connected Classifier Head
        # --------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(_flat_size, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Raw IQ batch, shape (B, 2, 128).

        Returns:
            torch.Tensor: Logit scores, shape (B, num_classes).
        """
        features = self.conv_blocks(x)      # (B, 256, 8)
        logits   = self.classifier(features) # (B, num_classes)
        return logits

    # ----------------------------------------------------------
    # Utility: Parameter count
    # ----------------------------------------------------------

    def get_param_count(self) -> int:
        """
        Returns the total number of trainable parameters in this model.

        Prints a formatted summary to stdout and also returns the integer
        count for programmatic use.

        Returns:
            int: Total trainable parameter count.
        """
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[CNN1DClassifier]      Trainable parameters: {total:,}")
        return total


# =========================================================
# Architecture B — Helper: Sinusoidal Positional Encoding
# =========================================================

class PositionalEncoding(nn.Module):
    """
    Injects fixed sinusoidal positional embeddings into a sequence tensor.

    Adds position-aware information to each token so the Transformer
    can reason about the temporal ordering of feature vectors.

    Encoding formula (Vaswani et al., 2017):
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Expects batch-first input: (B, seq_len, d_model).

    Args:
        d_model  (int):   Embedding dimension (must match model d_model).
        max_len  (int):   Maximum sequence length supported. Default: 512.
        dropout  (float): Dropout applied after adding positional encoding.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int  = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build the fixed PE table: shape (max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()          # (L, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )                                                                # (d_model/2,)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims -> sin
        pe[:, 1::2] = torch.cos(position * div_term)  # odd  dims -> cos

        # Register as non-trainable buffer (saved with model state_dict)
        # Shape: (1, max_len, d_model) for broadcasting over batch
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Token sequence, shape (B, seq_len, d_model).

        Returns:
            torch.Tensor: Position-encoded sequence, same shape.
        """
        x = x + self.pe[:, : x.size(1), :]   # broadcast over batch
        return self.dropout(x)


# =========================================================
# Architecture B: 1D CNN-Transformer Hybrid Classifier
# =========================================================

class CNNTransformerHybrid(nn.Module):
    """
    Hybrid 1D CNN + Transformer architecture for RF modulation classification.

    Combines the local pattern extraction strength of 1D convolutions with
    the global context modeling capacity of multi-head self-attention.

    Architecture:
        Input        (B, 2, 128)
        -- CNN Frontend (local feature extraction) --
        Conv1d(2->64,  k=7, stride=2, pad=3) -> ReLU -> BN  => (B, 64,  64)
        Conv1d(64->128,k=5, stride=2, pad=2) -> ReLU -> BN  => (B, 128, 32)
        -- Temporal Alignment --
        Transpose(1,2)                                        => (B, 32, 128)
        PositionalEncoding(d_model=128)
        -- Transformer Encoder (global context) --
        TransformerEncoder(
            num_layers=2, nhead=8,
            dim_feedforward=256, dropout=0.1
        )                                                     => (B, 32, 128)
        -- Aggregation --
        Global Average Pooling over seq dim                   => (B, 128)
        -- Classifier Head --
        Linear(128 -> num_classes)                            => (B, C)

    Design notes:
        - d_model = 128; nhead = 8; dim_per_head = 16.
        - batch_first=True used throughout to avoid costly transposes.
        - Global avg pooling replaces [CLS] token to cut parameter count.

    Args:
        num_classes (int): Number of output classification logits.
                           Default: 11 (RadioML 2016.10a).
        transformer_layers (int): Number of Transformer encoder layers. Default: 2.
        nhead              (int): Number of self-attention heads.        Default: 8.
        dim_feedforward    (int): FF sublayer hidden size.               Default: 256.
        attn_dropout       (float): Dropout inside Transformer.          Default: 0.1.

    Example:
        >>> model = CNNTransformerHybrid(num_classes=11)
        >>> x = torch.randn(32, 2, 128)
        >>> logits = model(x)            # (32, 11)
        >>> model.get_param_count()
    """

    _D_MODEL = 128   # CNN output channels == Transformer embedding dim

    def __init__(
        self,
        num_classes:        int   = _DEFAULT_CLASSES,
        transformer_layers: int   = 2,
        nhead:              int   = 8,
        dim_feedforward:    int   = 256,
        attn_dropout:       float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        d_model = self._D_MODEL  # 128

        # --------------------------------------------------
        # CNN Frontend -- strided convolutions for downsampling
        # No MaxPool: stride controls spatial reduction
        # --------------------------------------------------
        self.cnn_frontend = nn.Sequential(
            # (B, 2, 128) -> (B, 64, 64)
            nn.Conv1d(
                _INPUT_CHANNELS, 64,
                kernel_size=7, stride=2, padding=3,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),

            # (B, 64, 64) -> (B, 128, 32)
            nn.Conv1d(
                64, d_model,
                kernel_size=5, stride=2, padding=2,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(d_model),
        )

        # --------------------------------------------------
        # Sinusoidal Positional Encoding
        # max_len >= 32 (our sequence length after CNN frontend)
        # --------------------------------------------------
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=512,
            dropout=attn_dropout,
        )

        # --------------------------------------------------
        # Transformer Encoder
        # batch_first=True: input/output shape (B, seq, d_model)
        # --------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=attn_dropout,
            activation="relu",
            batch_first=True,    # (B, T, C) convention
            norm_first=False,    # post-norm (original Transformer)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,  # avoids padding-related warnings
        )

        # --------------------------------------------------
        # Classifier Head
        # Global avg pool collapses the sequence dimension
        # --------------------------------------------------
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Raw IQ batch, shape (B, 2, 128).

        Returns:
            torch.Tensor: Logit scores, shape (B, num_classes).
        """
        # 1. Local feature extraction via strided CNN
        features = self.cnn_frontend(x)      # (B, 128, 32)

        # 2. Transpose to (B, T, C) for Transformer  [T=32, C=128]
        tokens = features.transpose(1, 2)    # (B, 32, 128)

        # 3. Inject sinusoidal positional information
        tokens = self.pos_encoder(tokens)    # (B, 32, 128)

        # 4. Global context modeling via multi-head self-attention
        encoded = self.transformer_encoder(tokens)  # (B, 32, 128)

        # 5. Global average pooling over the token/time dimension
        context = encoded.mean(dim=1)        # (B, 128)

        # 6. Linear projection to class logits
        logits = self.classifier(context)    # (B, num_classes)
        return logits

    # ----------------------------------------------------------
    # Utility: Parameter count
    # ----------------------------------------------------------

    def get_param_count(self) -> int:
        """
        Returns the total number of trainable parameters in this model.

        Prints a formatted summary to stdout and also returns the integer
        count for programmatic use.

        Returns:
            int: Total trainable parameter count.
        """
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[CNNTransformerHybrid] Trainable parameters: {total:,}")
        return total


# =========================================================
# Factory Helper — returns a model by name string
# =========================================================

_REGISTRY = {
    "cnn1d":          CNN1DClassifier,
    "cnn_transformer": CNNTransformerHybrid,
}


def build_model(
    name:        str,
    num_classes: int = _DEFAULT_CLASSES,
    **kwargs,
) -> nn.Module:
    """
    Instantiate a registered 1D classifier by name.

    Args:
        name (str):        Model key. One of: 'cnn1d', 'cnn_transformer'.
        num_classes (int): Number of output classes.
        **kwargs:          Passed through to the model constructor.

    Returns:
        nn.Module: Instantiated, untrained model.

    Raises:
        KeyError: If `name` is not in the model registry.

    Example:
        >>> model = build_model("cnn1d", num_classes=11)
        >>> model = build_model("cnn_transformer", num_classes=11, nhead=8)
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. "
            f"Available models: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](num_classes=num_classes, **kwargs)
