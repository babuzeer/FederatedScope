"""
GGEUR Text RNN/LSTM Classifier

This module provides lightweight RNN/LSTM classifiers for fixed-size feature
embeddings (e.g., BERT sentence embeddings) produced by a frozen feature
extractor. It is designed to be used with the GGEUR pipeline where the
trainable model should NOT be the pretrained extractor.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from federatedscope.register import register_model


class GGEURTextRNNClassifier(nn.Module):
    """
    A simple sequence classifier for embedding features.

    Input:
      - x: (B, D) or (B, T, D) float tensor
    Output:
      - logits: (B, C)
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_classes: int,
                 num_layers: int = 1,
                 dropout: float = 0.0,
                 rnn_type: str = "lstm"):
        super().__init__()

        rnn_type = str(rnn_type).lower()
        if rnn_type not in {"rnn", "lstm"}:
            raise ValueError(f"Unsupported rnn_type={rnn_type}, expected 'rnn' or 'lstm'")

        num_layers = int(num_layers) if int(num_layers) > 0 else 1
        dropout = float(dropout)
        rnn_dropout = dropout if num_layers > 1 else 0.0

        if rnn_type == "rnn":
            self.rnn = nn.RNN(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        else:
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)
        elif x.dim() != 3:
            raise ValueError(f"Expected input dim 2 or 3, got {x.dim()}")

        out, _ = self.rnn(x)
        last = out[:, -1, :]  # (B, H)
        last = self.dropout(last)
        return self.classifier(last)


def _build(model_config, input_shape, rnn_type: str):
    # Prefer explicit config; fallback to inferred input_shape if available
    input_dim = int(getattr(model_config, "in_channels", 0))
    if input_dim <= 0 and input_shape is not None:
        # input_shape could be (B, D) or (B, T, D)
        input_dim = int(input_shape[-1])

    hidden_dim = int(getattr(model_config, "hidden", 256))
    num_layers = int(getattr(model_config, "layer", 1))
    dropout = float(getattr(model_config, "dropout", 0.0))
    num_classes = int(getattr(model_config, "num_classes", getattr(model_config, "out_channels", 2)))

    return GGEURTextRNNClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout,
        rnn_type=rnn_type,
    )


def call_ggeur_text_rnn(model_config, input_shape):
    model_type = str(model_config.type).lower()
    if model_type == "ggeur_rnn":
        return _build(model_config, input_shape, rnn_type="rnn")
    if model_type == "ggeur_lstm":
        return _build(model_config, input_shape, rnn_type="lstm")
    return None


register_model("ggeur_rnn", call_ggeur_text_rnn)
register_model("ggeur_lstm", call_ggeur_text_rnn)

