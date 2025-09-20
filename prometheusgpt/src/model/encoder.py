"""
PrometheusGPT Mini - Encoder
Author: MagistrTheOne, Krasnodar, 2025

Encoder часть seq2seq модели на базе Transformer.
"""

import torch
import torch.nn as nn
from typing import Optional

from .layers import TransformerBlock, Embedding
from .config import model_config


class Encoder(nn.Module):
    """Transformer Encoder для обработки входной последовательности"""

    def __init__(self, config: model_config = None):
        super().__init__()

        if config is None:
            config = model_config

        # Эмбеддинг слой
        self.embedding = Embedding(config.vocab_size, config.d_model, config.max_seq_length)

        # Стек Transformer блоков
        self.layers = nn.ModuleList([
            TransformerBlock(
                config.d_model, config.n_heads, config.d_ff, config.dropout,
                config.use_gradient_checkpointing, config.use_dynamic_attention
            )
            for _ in range(config.n_layers)
        ])

        # Финальная Layer Norm
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: входные токены [batch_size, seq_len]
            mask: маска для padding [batch_size, 1, 1, seq_len]
        Returns:
            encoder_output: [batch_size, seq_len, d_model]
        """

        # Эмбеддинг + позиционные эмбеддинги
        x = self.embedding(x)  # [batch_size, seq_len, d_model]

        # Проходим через все encoder слои
        for layer in self.layers:
            x = layer(x, mask)  # [batch_size, seq_len, d_model]

        # Финальная нормализация
        x = self.norm(x)  # [batch_size, seq_len, d_model]

        return x

    def get_attention_weights(self) -> list:
        """Получить attention weights из всех слоев для анализа"""
        return [layer.attention.attn_weights for layer in self.layers]
