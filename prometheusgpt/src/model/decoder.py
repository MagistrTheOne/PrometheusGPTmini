"""
PrometheusGPT Mini - Decoder
Author: MagistrTheOne, Krasnodar, 2025

Decoder часть seq2seq модели с cross-attention.
"""

import torch
import torch.nn as nn
from typing import Optional

from .layers import TransformerBlock, Embedding
from .config import model_config


class Decoder(nn.Module):
    """Transformer Decoder с cross-attention для seq2seq"""

    def __init__(self, config: model_config = None):
        super().__init__()

        if config is None:
            config = model_config

        # Эмбеддинг слой
        self.embedding = Embedding(config.vocab_size, config.d_model, config.max_seq_length)

        # Стек Transformer блоков с cross-attention
        self.layers = nn.ModuleList([
            DecoderBlock(
                config.d_model, config.n_heads, config.d_ff, config.dropout,
                config.use_gradient_checkpointing
            )
            for _ in range(config.n_layers)
        ])

        # Финальная Layer Norm
        self.norm = nn.LayerNorm(config.d_model)

        # Выходной слой для генерации токенов
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: целевые токены [batch_size, tgt_seq_len]
            encoder_output: выход энкодера [batch_size, src_seq_len, d_model]
            src_mask: маска для source последовательности
            tgt_mask: маска для target последовательности (causal)
        Returns:
            output: logits для следующего токена [batch_size, tgt_seq_len, vocab_size]
        """

        # Эмбеддинг + позиционные эмбеддинги
        x = self.embedding(x)  # [batch_size, tgt_seq_len, d_model]

        # Проходим через все decoder слои
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        # Финальная нормализация
        x = self.norm(x)  # [batch_size, tgt_seq_len, d_model]

        # Проекция на словарь для получения logits
        output = self.output_projection(x)  # [batch_size, tgt_seq_len, vocab_size]

        return output

    def get_attention_weights(self) -> tuple:
        """Получить attention weights из всех слоев"""
        self_attn_weights = []
        cross_attn_weights = []

        for layer in self.layers:
            if hasattr(layer, 'self_attention'):
                self_attn_weights.append(layer.self_attention.attn_weights)
            if hasattr(layer, 'cross_attention'):
                cross_attn_weights.append(layer.cross_attention.attn_weights)

        return self_attn_weights, cross_attn_weights


class DecoderBlock(nn.Module):
    """Decoder блок с self-attention и cross-attention"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 use_gradient_checkpointing: bool = False):
        super().__init__()

        # Self-Attention (masked для autoregressive)
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-Attention (attention к encoder output)
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        # Gradient checkpointing
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decoder block: Self-Attn -> Cross-Attn -> Feed-Forward
        """

        # Gradient checkpointing для экономии памяти
        if self.use_gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward

            # 1. Self-Attention с causal маской
            attn_output, _ = torch.utils.checkpoint.checkpoint(
                create_custom_forward(lambda x, enc, mask: self.self_attention(x, x, x, attn_mask=tgt_mask, key_padding_mask=None)),
                x, encoder_output, src_mask
            )
            x = self.norm1(x + self.dropout(attn_output))

            # 2. Cross-Attention к encoder output
            attn_output, _ = torch.utils.checkpoint.checkpoint(
                create_custom_forward(lambda x, enc, mask: self.cross_attention(x, encoder_output, encoder_output, key_padding_mask=src_mask)),
                x, encoder_output, src_mask
            )
            x = self.norm2(x + self.dropout(attn_output))

            # 3. Feed-Forward
            ff_output = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.feed_forward), x
            )
            x = self.norm3(x + self.dropout(ff_output))
        else:
            # Стандартный forward pass
            # 1. Self-Attention с causal маской
            attn_output, _ = self.self_attention(x, x, x, attn_mask=tgt_mask, key_padding_mask=None)
            x = self.norm1(x + self.dropout(attn_output))

            # 2. Cross-Attention к encoder output
            attn_output, _ = self.cross_attention(x, encoder_output, encoder_output,
                                                key_padding_mask=src_mask)
            x = self.norm2(x + self.dropout(attn_output))

            # 3. Feed-Forward
            ff_output = self.feed_forward(x)
            x = self.norm3(x + self.dropout(ff_output))

        return x
