"""
PrometheusGPT Mini - Base Layers
Author: MagistrTheOne, Krasnodar, 2025

Базовые слои для Transformer архитектуры.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class PositionalEncoding(nn.Module):
    """Positional Encoding для учета порядка токенов"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # Создаем матрицу позиций [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # четные позиции
        pe[:, 1::2] = torch.cos(position * div_term)  # нечетные позиции

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Добавляем позиционные эмбеддинги к входу"""
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class FeedForward(nn.Module):
    """Feed-Forward сеть внутри Transformer блока"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed-forward: Linear -> ReLU -> Dropout -> Linear"""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class LayerNorm(nn.Module):
    """Layer Normalization с epsilon для стабильности"""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Layer Norm: (x - mean) / std * gamma + beta"""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class Embedding(nn.Module):
    """Эмбеддинг слой с поддержкой позиционных эмбеддингов"""

    def __init__(self, vocab_size: int, d_model: int, max_len: int = 5000):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Токен эмбеддинги + позиционные эмбеддинги + dropout"""
        seq_len = x.size(1)

        # Токен эмбеддинги
        token_emb = self.token_embedding(x)  # [batch, seq_len, d_model]

        # Добавляем позиционные эмбеддинги
        positional_emb = self.positional_encoding(token_emb)

        # Dropout для регуляризации
        return self.dropout(positional_emb)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention механизм"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # размерность каждой головы

        # Линейные проекции для Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: [batch, seq_len_q, d_model]
            key: [batch, seq_len_k, d_model]
            value: [batch, seq_len_v, d_model]
            mask: [batch, 1, seq_len_q, seq_len_k] или [batch, n_heads, seq_len_q, seq_len_k]
        """

        batch_size = query.size(0)

        # Линейные проекции
        Q = self.w_q(query)  # [batch, seq_len, d_model]
        K = self.w_k(key)
        V = self.w_v(value)

        # Разделяем на головы: [batch, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Attention: Q * K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Применяем маску (если есть)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax для получения attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Применяем attention к values
        attn_output = torch.matmul(attn_weights, V)  # [batch, n_heads, seq_len, d_k]

        # Собираем головы обратно
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, seq_len, n_heads, d_k]
        attn_output = attn_output.view(batch_size, -1, self.d_model)  # [batch, seq_len, d_model]

        # Финальная линейная проекция
        return self.w_o(attn_output)


class TransformerBlock(nn.Module):
    """Один блок Transformer (Multi-Head Attention + Feed-Forward)"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 use_gradient_checkpointing: bool = False, use_dynamic_attention: bool = False):
        super().__init__()

        # Multi-Head Attention
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)

        # Layer Normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        # Feed-Forward
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Dropout для residual connections
        self.dropout = nn.Dropout(dropout)

        # Опции для оптимизации
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_dynamic_attention = use_dynamic_attention

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Transformer block: Attention -> Add&Norm -> FeedForward -> Add&Norm"""

        # Gradient checkpointing для экономии памяти
        if self.use_gradient_checkpointing and self.training:
            # Используем torch.utils.checkpoint для forward pass
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward

            # Self-Attention с checkpointing
            attn_output = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.attention), x, x, x, mask
            )
            x = self.norm1(x + self.dropout(attn_output))

            # Feed-Forward с checkpointing
            ff_output = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.feed_forward), x
            )
            x = self.norm2(x + self.dropout(ff_output))
        else:
            # Стандартный forward pass
            # Self-Attention с residual connection
            attn_output = self.attention(x, x, x, mask)
            x = self.norm1(x + self.dropout(attn_output))

            # Feed-Forward с residual connection
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_output))

        return x
