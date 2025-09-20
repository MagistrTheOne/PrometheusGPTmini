"""
PrometheusGPT Mini - Model Configuration
Author: MagistrTheOne, Krasnodar, 2025

Конфигурация параметров модели для ~8M параметров.
Архитектура: Transformer-based seq2seq модель.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Конфигурация архитектуры модели для ~8M параметров"""

    # Размер словаря (уменьшен для экономии параметров)
    vocab_size: int = 16000

    # Размерность эмбеддингов (уменьшена для экономии)
    d_model: int = 256

    # Количество слоев encoder/decoder (уменьшено)
    n_layers: int = 4

    # Количество голов внимания (уменьшено)
    n_heads: int = 4

    # Размерность feed-forward слоя (уменьшена)
    d_ff: int = 1024

    # Коэффициент dropout
    dropout: float = 0.1

    # Максимальная длина последовательности
    max_seq_length: int = 256

    # Размерность ключа/значения в attention
    d_k: Optional[int] = None
    d_v: Optional[int] = None

    # Инициализация весов
    init_method: str = "xavier"

    def __post_init__(self):
        """Автоматический расчет d_k и d_v"""
        if self.d_k is None:
            self.d_k = self.d_model // self.n_heads
        if self.d_v is None:
            self.d_v = self.d_model // self.n_heads

    @property
    def total_params(self) -> int:
        """Примерный расчет общего количества параметров"""
        # Эмбеддинги: vocab_size * d_model
        embed_params = self.vocab_size * self.d_model

        # Encoder слои
        encoder_params = self.n_layers * (
            # Multi-head attention
            self.d_model * self.d_k * self.n_heads * 3 +  # Q, K, V projections + output
            self.d_model * self.d_model  # Output projection
        )

        # Decoder слои (дополнительно cross-attention)
        decoder_params = self.n_layers * (
            # Self-attention
            self.d_model * self.d_k * self.n_heads * 3 +  # Q, K, V projections + output
            # Cross-attention
            self.d_model * self.d_k * self.n_heads * 3 +  # Q, K, V projections + output
            # Feed-forward
            self.d_model * self.d_ff * 2 +  # 2 линейных слоя
            self.d_model * self.d_model  # Output projection
        )

        # Position embeddings
        pos_embed_params = self.max_seq_length * self.d_model

        # Output layer (для генерации)
        output_params = self.d_model * self.vocab_size

        total = embed_params + encoder_params + decoder_params + pos_embed_params + output_params
        return total


@dataclass
class TrainingConfig:
    """Конфигурация параметров обучения"""

    # Batch size (8-16 для RTX 2080 Super)
    batch_size: int = 8

    # Learning rate
    learning_rate: float = 1e-4

    # Количество эпох
    num_epochs: int = 10

    # Warmup steps для learning rate
    warmup_steps: int = 1000

    # Weight decay для регуляризации
    weight_decay: float = 0.01

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Mixed precision training
    use_mixed_precision: bool = True

    # Gradient checkpointing для экономии памяти
    use_gradient_checkpointing: bool = True

    # Seed для воспроизводимости
    seed: int = 42


# Глобальные конфигурации
model_config = ModelConfig()
training_config = TrainingConfig()

if __name__ == "__main__":
    # Пример использования
    print("=== PrometheusGPT Mini Configuration ===")
    print(f"Author: MagistrTheOne, Krasnodar, 2025")
    print(f"Total parameters: ~{model_config.total_params / 1_000_000:.1f}M")
    print(f"Model config: {model_config}")
    print(f"Training config: {training_config}")
