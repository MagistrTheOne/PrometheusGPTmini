"""
Конфигурация для PrometheusGPT Mini
Гибридная мультиязычная LLM с 8 млн параметров
"""

import torch

class ModelConfig:
    """Конфигурация модели PrometheusGPT Mini"""

    # Архитектура
    vocab_size = 32000  # Размер словаря для мультиязычной токенизации
    d_model = 256       # Размер скрытого слоя
    num_layers = 6      # Количество трансформерных слоев
    num_heads = 4       # Количество голов внимания
    d_ff = 1024         # Размер feed-forward сети
    dropout = 0.1       # Dropout rate

    # HM-MoE (Hybrid Multilingual Mixture-of-Experts)
    num_experts = 4     # Количество экспертов (по языковым парам)
    expert_size = 512   # Размер каждого эксперта
    top_k_experts = 2   # Количество активируемых экспертов на токен

    # Позиционное кодирование
    max_seq_len = 512   # Максимальная длина последовательности

    # Обучение
    batch_size = 2      # Размер батча (ограничено VRAM)
    gradient_accumulation_steps = 4  # Накопление градиентов
    effective_batch_size = batch_size * gradient_accumulation_steps

    learning_rate = 5e-4
    warmup_steps = 1000
    max_steps = 50000   # Общее количество шагов обучения

    # Оптимизация для RTX 2080
    use_fp16 = True     # Смешанная точность
    use_gradient_checkpointing = True  # Градиентный чекпоинтинг

    # Данные
    languages = ['en', 'ru', 'es', 'fr', 'de']  # Поддерживаемые языки
    data_split = {'train': 0.8, 'val': 0.1, 'test': 0.1}

    # Пути
    model_dir = "models/"
    data_dir = "data/"
    checkpoint_dir = "checkpoints/"
    log_dir = "logs/"

    @classmethod
    def get_total_params(cls):
        """Расчет общего количества параметров (приблизительно)"""
        # Embedding: vocab_size * d_model
        embedding_params = cls.vocab_size * cls.d_model

        # Attention per layer: 4 * (d_model * d_model) for Q,K,V,O
        attention_params = 4 * (cls.d_model * cls.d_model) * cls.num_layers

        # FFN per layer: 2 * (d_model * d_ff)
        ffn_params = 2 * (cls.d_model * cls.d_ff) * cls.num_layers

        # HM-MoE: router + experts
        moe_params = cls.d_model * cls.num_experts  # router
        moe_params += cls.num_experts * (2 * cls.d_model * cls.expert_size)  # experts

        # Output: d_model * vocab_size
        output_params = cls.d_model * cls.vocab_size

        total = embedding_params + attention_params + ffn_params + moe_params + output_params
        return total

    @classmethod
    def print_config(cls):
        """Вывод конфигурации"""
        print("=== PrometheusGPT Mini Configuration ===")
        print(f"Total Parameters: {cls.get_total_params():,}")
        print(f"Architecture: {cls.num_layers} layers, d_model={cls.d_model}, heads={cls.num_heads}")
        print(f"HM-MoE: {cls.num_experts} experts, top-k={cls.top_k_experts}")
        print(f"Languages: {', '.join(cls.languages)}")
        print(f"Batch size: {cls.batch_size} (effective: {cls.effective_batch_size})")
        print(f"Max sequence length: {cls.max_seq_len}")
        print("=" * 40)
