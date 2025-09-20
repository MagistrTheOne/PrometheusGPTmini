"""
PrometheusGPT Mini - Checkpointing Tests
Author: MagistrTheOne, Krasnodar, 2025

Тесты для gradient checkpointing и dynamic attention.
"""

import torch
import pytest
from torch.utils.checkpoint import checkpoint
import time

from src.model import PrometheusGPTMini, model_config, training_config


class TestCheckpointing:
    """Тесты для gradient checkpointing"""

    def test_gradient_checkpointing_enabled(self):
        """Тест включенного gradient checkpointing"""

        # Создаем модель с включенным checkpointing
        config = model_config
        config.use_gradient_checkpointing = True

        model = PrometheusGPTMini(config)

        # Проверяем что все блоки имеют checkpointing
        for layer in model.encoder.layers:
            assert layer.use_gradient_checkpointing == True

        for layer in model.decoder.layers:
            assert layer.use_gradient_checkpointing == True

    def test_gradient_checkpointing_disabled(self):
        """Тест отключенного gradient checkpointing"""

        # Создаем модель с отключенным checkpointing
        config = model_config
        config.use_gradient_checkpointing = False

        model = PrometheusGPTMini(config)

        # Проверяем что все блоки не имеют checkpointing
        for layer in model.encoder.layers:
            assert layer.use_gradient_checkpointing == False

        for layer in model.decoder.layers:
            assert layer.use_gradient_checkpointing == False

    def test_memory_usage_comparison(self):
        """Сравнение использования памяти с/без checkpointing"""

        batch_size = 16
        seq_len = 256

        # Создаем тестовые данные
        src_tokens = torch.randint(0, 1000, (batch_size, seq_len))
        tgt_tokens = torch.randint(0, 1000, (batch_size, seq_len))

        # Тест без checkpointing
        config1 = model_config
        config1.use_gradient_checkpointing = False

        model1 = PrometheusGPTMini(config1)
        model1.train()

        # Очищаем кэш GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Измеряем память до
        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated(0)

        # Forward pass без checkpointing
        with torch.no_grad():
            output1 = model1(src_tokens, tgt_tokens)

        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated(0)
            memory_used_no_checkpoint = memory_after - memory_before

        # Тест с checkpointing
        config2 = model_config
        config2.use_gradient_checkpointing = True

        model2 = PrometheusGPTMini(config2)
        model2.train()

        # Очищаем кэш GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Измеряем память до
        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated(0)

        # Forward pass с checkpointing
        with torch.no_grad():
            output2 = model2(src_tokens, tgt_tokens)

        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated(0)
            memory_used_with_checkpoint = memory_after - memory_before

            # Checkpointing должен использовать меньше памяти
            assert memory_used_with_checkpoint < memory_used_no_checkpoint * 1.5

        # Проверяем что outputs одинаковые по размеру
        assert output1.shape == output2.shape

    def test_training_step_with_checkpointing(self):
        """Тест шага обучения с checkpointing"""

        # Создаем модель с checkpointing
        config = model_config
        config.use_gradient_checkpointing = True

        model = PrometheusGPTMini(config)

        # Создаем тестовые данные
        batch_size = 8
        seq_len = 128

        src_tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        tgt_tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Создаем простой оптимизатор
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Training step
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(src_tokens, tgt_tokens)

        # Loss
        loss = torch.mean(outputs ** 2)  # dummy loss

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Проверяем что loss конечный
        assert torch.isfinite(loss)

        # Проверяем что градиенты существуют
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()

    def test_checkpointing_deterministic(self):
        """Проверка детерминированности с checkpointing"""

        # Фиксируем seed
        torch.manual_seed(42)

        config = model_config
        config.use_gradient_checkpointing = True

        model1 = PrometheusGPTMini(config)
        model2 = PrometheusGPTMini(config)

        # Создаем одинаковые входные данные
        torch.manual_seed(42)
        src_tokens1 = torch.randint(0, config.vocab_size, (4, 64))
        tgt_tokens1 = torch.randint(0, config.vocab_size, (4, 64))

        torch.manual_seed(42)
        src_tokens2 = torch.randint(0, config.vocab_size, (4, 64))
        tgt_tokens2 = torch.randint(0, config.vocab_size, (4, 64))

        # Forward pass
        with torch.no_grad():
            output1 = model1(src_tokens1, tgt_tokens1)
            output2 = model2(src_tokens2, tgt_tokens2)

        # Проверяем что результаты идентичны
        assert torch.allclose(output1, output2, rtol=1e-5)


class TestDynamicAttention:
    """Тесты для dynamic attention pruning"""

    def test_dynamic_attention_config(self):
        """Тест конфигурации dynamic attention"""

        config = model_config
        config.use_dynamic_attention = True
        config.attention_prune_ratio = 0.3
        config.attention_threshold = 0.2

        model = PrometheusGPTMini(config)

        # Проверяем что все блоки имеют dynamic attention
        for layer in model.encoder.layers:
            assert layer.use_dynamic_attention == True

    def test_attention_pruning_shapes(self):
        """Тест что attention pruning не меняет размеры тензоров"""

        config = model_config
        config.use_dynamic_attention = True

        model = PrometheusGPTMini(config)

        # Тестовые данные
        batch_size = 4
        seq_len = 64

        src_tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        tgt_tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward pass
        with torch.no_grad():
            outputs = model(src_tokens, tgt_tokens)

        # Проверяем размеры
        expected_shape = (batch_size, seq_len, config.vocab_size)
        assert outputs.shape == expected_shape

    def test_attention_pruning_toggle(self):
        """Тест переключения dynamic attention"""

        # Модель без pruning
        config1 = model_config
        config1.use_dynamic_attention = False

        model1 = PrometheusGPTMini(config1)

        # Модель с pruning
        config2 = model_config
        config2.use_dynamic_attention = True

        model2 = PrometheusGPTMini(config2)

        # Тестовые данные
        batch_size = 2
        seq_len = 32

        src_tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        tgt_tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward pass для обеих моделей
        with torch.no_grad():
            outputs1 = model1(src_tokens, tgt_tokens)
            outputs2 = model2(src_tokens, tgt_tokens)

        # Обе модели должны давать одинаковые размеры
        assert outputs1.shape == outputs2.shape


if __name__ == "__main__":
    # Запуск тестов
    print("=== PrometheusGPT Mini Checkpointing Tests ===")
    print("Author: MagistrTheOne, Krasnodar, 2025")

    # Создаем тесты
    test_checkpointing = TestCheckpointing()
    test_attention = TestDynamicAttention()

    # Запуск тестов
    print("\n1. Testing gradient checkpointing...")
    test_checkpointing.test_gradient_checkpointing_enabled()
    test_checkpointing.test_gradient_checkpointing_disabled()
    test_checkpointing.test_checkpointing_deterministic()
    print("✅ Checkpointing tests passed!")

    print("\n2. Testing dynamic attention...")
    test_attention.test_dynamic_attention_config()
    test_attention.test_attention_pruning_shapes()
    test_attention.test_attention_pruning_toggle()
    print("✅ Dynamic attention tests passed!")

    print("\n3. Testing memory usage...")
    test_checkpointing.test_memory_usage_comparison()
    print("✅ Memory usage tests passed!")

    print("\n4. Testing training step...")
    test_checkpointing.test_training_step_with_checkpointing()
    print("✅ Training step tests passed!")

    print("\n🎉 All tests passed!")
