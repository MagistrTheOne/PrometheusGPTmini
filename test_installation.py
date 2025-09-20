#!/usr/bin/env python3
"""
Тест установки PrometheusGPT Mini
Проверяет работоспособность всех компонентов
"""

import sys
import torch
import numpy as np
from config import ModelConfig
from src.model import PrometheusGPT, count_parameters
from src.tokenizer import MultilingualTokenizer
from src.data import MultilingualDataset
from src.evaluator import ModelEvaluator


def test_imports():
    """Тест импортов"""
    print("Testing imports...")
    try:
        import torch
        import transformers
        import datasets
        import tokenizers
        import sentencepiece
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config():
    """Тест конфигурации"""
    print("\nTesting configuration...")
    try:
        config = ModelConfig()
        config.print_config()
        print("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        return False


def test_model():
    """Тест модели"""
    print("\nTesting model architecture...")
    try:
        config = ModelConfig()
        model = PrometheusGPT(config)

        # Подсчет параметров
        total_params = count_parameters(model)
        print(f"✓ Model created with {total_params:,} parameters")

        # Тест forward pass
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            outputs = model(input_ids)
            print(f"✓ Forward pass successful, output shape: {outputs['logits'].shape}")

        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False


def test_tokenizer():
    """Тест токенизатора"""
    print("\nTesting tokenizer...")
    try:
        config = ModelConfig()
        tokenizer = MultilingualTokenizer(config)

        # Тест кодирования/декодирования
        test_text = "Hello world! Привет мир!"
        tokens = tokenizer.encode(test_text, 'en')
        decoded = tokenizer.decode(tokens)

        print(f"✓ Tokenizer created, vocab size: {tokenizer.get_vocab_size()}")
        print(f"✓ Encoding/decoding test: '{test_text}' -> {len(tokens)} tokens -> '{decoded}'")

        # Тест пакетной обработки
        batch = tokenizer.encode_batch([test_text], ['en'])
        print(f"✓ Batch processing test passed")

        return True
    except Exception as e:
        print(f"✗ Tokenizer test failed: {e}")
        return False


def test_data():
    """Тест подготовки данных"""
    print("\nTesting data preparation...")
    try:
        config = ModelConfig()
        tokenizer = MultilingualTokenizer(config)

        # Создаем небольшой датасет для теста
        dataset = MultilingualDataset(config, tokenizer, split="train")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✓ Dataset created with {len(dataset)} samples")
            print(f"✓ Sample keys: {list(sample.keys())}")
            print(f"✓ Sample input shape: {sample['input_ids'].shape}")
        else:
            print("⚠ Dataset is empty (expected for test environment)")

        return True
    except Exception as e:
        print(f"✗ Data test failed: {e}")
        return False


def test_evaluator():
    """Тест системы оценки"""
    print("\nTesting evaluator...")
    try:
        config = ModelConfig()
        model = PrometheusGPT(config)
        tokenizer = MultilingualTokenizer(config)

        evaluator = ModelEvaluator(config, model, tokenizer)
        print("✓ Evaluator created successfully")

        # Тест скорости инференса
        speed_results = evaluator.benchmark_inference_speed([1], [32])
        print(f"✓ Inference speed test: {list(speed_results.keys())[0]} = {list(speed_results.values())[0]:.1f}")

        return True
    except Exception as e:
        print(f"✗ Evaluator test failed: {e}")
        return False


def test_cuda():
    """Тест CUDA доступности"""
    print("\nTesting CUDA setup...")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3

        print(f"✓ CUDA available: {device_count} device(s)")
        print(f"✓ Current device: {current_device} ({device_name})")
        print(f"✓ GPU memory: {memory:.1f} GB")

        return True
    else:
        print("⚠ CUDA not available - training will be slow on CPU")
        return False


def main():
    """Основная функция тестирования"""
    print("=" * 60)
    print("PrometheusGPT Mini - Installation Test")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("CUDA Setup", test_cuda),
        ("Model Architecture", test_model),
        ("Tokenizer", test_tokenizer),
        ("Data Preparation", test_data),
        ("Evaluator", test_evaluator),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Итоги
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print("25")
        if result:
            passed += 1

    print(f"\nPassed: {passed}/{total} tests")

    if passed == total:
        print("🎉 All tests passed! PrometheusGPT Mini is ready for training.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("   You may still be able to run the model with limited functionality.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
