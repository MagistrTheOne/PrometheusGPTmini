#!/usr/bin/env python3
"""
Скрипт для оценки PrometheusGPT Mini
"""

import argparse
import logging
import os
import torch
from config import ModelConfig
from src.model import PrometheusGPT
from src.tokenizer import MultilingualTokenizer
from src.evaluator import ModelEvaluator
from src.data import MultilingualDataset
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(config: ModelConfig, checkpoint_path: str) -> PrometheusGPT:
    """Загрузка модели из чекпоинта"""
    model = PrometheusGPT(config)

    if torch.cuda.is_available():
        model = model.cuda()
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Model loaded from {checkpoint_path}")
    logger.info(f"Checkpoint step: {checkpoint.get('step', 'unknown')}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate PrometheusGPT Mini")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                       help="Output file for results")
    parser.add_argument("--generate-samples", action="store_true",
                       help="Generate sample outputs")

    args = parser.parse_args()

    # Проверяем существование чекпоинта
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return

    # Загружаем конфигурацию
    config = ModelConfig()

    # Загружаем модель
    model = load_model(config, args.checkpoint)

    # Создаем токенизатор
    tokenizer = MultilingualTokenizer(config)

    # Создаем оценщика
    evaluator = ModelEvaluator(config, model, tokenizer)

    # Создаем тестовый датасет
    test_dataset = MultilingualDataset(config, tokenizer, split="test")
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Запускаем полную оценку
    logger.info("Running full evaluation...")
    results = evaluator.run_full_evaluation(test_loader)

    # Генерируем примеры если запрошено
    if args.generate_samples:
        logger.info("Generating sample outputs...")
        sample_prompts = {
            'en': "The future of AI is",
            'ru': "Будущее ИИ заключается",
            'es': "El futuro de la IA es",
            'fr': "L'avenir de l'IA est",
            'de': "Die Zukunft der KI ist"
        }

        print("\n" + "="*50)
        print("SAMPLE GENERATIONS")
        print("="*50)

        for lang, prompt in sample_prompts.items():
            input_tokens = tokenizer.encode(prompt, lang, add_special_tokens=True)
            input_tensor = torch.tensor([input_tokens])
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()

            generated = model.generate(input_tensor, max_new_tokens=30, temperature=0.8)
            generated_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)

            print(f"\n{lang.upper()} ({lang}):")
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated_text}")
            print("-" * 30)

    # Сохраняем результаты
    import json
    with open(args.output, 'w', encoding='utf-8') as f:
        # Конвертируем numpy/scalar значения в обычные float
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, (np.floating, np.integer)):
                serializable_results[key] = float(value)
            elif isinstance(value, torch.Tensor):
                serializable_results[key] = float(value.item())
            else:
                serializable_results[key] = value

        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {args.output}")

    # Выводим основные метрики
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Overall Perplexity: {results.get('perplexity', 'N/A'):.2f}")

    for lang in config.languages:
        bleu = results.get(f'{lang}_bleu')
        if bleu is not None:
            print(f"{lang.upper()} BLEU: {bleu:.4f}")

        ppl = results.get(f'{lang}_perplexity')
        if ppl is not None:
            print(f"{lang.upper()} Perplexity: {ppl:.2f}")

    # Инференс скорость
    speed_metrics = {k: v for k, v in results.items() if 'tokens_sec' in k}
    if speed_metrics:
        print(f"\nInference Speed (tokens/sec):")
        for key, value in speed_metrics.items():
            batch_size = key.split('_')[1]
            seq_len = key.split('_')[3]
            print(f"  Batch {batch_size}, Seq {seq_len}: {value:.1f}")

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
