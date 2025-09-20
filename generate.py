#!/usr/bin/env python3
"""
Скрипт для генерации текста с помощью PrometheusGPT Mini
"""

import argparse
import logging
import torch
import os
from config import ModelConfig
from src.model import PrometheusGPT
from src.tokenizer import MultilingualTokenizer

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
    return model


def generate_text(model: PrometheusGPT, tokenizer: MultilingualTokenizer,
                 prompt: str, lang: str, max_new_tokens: int = 50,
                 temperature: float = 0.8, top_k: int = 40) -> str:
    """
    Генерация текста на основе промпта

    Args:
        model: Загруженная модель
        tokenizer: Токенизатор
        prompt: Начальный текст
        lang: Код языка
        max_new_tokens: Максимальное количество новых токенов
        temperature: Температура для sampling
        top_k: Top-k для ограничения словаря

    Returns:
        Сгенерированный текст
    """
    # Токенизируем промпт
    input_tokens = tokenizer.encode(prompt, lang, add_special_tokens=True)
    input_tensor = torch.tensor([input_tokens])

    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    # Генерируем
    with torch.no_grad():
        generated = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )

    # Декодируем результат
    generated_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)

    return generated_text


def interactive_generation(model: PrometheusGPT, tokenizer: MultilingualTokenizer,
                          config: ModelConfig):
    """Интерактивная генерация текста"""
    print("\n" + "="*60)
    print("PrometheusGPT Mini - Interactive Text Generation")
    print("="*60)
    print("Available languages:", ', '.join(config.languages))
    print("Commands:")
    print("  'quit' - exit")
    print("  'lang <code>' - change language")
    print("  or just type your prompt")
    print()

    current_lang = 'en'

    while True:
        try:
            user_input = input(f"[{current_lang}] Prompt: ").strip()

            if user_input.lower() == 'quit':
                break

            if user_input.startswith('lang '):
                new_lang = user_input.split()[1].lower()
                if new_lang in config.languages:
                    current_lang = new_lang
                    print(f"Language changed to {current_lang}")
                else:
                    print(f"Unknown language. Available: {', '.join(config.languages)}")
                continue

            if not user_input:
                continue

            # Генерируем текст
            print("Generating...")
            generated_text = generate_text(
                model, tokenizer, user_input, current_lang,
                max_new_tokens=50, temperature=0.8
            )

            print("\nGenerated text:")
            print("-" * 40)
            print(generated_text)
            print("-" * 40)
            print()

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="Generate text with PrometheusGPT Mini")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Text prompt for generation")
    parser.add_argument("--lang", type=str, default="en",
                       help="Language code (en, ru, es, fr, de)")
    parser.add_argument("--max-tokens", type=int, default=50,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature (0.1-1.0)")
    parser.add_argument("--top-k", type=int, default=40,
                       help="Top-k for sampling")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")

    args = parser.parse_args()

    # Проверяем существование чекпоинта
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return

    # Проверяем язык
    config = ModelConfig()
    if args.lang not in config.languages:
        logger.error(f"Unsupported language: {args.lang}")
        logger.error(f"Available languages: {', '.join(config.languages)}")
        return

    # Загружаем модель и токенизатор
    model = load_model(config, args.checkpoint)
    tokenizer = MultilingualTokenizer(config)

    if args.interactive:
        # Интерактивный режим
        interactive_generation(model, tokenizer, config)
    else:
        # Одноразовая генерация
        if not args.prompt:
            logger.error("Please provide a prompt with --prompt or use --interactive")
            return

        print(f"Generating text for prompt: '{args.prompt}' (lang: {args.lang})")
        print("Parameters: max_tokens={}, temperature={}, top_k={}".format(
            args.max_tokens, args.temperature, args.top_k))

        generated_text = generate_text(
            model, tokenizer, args.prompt, args.lang,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )

        print("\nGenerated text:")
        print("=" * 50)
        print(generated_text)
        print("=" * 50)


if __name__ == "__main__":
    main()
