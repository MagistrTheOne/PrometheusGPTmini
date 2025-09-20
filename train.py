#!/usr/bin/env python3
"""
Основной скрипт для обучения PrometheusGPT Mini
"""

import argparse
import logging
import os
from config import ModelConfig
from src.trainer import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train PrometheusGPT Mini")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training from")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to custom config file")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    parser.add_argument("--experiment-name", type=str, default="prometheus-gpt-mini",
                       help="Name for the experiment")

    args = parser.parse_args()

    # Загружаем конфигурацию
    config = ModelConfig()

    if args.config and os.path.exists(args.config):
        # Загружаем кастомную конфигурацию (опционально)
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_config", args.config)
        custom_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_config)
        # Обновляем конфигурацию
        for key, value in custom_config.__dict__.items():
            if not key.startswith('_') and hasattr(config, key):
                setattr(config, key, value)

    # Выводим конфигурацию
    config.print_config()

    # Создаем тренера
    trainer = Trainer(config, use_wandb=not args.no_wandb)

    # Запускаем обучение
    logger.info("Starting training...")
    trainer.train(resume_from=args.resume)

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
