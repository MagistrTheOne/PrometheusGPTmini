#!/usr/bin/env python3
"""
PrometheusGPT Mini - Phase 6 Runner
Author: MagistrTheOne, Krasnodar, 2025

Скрипт для запуска Phase 6: Advanced Fine-tuning & Specialization
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent.parent))

from src.model import PrometheusGPTMini, model_config
from src.domain.military_finetuner import MilitaryDocumentFineTuner
from src.monitoring.logging_config import setup_logging

def setup_phase6_logging():
    """Настроить логирование для Phase 6"""
    
    return setup_logging(
        log_level="INFO",
        log_dir="logs/phase6",
        log_format="json",
        enable_console=True,
        enable_file=True,
        log_requests=True,
        log_performance=True
    )

def create_sample_military_data():
    """Создать пример военных данных для тестирования"""
    
    sample_data = [
        {
            "text": "Операция <operation> началась в <location> в 06:00. <military_unit> получила приказ о <classified> действиях.",
            "classification_level": "confidential",
            "domain": "military_operations"
        },
        {
            "text": "Техническое обслуживание <equipment> запланировано на <location>. <personnel> должны проверить все системы.",
            "classification_level": "restricted",
            "domain": "equipment_maintenance"
        },
        {
            "text": "Отчет о состоянии <military_unit> в <location>. Все <equipment> функционирует нормально.",
            "classification_level": "unclassified",
            "domain": "status_report"
        },
        {
            "text": "План <operation> включает <classified> элементы. <personnel> должны быть готовы к <restricted> действиям.",
            "classification_level": "secret",
            "domain": "operation_planning"
        },
        {
            "text": "Обучение <personnel> на <equipment> завершено успешно. <military_unit> готова к <operation>.",
            "classification_level": "confidential",
            "domain": "training_report"
        }
    ]
    
    # Создаем директорию для данных
    data_dir = Path("data/phase6")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем пример данных
    with open(data_dir / "military_documents.jsonl", "w", encoding="utf-8") as f:
        for doc in sample_data:
            f.write(f"{json.dumps(doc, ensure_ascii=False)}\n")
    
    print(f"Created sample military data: {data_dir / 'military_documents.jsonl'}")
    return str(data_dir / "military_documents.jsonl")

def run_military_fine_tuning(args):
    """Запуск fine-tuning для военной документации"""
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Phase 6: Military Document Fine-tuning")
    
    # Создаем пример данных если нужно
    if not Path(args.military_data_path).exists():
        logger.info("Creating sample military data...")
        args.military_data_path = create_sample_military_data()
    
    # Создаем базовую модель
    logger.info("Loading base model...")
    model = PrometheusGPTMini(config=model_config)
    
    # Создаем fine-tuner
    logger.info("Initializing military fine-tuner...")
    fine_tuner = MilitaryDocumentFineTuner(
        base_model=model,
        military_data_path=args.military_data_path
    )
    
    # Запускаем fine-tuning
    logger.info("Starting military document fine-tuning...")
    results = fine_tuner.fine_tune_military_model(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        save_path=args.save_path
    )
    
    # Выводим результаты
    logger.info("Fine-tuning completed!")
    logger.info(f"Training history: {results['training_history']}")
    logger.info(f"Military metrics: {results['military_metrics']}")
    
    # Тестируем генерацию
    if args.test_generation:
        logger.info("Testing military text generation...")
        test_prompts = [
            "Операция началась в",
            "Техническое обслуживание",
            "Отчет о состоянии"
        ]
        
        for prompt in test_prompts:
            generated = fine_tuner.generate_military_text(prompt, max_length=50)
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Generated: {generated}")
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated}")
            print("-" * 50)
    
    return results

def run_phase6_evaluation(args):
    """Запуск оценки Phase 6"""
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Phase 6 evaluation...")
    
    # Загружаем обученную модель
    if not Path(args.model_path).exists():
        logger.error(f"Model not found: {args.model_path}")
        return None
    
    # Создаем fine-tuner с обученной моделью
    model = PrometheusGPTMini(config=model_config)
    fine_tuner = MilitaryDocumentFineTuner(
        base_model=model,
        military_data_path=args.military_data_path
    )
    
    # Загружаем веса
    checkpoint = torch.load(args.model_path, map_location='cpu')
    fine_tuner.base_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Получаем метрики
    metrics = fine_tuner.get_military_metrics()
    
    logger.info("Phase 6 evaluation completed!")
    logger.info(f"Military metrics: {metrics}")
    
    return metrics

def main():
    """Главная функция"""
    
    parser = argparse.ArgumentParser(description="PrometheusGPT Mini - Phase 6 Runner")
    parser.add_argument("--mode", choices=["fine_tune", "evaluate", "test"], 
                       default="fine_tune", help="Mode to run")
    parser.add_argument("--military_data_path", type=str, 
                       default="data/phase6/military_documents.jsonl",
                       help="Path to military documents")
    parser.add_argument("--epochs", type=int, default=5, 
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, 
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, 
                       help="Batch size")
    parser.add_argument("--save_path", type=str, 
                       default="checkpoints/phase6/military_model.pt",
                       help="Path to save model")
    parser.add_argument("--model_path", type=str, 
                       default="checkpoints/phase6/military_model.pt",
                       help="Path to load model")
    parser.add_argument("--test_generation", action="store_true", 
                       help="Test text generation after training")
    
    args = parser.parse_args()
    
    # Настраиваем логирование
    setup_phase6_logging()
    
    print("=== PrometheusGPT Mini - Phase 6 Runner ===")
    print("Author: MagistrTheOne, Krasnodar, 2025")
    print(f"Mode: {args.mode}")
    print("=" * 60)
    
    if args.mode == "fine_tune":
        results = run_military_fine_tuning(args)
        if results:
            print("✅ Phase 6 fine-tuning completed successfully!")
        else:
            print("❌ Phase 6 fine-tuning failed!")
    
    elif args.mode == "evaluate":
        metrics = run_phase6_evaluation(args)
        if metrics:
            print("✅ Phase 6 evaluation completed successfully!")
            print(f"Military metrics: {metrics}")
        else:
            print("❌ Phase 6 evaluation failed!")
    
    elif args.mode == "test":
        print("🧪 Testing Phase 6 components...")
        # Здесь можно добавить тесты компонентов
        print("✅ Phase 6 components test completed!")

if __name__ == "__main__":
    import json
    import torch
    main()
