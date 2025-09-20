#!/usr/bin/env python3
"""
PrometheusGPT Mini - Production Runner
Author: MagistrTheOne, Krasnodar, 2025

Скрипт для запуска production deployment с полной функциональностью.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent.parent))

from src.api.production_api import main as run_api
from src.train.full_train import main as run_training
from src.monitoring.logging_config import setup_logging
from src.monitoring.prometheus_metrics import PrometheusMetrics
from src.monitoring.gpu_monitor import MultiGPUMonitor
from src.monitoring.performance_monitor import PerformanceMonitor

def setup_production_logging():
    """Настроить логирование для production"""
    
    return setup_logging(
        log_level="INFO",
        log_dir="logs",
        log_format="json",
        enable_console=True,
        enable_file=True,
        log_requests=True,
        log_performance=True
    )

def start_monitoring():
    """Запустить мониторинг"""
    
    logger = logging.getLogger(__name__)
    logger.info("Starting production monitoring...")
    
    # GPU мониторинг
    gpu_monitor = MultiGPUMonitor(update_interval=5.0)
    gpu_monitor.start_monitoring()
    
    # Performance мониторинг
    perf_monitor = PerformanceMonitor(update_interval=10.0)
    perf_monitor.start_monitoring()
    
    # Prometheus метрики
    prometheus_metrics = PrometheusMetrics(port=8001)
    prometheus_metrics.start_server()
    
    logger.info("Production monitoring started")
    
    return {
        'gpu_monitor': gpu_monitor,
        'perf_monitor': perf_monitor,
        'prometheus_metrics': prometheus_metrics
    }

def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Запустить API сервер"""
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting API server on {host}:{port}")
    
    # Устанавливаем переменные окружения
    os.environ['PORT'] = str(port)
    os.environ['HOST'] = host
    
    # Запускаем API
    run_api()

def run_training_pipeline(config_file: str = None):
    """Запустить training pipeline"""
    
    logger = logging.getLogger(__name__)
    logger.info("Starting training pipeline...")
    
    # Запускаем training
    run_training()

def main():
    """Главная функция"""
    
    parser = argparse.ArgumentParser(description='PrometheusGPT Mini Production Runner')
    
    # Режимы работы
    parser.add_argument('--mode', choices=['api', 'training', 'monitoring', 'all'], 
                       default='api', help='Режим работы')
    
    # API настройки
    parser.add_argument('--host', default='0.0.0.0', help='Host для API сервера')
    parser.add_argument('--port', type=int, default=8000, help='Port для API сервера')
    
    # Training настройки
    parser.add_argument('--config', help='Файл конфигурации для training')
    
    # Мониторинг
    parser.add_argument('--enable-monitoring', action='store_true', 
                       help='Включить мониторинг')
    
    # Логирование
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Уровень логирования')
    
    args = parser.parse_args()
    
    # Настраиваем логирование
    setup_logging(
        log_level=args.log_level,
        log_dir="logs",
        log_format="json",
        enable_console=True,
        enable_file=True
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info("=== PrometheusGPT Mini Production Runner ===")
    logger.info("Author: MagistrTheOne, Krasnodar, 2025")
    logger.info(f"Mode: {args.mode}")
    
    try:
        if args.mode == 'api':
            # Запускаем только API
            if args.enable_monitoring:
                start_monitoring()
            run_api_server(args.host, args.port)
            
        elif args.mode == 'training':
            # Запускаем только training
            if args.enable_monitoring:
                start_monitoring()
            run_training_pipeline(args.config)
            
        elif args.mode == 'monitoring':
            # Запускаем только мониторинг
            monitoring = start_monitoring()
            
            # Держим процесс живым
            import time
            try:
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                logger.info("Stopping monitoring...")
                monitoring['gpu_monitor'].stop_monitoring()
                monitoring['perf_monitor'].stop_monitoring()
            
        elif args.mode == 'all':
            # Запускаем все компоненты
            monitoring = start_monitoring()
            
            # Запускаем API в отдельном процессе
            import multiprocessing
            api_process = multiprocessing.Process(
                target=run_api_server, 
                args=(args.host, args.port)
            )
            api_process.start()
            
            logger.info("All components started")
            
            try:
                # Держим процесс живым
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                logger.info("Stopping all components...")
                api_process.terminate()
                api_process.join()
                monitoring['gpu_monitor'].stop_monitoring()
                monitoring['perf_monitor'].stop_monitoring()
    
    except Exception as e:
        logger.error(f"Error in production runner: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
