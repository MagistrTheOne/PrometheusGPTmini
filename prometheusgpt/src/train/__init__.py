"""
PrometheusGPT Mini - Training Module
Author: MagistrTheOne, Krasnodar, 2025

Модуль для обучения: trainer, monitoring, checkpointing.
"""

from .train_pipeline import AdvancedTrainer, TrainingMonitor, create_demo_training_setup, run_smoke_test

__all__ = [
    'AdvancedTrainer',
    'TrainingMonitor',
    'create_demo_training_setup',
    'run_smoke_test'
]
