"""
PrometheusGPT Mini - Training Integration with Experiment Tracking
Author: MagistrTheOne, Krasnodar, 2025

Интеграция experiment tracking с training pipeline.
"""

import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

from .base_tracker import BaseTracker
from .clearml_tracker import ClearMLTracker
from .mlflow_tracker import MLflowTracker

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Универсальный трекер экспериментов с поддержкой нескольких бэкендов"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: конфигурация трекера
        """
        
        self.config = config
        self.trackers: List[BaseTracker] = []
        self.is_initialized = False
        
        # Инициализируем трекеры
        self._setup_trackers()
    
    def _setup_trackers(self):
        """Настройка трекеров"""
        
        project_name = self.config.get('project_name', 'PrometheusGPT')
        experiment_name = self.config.get('experiment_name')
        
        # ClearML трекер
        if self.config.get('use_clearml', False):
            try:
                clearml_config = self.config.get('clearml', {})
                clearml_tracker = ClearMLTracker(
                    project_name=project_name,
                    experiment_name=experiment_name,
                    task_type=clearml_config.get('task_type', 'training'),
                    tags=clearml_config.get('tags', [])
                )
                self.trackers.append(clearml_tracker)
                logger.info("ClearML tracker added")
            except Exception as e:
                logger.warning(f"Failed to setup ClearML tracker: {e}")
        
        # MLflow трекер
        if self.config.get('use_mlflow', False):
            try:
                mlflow_config = self.config.get('mlflow', {})
                mlflow_tracker = MLflowTracker(
                    project_name=project_name,
                    experiment_name=experiment_name,
                    tracking_uri=mlflow_config.get('tracking_uri'),
                    tags=mlflow_config.get('tags', {})
                )
                self.trackers.append(mlflow_tracker)
                logger.info("MLflow tracker added")
            except Exception as e:
                logger.warning(f"Failed to setup MLflow tracker: {e}")
    
    def initialize(self) -> bool:
        """Инициализировать все трекеры"""
        
        if self.is_initialized:
            return True
        
        success_count = 0
        for tracker in self.trackers:
            if tracker.initialize():
                success_count += 1
        
        self.is_initialized = success_count > 0
        
        if self.is_initialized:
            logger.info(f"Experiment tracking initialized with {success_count} trackers")
        else:
            logger.warning("No experiment trackers could be initialized")
        
        return self.is_initialized
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Логировать параметры во все трекеры"""
        
        if not self.is_initialized:
            logger.warning("Experiment tracker not initialized")
            return
        
        for tracker in self.trackers:
            if tracker.is_initialized:
                try:
                    tracker.log_parameters(params)
                except Exception as e:
                    logger.error(f"Failed to log parameters to {type(tracker).__name__}: {e}")
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        """Логировать метрики во все трекеры"""
        
        if not self.is_initialized:
            logger.warning("Experiment tracker not initialized")
            return
        
        for tracker in self.trackers:
            if tracker.is_initialized:
                try:
                    tracker.log_metrics(metrics, step=step)
                except Exception as e:
                    logger.error(f"Failed to log metrics to {type(tracker).__name__}: {e}")
    
    def log_artifacts(self, artifacts: Dict[str, Union[str, Path]]) -> None:
        """Логировать артефакты во все трекеры"""
        
        if not self.is_initialized:
            logger.warning("Experiment tracker not initialized")
            return
        
        for tracker in self.trackers:
            if tracker.is_initialized:
                try:
                    tracker.log_artifacts(artifacts)
                except Exception as e:
                    logger.error(f"Failed to log artifacts to {type(tracker).__name__}: {e}")
    
    def log_text(self, text: str, name: str) -> None:
        """Логировать текст во все трекеры"""
        
        if not self.is_initialized:
            logger.warning("Experiment tracker not initialized")
            return
        
        for tracker in self.trackers:
            if tracker.is_initialized:
                try:
                    tracker.log_text(text, name)
                except Exception as e:
                    logger.error(f"Failed to log text to {type(tracker).__name__}: {e}")
    
    def log_figure(self, figure, name: str, step: Optional[int] = None) -> None:
        """Логировать график во все трекеры"""
        
        if not self.is_initialized:
            logger.warning("Experiment tracker not initialized")
            return
        
        for tracker in self.trackers:
            if tracker.is_initialized:
                try:
                    tracker.log_figure(figure, name, step=step)
                except Exception as e:
                    logger.error(f"Failed to log figure to {type(tracker).__name__}: {e}")
    
    def log_training_step(self, step: int, loss: float, learning_rate: float, 
                         step_time: float, **kwargs) -> None:
        """Логировать шаг обучения"""
        
        if not self.is_initialized:
            return
        
        for tracker in self.trackers:
            if tracker.is_initialized:
                try:
                    tracker.log_training_step(step, loss, learning_rate, step_time, **kwargs)
                except Exception as e:
                    logger.error(f"Failed to log training step to {type(tracker).__name__}: {e}")
    
    def log_validation(self, step: int, val_loss: float, val_ppl: float, 
                      val_tokens: int, **kwargs) -> None:
        """Логировать валидацию"""
        
        if not self.is_initialized:
            return
        
        for tracker in self.trackers:
            if tracker.is_initialized:
                try:
                    tracker.log_validation(step, val_loss, val_ppl, val_tokens, **kwargs)
                except Exception as e:
                    logger.error(f"Failed to log validation to {type(tracker).__name__}: {e}")
    
    def log_model_info(self, model, model_name: str = "model") -> None:
        """Логировать информацию о модели"""
        
        if not self.is_initialized:
            return
        
        for tracker in self.trackers:
            if tracker.is_initialized:
                try:
                    tracker.log_model_info(model, model_name)
                except Exception as e:
                    logger.error(f"Failed to log model info to {type(tracker).__name__}: {e}")
    
    def log_system_info(self) -> None:
        """Логировать системную информацию"""
        
        if not self.is_initialized:
            return
        
        for tracker in self.trackers:
            if tracker.is_initialized:
                try:
                    tracker.log_system_info()
                except Exception as e:
                    logger.error(f"Failed to log system info to {type(tracker).__name__}: {e}")
    
    def log_gpu_usage(self, gpu_metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Логировать использование GPU"""
        
        if not self.is_initialized:
            return
        
        for tracker in self.trackers:
            if tracker.is_initialized:
                try:
                    if hasattr(tracker, 'log_gpu_usage'):
                        tracker.log_gpu_usage(gpu_metrics, step=step)
                except Exception as e:
                    logger.error(f"Failed to log GPU usage to {type(tracker).__name__}: {e}")
    
    def log_training_curves(self, train_losses: list, val_losses: list, 
                           train_steps: list, val_steps: list) -> None:
        """Логировать кривые обучения"""
        
        if not self.is_initialized:
            return
        
        for tracker in self.trackers:
            if tracker.is_initialized:
                try:
                    if hasattr(tracker, 'log_training_curves'):
                        tracker.log_training_curves(train_losses, val_losses, train_steps, val_steps)
                except Exception as e:
                    logger.error(f"Failed to log training curves to {type(tracker).__name__}: {e}")
    
    def log_model_checkpoint(self, model_path: Union[str, Path], 
                           metrics: Optional[Dict[str, float]] = None) -> None:
        """Логировать чекпоинт модели"""
        
        if not self.is_initialized:
            return
        
        for tracker in self.trackers:
            if tracker.is_initialized:
                try:
                    if hasattr(tracker, 'log_model_checkpoint'):
                        tracker.log_model_checkpoint(model_path, metrics)
                except Exception as e:
                    logger.error(f"Failed to log model checkpoint to {type(tracker).__name__}: {e}")
    
    def finish(self) -> None:
        """Завершить все эксперименты"""
        
        for tracker in self.trackers:
            if tracker.is_initialized:
                try:
                    tracker.finish()
                except Exception as e:
                    logger.error(f"Failed to finish {type(tracker).__name__}: {e}")
        
        self.is_initialized = False
        logger.info("All experiment trackers finished")
    
    def get_tracker_urls(self) -> Dict[str, str]:
        """Получить URLs всех трекеров"""
        
        urls = {}
        
        for tracker in self.trackers:
            if tracker.is_initialized:
                try:
                    if hasattr(tracker, 'get_task_url'):
                        url = tracker.get_task_url()
                        if url:
                            urls[f"{type(tracker).__name__}_url"] = url
                    elif hasattr(tracker, 'get_run_url'):
                        url = tracker.get_run_url()
                        if url:
                            urls[f"{type(tracker).__name__}_url"] = url
                except Exception as e:
                    logger.error(f"Failed to get URL from {type(tracker).__name__}: {e}")
        
        return urls


def create_experiment_tracker(config: Dict[str, Any]) -> ExperimentTracker:
    """Создать трекер экспериментов из конфигурации"""
    
    return ExperimentTracker(config)


def get_default_tracking_config() -> Dict[str, Any]:
    """Получить конфигурацию по умолчанию для трекинга"""
    
    return {
        'project_name': 'PrometheusGPT',
        'experiment_name': None,  # будет сгенерировано автоматически
        
        # ClearML настройки
        'use_clearml': True,
        'clearml': {
            'task_type': 'training',
            'tags': ['prometheusgpt', 'llm', 'transformer']
        },
        
        # MLflow настройки
        'use_mlflow': False,  # по умолчанию отключен
        'mlflow': {
            'tracking_uri': None,  # локальный MLflow
            'tags': {
                'project': 'PrometheusGPT',
                'model_type': 'transformer',
                'author': 'MagistrTheOne'
            }
        }
    }
