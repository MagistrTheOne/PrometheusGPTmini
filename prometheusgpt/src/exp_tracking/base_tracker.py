"""
PrometheusGPT Mini - Base Experiment Tracker
Author: MagistrTheOne, Krasnodar, 2025

Базовый класс для отслеживания экспериментов.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseTracker(ABC):
    """Базовый класс для отслеживания экспериментов"""

    def __init__(self, project_name: str = "PrometheusGPT", experiment_name: Optional[str] = None):
        """
        Args:
            project_name: название проекта
            experiment_name: название эксперимента
        """
        
        self.project_name = project_name
        self.experiment_name = experiment_name or f"experiment_{self._get_timestamp()}"
        self.is_initialized = False
        
    def _get_timestamp(self) -> str:
        """Получить временную метку"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @abstractmethod
    def initialize(self) -> bool:
        """Инициализировать трекер"""
        pass
    
    @abstractmethod
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Логировать параметры эксперимента"""
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        """Логировать метрики"""
        pass
    
    @abstractmethod
    def log_artifacts(self, artifacts: Dict[str, Union[str, Path]]) -> None:
        """Логировать артефакты (файлы, модели)"""
        pass
    
    @abstractmethod
    def log_text(self, text: str, name: str) -> None:
        """Логировать текст"""
        pass
    
    @abstractmethod
    def log_figure(self, figure, name: str, step: Optional[int] = None) -> None:
        """Логировать график/изображение"""
        pass
    
    @abstractmethod
    def finish(self) -> None:
        """Завершить эксперимент"""
        pass
    
    def log_model_info(self, model, model_name: str = "model") -> None:
        """Логировать информацию о модели"""
        
        if hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self.log_metrics({
                f"{model_name}_total_params": total_params,
                f"{model_name}_trainable_params": trainable_params,
                f"{model_name}_size_mb": total_params * 4 / 1024 / 1024  # примерный размер в MB
            })
    
    def log_training_step(self, step: int, loss: float, learning_rate: float, 
                         step_time: float, **kwargs) -> None:
        """Логировать шаг обучения"""
        
        metrics = {
            "train/loss": loss,
            "train/learning_rate": learning_rate,
            "train/step_time": step_time,
            "train/step": step
        }
        
        # Добавляем дополнительные метрики
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                metrics[f"train/{key}"] = value
        
        self.log_metrics(metrics, step=step)
    
    def log_validation(self, step: int, val_loss: float, val_ppl: float, 
                      val_tokens: int, **kwargs) -> None:
        """Логировать валидацию"""
        
        metrics = {
            "val/loss": val_loss,
            "val/perplexity": val_ppl,
            "val/tokens": val_tokens,
            "val/step": step
        }
        
        # Добавляем дополнительные метрики
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                metrics[f"val/{key}"] = value
        
        self.log_metrics(metrics, step=step)
    
    def log_system_info(self) -> None:
        """Логировать системную информацию"""
        
        import torch
        import psutil
        import platform
        
        system_info = {
            "system/platform": platform.platform(),
            "system/python_version": platform.python_version(),
            "system/cpu_count": psutil.cpu_count(),
            "system/memory_gb": psutil.virtual_memory().total / 1024**3,
        }
        
        if torch.cuda.is_available():
            system_info.update({
                "system/cuda_available": True,
                "system/cuda_version": torch.version.cuda,
                "system/gpu_count": torch.cuda.device_count(),
                "system/gpu_name": torch.cuda.get_device_name(0),
                "system/gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
            })
        else:
            system_info["system/cuda_available"] = False
        
        self.log_parameters(system_info)
