"""
PrometheusGPT Mini - ClearML Experiment Tracker
Author: MagistrTheOne, Krasnodar, 2025

Интеграция с ClearML для отслеживания экспериментов.
"""

import os
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from .base_tracker import BaseTracker

logger = logging.getLogger(__name__)

try:
    from clearml import Task, Logger
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    logger.warning("ClearML not available. Install with: pip install clearml")


class ClearMLTracker(BaseTracker):
    """ClearML трекер для отслеживания экспериментов"""

    def __init__(self, project_name: str = "PrometheusGPT", experiment_name: Optional[str] = None,
                 task_type: str = "training", tags: Optional[list] = None):
        """
        Args:
            project_name: название проекта
            experiment_name: название эксперимента
            task_type: тип задачи (training, testing, etc.)
            tags: теги для эксперимента
        """
        
        super().__init__(project_name, experiment_name)
        
        if not CLEARML_AVAILABLE:
            raise ImportError("ClearML not available. Install with: pip install clearml")
        
        self.task_type = task_type
        self.tags = tags or []
        self.task = None
        self.logger = None
        
    def initialize(self) -> bool:
        """Инициализировать ClearML трекер"""
        
        try:
            # Создаем задачу
            self.task = Task.init(
                project_name=self.project_name,
                task_name=self.experiment_name,
                task_type=self.task_type,
                tags=self.tags,
                auto_connect_frameworks=True,
                auto_connect_streams=True
            )
            
            # Получаем логгер
            self.logger = self.task.get_logger()
            
            # Логируем системную информацию
            self.log_system_info()
            
            self.is_initialized = True
            logger.info(f"ClearML tracker initialized: {self.project_name}/{self.experiment_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ClearML tracker: {e}")
            return False
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Логировать параметры эксперимента"""
        
        if not self.is_initialized:
            logger.warning("ClearML tracker not initialized")
            return
        
        try:
            # Логируем параметры
            for key, value in params.items():
                self.task.connect({key: value})
            
            logger.debug(f"Logged {len(params)} parameters to ClearML")
            
        except Exception as e:
            logger.error(f"Failed to log parameters to ClearML: {e}")
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        """Логировать метрики"""
        
        if not self.is_initialized:
            logger.warning("ClearML tracker not initialized")
            return
        
        try:
            # Логируем метрики
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.logger.report_scalar(
                        title=key.split('/')[0] if '/' in key else "metrics",
                        series=key.split('/')[1] if '/' in key else key,
                        value=value,
                        iteration=step or 0
                    )
            
            logger.debug(f"Logged {len(metrics)} metrics to ClearML")
            
        except Exception as e:
            logger.error(f"Failed to log metrics to ClearML: {e}")
    
    def log_artifacts(self, artifacts: Dict[str, Union[str, Path]]) -> None:
        """Логировать артефакты"""
        
        if not self.is_initialized:
            logger.warning("ClearML tracker not initialized")
            return
        
        try:
            for name, path in artifacts.items():
                if isinstance(path, str):
                    path = Path(path)
                
                if path.exists():
                    # Загружаем артефакт
                    self.task.upload_artifact(
                        name=name,
                        artifact_object=str(path)
                    )
                    logger.info(f"Uploaded artifact {name}: {path}")
                else:
                    logger.warning(f"Artifact not found: {path}")
            
        except Exception as e:
            logger.error(f"Failed to log artifacts to ClearML: {e}")
    
    def log_text(self, text: str, name: str) -> None:
        """Логировать текст"""
        
        if not self.is_initialized:
            logger.warning("ClearML tracker not initialized")
            return
        
        try:
            self.logger.report_text(
                title="text_logs",
                series=name,
                value=text
            )
            logger.debug(f"Logged text {name} to ClearML")
            
        except Exception as e:
            logger.error(f"Failed to log text to ClearML: {e}")
    
    def log_figure(self, figure, name: str, step: Optional[int] = None) -> None:
        """Логировать график/изображение"""
        
        if not self.is_initialized:
            logger.warning("ClearML tracker not initialized")
            return
        
        try:
            # Если передан matplotlib figure
            if hasattr(figure, 'savefig'):
                # Сохраняем во временный файл
                temp_path = Path(f"/tmp/{name}_{step or 0}.png")
                figure.savefig(temp_path, dpi=150, bbox_inches='tight')
                
                # Загружаем в ClearML
                self.task.upload_artifact(
                    name=f"figure_{name}",
                    artifact_object=str(temp_path)
                )
                
                # Удаляем временный файл
                temp_path.unlink(missing_ok=True)
                
            # Если передан numpy array
            elif isinstance(figure, np.ndarray):
                self.logger.report_image(
                    title="images",
                    series=name,
                    iteration=step or 0,
                    image=figure
                )
            
            logger.debug(f"Logged figure {name} to ClearML")
            
        except Exception as e:
            logger.error(f"Failed to log figure to ClearML: {e}")
    
    def log_model_checkpoint(self, model_path: Union[str, Path], metrics: Optional[Dict[str, float]] = None) -> None:
        """Логировать чекпоинт модели"""
        
        if not self.is_initialized:
            logger.warning("ClearML tracker not initialized")
            return
        
        try:
            model_path = Path(model_path)
            
            if model_path.exists():
                # Загружаем модель как артефакт
                self.task.upload_artifact(
                    name="model_checkpoint",
                    artifact_object=str(model_path)
                )
                
                # Логируем метрики модели если есть
                if metrics:
                    self.log_metrics(metrics)
                
                logger.info(f"Uploaded model checkpoint: {model_path}")
            else:
                logger.warning(f"Model checkpoint not found: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to log model checkpoint to ClearML: {e}")
    
    def log_training_curves(self, train_losses: list, val_losses: list, 
                           train_steps: list, val_steps: list) -> None:
        """Логировать кривые обучения"""
        
        if not self.is_initialized:
            logger.warning("ClearML tracker not initialized")
            return
        
        try:
            # Создаем график
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # График потерь
            ax1.plot(train_steps, train_losses, label='Train Loss', color='blue')
            if val_losses and val_steps:
                ax1.plot(val_steps, val_losses, label='Val Loss', color='red')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True)
            
            # График perplexity
            train_ppl = [np.exp(loss) for loss in train_losses]
            ax2.plot(train_steps, train_ppl, label='Train PPL', color='blue')
            if val_losses and val_steps:
                val_ppl = [np.exp(loss) for loss in val_losses]
                ax2.plot(val_steps, val_ppl, label='Val PPL', color='red')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Perplexity')
            ax2.set_title('Training and Validation Perplexity')
            ax2.legend()
            ax2.grid(True)
            
            # Логируем график
            self.log_figure(fig, "training_curves")
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Failed to log training curves to ClearML: {e}")
    
    def log_gpu_usage(self, gpu_metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Логировать использование GPU"""
        
        if not self.is_initialized:
            logger.warning("ClearML tracker not initialized")
            return
        
        try:
            gpu_metrics_with_prefix = {f"gpu/{k}": v for k, v in gpu_metrics.items()}
            self.log_metrics(gpu_metrics_with_prefix, step=step)
            
        except Exception as e:
            logger.error(f"Failed to log GPU usage to ClearML: {e}")
    
    def finish(self) -> None:
        """Завершить эксперимент"""
        
        if not self.is_initialized:
            logger.warning("ClearML tracker not initialized")
            return
        
        try:
            # Завершаем задачу
            self.task.close()
            logger.info("ClearML experiment finished")
            
        except Exception as e:
            logger.error(f"Failed to finish ClearML experiment: {e}")
    
    def get_task_id(self) -> Optional[str]:
        """Получить ID задачи"""
        
        if self.task:
            return self.task.id
        return None
    
    def get_task_url(self) -> Optional[str]:
        """Получить URL задачи в веб-интерфейсе"""
        
        if self.task:
            return self.task.get_output_log_web_page()
        return None
