"""
PrometheusGPT Mini - MLflow Experiment Tracker
Author: MagistrTheOne, Krasnodar, 2025

Интеграция с MLflow для отслеживания экспериментов.
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
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Install with: pip install mlflow")


class MLflowTracker(BaseTracker):
    """MLflow трекер для отслеживания экспериментов"""

    def __init__(self, project_name: str = "PrometheusGPT", experiment_name: Optional[str] = None,
                 tracking_uri: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Args:
            project_name: название проекта
            experiment_name: название эксперимента
            tracking_uri: URI для MLflow tracking server
            tags: теги для эксперимента
        """
        
        super().__init__(project_name, experiment_name)
        
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not available. Install with: pip install mlflow")
        
        self.tracking_uri = tracking_uri
        self.tags = tags or {}
        self.run = None
        
        # Настраиваем MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
    
    def initialize(self) -> bool:
        """Инициализировать MLflow трекер"""
        
        try:
            # Создаем или получаем эксперимент
            experiment = mlflow.get_experiment_by_name(self.project_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.project_name)
            else:
                experiment_id = experiment.experiment_id
            
            # Начинаем run
            self.run = mlflow.start_run(
                experiment_id=experiment_id,
                run_name=self.experiment_name,
                tags=self.tags
            )
            
            # Логируем системную информацию
            self.log_system_info()
            
            self.is_initialized = True
            logger.info(f"MLflow tracker initialized: {self.project_name}/{self.experiment_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MLflow tracker: {e}")
            return False
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Логировать параметры эксперимента"""
        
        if not self.is_initialized:
            logger.warning("MLflow tracker not initialized")
            return
        
        try:
            # Конвертируем параметры в строки (MLflow требует)
            str_params = {}
            for key, value in params.items():
                if isinstance(value, (str, int, float, bool)):
                    str_params[key] = str(value)
                else:
                    str_params[key] = str(value)
            
            mlflow.log_params(str_params)
            logger.debug(f"Logged {len(params)} parameters to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log parameters to MLflow: {e}")
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        """Логировать метрики"""
        
        if not self.is_initialized:
            logger.warning("MLflow tracker not initialized")
            return
        
        try:
            # Логируем метрики
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=step)
            
            logger.debug(f"Logged {len(metrics)} metrics to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log metrics to MLflow: {e}")
    
    def log_artifacts(self, artifacts: Dict[str, Union[str, Path]]) -> None:
        """Логировать артефакты"""
        
        if not self.is_initialized:
            logger.warning("MLflow tracker not initialized")
            return
        
        try:
            for name, path in artifacts.items():
                if isinstance(path, str):
                    path = Path(path)
                
                if path.exists():
                    if path.is_file():
                        mlflow.log_artifact(str(path), artifact_path=name)
                    elif path.is_dir():
                        mlflow.log_artifacts(str(path), artifact_path=name)
                    logger.info(f"Logged artifact {name}: {path}")
                else:
                    logger.warning(f"Artifact not found: {path}")
            
        except Exception as e:
            logger.error(f"Failed to log artifacts to MLflow: {e}")
    
    def log_text(self, text: str, name: str) -> None:
        """Логировать текст"""
        
        if not self.is_initialized:
            logger.warning("MLflow tracker not initialized")
            return
        
        try:
            # Создаем временный файл
            temp_path = Path(f"/tmp/{name}.txt")
            temp_path.write_text(text, encoding='utf-8')
            
            # Логируем как артефакт
            mlflow.log_artifact(str(temp_path), artifact_path="text_logs")
            
            # Удаляем временный файл
            temp_path.unlink(missing_ok=True)
            
            logger.debug(f"Logged text {name} to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log text to MLflow: {e}")
    
    def log_figure(self, figure, name: str, step: Optional[int] = None) -> None:
        """Логировать график/изображение"""
        
        if not self.is_initialized:
            logger.warning("MLflow tracker not initialized")
            return
        
        try:
            # Если передан matplotlib figure
            if hasattr(figure, 'savefig'):
                # Сохраняем во временный файл
                temp_path = Path(f"/tmp/{name}_{step or 0}.png")
                figure.savefig(temp_path, dpi=150, bbox_inches='tight')
                
                # Логируем как артефакт
                mlflow.log_artifact(str(temp_path), artifact_path="figures")
                
                # Удаляем временный файл
                temp_path.unlink(missing_ok=True)
                
            # Если передан numpy array
            elif isinstance(figure, np.ndarray):
                # Сохраняем как изображение
                temp_path = Path(f"/tmp/{name}_{step or 0}.png")
                plt.imsave(temp_path, figure)
                
                # Логируем как артефакт
                mlflow.log_artifact(str(temp_path), artifact_path="images")
                
                # Удаляем временный файл
                temp_path.unlink(missing_ok=True)
            
            logger.debug(f"Logged figure {name} to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log figure to MLflow: {e}")
    
    def log_model(self, model, model_name: str = "model", 
                  signature=None, input_example=None) -> None:
        """Логировать модель PyTorch"""
        
        if not self.is_initialized:
            logger.warning("MLflow tracker not initialized")
            return
        
        try:
            # Логируем модель PyTorch
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=model_name,
                signature=signature,
                input_example=input_example
            )
            
            logger.info(f"Logged PyTorch model {model_name} to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log model to MLflow: {e}")
    
    def log_model_checkpoint(self, model_path: Union[str, Path], 
                           metrics: Optional[Dict[str, float]] = None) -> None:
        """Логировать чекпоинт модели"""
        
        if not self.is_initialized:
            logger.warning("MLflow tracker not initialized")
            return
        
        try:
            model_path = Path(model_path)
            
            if model_path.exists():
                # Логируем как артефакт
                mlflow.log_artifact(str(model_path), artifact_path="checkpoints")
                
                # Логируем метрики модели если есть
                if metrics:
                    self.log_metrics(metrics)
                
                logger.info(f"Logged model checkpoint: {model_path}")
            else:
                logger.warning(f"Model checkpoint not found: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to log model checkpoint to MLflow: {e}")
    
    def log_training_curves(self, train_losses: list, val_losses: list, 
                           train_steps: list, val_steps: list) -> None:
        """Логировать кривые обучения"""
        
        if not self.is_initialized:
            logger.warning("MLflow tracker not initialized")
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
            logger.error(f"Failed to log training curves to MLflow: {e}")
    
    def log_gpu_usage(self, gpu_metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Логировать использование GPU"""
        
        if not self.is_initialized:
            logger.warning("MLflow tracker not initialized")
            return
        
        try:
            gpu_metrics_with_prefix = {f"gpu/{k}": v for k, v in gpu_metrics.items()}
            self.log_metrics(gpu_metrics_with_prefix, step=step)
            
        except Exception as e:
            logger.error(f"Failed to log GPU usage to MLflow: {e}")
    
    def finish(self) -> None:
        """Завершить эксперимент"""
        
        if not self.is_initialized:
            logger.warning("MLflow tracker not initialized")
            return
        
        try:
            # Завершаем run
            mlflow.end_run()
            logger.info("MLflow experiment finished")
            
        except Exception as e:
            logger.error(f"Failed to finish MLflow experiment: {e}")
    
    def get_run_id(self) -> Optional[str]:
        """Получить ID run"""
        
        if self.run:
            return self.run.info.run_id
        return None
    
    def get_run_url(self) -> Optional[str]:
        """Получить URL run в веб-интерфейсе"""
        
        if self.run and self.tracking_uri:
            return f"{self.tracking_uri}/#/experiments/{self.run.info.experiment_id}/runs/{self.run.info.run_id}"
        return None
