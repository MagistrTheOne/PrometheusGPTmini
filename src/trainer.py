"""
Обучающий пайплайн для PrometheusGPT Mini
Оптимизации для RTX 2080: FP16, gradient checkpointing, efficient batching
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import logging
from tqdm import tqdm
import wandb
from typing import Dict, Optional, Tuple
import time
from config import ModelConfig
from src.model import PrometheusGPT, count_parameters
from src.data import create_data_loaders
from src.tokenizer import MultilingualTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Тренер для PrometheusGPT Mini с оптимизациями для RTX 2080
    """

    def __init__(self, config: ModelConfig, use_wandb: bool = True):
        self.config = config
        self.use_wandb = use_wandb

        # Создаем модель
        self.model = PrometheusGPT(config)
        self.model = self.model.cuda() if torch.cuda.is_available() else self.model

        # Токенизатор
        self.tokenizer = MultilingualTokenizer(config)

        # Оптимизатор
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        # LR scheduler с warmup
        self.scheduler = self._create_scheduler()

        # Mixed precision scaler
        self.scaler = GradScaler() if config.use_fp16 else None

        # Gradient clipping
        self.max_grad_norm = 1.0

        # Создаем директории
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        # Лучшая perplexity для сохранения чекпоинтов
        self.best_perplexity = float('inf')

        # Инициализация wandb
        if self.use_wandb:
            wandb.init(
                project="prometheus-gpt-mini",
                config={
                    "model": "PrometheusGPT-Mini-HM-MoE",
                    "vocab_size": config.vocab_size,
                    "d_model": config.d_model,
                    "num_layers": config.num_layers,
                    "num_experts": config.num_experts,
                    "batch_size": config.effective_batch_size,
                    "learning_rate": config.learning_rate,
                    "max_steps": config.max_steps,
                }
            )

        logger.info(f"Trainer initialized with {count_parameters(self.model):,} parameters")

    def _create_scheduler(self):
        """Создание LR scheduler с warmup"""
        # Simple cosine annealing без warmup (можно добавить warmup позже)
        return CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_steps,
            eta_min=self.config.learning_rate * 0.1
        )

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Обучение одной эпохи

        Args:
            train_loader: DataLoader для тренировочных данных

        Returns:
            Словарь с метриками эпохи
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc="Training")
        start_time = time.time()

        for batch_idx, batch in enumerate(progress_bar):
            # Перемещаем на GPU
            input_ids = batch['input_ids'].cuda() if torch.cuda.is_available() else batch['input_ids']
            labels = batch['labels'].cuda() if torch.cuda.is_available() else batch['labels']

            # Gradient accumulation
            loss_accumulated = 0.0

            for accum_step in range(self.config.gradient_accumulation_steps):
                # Выбираем подбатч
                start_idx = accum_step * self.config.batch_size
                end_idx = start_idx + self.config.batch_size

                batch_input_ids = input_ids[start_idx:end_idx]
                batch_labels = labels[start_idx:end_idx]

                if batch_input_ids.size(0) == 0:
                    break

                # Forward pass с mixed precision
                with autocast(enabled=self.config.use_fp16):
                    outputs = self.model(batch_input_ids, labels=batch_labels)
                    loss = outputs['loss'] / self.config.gradient_accumulation_steps

                loss_accumulated += loss.item()

                # Backward pass
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

            # Gradient clipping и оптимизация
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            self.optimizer.zero_grad()

            # Scheduler step
            self.scheduler.step()

            # Статистика
            total_loss += loss_accumulated
            num_batches += 1

            # Обновляем progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f"{loss_accumulated:.4f}",
                'lr': f"{current_lr:.2e}",
                'gpu_mem': f"{torch.cuda.memory_allocated()/1024**3:.1f}GB" if torch.cuda.is_available() else "N/A"
            })

            # Ранний выход для тестирования
            if batch_idx >= 10:  # Для отладки
                break

        epoch_time = time.time() - start_time
        avg_loss = total_loss / num_batches

        metrics = {
            'train_loss': avg_loss,
            'train_perplexity': torch.exp(torch.tensor(avg_loss)).item(),
            'epoch_time': epoch_time,
            'samples_per_sec': len(train_loader) * self.config.effective_batch_size / epoch_time
        }

        return metrics

    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """
        Валидация модели

        Args:
            val_loader: DataLoader для валидационных данных

        Returns:
            Словарь с метриками валидации
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(val_loader, desc="Validating")

        for batch in progress_bar:
            # Перемещаем на GPU
            input_ids = batch['input_ids'].cuda() if torch.cuda.is_available() else batch['input_ids']
            labels = batch['labels'].cuda() if torch.cuda.is_available() else batch['labels']

            # Forward pass
            outputs = self.model(input_ids, labels=labels)
            loss = outputs['loss']

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({'val_loss': f"{loss.item():.4f}"})

            # Ранний выход для тестирования
            if num_batches >= 5:  # Для отладки
                break

        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {
            'val_loss': avg_loss,
            'val_perplexity': perplexity
        }

    def save_checkpoint(self, step: int, metrics: Dict[str, float]):
        """Сохранение чекпоинта"""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_step_{step}.pt"
        )

        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': metrics,
            'config': self.config.__dict__
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Сохраняем лучший чекпоинт
        if metrics.get('val_perplexity', float('inf')) < self.best_perplexity:
            self.best_perplexity = metrics['val_perplexity']
            best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Загрузка чекпоинта"""
        checkpoint = torch.load(checkpoint_path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        step = checkpoint['step']
        metrics = checkpoint.get('metrics', {})

        logger.info(f"Checkpoint loaded from step {step}")
        return step, metrics

    def train(self, resume_from: Optional[str] = None):
        """
        Основной цикл обучения

        Args:
            resume_from: Путь к чекпоинту для возобновления обучения
        """
        # Загружаем чекпоинт если указан
        start_step = 0
        if resume_from:
            start_step, _ = self.load_checkpoint(resume_from)

        # Создаем data loaders
        train_loader, val_loader, _ = create_data_loaders(self.config)

        logger.info("Starting training...")
        logger.info(f"GPU available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        global_step = start_step

        try:
            while global_step < self.config.max_steps:
                # Train epoch
                train_metrics = self.train_epoch(train_loader)
                global_step += 1

                # Validation каждые 100 шагов
                if global_step % 100 == 0:
                    val_metrics = self.validate(val_loader)
                    metrics = {**train_metrics, **val_metrics}

                    # Logging
                    logger.info(f"Step {global_step}: Train Loss: {metrics['train_loss']:.4f}, "
                              f"Val Loss: {metrics['val_loss']:.4f}, "
                              f"Val PPL: {metrics['val_perplexity']:.2f}")

                    # Wandb logging
                    if self.use_wandb:
                        wandb.log(metrics, step=global_step)

                    # Сохраняем чекпоинт
                    self.save_checkpoint(global_step, metrics)

                # Ранний выход для тестирования
                if global_step >= 2:  # Для отладки
                    break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Финальный чекпоинт
            final_metrics = self.validate(val_loader)
            self.save_checkpoint(global_step, final_metrics)

            if self.use_wandb:
                wandb.finish()

            logger.info("Training completed!")


def main():
    """Основная функция для запуска обучения"""
    config = ModelConfig()
    config.print_config()

    # Создаем тренера
    trainer = Trainer(config, use_wandb=False)  # Отключаем wandb для тестирования

    # Запускаем обучение
    trainer.train()


if __name__ == "__main__":
    main()
