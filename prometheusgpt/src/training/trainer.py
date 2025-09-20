"""
PrometheusGPT Mini - Training Pipeline
Author: MagistrTheOne, Krasnodar, 2025

Конвейер обучения с оптимизацией для RTX 2080 Super.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
import time
import logging
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import json

from ..model import PrometheusGPTMini, model_config, training_config
from .dataset import create_dataloader

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Trainer:
    """Тренер для PrometheusGPT Mini с оптимизацией под RTX 2080 Super"""

    def __init__(self, model: PrometheusGPTMini, config: training_config = None):
        """
        Args:
            model: модель для обучения
            config: конфигурация обучения
        """

        self.model = model
        self.config = config or training_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Переносим модель на устройство
        self.model.to(self.device)

        # Оптимизатор с weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )

        # Learning rate scheduler с warmup
        self.scheduler = self._create_scheduler()

        # Mixed precision для экономии памяти
        self.scaler = GradScaler(enabled=self.config.use_mixed_precision)

        # История обучения
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_times': []
        }

        # Метрики
        self.metrics = {
            'best_val_loss': float('inf'),
            'epochs_without_improvement': 0,
            'total_training_time': 0
        }

        logger.info(f"Trainer initialized. Device: {self.device}")
        logger.info(f"Model parameters: {model.get_model_info()['total_parameters']}")

    def _create_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Создать scheduler с warmup"""

        def lr_lambda(epoch):
            # Warmup phase
            if epoch < self.config.warmup_steps:
                return float(epoch) / float(max(1, self.config.warmup_steps))

            # Cosine annealing
            return 0.5 * (1.0 + torch.cos(torch.tensor(
                (epoch - self.config.warmup_steps) * 3.14159 /
                (self.config.num_epochs - self.config.warmup_steps)
            )))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self, dataloader: torch.utils.data.DataLoader,
                   epoch: int) -> Dict[str, float]:
        """Обучить одну эпоху"""

        self.model.train()
        total_loss = 0
        num_batches = 0
        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            # Подготовка данных
            batch = self._prepare_batch(batch)

            # Обнуляем градиенты
            self.optimizer.zero_grad()

            # Forward pass с mixed precision
            with autocast(enabled=self.config.use_mixed_precision):
                outputs = self.model(
                    batch['src_input_ids'],
                    batch['tgt_input_ids'],
                    batch.get('src_attention_mask'),
                    batch.get('tgt_attention_mask')
                )

                # Вычисляем loss
                loss = self._compute_loss(outputs, batch['tgt_target_ids'])

            # Backward pass с gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.config.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Обновляем статистику
            total_loss += loss.item()
            num_batches += 1

            # Логируем прогресс
            if batch_idx % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                          f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

        # Средний loss за эпоху
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start_time

        # Обновляем scheduler
        self.scheduler.step()

        # Сохраняем метрики
        self.history['train_loss'].append(avg_loss)
        self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        self.history['epoch_times'].append(epoch_time)

        logger.info(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}, "
                   f"Time: {epoch_time:.2f}s")

        return {'train_loss': avg_loss, 'epoch_time': epoch_time}

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Подготовить батч для обучения"""

        # Переносим все тензоры на устройство
        prepared_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared_batch[key] = value.to(self.device)
            else:
                prepared_batch[key] = value

        return prepared_batch

    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Вычислить loss для seq2seq модели"""

        # outputs: [batch_size, seq_len, vocab_size]
        # targets: [batch_size, seq_len]

        batch_size, seq_len, vocab_size = outputs.shape

        # Reshape для cross-entropy: [batch_size * seq_len, vocab_size]
        outputs = outputs.view(-1, vocab_size)
        targets = targets.view(-1)

        # Cross-entropy loss
        loss = nn.CrossEntropyLoss(ignore_index=0)(outputs, targets)  # ignore PAD token

        return loss

    def validate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Валидация модели"""

        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = self._prepare_batch(batch)

                # Forward pass
                outputs = self.model(
                    batch['src_input_ids'],
                    batch['tgt_input_ids'],
                    batch.get('src_attention_mask'),
                    batch.get('tgt_attention_mask')
                )

                # Loss
                loss = self._compute_loss(outputs, batch['tgt_target_ids'])
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        self.history['val_loss'].append(avg_loss)

        logger.info(f"Validation Loss: {avg_loss:.4f}")

        return {'val_loss': avg_loss}

    def train(self, train_dataloader: torch.utils.data.DataLoader,
              val_dataloader: Optional[torch.utils.data.DataLoader] = None,
              save_path: str = "models/prometheusgpt.pt",
              early_stopping_patience: int = 5) -> Dict[str, Any]:
        """
        Основной цикл обучения

        Args:
            train_dataloader: DataLoader для обучения
            val_dataloader: DataLoader для валидации
            save_path: путь для сохранения модели
            early_stopping_patience: количество эпох без улучшения
        """

        logger.info("=== Starting Training ===")
        logger.info(f"Author: MagistrTheOne, Krasnodar, 2025")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training config: {self.config.__dict__}")

        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            # Обучение
            train_metrics = self.train_epoch(train_dataloader, epoch)

            # Валидация
            val_metrics = {}
            if val_dataloader is not None:
                val_metrics = self.validate(val_dataloader)

            # Early stopping
            if val_dataloader is not None:
                current_val_loss = val_metrics['val_loss']
                if current_val_loss < self.metrics['best_val_loss']:
                    self.metrics['best_val_loss'] = current_val_loss
                    self.metrics['epochs_without_improvement'] = 0
                    self.save_model(save_path)
                    logger.info(f"New best model saved: {save_path}")
                else:
                    self.metrics['epochs_without_improvement'] += 1

                if self.metrics['epochs_without_improvement'] >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                # Сохраняем модель каждый 5 эпох
                if epoch % 5 == 0:
                    self.save_model(f"{save_path}_epoch_{epoch}.pt")

        # Финальное время
        self.metrics['total_training_time'] = time.time() - start_time

        logger.info("=== Training Completed ===")
        logger.info(f"Total time: {self.metrics['total_training_time']:.2f}s")
        logger.info(f"Best validation loss: {self.metrics['best_val_loss']:.4f}")

        return {
            'history': self.history,
            'metrics': self.metrics,
            'final_model_path': save_path
        }

    def save_model(self, path: str):
        """Сохранить модель и состояние тренера"""

        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.config,
            'history': self.history,
            'metrics': self.metrics,
            'author': 'MagistrTheOne, Krasnodar, 2025'
        }, path)

        logger.info(f"Model saved: {path}")

    def load_model(self, path: str):
        """Загрузить модель и состояние тренера"""

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if 'history' in checkpoint:
            self.history = checkpoint['history']
        if 'metrics' in checkpoint:
            self.metrics = checkpoint['metrics']

        logger.info(f"Model loaded: {path}")

    def get_memory_usage(self) -> Dict[str, float]:
        """Получить использование памяти GPU"""

        if not torch.cuda.is_available():
            return {'cpu_memory': 0}

        allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(0) / 1024**3   # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB

        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'utilization_percent': (allocated / total) * 100
        }


def create_demo_trainer() -> Trainer:
    """Создать тренер с демо моделью и датасетом"""

    from ..tokenizer import BPETokenizer
    from .dataset import TranslationDataset, create_demo_dataset, create_dataloader

    logger.info("Creating demo trainer...")

    # Создаем модель
    model = PrometheusGPTMini()

    # Создаем токенизаторы
    ru_texts, en_texts, train_ru, train_en = create_demo_dataset()
    ru_tokenizer = BPETokenizer()
    en_tokenizer = BPETokenizer()

    ru_tokenizer.train(ru_texts, "ru_tokenizer", vocab_size=1000)
    en_tokenizer.train(en_texts, "en_tokenizer", vocab_size=1000)

    # Создаем датасет и даталоадеры
    train_dataset = TranslationDataset(train_ru, train_en, ru_tokenizer, en_tokenizer)
    train_dataloader = create_dataloader(train_dataset, batch_size=8)

    # Создаем тренер
    trainer = Trainer(model)

    logger.info("Demo trainer created successfully!")

    return trainer, train_dataloader


if __name__ == "__main__":
    # Демонстрация тренера
    print("=== PrometheusGPT Mini Trainer Demo ===")
    print("Author: MagistrTheOne, Krasnodar, 2025")

    # Создаем тренер
    trainer, dataloader = create_demo_trainer()

    print(f"\nTraining info:")
    print(f"Model device: {trainer.device}")
    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Number of batches: {len(dataloader)}")

    # Пример батча
    batch = next(iter(dataloader))
    print(f"\nSample batch shapes:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")

    # Память GPU
    if torch.cuda.is_available():
        memory = trainer.get_memory_usage()
        print(f"\nGPU Memory:")
        for key, value in memory.items():
            print(f"  {key}: {value}")

    print(f"\n✅ Trainer demo completed!")
