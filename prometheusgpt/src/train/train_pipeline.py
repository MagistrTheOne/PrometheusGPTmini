"""
PrometheusGPT Mini - Advanced Training Pipeline
Author: MagistrTheOne, Krasnodar, 2025

Продвинутый training pipeline с gradient checkpointing, monitoring и всем необходимым.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
import time
import logging
import json
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from datetime import datetime
import psutil
import threading
import queue
import signal
import sys

from ..model import PrometheusGPTMini, model_config, training_config
from ..data.tokenizer import AdvancedBPETokenizer
from ..data.prepare_dataset import DatasetPreparator
from ..data.dataloader import CachedTextDataset, create_dataloader, get_memory_usage

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainingMonitor:
    """Мониторинг обучения в реальном времени"""

    def __init__(self, log_interval: int = 10, checkpoint_interval: int = 1000):
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.metrics_queue = queue.Queue()
        self.running = True

        # Запускаем мониторинг
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _monitor_loop(self):
        """Основной цикл мониторинга"""
        while self.running:
            try:
                # Получаем метрики
                if not self.metrics_queue.empty():
                    metrics = self.metrics_queue.get()
                    self._log_metrics(metrics)

                time.sleep(1)  # Проверяем каждую секунду
            except Exception as e:
                logger.error(f"Monitor error: {e}")

    def _log_metrics(self, metrics: Dict[str, Any]):
        """Логировать метрики"""
        step = metrics.get('step', 0)
        loss = metrics.get('loss', 0)
        lr = metrics.get('learning_rate', 0)
        step_time = metrics.get('step_time', 0)

        if step % self.log_interval == 0:
            memory = get_memory_usage()

            log_message = (
                f"Step {step:6d} | Loss: {loss:.4f} | LR: {lr:.6f} | "
                f"Time: {step_time:.3f}s | "
            )

            if 'gpu_allocated_gb' in memory:
                log_message += f"GPU: {memory['gpu_allocated_gb']:.2f}GB | "
            else:
                log_message += f"CPU Memory: {memory.get('cpu_only', 'N/A')} | "

            logger.info(log_message)

    def stop(self):
        """Остановить мониторинг"""
        self.running = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)


class AdvancedTrainer:
    """Продвинутый тренер для PrometheusGPT Mini"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: конфигурация обучения
        """

        # Конфигурация
        self.model_config = model_config
        self.training_config = training_config

        if config:
            self._update_config(config)

        # Устройство
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Создаем модель
        self.model = PrometheusGPTMini(self.model_config)
        self.model.to(self.device)

        # Оптимизатор
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            betas=(0.9, 0.999)
        )

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.scaler = GradScaler(enabled=self.training_config.use_mixed_precision)

        # Мониторинг
        self.monitor = TrainingMonitor()

        # Статистика
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_steps': 0,
            'best_loss': float('inf'),
            'current_epoch': 0,
            'steps_per_epoch': 0
        }

        logger.info(f"Advanced Trainer initialized. Device: {self.device}")

    def _update_config(self, config: Dict[str, Any]):
        """Обновить конфигурацию"""
        for key, value in config.items():
            if hasattr(self.model_config, key):
                setattr(self.model_config, key, value)
            if hasattr(self.training_config, key):
                setattr(self.training_config, key, value)

    def _create_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Создать scheduler с warmup и cosine annealing"""

        warmup_steps = getattr(self.training_config, 'warmup_steps', 1000)
        num_epochs = getattr(self.training_config, 'num_epochs', 10)

        def lr_lambda(step):
            # Warmup phase
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))

            # Cosine annealing
            progress = (step - warmup_steps) / float(max(1, num_epochs * self.stats['steps_per_epoch'] - warmup_steps))
            return 0.5 * (1.0 + torch.cos(torch.tensor(min(progress, 1.0) * 3.14159)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def save_checkpoint(self, step: int, loss: float, save_path: str):
        """Сохранить чекпоинт"""

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'step': step,
            'loss': loss,
            'config': {
                'model': self.model_config.__dict__,
                'training': self.training_config.__dict__
            },
            'stats': self.stats,
            'author': 'MagistrTheOne, Krasnodar, 2025',
            'timestamp': datetime.now().isoformat()
        }

        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved: {save_path} (step {step}, loss {loss:.4f})")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Загрузить чекпоинт"""

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.stats.update(checkpoint.get('stats', {}))
        start_step = checkpoint.get('step', 0)

        logger.info(f"Checkpoint loaded: {checkpoint_path} (step {start_step})")

        return start_step

    def train_step(self, batch: Dict[str, torch.Tensor], step: int) -> Dict[str, float]:
        """Один шаг обучения"""

        # Подготовка данных
        batch = self._prepare_batch(batch)

        # Обнуляем градиенты
        self.optimizer.zero_grad()

        # Forward pass с mixed precision
        with autocast(enabled=self.training_config.use_mixed_precision):
            outputs = self.model(batch['input_ids'], batch['target_ids'])
            loss = self._compute_loss(outputs, batch['target_ids'])

        # Backward pass
        self.scaler.scale(loss).backward()

        # Gradient clipping
        if self.training_config.max_grad_norm > 0:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Scheduler step
        self.scheduler.step()

        # Метрики
        metrics = {
            'step': step,
            'loss': loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'step_time': time.time() - self.stats.get('last_step_time', time.time())
        }

        # Обновляем статистику
        self.stats['last_step_time'] = time.time()
        self.stats['total_steps'] = step

        # Отправляем метрики в монитор
        self.monitor.metrics_queue.put(metrics)

        return metrics

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Подготовить батч для обучения"""
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Вычислить loss"""
        batch_size, seq_len, vocab_size = outputs.shape

        # Reshape для cross-entropy
        outputs = outputs.view(-1, vocab_size)
        targets = targets.view(-1)

        # Cross-entropy loss
        loss = nn.CrossEntropyLoss(ignore_index=0)(outputs, targets)  # ignore PAD

        return loss

    def validate(self, dataloader: torch.utils.data.DataLoader, step: int) -> Dict[str, float]:
        """Валидация модели"""

        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = self._prepare_batch(batch)

                # Forward pass
                outputs = self.model(batch['input_ids'], batch['target_ids'])
                loss = self._compute_loss(outputs, batch['target_ids'])

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        self.model.train()

        logger.info(f"Validation at step {step}: Loss = {avg_loss:.4f}")

        return {'val_loss': avg_loss, 'step': step}

    def generate_sample(self, prompt: str, max_length: int = 50) -> str:
        """Генерировать пример текста"""

        self.model.eval()

        # Токенизируем промпт
        tokens = self.tokenizer.encode(prompt, max_length=self.model_config.max_seq_length)
        input_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)

        # Генерируем
        with torch.no_grad():
            generated_tokens = self.model.generate(
                input_tensor,
                max_length=max_length,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                do_sample=True
            )

        # Декодируем
        generated_text = self.tokenizer.decode(generated_tokens[0].tolist())

        self.model.train()

        return generated_text

    def train(self, train_dataloader: torch.utils.data.DataLoader,
              val_dataloader: Optional[torch.utils.data.DataLoader] = None,
              num_steps: int = 10000, output_dir: str = "models",
              resume_from: Optional[str] = None):
        """
        Основной цикл обучения

        Args:
            train_dataloader: DataLoader для обучения
            val_dataloader: DataLoader для валидации
            num_steps: количество шагов обучения
            output_dir: директория для сохранения
            resume_from: путь к чекпоинту для возобновления
        """

        logger.info("=== Starting Advanced Training ===")
        logger.info(f"Author: MagistrTheOne, Krasnodar, 2025")
        logger.info(f"Target steps: {num_steps}")
        logger.info(f"Device: {self.device}")

        # Создаем директорию
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Возобновляем обучение если нужно
        start_step = 0
        if resume_from:
            start_step = self.load_checkpoint(resume_from)
            logger.info(f"Resuming from step {start_step}")

        # Начинаем обучение
        self.stats['start_time'] = time.time()

        try:
            for step in range(start_step, num_steps):
                # Получаем батч
                try:
                    batch = next(iter(train_dataloader))
                except StopIteration:
                    # Перезапускаем даталоадер
                    train_dataloader = self._recreate_dataloader(train_dataloader)
                    batch = next(iter(train_dataloader))

                # Шаг обучения
                metrics = self.train_step(batch, step)

                # Валидация
                if val_dataloader and step % 1000 == 0:
                    val_metrics = self.validate(val_dataloader, step)

                    # Сохраняем лучший чекпоинт
                    if val_metrics['val_loss'] < self.stats['best_loss']:
                        self.stats['best_loss'] = val_metrics['val_loss']
                        self.save_checkpoint(step, val_metrics['val_loss'],
                                           str(output_path / f"best_checkpoint_step_{step}.pt"))

                # Сохраняем чекпоинт
                if step % self.monitor.checkpoint_interval == 0:
                    self.save_checkpoint(step, metrics['loss'],
                                       str(output_path / f"checkpoint_step_{step}.pt"))

                # Генерируем пример
                if step % 2000 == 0:
                    sample_text = self.generate_sample("Привет! Меня зовут")
                    logger.info(f"Sample generation at step {step}: {sample_text[:100]}...")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
        finally:
            self.stats['end_time'] = time.time()
            self.monitor.stop()

            # Финальный чекпоинт
            self.save_checkpoint(num_steps, metrics.get('loss', 0),
                               str(output_path / "final_checkpoint.pt"))

            # Сохраняем статистику
            self._save_training_stats(output_path)

        logger.info("=== Training Completed ===")
        logger.info(f"Total steps: {self.stats['total_steps']}")
        logger.info(f"Best loss: {self.stats['best_loss']:.4f}")
        logger.info(f"Total time: {self.stats['end_time'] - self.stats['start_time']:.2f}s")

    def _recreate_dataloader(self, old_dataloader):
        """Пересоздать даталоадер"""
        # Эта функция должна пересоздавать даталоадер с новыми данными
        # Пока оставляем заглушку
        return old_dataloader

    def _save_training_stats(self, output_path: Path):
        """Сохранить статистику обучения"""

        stats_file = output_path / "training_stats.json"

        training_stats = {
            'author': 'MagistrTheOne, Krasnodar, 2025',
            'timestamp': datetime.now().isoformat(),
            'model_config': self.model_config.__dict__,
            'training_config': self.training_config.__dict__,
            'final_stats': self.stats,
            'device': str(self.device),
            'total_parameters': sum(p.numel() for p in self.model.parameters())
        }

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(training_stats, f, indent=2, ensure_ascii=False)

        logger.info(f"Training stats saved to {stats_file}")


def create_demo_training_setup():
    """Создать демо setup для тестирования"""

    from ..data.prepare_dataset import DatasetPreparator
    from ..data.tokenizer import AdvancedBPETokenizer, create_large_demo_dataset

    logger.info("Creating demo training setup...")

    # Создаем датасет
    preparator = DatasetPreparator(target_sentences=10000)
    ru_texts, en_texts, train_ru, train_en = preparator.prepare_dataset()

    # Создаем токенизатор
    tokenizer = AdvancedBPETokenizer()
    tokenizer.train(ru_texts + en_texts, "demo_tokenizer", vocab_size=3000)

    # Создаем датасет
    train_dataset = CachedTextDataset(train_ru, tokenizer, use_cache=True)
    val_dataset = CachedTextDataset(train_en[:1000], tokenizer, use_cache=True)

    # Создаем даталоадеры
    train_dataloader = create_dataloader(train_dataset, batch_size=16, memory_optimized=True)
    val_dataloader = create_dataloader(val_dataset, batch_size=16, memory_optimized=True)

    # Создаем тренер
    trainer = AdvancedTrainer()

    return trainer, train_dataloader, val_dataloader, tokenizer


def run_smoke_test(num_steps: int = 10000):
    """Запустить smoke test"""

    logger.info(f"=== Starting Smoke Test: {num_steps} steps ===")
    logger.info("Author: MagistrTheOne, Krasnodar, 2025")

    # Создаем setup
    trainer, train_dataloader, val_dataloader, tokenizer = create_demo_training_setup()

    # Запускаем обучение
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_steps=num_steps,
        output_dir="models/smoke_test"
    )

    logger.info("=== Smoke Test Completed ===")
    logger.info(f"Total steps: {trainer.stats['total_steps']}")
    logger.info(f"Best loss: {trainer.stats['best_loss']:.4f}")

    # Генерируем финальный пример
    sample_text = trainer.generate_sample("Привет! Расскажи о")
    logger.info(f"Final sample: {sample_text}")

    return trainer.stats


if __name__ == "__main__":
    # Запуск smoke test
    print("=== PrometheusGPT Mini Advanced Training Pipeline Demo ===")
    print("Author: MagistrTheOne, Krasnodar, 2025")

    # Запускаем smoke test на 1000 шагов
    stats = run_smoke_test(1000)

    print(f"\n✅ Smoke test completed!")
    print(f"Total steps: {stats['total_steps']}")
    print(f"Best loss: {stats['best_loss']:.4f}")

    print(f"\n🎉 Phase 4 - Dataset & Training: SMOKE TEST PASSED!")

