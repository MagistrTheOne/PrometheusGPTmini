"""
PrometheusGPT Mini - Full Dataset Training Script
Author: MagistrTheOne, Krasnodar, 2025

Production-ready training script для полного датасета (1-2M предложений) с multi-GPU поддержкой.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
import time
import logging
import json
import argparse
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from datetime import datetime
import psutil
import threading
import queue
import signal
import numpy as np
from tqdm import tqdm

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model import PrometheusGPTMini, model_config, training_config
from src.data.tokenizer import AdvancedBPETokenizer
from src.data.prepare_dataset import DatasetPreparator
from src.data.dataloader import CachedTextDataset, create_dataloader, get_memory_usage, StreamingDataLoader
from src.train.train_pipeline import TrainingMonitor

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FullDatasetTrainer:
    """Production trainer для полного датасета с multi-GPU поддержкой"""

    def __init__(self, config: Dict[str, Any], rank: int = 0, world_size: int = 1):
        """
        Args:
            config: конфигурация обучения
            rank: ранг процесса (для DDP)
            world_size: общее количество процессов
        """
        
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        
        # Создаем директории
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.logs_dir = Path(config.get('logs_dir', 'logs'))
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.logs_dir.mkdir(exist_ok=True, parents=True)
        
        # Инициализация компонентов
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.tokenizer = None
        self.train_loader = None
        self.val_loader = None
        self.monitor = None
        
        # Метрики
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_start_time = None
        
        # Настройка DDP
        if world_size > 1:
            self._setup_ddp()
    
    def _setup_ddp(self):
        """Настройка Distributed Data Parallel"""
        
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', rank=self.rank, world_size=self.world_size)
        
        torch.cuda.set_device(self.rank)
        logger.info(f"DDP setup: rank={self.rank}, world_size={self.world_size}")
    
    def _setup_model(self):
        """Инициализация модели"""
        
        logger.info("Setting up model...")
        
        # Создаем модель
        self.model = PrometheusGPTMini(
            vocab_size=self.config['vocab_size'],
            d_model=self.config['d_model'],
            n_layers=self.config['n_layers'],
            n_heads=self.config['n_heads'],
            d_ff=self.config['d_ff'],
            max_seq_length=self.config['max_seq_length'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        # DDP wrapper
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.rank])
        
        # Подсчет параметров
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        logger.info(f"Model size: ~{total_params / 1_000_000:.1f}M parameters")
    
    def _setup_optimizer(self):
        """Инициализация оптимизатора и scheduler"""
        
        logger.info("Setting up optimizer...")
        
        # Оптимизатор с weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 0.01),
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        # Learning rate scheduler с warmup
        total_steps = self.config['num_epochs'] * len(self.train_loader)
        warmup_steps = self.config.get('warmup_steps', min(1000, total_steps // 10))
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Mixed precision scaler
        self.scaler = GradScaler(enabled=self.config.get('use_mixed_precision', True))
        
        logger.info(f"Optimizer: AdamW, LR={self.config['learning_rate']}, warmup={warmup_steps}")
    
    def _setup_tokenizer(self):
        """Инициализация токенизатора"""
        
        logger.info("Setting up tokenizer...")
        
        self.tokenizer = AdvancedBPETokenizer()
        
        # Загружаем или создаем токенизатор
        tokenizer_path = Path(self.config.get('tokenizer_path', 'tokenizer'))
        if tokenizer_path.exists():
            self.tokenizer.load(str(tokenizer_path))
            logger.info(f"Loaded tokenizer from {tokenizer_path}")
        else:
            logger.warning(f"Tokenizer not found at {tokenizer_path}, using default")
    
    def _setup_data(self):
        """Инициализация данных"""
        
        logger.info("Setting up data loaders...")
        
        # Пути к данным
        data_dir = Path(self.config.get('data_dir', 'data'))
        train_files = [
            data_dir / 'train_ru.txt',
            data_dir / 'train_en.txt'
        ]
        val_files = [
            data_dir / 'val_ru.txt',
            data_dir / 'val_en.txt'
        ]
        
        # Проверяем существование файлов
        missing_files = [f for f in train_files + val_files if not f.exists()]
        if missing_files:
            logger.warning(f"Missing data files: {missing_files}")
            logger.info("Creating demo dataset...")
            self._create_demo_dataset(data_dir)
        
        # Создаем streaming dataloader для больших данных
        if self.config.get('use_streaming', True):
            self.train_loader = StreamingDataLoader(
                file_paths=[str(f) for f in train_files if f.exists()],
                tokenizer=self.tokenizer,
                batch_size=self.config['batch_size'],
                max_length=self.config['max_seq_length'],
                buffer_size=self.config.get('buffer_size', 100000)
            )
            
            self.val_loader = StreamingDataLoader(
                file_paths=[str(f) for f in val_files if f.exists()],
                tokenizer=self.tokenizer,
                batch_size=self.config['batch_size'],
                max_length=self.config['max_seq_length'],
                buffer_size=self.config.get('buffer_size', 10000)
            )
        else:
            # Обычный dataloader для меньших данных
            train_texts = self._load_texts(train_files)
            val_texts = self._load_texts(val_files)
            
            train_dataset = CachedTextDataset(
                train_texts, self.tokenizer, 
                max_length=self.config['max_seq_length'],
                use_cache=True
            )
            val_dataset = CachedTextDataset(
                val_texts, self.tokenizer,
                max_length=self.config['max_seq_length'],
                use_cache=True
            )
            
            self.train_loader = create_dataloader(
                train_dataset, 
                batch_size=self.config['batch_size'],
                shuffle=True,
                memory_optimized=True
            )
            self.val_loader = create_dataloader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                memory_optimized=True
            )
        
        logger.info(f"Data loaders created: train={len(self.train_loader) if hasattr(self.train_loader, '__len__') else 'streaming'}")
    
    def _create_demo_dataset(self, data_dir: Path):
        """Создать демо датасет если файлы отсутствуют"""
        
        logger.info("Creating demo dataset...")
        
        preparator = DatasetPreparator(
            output_dir=str(data_dir),
            target_sentences=self.config.get('demo_sentences', 100000)
        )
        
        ru_texts, en_texts, train_ru, train_en = preparator.prepare_dataset()
        logger.info(f"Created demo dataset: {len(ru_texts)} total, {len(train_ru)} train")
    
    def _load_texts(self, file_paths: List[Path]) -> List[str]:
        """Загрузить тексты из файлов"""
        
        texts = []
        for file_path in file_paths:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts.extend([line.strip() for line in f if line.strip()])
        
        return texts
    
    def _setup_monitoring(self):
        """Инициализация мониторинга"""
        
        if self.rank == 0:  # Только главный процесс
            self.monitor = TrainingMonitor(
                log_interval=self.config.get('log_interval', 10),
                checkpoint_interval=self.config.get('checkpoint_interval', 1000)
            )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Один шаг обучения"""
        
        self.model.train()
        
        # Переносим данные на устройство
        input_ids = batch['input_ids'].to(self.device, non_blocking=True)
        target_ids = batch['target_ids'].to(self.device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
        
        # Forward pass с mixed precision
        with autocast(enabled=self.config.get('use_mixed_precision', True)):
            outputs = self.model(input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.special_tokens['<pad>'])(
                outputs.view(-1, outputs.size(-1)), target_ids.view(-1)
            )
        
        # Backward pass
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        if self.config.get('max_grad_norm', 1.0) > 0:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        
        # Метрики
        current_lr = self.optimizer.param_groups[0]['lr']
        
        return {
            'loss': loss.item(),
            'learning_rate': current_lr,
            'step_time': time.time() - getattr(self, '_step_start_time', time.time())
        }
    
    def validate(self) -> Dict[str, float]:
        """Валидация модели"""
        
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            val_batches = 0
            for batch in self.val_loader:
                if val_batches >= self.config.get('max_val_batches', 100):
                    break
                
                # Переносим данные
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                target_ids = batch['target_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                
                # Forward pass
                with autocast(enabled=self.config.get('use_mixed_precision', True)):
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.special_tokens['<pad>'])(
                        outputs.view(-1, outputs.size(-1)), target_ids.view(-1)
                    )
                
                # Подсчет метрик
                total_loss += loss.item()
                total_tokens += (target_ids != self.tokenizer.special_tokens['<pad>']).sum().item()
                val_batches += 1
        
        avg_loss = total_loss / max(val_batches, 1)
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'val_loss': avg_loss,
            'val_ppl': ppl,
            'val_tokens': total_tokens
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Сохранить чекпоинт"""
        
        if self.rank != 0:  # Только главный процесс
            return
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Сохраняем обычный чекпоинт
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.epoch}_step_{self.global_step}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Сохраняем лучший чекпоинт
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved: val_loss={self.best_val_loss:.4f}")
        
        # Сохраняем последний чекпоинт
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Загрузить чекпоинт"""
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Загружаем состояние модели
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Загружаем состояние оптимизатора
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Восстанавливаем метрики
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Checkpoint loaded: epoch={self.epoch}, step={self.global_step}, best_val_loss={self.best_val_loss:.4f}")
    
    def train(self):
        """Основной цикл обучения"""
        
        logger.info("=== Starting Full Dataset Training ===")
        logger.info(f"Author: MagistrTheOne, Krasnodar, 2025")
        logger.info(f"Device: {self.device}, Rank: {self.rank}/{self.world_size}")
        logger.info(f"Config: {self.config}")
        
        # Инициализация компонентов
        self._setup_model()
        self._setup_optimizer()
        self._setup_tokenizer()
        self._setup_data()
        self._setup_monitoring()
        
        # Загрузка чекпоинта если указан
        if self.config.get('resume_from'):
            self.load_checkpoint(self.config['resume_from'])
        
        self.training_start_time = time.time()
        
        # Основной цикл обучения
        for epoch in range(self.epoch, self.config['num_epochs']):
            self.epoch = epoch
            
            logger.info(f"Starting epoch {epoch + 1}/{self.config['num_epochs']}")
            
            # Обучение
            epoch_loss = 0.0
            epoch_steps = 0
            
            for batch_idx, batch in enumerate(self.train_loader):
                self._step_start_time = time.time()
                
                # Шаг обучения
                metrics = self.train_step(batch)
                epoch_loss += metrics['loss']
                epoch_steps += 1
                self.global_step += 1
                
                # Логирование
                if self.rank == 0 and self.global_step % self.config.get('log_interval', 10) == 0:
                    memory = get_memory_usage()
                    logger.info(
                        f"Epoch {epoch+1}, Step {self.global_step} | "
                        f"Loss: {metrics['loss']:.4f} | LR: {metrics['learning_rate']:.6f} | "
                        f"Time: {metrics['step_time']:.3f}s | "
                        f"GPU: {memory.get('gpu_allocated_gb', 0):.2f}GB"
                    )
                
                # Валидация
                if self.global_step % self.config.get('val_interval', 1000) == 0:
                    val_metrics = self.validate()
                    
                    if self.rank == 0:
                        logger.info(
                            f"Validation | Loss: {val_metrics['val_loss']:.4f} | "
                            f"PPL: {val_metrics['val_ppl']:.2f} | Tokens: {val_metrics['val_tokens']}"
                        )
                        
                        # Сохраняем лучшую модель
                        is_best = val_metrics['val_loss'] < self.best_val_loss
                        if is_best:
                            self.best_val_loss = val_metrics['val_loss']
                        
                        self.save_checkpoint(is_best=is_best)
                
                # Чекпоинт
                if self.global_step % self.config.get('checkpoint_interval', 1000) == 0:
                    if self.rank == 0:
                        self.save_checkpoint()
            
            # Логирование эпохи
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            logger.info(f"Epoch {epoch+1} completed | Avg Loss: {avg_epoch_loss:.4f}")
        
        # Финальное сохранение
        if self.rank == 0:
            self.save_checkpoint()
            training_time = time.time() - self.training_start_time
            logger.info(f"Training completed in {training_time/3600:.2f} hours")
        
        # Очистка DDP
        if self.world_size > 1:
            dist.destroy_process_group()


def setup_distributed_training(rank: int, world_size: int, config: Dict[str, Any]):
    """Настройка распределенного обучения"""
    
    trainer = FullDatasetTrainer(config, rank, world_size)
    trainer.train()


def main():
    """Главная функция"""
    
    parser = argparse.ArgumentParser(description='PrometheusGPT Mini Full Dataset Training')
    
    # Основные параметры
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--logs_dir', type=str, default='logs', help='Logs directory')
    parser.add_argument('--tokenizer_path', type=str, default='tokenizer', help='Tokenizer path')
    
    # Параметры модели
    parser.add_argument('--vocab_size', type=int, default=30000, help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feed-forward dimension')
    parser.add_argument('--max_seq_length', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Параметры обучения
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    
    # Опции
    parser.add_argument('--use_mixed_precision', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--use_streaming', action='store_true', default=True, help='Use streaming dataloader')
    parser.add_argument('--resume_from', type=str, help='Resume from checkpoint')
    
    # Интервалы
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--val_interval', type=int, default=1000, help='Validation interval')
    parser.add_argument('--checkpoint_interval', type=int, default=1000, help='Checkpoint interval')
    
    # Multi-GPU
    parser.add_argument('--world_size', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--rank', type=int, default=0, help='Process rank')
    
    args = parser.parse_args()
    
    # Создаем конфигурацию
    config = vars(args)
    
    # Multi-GPU обучение
    if args.world_size > 1:
        mp.spawn(
            setup_distributed_training,
            args=(args.world_size, config),
            nprocs=args.world_size,
            join=True
        )
    else:
        # Одно-GPU обучение
        trainer = FullDatasetTrainer(config)
        trainer.train()


if __name__ == "__main__":
    main()
