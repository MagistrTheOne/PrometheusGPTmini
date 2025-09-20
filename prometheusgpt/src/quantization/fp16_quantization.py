"""
PrometheusGPT Mini - FP16 Quantization
Author: MagistrTheOne, Krasnodar, 2025

FP16 квантизация для оптимизации памяти с сохранением качества.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, List
import copy

from .quantization_utils import QuantizationUtils

logger = logging.getLogger(__name__)


class FP16Quantizer:
    """FP16 квантизатор для PyTorch моделей"""
    
    def __init__(self, device: str = 'cuda'):
        """
        Args:
            device: устройство для квантизации ('cuda' или 'cpu')
        """
        
        self.device = device
        self.quantized_model = None
        self.original_dtype = None
        
        logger.info(f"FP16 quantizer initialized for device: {device}")
    
    def quantize_model(self, model: nn.Module, 
                      preserve_embeddings: bool = True) -> nn.Module:
        """
        Квантизировать модель в FP16
        
        Args:
            model: модель для квантизации
            preserve_embeddings: сохранить embeddings в FP32 для стабильности
        
        Returns:
            FP16 квантизированная модель
        """
        
        logger.info("Starting FP16 quantization...")
        
        # Создаем копию модели
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        
        # Сохраняем оригинальный dtype
        self.original_dtype = next(model_copy.parameters()).dtype
        
        # Квантизируем модель
        quantized_model = self._convert_to_fp16(model_copy, preserve_embeddings)
        
        # Переносим на устройство
        if self.device == 'cuda' and torch.cuda.is_available():
            quantized_model = quantized_model.cuda()
        
        self.quantized_model = quantized_model
        
        logger.info("FP16 quantization completed")
        
        return quantized_model
    
    def _convert_to_fp16(self, model: nn.Module, 
                        preserve_embeddings: bool = True) -> nn.Module:
        """Конвертировать модель в FP16"""
        
        def convert_module(module):
            """Рекурсивно конвертировать модуль"""
            
            for name, child in module.named_children():
                # Пропускаем embedding слои если нужно
                if preserve_embeddings and isinstance(child, nn.Embedding):
                    logger.debug(f"Preserving {name} as {child.weight.dtype}")
                    continue
                
                # Конвертируем в FP16
                if hasattr(child, 'weight') and child.weight is not None:
                    child.weight.data = child.weight.data.half()
                
                if hasattr(child, 'bias') and child.bias is not None:
                    child.bias.data = child.bias.data.half()
                
                # Рекурсивно обрабатываем дочерние модули
                convert_module(child)
        
        # Конвертируем модель
        convert_module(model)
        
        # Устанавливаем dtype для модели
        model = model.half()
        
        return model
    
    def quantize_with_autocast(self, model: nn.Module) -> nn.Module:
        """
        Квантизировать модель с использованием autocast (рекомендуется)
        
        Args:
            model: модель для квантизации
        
        Returns:
            Модель с поддержкой FP16 через autocast
        """
        
        logger.info("Setting up FP16 quantization with autocast...")
        
        # Создаем копию модели
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        
        # Создаем wrapper для autocast
        class FP16ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, *args, **kwargs):
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    return self.model(*args, **kwargs)
        
        wrapped_model = FP16ModelWrapper(model_copy)
        
        # Переносим на устройство
        if self.device == 'cuda' and torch.cuda.is_available():
            wrapped_model = wrapped_model.cuda()
        
        self.quantized_model = wrapped_model
        
        logger.info("FP16 quantization with autocast completed")
        
        return wrapped_model
    
    def evaluate_quantization(self, original_model: nn.Module,
                            test_input: torch.Tensor,
                            tolerance: float = 1e-2) -> Dict[str, Any]:
        """Оценить качество FP16 квантизации"""
        
        if self.quantized_model is None:
            raise ValueError("No quantized model available. Run quantization first.")
        
        # Переносим input на правильное устройство и dtype
        if self.device == 'cuda' and torch.cuda.is_available():
            test_input = test_input.cuda()
        
        # Сравниваем модели
        comparison = QuantizationUtils.compare_models(
            original_model, self.quantized_model, test_input, tolerance
        )
        
        # Получаем информацию о размерах
        original_size = QuantizationUtils.get_model_size_mb(original_model)
        quantized_size = QuantizationUtils.get_model_size_mb(self.quantized_model)
        
        # Вычисляем экономию
        savings = QuantizationUtils.calculate_quantization_savings(
            original_size, quantized_size
        )
        
        return {
            'quantization_type': 'FP16',
            'device': self.device,
            'model_comparison': comparison,
            'size_savings': savings,
            'quantization_successful': comparison['within_tolerance']
        }
    
    def save_quantized_model(self, filepath: str):
        """Сохранить FP16 квантизированную модель"""
        
        if self.quantized_model is None:
            raise ValueError("No quantized model to save")
        
        # Сохраняем в FP16
        torch.save(self.quantized_model.state_dict(), filepath)
        logger.info(f"FP16 quantized model saved to {filepath}")
    
    def load_quantized_model(self, model_class: nn.Module, filepath: str) -> nn.Module:
        """Загрузить FP16 квантизированную модель"""
        
        # Создаем экземпляр модели
        model = model_class()
        
        # Загружаем веса
        state_dict = torch.load(filepath, map_location='cpu')
        model.load_state_dict(state_dict)
        
        # Конвертируем в FP16
        model = model.half()
        
        # Переносим на устройство
        if self.device == 'cuda' and torch.cuda.is_available():
            model = model.cuda()
        
        self.quantized_model = model
        
        logger.info(f"FP16 quantized model loaded from {filepath}")
        
        return model
    
    def get_quantization_info(self) -> Dict[str, Any]:
        """Получить информацию о квантизации"""
        
        if self.quantized_model is None:
            return {'status': 'no_quantized_model'}
        
        return {
            'quantization_type': 'FP16',
            'device': self.device,
            'model_available': True,
            'model_size_mb': QuantizationUtils.get_model_size_mb(self.quantized_model),
            'original_dtype': str(self.original_dtype) if self.original_dtype else None
        }


class MixedPrecisionQuantizer:
    """Квантизация со смешанной точностью (Mixed Precision)"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.quantized_model = None
        self.scaler = None
    
    def quantize_with_grad_scaler(self, model: nn.Module) -> nn.Module:
        """
        Квантизировать модель с градиентным скейлером для обучения
        
        Args:
            model: модель для квантизации
        
        Returns:
            Модель с поддержкой mixed precision
        """
        
        logger.info("Setting up mixed precision quantization...")
        
        # Создаем копию модели
        model_copy = copy.deepcopy(model)
        
        # Создаем градиентный скейлер
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Создаем wrapper для mixed precision
        class MixedPrecisionWrapper(nn.Module):
            def __init__(self, model, scaler):
                super().__init__()
                self.model = model
                self.scaler = scaler
            
            def forward(self, *args, **kwargs):
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    return self.model(*args, **kwargs)
            
            def training_step(self, loss, optimizer):
                """Шаг обучения с mixed precision"""
                # Backward pass с градиентным скейлером
                self.scaler.scale(loss).backward()
                
                # Обновляем веса
                self.scaler.step(optimizer)
                self.scaler.update()
                
                # Очищаем градиенты
                optimizer.zero_grad()
        
        wrapped_model = MixedPrecisionWrapper(model_copy, self.scaler)
        
        # Переносим на устройство
        if self.device == 'cuda' and torch.cuda.is_available():
            wrapped_model = wrapped_model.cuda()
        
        self.quantized_model = wrapped_model
        
        logger.info("Mixed precision quantization completed")
        
        return wrapped_model
    
    def get_scaler(self):
        """Получить градиентный скейлер"""
        
        return self.scaler
