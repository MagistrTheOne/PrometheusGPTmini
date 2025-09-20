"""
PrometheusGPT Mini - INT8 Quantization
Author: MagistrTheOne, Krasnodar, 2025

INT8 квантизация для оптимизации памяти и ускорения inference.
"""

import torch
import torch.nn as nn
import torch.quantization as quant
import logging
from typing import Dict, Any, Optional, List, Tuple
import copy

from .quantization_utils import QuantizationUtils

logger = logging.getLogger(__name__)


class INT8Quantizer:
    """INT8 квантизатор для PyTorch моделей"""
    
    def __init__(self, backend: str = 'qnnpack'):
        """
        Args:
            backend: бэкенд для квантизации ('qnnpack' для CPU, 'fbgemm' для x86)
        """
        
        self.backend = backend
        self.quantized_model = None
        self.quantization_config = None
        
        # Настройка бэкенда
        if backend == 'qnnpack':
            quant.backend_config.qnnpack.prepare = quant.backend_config.qnnpack.prepare
            quant.backend_config.qnnpack.convert = quant.backend_config.qnnpack.convert
        elif backend == 'fbgemm':
            quant.backend_config.fbgemm.prepare = quant.backend_config.fbgemm.prepare
            quant.backend_config.fbgemm.convert = quant.backend_config.fbgemm.convert
        
        logger.info(f"INT8 quantizer initialized with backend: {backend}")
    
    def quantize_model(self, model: nn.Module, 
                      calibration_data: Optional[List[torch.Tensor]] = None,
                      quantize_embeddings: bool = True,
                      quantize_linear: bool = True,
                      quantize_conv: bool = True) -> nn.Module:
        """
        Квантизировать модель в INT8
        
        Args:
            model: модель для квантизации
            calibration_data: данные для калибровки (опционально)
            quantize_embeddings: квантизировать embedding слои
            quantize_linear: квантизировать linear слои
            quantize_conv: квантизировать conv слои
        
        Returns:
            Квантизированная модель
        """
        
        logger.info("Starting INT8 quantization...")
        
        # Создаем копию модели
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        
        # Настраиваем квантизацию
        self._setup_quantization_config(
            quantize_embeddings, quantize_linear, quantize_conv
        )
        
        # Подготавливаем модель для квантизации
        prepared_model = self._prepare_model(model_copy)
        
        # Калибровка (если есть данные)
        if calibration_data:
            self._calibrate_model(prepared_model, calibration_data)
        
        # Конвертируем в квантизированную модель
        quantized_model = self._convert_model(prepared_model)
        
        self.quantized_model = quantized_model
        
        logger.info("INT8 quantization completed")
        
        return quantized_model
    
    def _setup_quantization_config(self, quantize_embeddings: bool,
                                 quantize_linear: bool, quantize_conv: bool):
        """Настроить конфигурацию квантизации"""
        
        config = {}
        
        if quantize_embeddings:
            config[nn.Embedding] = quant.quantize_dynamic
        
        if quantize_linear:
            config[nn.Linear] = quant.quantize_dynamic
        
        if quantize_conv:
            config[nn.Conv1d] = quant.quantize_dynamic
            config[nn.Conv2d] = quant.quantize_dynamic
        
        self.quantization_config = config
    
    def _prepare_model(self, model: nn.Module) -> nn.Module:
        """Подготовить модель для квантизации"""
        
        # Настраиваем квантизацию
        model.qconfig = quant.get_default_qconfig(self.backend)
        
        # Подготавливаем модель
        prepared_model = quant.prepare(model, inplace=False)
        
        return prepared_model
    
    def _calibrate_model(self, model: nn.Module, calibration_data: List[torch.Tensor]):
        """Калибровать модель на данных"""
        
        logger.info(f"Calibrating model on {len(calibration_data)} samples...")
        
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(calibration_data):
                if isinstance(data, (list, tuple)):
                    model(*data)
                else:
                    model(data)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Calibrated {i + 1}/{len(calibration_data)} samples")
    
    def _convert_model(self, model: nn.Module) -> nn.Module:
        """Конвертировать модель в квантизированную"""
        
        quantized_model = quant.convert(model, inplace=False)
        
        return quantized_model
    
    def quantize_dynamic(self, model: nn.Module, 
                        qconfig_spec: Optional[Dict] = None) -> nn.Module:
        """
        Динамическая квантизация (более простая)
        
        Args:
            model: модель для квантизации
            qconfig_spec: спецификация квантизации
        
        Returns:
            Динамически квантизированная модель
        """
        
        logger.info("Starting dynamic INT8 quantization...")
        
        if qconfig_spec is None:
            # Квантизируем только Linear слои
            qconfig_spec = {nn.Linear}
        
        quantized_model = quant.quantize_dynamic(
            model, qconfig_spec, dtype=torch.qint8
        )
        
        self.quantized_model = quantized_model
        
        logger.info("Dynamic INT8 quantization completed")
        
        return quantized_model
    
    def quantize_static(self, model: nn.Module, 
                       calibration_data: List[torch.Tensor]) -> nn.Module:
        """
        Статическая квантизация (более точная, требует калибровки)
        
        Args:
            model: модель для квантизации
            calibration_data: данные для калибровки
        
        Returns:
            Статически квантизированная модель
        """
        
        logger.info("Starting static INT8 quantization...")
        
        # Настраиваем квантизацию
        model.qconfig = quant.get_default_qconfig(self.backend)
        
        # Подготавливаем модель
        prepared_model = quant.prepare(model, inplace=False)
        
        # Калибруем
        self._calibrate_model(prepared_model, calibration_data)
        
        # Конвертируем
        quantized_model = quant.convert(prepared_model, inplace=False)
        
        self.quantized_model = quantized_model
        
        logger.info("Static INT8 quantization completed")
        
        return quantized_model
    
    def evaluate_quantization(self, original_model: nn.Module,
                            test_input: torch.Tensor,
                            tolerance: float = 1e-3) -> Dict[str, Any]:
        """Оценить качество квантизации"""
        
        if self.quantized_model is None:
            raise ValueError("No quantized model available. Run quantization first.")
        
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
            'quantization_type': 'INT8',
            'backend': self.backend,
            'model_comparison': comparison,
            'size_savings': savings,
            'quantization_successful': comparison['within_tolerance']
        }
    
    def save_quantized_model(self, filepath: str):
        """Сохранить квантизированную модель"""
        
        if self.quantized_model is None:
            raise ValueError("No quantized model to save")
        
        torch.save(self.quantized_model.state_dict(), filepath)
        logger.info(f"Quantized model saved to {filepath}")
    
    def load_quantized_model(self, model_class: nn.Module, filepath: str) -> nn.Module:
        """Загрузить квантизированную модель"""
        
        # Создаем экземпляр модели
        model = model_class()
        
        # Загружаем веса
        state_dict = torch.load(filepath, map_location='cpu')
        model.load_state_dict(state_dict)
        
        self.quantized_model = model
        
        logger.info(f"Quantized model loaded from {filepath}")
        
        return model
    
    def get_quantization_info(self) -> Dict[str, Any]:
        """Получить информацию о квантизации"""
        
        if self.quantized_model is None:
            return {'status': 'no_quantized_model'}
        
        return {
            'quantization_type': 'INT8',
            'backend': self.backend,
            'model_available': True,
            'model_size_mb': QuantizationUtils.get_model_size_mb(self.quantized_model)
        }


class PostTrainingQuantizer:
    """Квантизация после обучения (Post-Training Quantization)"""
    
    def __init__(self, backend: str = 'qnnpack'):
        self.backend = backend
        self.quantizer = INT8Quantizer(backend)
    
    def quantize_with_calibration(self, model: nn.Module,
                                calibration_dataset: List[torch.Tensor],
                                method: str = 'static') -> nn.Module:
        """
        Квантизировать модель с калибровкой
        
        Args:
            model: модель для квантизации
            calibration_dataset: датасет для калибровки
            method: метод квантизации ('static' или 'dynamic')
        
        Returns:
            Квантизированная модель
        """
        
        if method == 'static':
            return self.quantizer.quantize_static(model, calibration_dataset)
        elif method == 'dynamic':
            return self.quantizer.quantize_dynamic(model)
        else:
            raise ValueError(f"Unknown quantization method: {method}")
    
    def create_calibration_dataset(self, dataloader, num_samples: int = 100) -> List[torch.Tensor]:
        """Создать датасет для калибровки"""
        
        calibration_data = []
        
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            
            if isinstance(batch, dict):
                # Если batch - это словарь, берем input_ids
                if 'input_ids' in batch:
                    calibration_data.append(batch['input_ids'])
                else:
                    # Берем первый тензор из batch
                    first_key = list(batch.keys())[0]
                    calibration_data.append(batch[first_key])
            elif isinstance(batch, (list, tuple)):
                calibration_data.append(batch[0])
            else:
                calibration_data.append(batch)
        
        logger.info(f"Created calibration dataset with {len(calibration_data)} samples")
        
        return calibration_data
