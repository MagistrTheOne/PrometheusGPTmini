"""
PrometheusGPT Mini - Dynamic Quantization
Author: MagistrTheOne, Krasnodar, 2025

Динамическая квантизация для runtime оптимизации.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, List, Union
import copy

from .quantization_utils import QuantizationUtils

logger = logging.getLogger(__name__)


class DynamicQuantizer:
    """Динамический квантизатор для runtime оптимизации"""
    
    def __init__(self, device: str = 'cuda'):
        """
        Args:
            device: устройство для квантизации
        """
        
        self.device = device
        self.quantized_model = None
        self.quantization_config = None
        
        logger.info(f"Dynamic quantizer initialized for device: {device}")
    
    def quantize_model(self, model: nn.Module,
                      target_modules: Optional[List[str]] = None,
                      quantization_dtype: torch.dtype = torch.qint8) -> nn.Module:
        """
        Динамически квантизировать модель
        
        Args:
            model: модель для квантизации
            target_modules: список модулей для квантизации
            quantization_dtype: тип квантизации
        
        Returns:
            Динамически квантизированная модель
        """
        
        logger.info("Starting dynamic quantization...")
        
        # Создаем копию модели
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        
        # Определяем модули для квантизации
        if target_modules is None:
            target_modules = self._get_default_target_modules(model_copy)
        
        # Создаем конфигурацию квантизации
        qconfig_spec = self._create_qconfig_spec(target_modules, quantization_dtype)
        
        # Применяем динамическую квантизацию
        quantized_model = torch.quantization.quantize_dynamic(
            model_copy, qconfig_spec, dtype=quantization_dtype
        )
        
        # Переносим на устройство
        if self.device == 'cuda' and torch.cuda.is_available():
            quantized_model = quantized_model.cuda()
        
        self.quantized_model = quantized_model
        self.quantization_config = {
            'target_modules': target_modules,
            'quantization_dtype': quantization_dtype
        }
        
        logger.info("Dynamic quantization completed")
        
        return quantized_model
    
    def _get_default_target_modules(self, model: nn.Module) -> List[str]:
        """Получить список модулей по умолчанию для квантизации"""
        
        target_modules = []
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                target_modules.append(name)
        
        logger.info(f"Found {len(target_modules)} modules for quantization")
        
        return target_modules
    
    def _create_qconfig_spec(self, target_modules: List[str], 
                           quantization_dtype: torch.dtype) -> Dict:
        """Создать спецификацию квантизации"""
        
        # Создаем mapping модулей для квантизации
        qconfig_spec = {}
        
        for module_name in target_modules:
            # Находим тип модуля
            module_type = self._get_module_type_by_name(module_name)
            if module_type:
                qconfig_spec[module_type] = torch.quantization.default_dynamic_qconfig
        
        return qconfig_spec
    
    def _get_module_type_by_name(self, module_name: str) -> Optional[type]:
        """Получить тип модуля по имени"""
        
        # Простое mapping для основных типов модулей
        module_type_mapping = {
            'linear': nn.Linear,
            'conv1d': nn.Conv1d,
            'conv2d': nn.Conv2d,
            'embedding': nn.Embedding
        }
        
        for key, module_type in module_type_mapping.items():
            if key in module_name.lower():
                return module_type
        
        return None
    
    def quantize_selective(self, model: nn.Module,
                         layer_patterns: List[str],
                         quantization_dtype: torch.dtype = torch.qint8) -> nn.Module:
        """
        Селективная квантизация по паттернам слоев
        
        Args:
            model: модель для квантизации
            layer_patterns: паттерны имен слоев для квантизации
            quantization_dtype: тип квантизации
        
        Returns:
            Селективно квантизированная модель
        """
        
        logger.info(f"Starting selective quantization with patterns: {layer_patterns}")
        
        # Создаем копию модели
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        
        # Находим модули по паттернам
        target_modules = self._find_modules_by_patterns(model_copy, layer_patterns)
        
        # Квантизируем
        return self.quantize_model(model_copy, target_modules, quantization_dtype)
    
    def _find_modules_by_patterns(self, model: nn.Module, 
                                patterns: List[str]) -> List[str]:
        """Найти модули по паттернам имен"""
        
        target_modules = []
        
        for name, module in model.named_modules():
            for pattern in patterns:
                if pattern in name:
                    target_modules.append(name)
                    break
        
        logger.info(f"Found {len(target_modules)} modules matching patterns")
        
        return target_modules
    
    def quantize_by_size(self, model: nn.Module,
                        min_params: int = 1000,
                        quantization_dtype: torch.dtype = torch.qint8) -> nn.Module:
        """
        Квантизировать модули по размеру (количеству параметров)
        
        Args:
            model: модель для квантизации
            min_params: минимальное количество параметров для квантизации
            quantization_dtype: тип квантизации
        
        Returns:
            Квантизированная модель
        """
        
        logger.info(f"Starting size-based quantization (min_params={min_params})")
        
        # Создаем копию модели
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        
        # Находим большие модули
        large_modules = []
        
        for name, module in model_copy.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                params = sum(p.numel() for p in module.parameters())
                if params >= min_params:
                    large_modules.append(name)
        
        logger.info(f"Found {len(large_modules)} large modules for quantization")
        
        # Квантизируем
        return self.quantize_model(model_copy, large_modules, quantization_dtype)
    
    def evaluate_quantization(self, original_model: nn.Module,
                            test_input: torch.Tensor,
                            tolerance: float = 1e-2) -> Dict[str, Any]:
        """Оценить качество динамической квантизации"""
        
        if self.quantized_model is None:
            raise ValueError("No quantized model available. Run quantization first.")
        
        # Переносим input на правильное устройство
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
            'quantization_type': 'Dynamic',
            'device': self.device,
            'quantization_config': self.quantization_config,
            'model_comparison': comparison,
            'size_savings': savings,
            'quantization_successful': comparison['within_tolerance']
        }
    
    def get_quantization_report(self, original_model: nn.Module,
                              test_input: torch.Tensor) -> Dict[str, Any]:
        """Получить подробный отчет о квантизации"""
        
        if self.quantized_model is None:
            raise ValueError("No quantized model available. Run quantization first.")
        
        # Создаем полный отчет
        report = QuantizationUtils.create_quantization_report(
            original_model, self.quantized_model, test_input
        )
        
        # Добавляем специфичную информацию
        report['quantization_type'] = 'Dynamic'
        report['device'] = self.device
        report['quantization_config'] = self.quantization_config
        
        return report
    
    def save_quantized_model(self, filepath: str):
        """Сохранить динамически квантизированную модель"""
        
        if self.quantized_model is None:
            raise ValueError("No quantized model to save")
        
        # Сохраняем модель и конфигурацию
        save_data = {
            'model_state_dict': self.quantized_model.state_dict(),
            'quantization_config': self.quantization_config,
            'quantization_type': 'Dynamic'
        }
        
        torch.save(save_data, filepath)
        logger.info(f"Dynamic quantized model saved to {filepath}")
    
    def load_quantized_model(self, model_class: nn.Module, filepath: str) -> nn.Module:
        """Загрузить динамически квантизированную модель"""
        
        # Загружаем данные
        save_data = torch.load(filepath, map_location='cpu')
        
        # Создаем экземпляр модели
        model = model_class()
        
        # Загружаем веса
        model.load_state_dict(save_data['model_state_dict'])
        
        # Восстанавливаем конфигурацию
        self.quantization_config = save_data.get('quantization_config')
        
        # Применяем квантизацию
        if self.quantization_config:
            target_modules = self.quantization_config.get('target_modules')
            quantization_dtype = self.quantization_config.get('quantization_dtype', torch.qint8)
            
            model = self.quantize_model(model, target_modules, quantization_dtype)
        else:
            # Fallback к стандартной квантизации
            model = self.quantize_model(model)
        
        logger.info(f"Dynamic quantized model loaded from {filepath}")
        
        return model
    
    def get_quantization_info(self) -> Dict[str, Any]:
        """Получить информацию о квантизации"""
        
        if self.quantized_model is None:
            return {'status': 'no_quantized_model'}
        
        return {
            'quantization_type': 'Dynamic',
            'device': self.device,
            'model_available': True,
            'model_size_mb': QuantizationUtils.get_model_size_mb(self.quantized_model),
            'quantization_config': self.quantization_config
        }
