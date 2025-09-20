"""
PrometheusGPT Mini - Quantization Utilities
Author: MagistrTheOne, Krasnodar, 2025

Утилиты для квантизации моделей.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)


class QuantizationUtils:
    """Утилиты для квантизации"""
    
    @staticmethod
    def get_model_size_mb(model: nn.Module) -> float:
        """Получить размер модели в MB"""
        
        total_params = sum(p.numel() for p in model.parameters())
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        
        total_size = param_size + buffer_size
        return total_size / 1024 / 1024
    
    @staticmethod
    def get_model_size_info(model: nn.Module) -> Dict[str, Any]:
        """Получить подробную информацию о размере модели"""
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Размеры по типам параметров
        param_sizes = {}
        for name, param in model.named_parameters():
            param_type = str(param.dtype)
            if param_type not in param_sizes:
                param_sizes[param_type] = 0
            param_sizes[param_type] += param.numel() * param.element_size()
        
        # Общий размер
        total_size_bytes = sum(param_sizes.values())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'total_size_mb': total_size_bytes / 1024 / 1024,
            'buffer_size_mb': buffer_size / 1024 / 1024,
            'param_sizes_by_type': {k: v / 1024 / 1024 for k, v in param_sizes.items()},
            'memory_efficiency': trainable_params / total_params if total_params > 0 else 0
        }
    
    @staticmethod
    def calculate_quantization_savings(original_size_mb: float, 
                                     quantized_size_mb: float) -> Dict[str, float]:
        """Вычислить экономию от квантизации"""
        
        savings_mb = original_size_mb - quantized_size_mb
        savings_percent = (savings_mb / original_size_mb) * 100 if original_size_mb > 0 else 0
        compression_ratio = original_size_mb / quantized_size_mb if quantized_size_mb > 0 else 1
        
        return {
            'original_size_mb': original_size_mb,
            'quantized_size_mb': quantized_size_mb,
            'savings_mb': savings_mb,
            'savings_percent': savings_percent,
            'compression_ratio': compression_ratio
        }
    
    @staticmethod
    def compare_models(model1: nn.Module, model2: nn.Module, 
                      input_tensor: torch.Tensor, 
                      tolerance: float = 1e-3) -> Dict[str, Any]:
        """Сравнить две модели"""
        
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            output1 = model1(input_tensor)
            output2 = model2(input_tensor)
        
        # Вычисляем различия
        mse = torch.mean((output1 - output2) ** 2).item()
        mae = torch.mean(torch.abs(output1 - output2)).item()
        max_diff = torch.max(torch.abs(output1 - output2)).item()
        
        # Корреляция
        correlation = torch.corrcoef(torch.stack([
            output1.flatten(), output2.flatten()
        ]))[0, 1].item()
        
        # Размеры моделей
        size1 = QuantizationUtils.get_model_size_mb(model1)
        size2 = QuantizationUtils.get_model_size_mb(model2)
        
        return {
            'mse': mse,
            'mae': mae,
            'max_difference': max_diff,
            'correlation': correlation,
            'within_tolerance': mae < tolerance,
            'model1_size_mb': size1,
            'model2_size_mb': size2,
            'size_reduction_mb': size1 - size2,
            'size_reduction_percent': ((size1 - size2) / size1) * 100 if size1 > 0 else 0
        }
    
    @staticmethod
    def get_layer_statistics(model: nn.Module) -> Dict[str, Any]:
        """Получить статистики по слоям модели"""
        
        layer_stats = {}
        
        for name, module in model.named_modules():
            if len(list(module.parameters())) > 0:
                params = list(module.parameters())
                total_params = sum(p.numel() for p in params)
                
                if total_params > 0:
                    # Статистики весов
                    weights = [p for p in params if p.dim() > 1]
                    if weights:
                        all_weights = torch.cat([w.flatten() for w in weights])
                        
                        layer_stats[name] = {
                            'total_parameters': total_params,
                            'weight_mean': all_weights.mean().item(),
                            'weight_std': all_weights.std().item(),
                            'weight_min': all_weights.min().item(),
                            'weight_max': all_weights.max().item(),
                            'weight_abs_mean': all_weights.abs().mean().item(),
                            'weight_abs_max': all_weights.abs().max().item(),
                            'module_type': type(module).__name__
                        }
        
        return layer_stats
    
    @staticmethod
    def find_quantization_candidates(model: nn.Module, 
                                   min_params: int = 1000) -> List[str]:
        """Найти слои-кандидаты для квантизации"""
        
        candidates = []
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                params = sum(p.numel() for p in module.parameters())
                if params >= min_params:
                    candidates.append(name)
        
        return candidates
    
    @staticmethod
    def create_quantization_report(original_model: nn.Module,
                                 quantized_model: nn.Module,
                                 test_input: torch.Tensor,
                                 tolerance: float = 1e-3) -> Dict[str, Any]:
        """Создать отчет о квантизации"""
        
        # Сравнение моделей
        comparison = QuantizationUtils.compare_models(
            original_model, quantized_model, test_input, tolerance
        )
        
        # Статистики слоев
        original_stats = QuantizationUtils.get_layer_statistics(original_model)
        quantized_stats = QuantizationUtils.get_layer_statistics(quantized_model)
        
        # Информация о размерах
        original_size_info = QuantizationUtils.get_model_size_info(original_model)
        quantized_size_info = QuantizationUtils.get_model_size_info(quantized_model)
        
        # Экономия
        savings = QuantizationUtils.calculate_quantization_savings(
            original_size_info['total_size_mb'],
            quantized_size_info['total_size_mb']
        )
        
        return {
            'quantization_successful': comparison['within_tolerance'],
            'model_comparison': comparison,
            'size_analysis': {
                'original': original_size_info,
                'quantized': quantized_size_info,
                'savings': savings
            },
            'layer_statistics': {
                'original': original_stats,
                'quantized': quantized_stats
            },
            'recommendations': QuantizationUtils._generate_recommendations(
                comparison, savings, original_stats, quantized_stats
            )
        }
    
    @staticmethod
    def _generate_recommendations(comparison: Dict[str, Any],
                                savings: Dict[str, Any],
                                original_stats: Dict[str, Any],
                                quantized_stats: Dict[str, Any]) -> List[str]:
        """Генерировать рекомендации по квантизации"""
        
        recommendations = []
        
        # Анализ качества
        if comparison['correlation'] > 0.99:
            recommendations.append("Excellent quantization quality - correlation > 0.99")
        elif comparison['correlation'] > 0.95:
            recommendations.append("Good quantization quality - correlation > 0.95")
        elif comparison['correlation'] > 0.90:
            recommendations.append("Acceptable quantization quality - correlation > 0.90")
        else:
            recommendations.append("Poor quantization quality - consider different approach")
        
        # Анализ экономии
        if savings['savings_percent'] > 50:
            recommendations.append(f"Excellent memory savings: {savings['savings_percent']:.1f}%")
        elif savings['savings_percent'] > 25:
            recommendations.append(f"Good memory savings: {savings['savings_percent']:.1f}%")
        elif savings['savings_percent'] > 10:
            recommendations.append(f"Moderate memory savings: {savings['savings_percent']:.1f}%")
        else:
            recommendations.append("Limited memory savings - consider if quantization is worth it")
        
        # Анализ точности
        if comparison['mae'] < 1e-4:
            recommendations.append("Very high precision maintained")
        elif comparison['mae'] < 1e-3:
            recommendations.append("High precision maintained")
        elif comparison['mae'] < 1e-2:
            recommendations.append("Acceptable precision loss")
        else:
            recommendations.append("Significant precision loss - monitor model performance")
        
        # Рекомендации по использованию
        if (comparison['correlation'] > 0.95 and 
            savings['savings_percent'] > 25 and 
            comparison['mae'] < 1e-3):
            recommendations.append("Recommended for production use")
        elif (comparison['correlation'] > 0.90 and 
              savings['savings_percent'] > 15):
            recommendations.append("Suitable for development and testing")
        else:
            recommendations.append("Consider alternative optimization methods")
        
        return recommendations
