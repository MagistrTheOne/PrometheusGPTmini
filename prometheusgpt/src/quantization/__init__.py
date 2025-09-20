"""
PrometheusGPT Mini - Quantization Module
Author: MagistrTheOne, Krasnodar, 2025

Модуль для квантизации модели: INT8, FP16, динамическая квантизация.
"""

from .quantization_utils import QuantizationUtils
from .int8_quantization import INT8Quantizer
from .fp16_quantization import FP16Quantizer
from .dynamic_quantization import DynamicQuantizer

__all__ = [
    'QuantizationUtils',
    'INT8Quantizer',
    'FP16Quantizer', 
    'DynamicQuantizer'
]
