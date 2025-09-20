"""
PrometheusGPT Mini - Prometheus Metrics
Author: MagistrTheOne, Krasnodar, 2025

Интеграция с Prometheus для экспорта метрик.
"""

import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """Экспорт метрик в Prometheus"""
    
    def __init__(self, port: int = 8001):
        """
        Args:
            port: порт для Prometheus HTTP сервера
        """
        
        if not PROMETHEUS_AVAILABLE:
            logger.warning("prometheus_client not available, metrics disabled")
            self.enabled = False
            return
        
        self.port = port
        self.enabled = True
        self.server_started = False
        
        # Создаем метрики
        self._create_metrics()
    
    def _create_metrics(self):
        """Создать Prometheus метрики"""
        
        # Информация о модели
        self.model_info = Info('prometheusgpt_model_info', 'Information about the model')
        
        # Счетчики запросов
        self.requests_total = Counter(
            'prometheusgpt_requests_total',
            'Total number of requests',
            ['endpoint', 'status']
        )
        
        self.tokens_generated_total = Counter(
            'prometheusgpt_tokens_generated_total',
            'Total number of tokens generated'
        )
        
        # Гистограммы latency
        self.request_duration_seconds = Histogram(
            'prometheusgpt_request_duration_seconds',
            'Request duration in seconds',
            ['endpoint'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]
        )
        
        self.generation_duration_seconds = Histogram(
            'prometheusgpt_generation_duration_seconds',
            'Text generation duration in seconds',
            ['endpoint'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )
        
        # Скорость генерации
        self.tokens_per_second = Histogram(
            'prometheusgpt_tokens_per_second',
            'Tokens generated per second',
            ['endpoint'],
            buckets=[1, 5, 10, 20, 50, 100, 200, 500, 1000]
        )
        
        # Системные метрики
        self.cpu_usage_percent = Gauge(
            'prometheusgpt_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        self.memory_usage_percent = Gauge(
            'prometheusgpt_memory_usage_percent',
            'Memory usage percentage'
        )
        
        self.memory_usage_bytes = Gauge(
            'prometheusgpt_memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        # GPU метрики
        self.gpu_memory_usage_percent = Gauge(
            'prometheusgpt_gpu_memory_usage_percent',
            'GPU memory usage percentage',
            ['device_id']
        )
        
        self.gpu_memory_usage_bytes = Gauge(
            'prometheusgpt_gpu_memory_usage_bytes',
            'GPU memory usage in bytes',
            ['device_id']
        )
        
        self.gpu_utilization_percent = Gauge(
            'prometheusgpt_gpu_utilization_percent',
            'GPU utilization percentage',
            ['device_id']
        )
        
        self.gpu_temperature_celsius = Gauge(
            'prometheusgpt_gpu_temperature_celsius',
            'GPU temperature in Celsius',
            ['device_id']
        )
        
        self.gpu_power_usage_watts = Gauge(
            'prometheusgpt_gpu_power_usage_watts',
            'GPU power usage in watts',
            ['device_id']
        )
        
        # Активные запросы
        self.active_requests = Gauge(
            'prometheusgpt_active_requests',
            'Number of active requests'
        )
        
        # Размер очереди
        self.queue_size = Gauge(
            'prometheusgpt_queue_size',
            'Size of request queue'
        )
        
        logger.info("Prometheus metrics created")
    
    def start_server(self):
        """Запустить HTTP сервер для Prometheus"""
        
        if not self.enabled:
            logger.warning("Prometheus metrics disabled")
            return
        
        if self.server_started:
            logger.warning("Prometheus server already started")
            return
        
        try:
            start_http_server(self.port)
            self.server_started = True
            logger.info(f"Prometheus metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
    
    def set_model_info(self, model_info: Dict[str, Any]):
        """Установить информацию о модели"""
        
        if not self.enabled:
            return
        
        try:
            # Конвертируем все значения в строки
            info_dict = {}
            for key, value in model_info.items():
                if isinstance(value, (str, int, float, bool)):
                    info_dict[key] = str(value)
                else:
                    info_dict[key] = str(value)
            
            self.model_info.info(info_dict)
        except Exception as e:
            logger.error(f"Failed to set model info: {e}")
    
    def record_request(self, endpoint: str, status: str, duration_seconds: float,
                      tokens_generated: int = 0, tokens_per_second: float = 0):
        """Записать метрики запроса"""
        
        if not self.enabled:
            return
        
        try:
            # Счетчики
            self.requests_total.labels(endpoint=endpoint, status=status).inc()
            
            if tokens_generated > 0:
                self.tokens_generated_total.inc(tokens_generated)
            
            # Гистограммы
            self.request_duration_seconds.labels(endpoint=endpoint).observe(duration_seconds)
            
            if tokens_generated > 0:
                self.generation_duration_seconds.labels(endpoint=endpoint).observe(duration_seconds)
                self.tokens_per_second.labels(endpoint=endpoint).observe(tokens_per_second)
        
        except Exception as e:
            logger.error(f"Failed to record request metrics: {e}")
    
    def update_system_metrics(self, system_metrics: Dict[str, Any]):
        """Обновить системные метрики"""
        
        if not self.enabled:
            return
        
        try:
            # CPU и память
            if 'cpu_usage_percent' in system_metrics:
                self.cpu_usage_percent.set(system_metrics['cpu_usage_percent'])
            
            if 'memory_usage_percent' in system_metrics:
                self.memory_usage_percent.set(system_metrics['memory_usage_percent'])
            
            if 'memory_usage_bytes' in system_metrics:
                self.memory_usage_bytes.set(system_metrics['memory_usage_bytes'])
        
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def update_gpu_metrics(self, gpu_metrics: Dict[int, Dict[str, Any]]):
        """Обновить GPU метрики"""
        
        if not self.enabled:
            return
        
        try:
            for device_id, metrics in gpu_metrics.items():
                device_id_str = str(device_id)
                
                # Память GPU
                if 'memory_usage_percent' in metrics:
                    self.gpu_memory_usage_percent.labels(device_id=device_id_str).set(
                        metrics['memory_usage_percent']
                    )
                
                if 'memory_used_gb' in metrics:
                    memory_bytes = metrics['memory_used_gb'] * 1024**3
                    self.gpu_memory_usage_bytes.labels(device_id=device_id_str).set(memory_bytes)
                
                # Утилизация GPU
                if 'utilization_gpu_percent' in metrics:
                    self.gpu_utilization_percent.labels(device_id=device_id_str).set(
                        metrics['utilization_gpu_percent']
                    )
                
                # Температура
                if 'temperature_celsius' in metrics:
                    self.gpu_temperature_celsius.labels(device_id=device_id_str).set(
                        metrics['temperature_celsius']
                    )
                
                # Потребление энергии
                if 'power_usage_watts' in metrics:
                    self.gpu_power_usage_watts.labels(device_id=device_id_str).set(
                        metrics['power_usage_watts']
                    )
        
        except Exception as e:
            logger.error(f"Failed to update GPU metrics: {e}")
    
    def set_active_requests(self, count: int):
        """Установить количество активных запросов"""
        
        if not self.enabled:
            return
        
        try:
            self.active_requests.set(count)
        except Exception as e:
            logger.error(f"Failed to set active requests: {e}")
    
    def set_queue_size(self, size: int):
        """Установить размер очереди"""
        
        if not self.enabled:
            return
        
        try:
            self.queue_size.set(size)
        except Exception as e:
            logger.error(f"Failed to set queue size: {e}")
    
    def create_custom_metric(self, name: str, metric_type: str, description: str, 
                           labels: Optional[list] = None):
        """Создать пользовательскую метрику"""
        
        if not self.enabled:
            return None
        
        try:
            if metric_type == 'counter':
                return Counter(name, description, labels or [])
            elif metric_type == 'gauge':
                return Gauge(name, description, labels or [])
            elif metric_type == 'histogram':
                return Histogram(name, description, labels or [])
            else:
                logger.error(f"Unknown metric type: {metric_type}")
                return None
        
        except Exception as e:
            logger.error(f"Failed to create custom metric: {e}")
            return None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Получить сводку метрик"""
        
        if not self.enabled:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "server_started": self.server_started,
            "port": self.port,
            "metrics_available": [
                "model_info",
                "requests_total",
                "tokens_generated_total",
                "request_duration_seconds",
                "generation_duration_seconds",
                "tokens_per_second",
                "cpu_usage_percent",
                "memory_usage_percent",
                "gpu_memory_usage_percent",
                "gpu_utilization_percent",
                "gpu_temperature_celsius",
                "gpu_power_usage_watts",
                "active_requests",
                "queue_size"
            ]
        }
