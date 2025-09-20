"""
PrometheusGPT Mini - Performance Monitoring
Author: MagistrTheOne, Krasnodar, 2025

Мониторинг производительности: latency, throughput, memory usage.
"""

import time
import logging
import threading
import psutil
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Метрики производительности"""
    
    timestamp: datetime
    request_id: str
    endpoint: str
    latency_ms: float
    tokens_generated: int
    tokens_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_memory_mb: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class SystemMetrics:
    """Системные метрики"""
    
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    network_io_bytes: int
    gpu_memory_usage_percent: float
    gpu_utilization_percent: float


class PerformanceMonitor:
    """Мониторинг производительности в реальном времени"""
    
    def __init__(self, max_history: int = 1000, update_interval: float = 5.0):
        """
        Args:
            max_history: максимальное количество записей в истории
            update_interval: интервал обновления системных метрик
        """
        
        self.max_history = max_history
        self.update_interval = update_interval
        
        # История метрик
        self.performance_history = deque(maxlen=max_history)
        self.system_history = deque(maxlen=max_history)
        
        # Текущие метрики
        self.current_request_id = 0
        self.active_requests = {}
        
        # Системный мониторинг
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Callbacks
        self.metrics_callbacks = []
        
        # Статистика
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_latency_ms': 0.0,
            'total_tokens_generated': 0,
            'total_processing_time': 0.0
        }
    
    def start_monitoring(self):
        """Запустить системный мониторинг"""
        
        if self.is_monitoring:
            logger.warning("Performance monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Остановить системный мониторинг"""
        
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Основной цикл системного мониторинга"""
        
        while self.is_monitoring:
            try:
                system_metrics = self._collect_system_metrics()
                self.system_history.append(system_metrics)
                
                # Вызываем callbacks
                for callback in self.metrics_callbacks:
                    try:
                        callback(system_metrics)
                    except Exception as e:
                        logger.error(f"Error in metrics callback: {e}")
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Собрать системные метрики"""
        
        # CPU и память
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Диск
        disk = psutil.disk_usage('/')
        
        # Сеть
        network = psutil.net_io_counters()
        network_io = network.bytes_sent + network.bytes_recv
        
        # GPU (если доступен)
        gpu_memory_percent = 0.0
        gpu_utilization_percent = 0.0
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated(0)
                gpu_total = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_percent = (gpu_memory / gpu_total) * 100
                
                # Попытка получить utilization (требует pynvml)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization_percent = utilization.gpu
                except:
                    pass
        except ImportError:
            pass
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage_percent=cpu_usage,
            memory_usage_percent=memory.percent,
            memory_available_gb=memory.available / 1024**3,
            disk_usage_percent=(disk.used / disk.total) * 100,
            network_io_bytes=network_io,
            gpu_memory_usage_percent=gpu_memory_percent,
            gpu_utilization_percent=gpu_utilization_percent
        )
    
    def start_request(self, endpoint: str) -> str:
        """Начать отслеживание запроса"""
        
        self.current_request_id += 1
        request_id = f"req_{self.current_request_id}_{int(time.time())}"
        
        self.active_requests[request_id] = {
            'endpoint': endpoint,
            'start_time': time.time(),
            'start_memory': psutil.Process().memory_info().rss / 1024**2,  # MB
            'start_gpu_memory': self._get_gpu_memory_mb()
        }
        
        return request_id
    
    def end_request(self, request_id: str, tokens_generated: int = 0, 
                   success: bool = True, error_message: Optional[str] = None):
        """Завершить отслеживание запроса"""
        
        if request_id not in self.active_requests:
            logger.warning(f"Request {request_id} not found in active requests")
            return
        
        request_info = self.active_requests.pop(request_id)
        
        # Вычисляем метрики
        end_time = time.time()
        latency_ms = (end_time - request_info['start_time']) * 1000
        
        current_memory = psutil.Process().memory_info().rss / 1024**2  # MB
        memory_usage_mb = current_memory - request_info['start_memory']
        
        current_gpu_memory = self._get_gpu_memory_mb()
        gpu_memory_mb = current_gpu_memory - request_info['start_gpu_memory']
        
        tokens_per_second = tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0
        
        # Создаем метрики
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            request_id=request_id,
            endpoint=request_info['endpoint'],
            latency_ms=latency_ms,
            tokens_generated=tokens_generated,
            tokens_per_second=tokens_per_second,
            memory_usage_mb=memory_usage_mb,
            cpu_usage_percent=psutil.cpu_percent(),
            gpu_memory_mb=gpu_memory_mb,
            success=success,
            error_message=error_message
        )
        
        # Добавляем в историю
        self.performance_history.append(metrics)
        
        # Обновляем статистику
        self._update_stats(metrics)
        
        # Логируем
        self._log_request_metrics(metrics)
    
    def _get_gpu_memory_mb(self) -> float:
        """Получить использование GPU памяти в MB"""
        
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated(0) / 1024**2
        except ImportError:
            pass
        
        return 0.0
    
    def _update_stats(self, metrics: PerformanceMetrics):
        """Обновить статистику"""
        
        self.stats['total_requests'] += 1
        
        if metrics.success:
            self.stats['successful_requests'] += 1
        else:
            self.stats['failed_requests'] += 1
        
        self.stats['total_latency_ms'] += metrics.latency_ms
        self.stats['total_tokens_generated'] += metrics.tokens_generated
        self.stats['total_processing_time'] += metrics.latency_ms / 1000
    
    def _log_request_metrics(self, metrics: PerformanceMetrics):
        """Логировать метрики запроса"""
        
        status = "SUCCESS" if metrics.success else "FAILED"
        
        logger.info(
            f"Request {metrics.request_id} | {metrics.endpoint} | {status} | "
            f"Latency: {metrics.latency_ms:.1f}ms | "
            f"Tokens: {metrics.tokens_generated} | "
            f"Speed: {metrics.tokens_per_second:.1f} tok/s | "
            f"Memory: {metrics.memory_usage_mb:.1f}MB | "
            f"GPU Memory: {metrics.gpu_memory_mb:.1f}MB"
        )
        
        if not metrics.success and metrics.error_message:
            logger.error(f"Request {metrics.request_id} failed: {metrics.error_message}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Получить статистику производительности"""
        
        if not self.performance_history:
            return {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'success_rate': 0.0,
                'average_latency_ms': 0.0,
                'average_tokens_per_second': 0.0,
                'total_tokens_generated': 0,
                'total_processing_time': 0.0
            }
        
        # Вычисляем статистики
        latencies = [m.latency_ms for m in self.performance_history]
        token_speeds = [m.tokens_per_second for m in self.performance_history if m.tokens_per_second > 0]
        
        successful_requests = sum(1 for m in self.performance_history if m.success)
        total_requests = len(self.performance_history)
        
        return {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': total_requests - successful_requests,
            'success_rate': (successful_requests / total_requests) * 100 if total_requests > 0 else 0,
            'average_latency_ms': statistics.mean(latencies) if latencies else 0,
            'median_latency_ms': statistics.median(latencies) if latencies else 0,
            'p95_latency_ms': self._percentile(latencies, 95) if latencies else 0,
            'p99_latency_ms': self._percentile(latencies, 99) if latencies else 0,
            'average_tokens_per_second': statistics.mean(token_speeds) if token_speeds else 0,
            'max_tokens_per_second': max(token_speeds) if token_speeds else 0,
            'total_tokens_generated': sum(m.tokens_generated for m in self.performance_history),
            'total_processing_time': sum(m.latency_ms for m in self.performance_history) / 1000
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Получить системную статистику"""
        
        if not self.system_history:
            return {}
        
        latest = self.system_history[-1]
        
        # Вычисляем средние значения за последние записи
        recent_count = min(10, len(self.system_history))
        recent_metrics = list(self.system_history)[-recent_count:]
        
        avg_cpu = statistics.mean([m.cpu_usage_percent for m in recent_metrics])
        avg_memory = statistics.mean([m.memory_usage_percent for m in recent_metrics])
        avg_gpu_memory = statistics.mean([m.gpu_memory_usage_percent for m in recent_metrics])
        avg_gpu_util = statistics.mean([m.gpu_utilization_percent for m in recent_metrics])
        
        return {
            'current_cpu_usage_percent': latest.cpu_usage_percent,
            'average_cpu_usage_percent': avg_cpu,
            'current_memory_usage_percent': latest.memory_usage_percent,
            'average_memory_usage_percent': avg_memory,
            'memory_available_gb': latest.memory_available_gb,
            'disk_usage_percent': latest.disk_usage_percent,
            'current_gpu_memory_usage_percent': latest.gpu_memory_usage_percent,
            'average_gpu_memory_usage_percent': avg_gpu_memory,
            'current_gpu_utilization_percent': latest.gpu_utilization_percent,
            'average_gpu_utilization_percent': avg_gpu_util,
            'network_io_bytes': latest.network_io_bytes
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Вычислить перцентиль"""
        
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        index = min(index, len(sorted_data) - 1)
        
        return sorted_data[index]
    
    def add_metrics_callback(self, callback: Callable[[SystemMetrics], None]):
        """Добавить callback для системных метрик"""
        
        self.metrics_callbacks.append(callback)
    
    def get_recent_requests(self, count: int = 10) -> List[Dict[str, Any]]:
        """Получить последние запросы"""
        
        recent = list(self.performance_history)[-count:]
        
        return [
            {
                'request_id': m.request_id,
                'endpoint': m.endpoint,
                'timestamp': m.timestamp.isoformat(),
                'latency_ms': m.latency_ms,
                'tokens_generated': m.tokens_generated,
                'tokens_per_second': m.tokens_per_second,
                'success': m.success,
                'error_message': m.error_message
            }
            for m in recent
        ]
    
    def clear_history(self):
        """Очистить историю метрик"""
        
        self.performance_history.clear()
        self.system_history.clear()
        
        # Сбрасываем статистику
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_latency_ms': 0.0,
            'total_tokens_generated': 0,
            'total_processing_time': 0.0
        }
        
        logger.info("Performance metrics history cleared")
