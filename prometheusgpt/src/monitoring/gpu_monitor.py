"""
PrometheusGPT Mini - GPU Monitoring
Author: MagistrTheOne, Krasnodar, 2025

Мониторинг GPU использования, памяти и производительности.
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import queue

try:
    import torch
    import pynvml
    TORCH_AVAILABLE = True
    PYNVML_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    PYNVML_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class GPUMetrics:
    """Метрики GPU"""
    
    device_id: int
    name: str
    memory_total: int  # bytes
    memory_used: int   # bytes
    memory_free: int   # bytes
    memory_allocated: int  # bytes (PyTorch)
    memory_reserved: int   # bytes (PyTorch)
    utilization_gpu: float  # percentage
    utilization_memory: float  # percentage
    temperature: float  # celsius
    power_usage: float  # watts
    clock_graphics: int  # MHz
    clock_memory: int    # MHz
    timestamp: datetime


class GPUMonitor:
    """Мониторинг GPU в реальном времени"""
    
    def __init__(self, device_id: int = 0, update_interval: float = 1.0):
        """
        Args:
            device_id: ID GPU устройства
            update_interval: интервал обновления в секундах
        """
        
        self.device_id = device_id
        self.update_interval = update_interval
        self.is_running = False
        self.monitor_thread = None
        self.metrics_queue = queue.Queue()
        self.latest_metrics = None
        
        # Инициализация
        self._initialize()
    
    def _initialize(self):
        """Инициализация мониторинга"""
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, GPU monitoring disabled")
            return
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, GPU monitoring disabled")
            return
        
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
                self.device_name = pynvml.nvmlDeviceGetName(self.handle).decode('utf-8')
                logger.info(f"GPU monitoring initialized for device {self.device_id}: {self.device_name}")
            except Exception as e:
                logger.error(f"Failed to initialize NVML: {e}")
                PYNVML_AVAILABLE = False
        else:
            logger.warning("pynvml not available, using PyTorch-only monitoring")
    
    def start_monitoring(self):
        """Запустить мониторинг"""
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, cannot start GPU monitoring")
            return
        
        if self.is_running:
            logger.warning("GPU monitoring already running")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"GPU monitoring started for device {self.device_id}")
    
    def stop_monitoring(self):
        """Остановить мониторинг"""
        
        if not self.is_running:
            return
        
        self.is_running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        logger.info("GPU monitoring stopped")
    
    def _monitor_loop(self):
        """Основной цикл мониторинга"""
        
        while self.is_running:
            try:
                metrics = self._collect_metrics()
                self.latest_metrics = metrics
                self.metrics_queue.put(metrics)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in GPU monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def _collect_metrics(self) -> GPUMetrics:
        """Собрать метрики GPU"""
        
        timestamp = datetime.now()
        
        # Базовые метрики из PyTorch
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.device_id)
            memory_reserved = torch.cuda.memory_reserved(self.device_id)
        else:
            memory_allocated = 0
            memory_reserved = 0
        
        # Расширенные метрики из NVML
        if PYNVML_AVAILABLE:
            try:
                # Информация о памяти
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                memory_total = memory_info.total
                memory_used = memory_info.used
                memory_free = memory_info.free
                
                # Использование GPU
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                utilization_gpu = utilization.gpu
                utilization_memory = utilization.memory
                
                # Температура
                temperature = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                
                # Потребление энергии
                power_usage = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # watts
                
                # Частоты
                clock_graphics = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS)
                clock_memory = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
                
            except Exception as e:
                logger.error(f"Error collecting NVML metrics: {e}")
                # Fallback значения
                memory_total = memory_allocated + memory_reserved
                memory_used = memory_allocated
                memory_free = memory_total - memory_used
                utilization_gpu = 0.0
                utilization_memory = 0.0
                temperature = 0.0
                power_usage = 0.0
                clock_graphics = 0
                clock_memory = 0
        else:
            # Fallback для случая без NVML
            memory_total = memory_allocated + memory_reserved
            memory_used = memory_allocated
            memory_free = memory_total - memory_used
            utilization_gpu = 0.0
            utilization_memory = 0.0
            temperature = 0.0
            power_usage = 0.0
            clock_graphics = 0
            clock_memory = 0
        
        return GPUMetrics(
            device_id=self.device_id,
            name=getattr(self, 'device_name', f'GPU-{self.device_id}'),
            memory_total=memory_total,
            memory_used=memory_used,
            memory_free=memory_free,
            memory_allocated=memory_allocated,
            memory_reserved=memory_reserved,
            utilization_gpu=utilization_gpu,
            utilization_memory=utilization_memory,
            temperature=temperature,
            power_usage=power_usage,
            clock_graphics=clock_graphics,
            clock_memory=clock_memory,
            timestamp=timestamp
        )
    
    def get_latest_metrics(self) -> Optional[GPUMetrics]:
        """Получить последние метрики"""
        
        return self.latest_metrics
    
    def get_metrics_history(self, max_count: int = 100) -> List[GPUMetrics]:
        """Получить историю метрик"""
        
        metrics_list = []
        
        while not self.metrics_queue.empty() and len(metrics_list) < max_count:
            try:
                metrics = self.metrics_queue.get_nowait()
                metrics_list.append(metrics)
            except queue.Empty:
                break
        
        return metrics_list
    
    def get_memory_usage_percent(self) -> float:
        """Получить процент использования памяти"""
        
        if not self.latest_metrics:
            return 0.0
        
        if self.latest_metrics.memory_total == 0:
            return 0.0
        
        return (self.latest_metrics.memory_used / self.latest_metrics.memory_total) * 100
    
    def get_allocated_memory_percent(self) -> float:
        """Получить процент выделенной PyTorch памяти"""
        
        if not self.latest_metrics:
            return 0.0
        
        if self.latest_metrics.memory_total == 0:
            return 0.0
        
        return (self.latest_metrics.memory_allocated / self.latest_metrics.memory_total) * 100
    
    def is_memory_pressure(self, threshold: float = 90.0) -> bool:
        """Проверить, есть ли давление на память"""
        
        return self.get_memory_usage_percent() > threshold
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Получить метрики в виде словаря"""
        
        if not self.latest_metrics:
            return {}
        
        metrics = self.latest_metrics
        
        return {
            'device_id': metrics.device_id,
            'device_name': metrics.name,
            'memory_total_gb': metrics.memory_total / 1024**3,
            'memory_used_gb': metrics.memory_used / 1024**3,
            'memory_free_gb': metrics.memory_free / 1024**3,
            'memory_allocated_gb': metrics.memory_allocated / 1024**3,
            'memory_reserved_gb': metrics.memory_reserved / 1024**3,
            'memory_usage_percent': self.get_memory_usage_percent(),
            'allocated_memory_percent': self.get_allocated_memory_percent(),
            'utilization_gpu_percent': metrics.utilization_gpu,
            'utilization_memory_percent': metrics.utilization_memory,
            'temperature_celsius': metrics.temperature,
            'power_usage_watts': metrics.power_usage,
            'clock_graphics_mhz': metrics.clock_graphics,
            'clock_memory_mhz': metrics.clock_memory,
            'timestamp': metrics.timestamp.isoformat()
        }
    
    def log_metrics(self):
        """Логировать текущие метрики"""
        
        if not self.latest_metrics:
            logger.warning("No GPU metrics available")
            return
        
        metrics = self.latest_metrics
        
        logger.info(
            f"GPU {metrics.device_id} ({metrics.name}) | "
            f"Memory: {metrics.memory_used/1024**3:.2f}/{metrics.memory_total/1024**3:.2f}GB "
            f"({self.get_memory_usage_percent():.1f}%) | "
            f"Allocated: {metrics.memory_allocated/1024**3:.2f}GB "
            f"({self.get_allocated_memory_percent():.1f}%) | "
            f"GPU Util: {metrics.utilization_gpu:.1f}% | "
            f"Temp: {metrics.temperature:.1f}°C | "
            f"Power: {metrics.power_usage:.1f}W"
        )


class MultiGPUMonitor:
    """Мониторинг нескольких GPU"""
    
    def __init__(self, device_ids: Optional[List[int]] = None, update_interval: float = 1.0):
        """
        Args:
            device_ids: список ID GPU устройств
            update_interval: интервал обновления
        """
        
        if device_ids is None:
            if torch.cuda.is_available():
                device_ids = list(range(torch.cuda.device_count()))
            else:
                device_ids = []
        
        self.device_ids = device_ids
        self.monitors = {}
        
        # Создаем мониторы для каждого GPU
        for device_id in device_ids:
            self.monitors[device_id] = GPUMonitor(device_id, update_interval)
    
    def start_monitoring(self):
        """Запустить мониторинг всех GPU"""
        
        for monitor in self.monitors.values():
            monitor.start_monitoring()
        
        logger.info(f"Multi-GPU monitoring started for devices: {self.device_ids}")
    
    def stop_monitoring(self):
        """Остановить мониторинг всех GPU"""
        
        for monitor in self.monitors.values():
            monitor.stop_monitoring()
        
        logger.info("Multi-GPU monitoring stopped")
    
    def get_all_metrics(self) -> Dict[int, Dict[str, Any]]:
        """Получить метрики всех GPU"""
        
        all_metrics = {}
        
        for device_id, monitor in self.monitors.items():
            all_metrics[device_id] = monitor.get_metrics_dict()
        
        return all_metrics
    
    def get_total_memory_usage(self) -> Dict[str, float]:
        """Получить общее использование памяти"""
        
        total_memory = 0
        total_used = 0
        total_allocated = 0
        
        for monitor in self.monitors.values():
            metrics = monitor.get_latest_metrics()
            if metrics:
                total_memory += metrics.memory_total
                total_used += metrics.memory_used
                total_allocated += metrics.memory_allocated
        
        return {
            'total_memory_gb': total_memory / 1024**3,
            'total_used_gb': total_used / 1024**3,
            'total_allocated_gb': total_allocated / 1024**3,
            'total_usage_percent': (total_used / total_memory * 100) if total_memory > 0 else 0,
            'total_allocated_percent': (total_allocated / total_memory * 100) if total_memory > 0 else 0
        }
    
    def log_all_metrics(self):
        """Логировать метрики всех GPU"""
        
        for device_id, monitor in self.monitors.items():
            logger.info(f"=== GPU {device_id} Metrics ===")
            monitor.log_metrics()
        
        # Общие метрики
        total_metrics = self.get_total_memory_usage()
        logger.info(
            f"Total GPU Memory: {total_metrics['total_used_gb']:.2f}/"
            f"{total_metrics['total_memory_gb']:.2f}GB "
            f"({total_metrics['total_usage_percent']:.1f}%)"
        )
