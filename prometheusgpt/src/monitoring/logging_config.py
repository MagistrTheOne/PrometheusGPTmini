"""
PrometheusGPT Mini - Logging Configuration
Author: MagistrTheOne, Krasnodar, 2025

Настройка логирования для production использования.
"""

import os
import logging
import logging.handlers
from typing import Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """JSON форматтер для структурированного логирования"""
    
    def format(self, record):
        """Форматировать запись в JSON"""
        
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Добавляем дополнительные поля если есть
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        
        if hasattr(record, 'endpoint'):
            log_entry['endpoint'] = record.endpoint
        
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        
        if hasattr(record, 'duration_ms'):
            log_entry['duration_ms'] = record.duration_ms
        
        if hasattr(record, 'tokens_generated'):
            log_entry['tokens_generated'] = record.tokens_generated
        
        if hasattr(record, 'gpu_memory_mb'):
            log_entry['gpu_memory_mb'] = record.gpu_memory_mb
        
        # Добавляем exception если есть
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False)


class RequestContextFilter(logging.Filter):
    """Фильтр для добавления контекста запроса"""
    
    def filter(self, record):
        """Добавить контекст запроса к записи"""
        
        # Добавляем информацию о процессе
        record.pid = os.getpid()
        
        # Добавляем информацию о потоке
        import threading
        record.thread_id = threading.get_ident()
        record.thread_name = threading.current_thread().name
        
        return True


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_format: str = "json",  # "json" или "text"
    max_file_size: int = 100 * 1024 * 1024,  # 100MB
    backup_count: int = 5,
    enable_console: bool = True,
    enable_file: bool = True,
    log_requests: bool = True,
    log_performance: bool = True
) -> Dict[str, Any]:
    """
    Настроить логирование для production
    
    Args:
        log_level: уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: директория для логов
        log_format: формат логов (json или text)
        max_file_size: максимальный размер файла лога
        backup_count: количество резервных файлов
        enable_console: включить вывод в консоль
        enable_file: включить запись в файл
        log_requests: логировать запросы
        log_performance: логировать метрики производительности
    
    Returns:
        Словарь с информацией о настройке логирования
    """
    
    # Создаем директорию для логов
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True, parents=True)
    
    # Настраиваем корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Очищаем существующие handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Создаем форматтеры
    if log_format == "json":
        formatter = JSONFormatter()
        text_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        text_formatter = formatter
    
    # Создаем фильтр контекста
    context_filter = RequestContextFilter()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(text_formatter)
        console_handler.addFilter(context_filter)
        root_logger.addHandler(console_handler)
    
    # File handlers
    if enable_file:
        # Основной лог файл
        main_log_file = log_path / "prometheusgpt.log"
        main_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        main_handler.setLevel(getattr(logging, log_level.upper()))
        main_handler.setFormatter(formatter)
        main_handler.addFilter(context_filter)
        root_logger.addHandler(main_handler)
        
        # Лог ошибок
        error_log_file = log_path / "prometheusgpt_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        error_handler.addFilter(context_filter)
        root_logger.addHandler(error_handler)
        
        # Лог запросов (если включен)
        if log_requests:
            request_log_file = log_path / "prometheusgpt_requests.log"
            request_handler = logging.handlers.RotatingFileHandler(
                request_log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            request_handler.setLevel(logging.INFO)
            request_handler.setFormatter(formatter)
            request_handler.addFilter(context_filter)
            
            # Создаем отдельный логгер для запросов
            request_logger = logging.getLogger('prometheusgpt.requests')
            request_logger.addHandler(request_handler)
            request_logger.setLevel(logging.INFO)
            request_logger.propagate = False
        
        # Лог производительности (если включен)
        if log_performance:
            performance_log_file = log_path / "prometheusgpt_performance.log"
            performance_handler = logging.handlers.RotatingFileHandler(
                performance_log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            performance_handler.setLevel(logging.INFO)
            performance_handler.setFormatter(formatter)
            performance_handler.addFilter(context_filter)
            
            # Создаем отдельный логгер для производительности
            performance_logger = logging.getLogger('prometheusgpt.performance')
            performance_logger.addHandler(performance_handler)
            performance_logger.setLevel(logging.INFO)
            performance_logger.propagate = False
    
    # Настраиваем логгеры для внешних библиотек
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('uvicorn.access').setLevel(logging.INFO)
    logging.getLogger('fastapi').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    
    # Создаем логгер для приложения
    app_logger = logging.getLogger('prometheusgpt')
    app_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Логируем информацию о настройке
    app_logger.info("Logging configured", extra={
        'log_level': log_level,
        'log_dir': str(log_path),
        'log_format': log_format,
        'max_file_size': max_file_size,
        'backup_count': backup_count,
        'enable_console': enable_console,
        'enable_file': enable_file,
        'log_requests': log_requests,
        'log_performance': log_performance
    })
    
    return {
        'log_level': log_level,
        'log_dir': str(log_path),
        'log_format': log_format,
        'handlers_count': len(root_logger.handlers),
        'app_logger': app_logger.name,
        'request_logger': 'prometheusgpt.requests' if log_requests else None,
        'performance_logger': 'prometheusgpt.performance' if log_performance else None
    }


def get_logger(name: str) -> logging.Logger:
    """Получить логгер с указанным именем"""
    
    return logging.getLogger(f'prometheusgpt.{name}')


def log_request(logger: logging.Logger, request_id: str, endpoint: str, 
               method: str, status_code: int, duration_ms: float, 
               **kwargs):
    """Логировать запрос"""
    
    logger.info(
        f"Request {method} {endpoint}",
        extra={
            'request_id': request_id,
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'duration_ms': duration_ms,
            **kwargs
        }
    )


def log_performance(logger: logging.Logger, operation: str, duration_ms: float,
                   tokens_generated: int = 0, tokens_per_second: float = 0,
                   memory_usage_mb: float = 0, gpu_memory_mb: float = 0,
                   **kwargs):
    """Логировать метрики производительности"""
    
    logger.info(
        f"Performance: {operation}",
        extra={
            'operation': operation,
            'duration_ms': duration_ms,
            'tokens_generated': tokens_generated,
            'tokens_per_second': tokens_per_second,
            'memory_usage_mb': memory_usage_mb,
            'gpu_memory_mb': gpu_memory_mb,
            **kwargs
        }
    )


def log_error(logger: logging.Logger, error: Exception, context: Optional[Dict[str, Any]] = None):
    """Логировать ошибку с контекстом"""
    
    extra = {'error_type': type(error).__name__, 'error_message': str(error)}
    if context:
        extra.update(context)
    
    logger.error(
        f"Error: {type(error).__name__}: {str(error)}",
        extra=extra,
        exc_info=True
    )
