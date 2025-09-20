"""
PrometheusGPT Mini - Production REST API
Author: MagistrTheOne, Krasnodar, 2025

Production-ready REST API с полной функциональностью для inference.
"""

import os
import sys
import torch
import torch.nn.functional as F
import asyncio
import time
import logging
import json
import psutil
from typing import List, Optional, Dict, Any, AsyncGenerator
from pathlib import Path
from datetime import datetime
import numpy as np

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, field_validator
import uvicorn

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model import PrometheusGPTMini, model_config, training_config
from src.data.tokenizer import AdvancedBPETokenizer
from src.data.dataloader import get_memory_usage

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Pydantic модели
class GenerateRequest(BaseModel):
    """Запрос на генерацию текста"""
    
    prompt: str = Field(..., min_length=1, max_length=1000, description="Текст для генерации")
    max_length: int = Field(50, ge=1, le=512, description="Максимальная длина генерируемого текста")
    temperature: float = Field(0.8, ge=0.1, le=2.0, description="Температура для sampling")
    top_k: int = Field(50, ge=1, le=1000, description="Top-k sampling")
    top_p: float = Field(0.9, ge=0.1, le=1.0, description="Top-p (nucleus) sampling")
    do_sample: bool = Field(True, description="Использовать sampling или greedy decoding")
    num_return_sequences: int = Field(1, ge=1, le=5, description="Количество генерируемых последовательностей")
    repetition_penalty: float = Field(1.0, ge=0.1, le=2.0, description="Штраф за повторения")
    length_penalty: float = Field(1.0, ge=0.1, le=2.0, description="Штраф за длину")
    early_stopping: bool = Field(True, description="Остановка при достижении EOS токена")
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError('Prompt cannot be empty')
        return v.strip()


class GenerateResponse(BaseModel):
    """Ответ сгенерированного текста"""
    
    text: str = Field(..., description="Сгенерированный текст")
    tokens_used: int = Field(..., description="Количество использованных токенов")
    generation_time: float = Field(..., description="Время генерации в секундах")
    prompt_tokens: int = Field(..., description="Количество токенов в промпте")
    model_info: Dict[str, Any] = Field(..., description="Информация о модели")
    
    model_config = {"protected_namespaces": ()}


class StreamingGenerateRequest(BaseModel):
    """Запрос на streaming генерацию"""
    
    prompt: str = Field(..., min_length=1, max_length=1000)
    max_length: int = Field(100, ge=1, le=512)
    temperature: float = Field(0.8, ge=0.1, le=2.0)
    top_k: int = Field(50, ge=1, le=1000)
    top_p: float = Field(0.9, ge=0.1, le=1.0)
    stream_interval: int = Field(1, ge=1, le=10, description="Интервал между токенами в потоке")


class HealthResponse(BaseModel):
    """Ответ проверки здоровья"""
    
    status: str = Field(..., description="Статус сервиса")
    timestamp: str = Field(..., description="Временная метка")
    model_loaded: bool = Field(..., description="Модель загружена")
    gpu_available: bool = Field(..., description="GPU доступен")
    gpu_memory_usage: Optional[Dict[str, float]] = Field(None, description="Использование GPU памяти")
    cpu_usage: float = Field(..., description="Использование CPU")
    memory_usage: Dict[str, float] = Field(..., description="Использование памяти")
    model_info: Dict[str, Any] = Field(..., description="Информация о модели")
    
    model_config = {"protected_namespaces": ()}


class ModelInfoResponse(BaseModel):
    """Информация о модели"""
    
    model_name: str = Field(..., description="Название модели")
    model_size: str = Field(..., description="Размер модели")
    total_parameters: int = Field(..., description="Общее количество параметров")
    trainable_parameters: int = Field(..., description="Количество обучаемых параметров")
    model_configuration: Dict[str, Any] = Field(..., description="Конфигурация модели")
    device: str = Field(..., description="Устройство выполнения")
    precision: str = Field(..., description="Точность вычислений")


class MetricsResponse(BaseModel):
    """Метрики производительности"""
    
    total_requests: int = Field(..., description="Общее количество запросов")
    successful_requests: int = Field(..., description="Успешные запросы")
    failed_requests: int = Field(..., description="Неудачные запросы")
    average_generation_time: float = Field(..., description="Среднее время генерации")
    average_tokens_per_second: float = Field(..., description="Средняя скорость генерации")
    uptime_seconds: float = Field(..., description="Время работы сервера")


# Создаем FastAPI приложение
app = FastAPI(
    title="PrometheusGPT Mini Production API",
    description="Production-ready API для мультиязычной LLM модели",
    version="1.0.0",
    contact={
        "name": "MagistrTheOne",
        "location": "Krasnodar, 2025"
    },
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Глобальные переменные
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
startup_time = time.time()

# Метрики
metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_generation_time": 0.0,
    "total_tokens_generated": 0
}


class ModelManager:
    """Менеджер модели для production использования"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False
        self.load_time = None
        
    def load_model(self, model_path: Optional[str] = None, tokenizer_path: Optional[str] = None):
        """Загрузить модель и токенизатор"""
        
        try:
            logger.info("Loading PrometheusGPT Mini model...")
            
            # Создаем модель
            self.model = PrometheusGPTMini(config=model_config)
            
            # Загружаем веса если есть
            if model_path and Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                logger.info(f"Model weights loaded from {model_path}")
            
            # Переносим на устройство
            self.model.to(self.device)
            self.model.eval()
            
            # Создаем токенизатор
            self.tokenizer = AdvancedBPETokenizer()
            
            # Пытаемся загрузить токенизатор из разных мест
            tokenizer_paths = [
                tokenizer_path,
                "demo_tokenizer.model",
                "tokenizer/demo_tokenizer.model",
                "data/demo_tokenizer.model"
            ]
            
            tokenizer_loaded = False
            for path in tokenizer_paths:
                if path and Path(path).exists():
                    try:
                        self.tokenizer.load(path)
                        logger.info(f"Tokenizer loaded from {path}")
                        tokenizer_loaded = True
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load tokenizer from {path}: {e}")
                        continue
            
            if not tokenizer_loaded:
                logger.warning("Tokenizer not found, using default")
            
            self.is_loaded = True
            self.load_time = time.time()
            
            logger.info("Model loaded successfully!")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate(self, prompt: str, max_length: int = 50, temperature: float = 0.8,
                top_k: int = 50, top_p: float = 0.9, do_sample: bool = True,
                repetition_penalty: float = 1.0, length_penalty: float = 1.0,
                early_stopping: bool = True) -> Dict[str, Any]:
        """Генерация текста"""
        
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Подготавливаем промпт с авторской информацией
            author_prompt = self.tokenizer.prepare_text_with_author(prompt)
            
            # Токенизируем
            input_tokens = self.tokenizer.encode(author_prompt, max_length=model_config.max_seq_length)
            input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(self.device)
            
            # Генерируем
            with torch.no_grad():
                generated_tokens = self._generate_tokens(
                    input_tensor, max_length, temperature, top_k, top_p, 
                    do_sample, repetition_penalty, length_penalty, early_stopping
                )
            
            # Декодируем
            generated_text = self.tokenizer.decode(generated_tokens[0].tolist())
            
            generation_time = time.time() - start_time
            
            return {
                "text": generated_text,
                "tokens_used": len(generated_tokens[0]),
                "generation_time": generation_time,
                "prompt_tokens": len(input_tokens)
            }
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise
    
    def _generate_tokens(self, input_tensor: torch.Tensor, max_length: int, 
                        temperature: float, top_k: int, top_p: float, 
                        do_sample: bool, repetition_penalty: float,
                        length_penalty: float, early_stopping: bool) -> torch.Tensor:
        """Генерация токенов с различными стратегиями"""
        
        generated = input_tensor.clone()
        eos_token_id = self.tokenizer.special_tokens.get('<eos>', 2)
        
        for _ in range(max_length - input_tensor.size(1)):
            # Forward pass
            outputs = self.model(generated)
            next_token_logits = outputs[:, -1, :]
            
            # Применяем repetition penalty
            if repetition_penalty != 1.0:
                for token_id in generated[0]:
                    if next_token_logits[0, token_id] < 0:
                        next_token_logits[0, token_id] *= repetition_penalty
                    else:
                        next_token_logits[0, token_id] /= repetition_penalty
            
            # Применяем temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Sampling
            if do_sample:
                # Top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits[0, top_k_indices] = top_k_logits
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Удаляем токены с кумулятивной вероятностью выше top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Добавляем новый токен
            generated = torch.cat([generated, next_token], dim=1)
            
            # Early stopping
            if early_stopping and next_token.item() == eos_token_id:
                break
        
        return generated
    
    async def generate_stream(self, prompt: str, max_length: int = 100, 
                            temperature: float = 0.8, top_k: int = 50, 
                            top_p: float = 0.9, stream_interval: int = 1) -> AsyncGenerator[str, None]:
        """Streaming генерация текста"""
        
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Подготавливаем промпт
            author_prompt = self.tokenizer.prepare_text_with_author(prompt)
            input_tokens = self.tokenizer.encode(author_prompt, max_length=model_config.max_seq_length)
            input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(self.device)
            
            generated = input_tensor.clone()
            eos_token_id = self.tokenizer.special_tokens.get('<eos>', 2)
            
            for i in range(max_length - input_tensor.size(1)):
                # Forward pass
                with torch.no_grad():
                    outputs = self.model(generated)
                    next_token_logits = outputs[:, -1, :] / temperature
                    
                    # Top-k и top-p filtering
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                        next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                        next_token_logits[0, top_k_indices] = top_k_logits
                    
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                # Добавляем токен
                generated = torch.cat([generated, next_token], dim=1)
                
                # Декодируем и отправляем новый токен
                if i % stream_interval == 0:
                    new_text = self.tokenizer.decode([next_token.item()])
                    if new_text.strip():
                        yield new_text
                
                # Early stopping
                if next_token.item() == eos_token_id:
                    break
                
                # Небольшая задержка для streaming эффекта
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
            yield f"Error: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получить информацию о модели"""
        
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": "PrometheusGPT Mini",
            "model_size": f"{total_params / 1_000_000:.1f}M parameters",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_config": {
                "vocab_size": model_config.vocab_size,
                "d_model": model_config.d_model,
                "n_layers": model_config.n_layers,
                "n_heads": model_config.n_heads,
                "d_ff": model_config.d_ff,
                "max_seq_length": model_config.max_seq_length,
                "dropout": model_config.dropout
            },
            "device": str(self.device),
            "precision": "float32",
            "load_time": self.load_time
        }


# Создаем менеджер модели
model_manager = ModelManager()


# Startup event
@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    
    logger.info("=== PrometheusGPT Mini Production API Starting ===")
    logger.info("Author: MagistrTheOne, Krasnodar, 2025")
    
    # Загружаем модель
    try:
        model_path = os.getenv("MODEL_PATH", "checkpoints/best_model.pt")
        tokenizer_path = os.getenv("TOKENIZER_PATH", "tokenizer")
        
        model_manager.load_model(model_path, tokenizer_path)
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("API will start without model - some endpoints will be unavailable")


# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка здоровья сервиса"""
    
    # Системная информация
    memory_info = psutil.virtual_memory()
    gpu_memory = None
    
    if torch.cuda.is_available():
        gpu_memory = {
            "allocated_gb": torch.cuda.memory_allocated(0) / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved(0) / 1024**3,
            "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
        }
    
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "degraded",
        timestamp=datetime.now().isoformat(),
        model_loaded=model_manager.is_loaded,
        gpu_available=torch.cuda.is_available(),
        gpu_memory_usage=gpu_memory,
        cpu_usage=psutil.cpu_percent(),
        memory_usage={
            "total_gb": memory_info.total / 1024**3,
            "available_gb": memory_info.available / 1024**3,
            "used_percent": memory_info.percent
        },
        model_info=model_manager.get_model_info()
    )


@app.get("/info", response_model=ModelInfoResponse)
async def model_info():
    """Информация о модели"""
    
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = model_manager.get_model_info()
    
    return ModelInfoResponse(
        model_name=info["model_name"],
        model_size=info["model_size"],
        total_parameters=info["total_parameters"],
        trainable_parameters=info["trainable_parameters"],
        model_configuration=info["model_config"],
        device=info["device"],
        precision=info["precision"]
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Генерация текста"""
    
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Обновляем метрики
    metrics["total_requests"] += 1
    
    try:
        result = model_manager.generate(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            do_sample=request.do_sample,
            repetition_penalty=request.repetition_penalty,
            length_penalty=request.length_penalty,
            early_stopping=request.early_stopping
        )
        
        # Обновляем метрики
        metrics["successful_requests"] += 1
        metrics["total_generation_time"] += result["generation_time"]
        metrics["total_tokens_generated"] += result["tokens_used"]
        
        return GenerateResponse(
            text=result["text"],
            tokens_used=result["tokens_used"],
            generation_time=result["generation_time"],
            prompt_tokens=result["prompt_tokens"],
            model_info=model_manager.get_model_info()
        )
        
    except Exception as e:
        metrics["failed_requests"] += 1
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@app.post("/generate_stream")
async def generate_text_stream(request: StreamingGenerateRequest):
    """Streaming генерация текста"""
    
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    async def generate():
        try:
            async for token in model_manager.generate_stream(
                prompt=request.prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                stream_interval=request.stream_interval
            ):
                yield f"data: {json.dumps({'token': token})}\n\n"
            
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Метрики производительности"""
    
    uptime = time.time() - startup_time
    avg_generation_time = (
        metrics["total_generation_time"] / max(metrics["successful_requests"], 1)
    )
    avg_tokens_per_second = (
        metrics["total_tokens_generated"] / max(metrics["total_generation_time"], 0.001)
    )
    
    return MetricsResponse(
        total_requests=metrics["total_requests"],
        successful_requests=metrics["successful_requests"],
        failed_requests=metrics["failed_requests"],
        average_generation_time=avg_generation_time,
        average_tokens_per_second=avg_tokens_per_second,
        uptime_seconds=uptime
    )


@app.get("/")
async def root():
    """Корневой endpoint"""
    
    return {
        "message": "PrometheusGPT Mini Production API",
        "author": "MagistrTheOne, Krasnodar, 2025",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model_manager.is_loaded,
        "device": str(device),
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "generate": "/generate",
            "generate_stream": "/generate_stream",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    print("=== PrometheusGPT Mini Production API ===")
    print("Author: MagistrTheOne, Krasnodar, 2025")
    print("Starting production server...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        workers=1,  # Для GPU моделей лучше использовать 1 worker
        log_level="info"
    )
