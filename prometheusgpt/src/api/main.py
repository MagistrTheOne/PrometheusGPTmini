"""
PrometheusGPT Mini - REST API
Author: MagistrTheOne, Krasnodar, 2025

FastAPI сервер для генерации текста с streaming поддержкой.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import asyncio
import os
from typing import List, Optional
import time
import psutil
import torch
import logging

logger = logging.getLogger(__name__)

# Импорт из других модулей
from ..model import PrometheusGPTMini, model_config, training_config
from ..tokenizer import BPETokenizer


class GenerateRequest(BaseModel):
    """Запрос на генерацию текста"""
    prompt: str
    max_length: int = 50
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    num_return_sequences: int = 1
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


class GenerateResponse(BaseModel):
    """Ответ сгенерированного текста"""
    text: str
    tokens_used: int
    generation_time: float


class HealthResponse(BaseModel):
    """Ответ проверки здоровья"""
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_memory_usage: Optional[dict]
    cpu_usage: float


app = FastAPI(
    title="PrometheusGPT Mini API",
    description="Мультиязычная LLM модель с собственной архитектурой",
    version="0.1.0",
    contact={
        "name": "MagistrTheOne",
        "location": "Krasnodar, 2025"
    }
)


# Глобальные переменные
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Функция инициализации модели
def initialize_model():
    """Инициализировать модель и токенизатор"""
    global model, tokenizer

    if model is None:
        logger.info("Initializing PrometheusGPT Mini model...")

        # Создаем модель
        model = PrometheusGPTMini()

        # Создаем токенизатор (пока демо)
        tokenizer = BPETokenizer()

        # Загружаем обученную модель если есть
        model_path = "models/prometheusgpt.pt"
        if os.path.exists(model_path):
            try:
                model.load_model(model_path)
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")

        # Переносим на устройство
        model.to(device)
        model.eval()

        logger.info("Model initialized successfully!")
        logger.info(f"Model info: {model.get_model_info()}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка здоровья сервиса"""
    # Инициализируем модель если нужно
    initialize_model()

    gpu_available = torch.cuda.is_available()

    if gpu_available:
        gpu_memory = torch.cuda.get_device_properties(0)
        gpu_memory_usage = {
            "total": gpu_memory.total_memory,
            "allocated": torch.cuda.memory_allocated(0),
            "cached": torch.cuda.memory_reserved(0)
        }
    else:
        gpu_memory_usage = None

    # Информация о модели
    model_info = model.get_model_info() if model else None

    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        gpu_available=gpu_available,
        gpu_memory_usage=gpu_memory_usage,
        cpu_usage=psutil.cpu_percent()
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Генерация текста по промпту"""
    # Инициализируем модель если нужно
    initialize_model()

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        # Добавляем информацию об авторе к промпту
        author_prompt = tokenizer.prepare_text_with_author(request.prompt)

        # Токенизируем
        input_tokens = tokenizer.encode(author_prompt)
        input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)

        # Генерируем
        with torch.no_grad():
            generated_tokens = model.generate(
                input_tensor,
                max_length=request.max_length,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                do_sample=request.do_sample
            )

        # Декодируем
        generated_text = tokenizer.decode(generated_tokens[0].tolist())

        generation_time = time.time() - start_time

        return GenerateResponse(
            text=generated_text,
            tokens_used=len(generated_tokens[0]),
            generation_time=generation_time
        )

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@app.post("/generate_stream")
async def generate_text_stream(request: GenerateRequest):
    """Streaming генерация текста"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    async def generate():
        try:
            # TODO: Реализовать streaming генерацию
            # Пока заглушка
            yield "Generated text based on: "
            await asyncio.sleep(0.1)
            yield f"{request.prompt[:50]}..."
            await asyncio.sleep(0.1)
            yield " (streaming complete)"

        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/")
async def root():
    """Корневой endpoint"""
    # Инициализируем модель если нужно
    initialize_model()

    model_info = model.get_model_info() if model else None

    return {
        "message": "PrometheusGPT Mini API",
        "author": "MagistrTheOne, Krasnodar, 2025",
        "status": "running",
        "model_info": model_info,
        "device": str(device)
    }


if __name__ == "__main__":
    print("=== PrometheusGPT Mini API ===")
    print(f"Author: MagistrTheOne, Krasnodar, 2025")
    print(f"Model parameters: ~{model_config.total_params / 1_000_000:.1f}M")
    print(f"Device: {device}")
    print("Starting server...")

    uvicorn.run(app, host="0.0.0.0", port=8000)
