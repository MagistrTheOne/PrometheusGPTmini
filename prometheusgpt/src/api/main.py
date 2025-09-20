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
from typing import List, Optional
import time
import psutil
import torch

# Импорт из других модулей (пока заглушки)
from ..model.config import model_config, training_config


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


# Глобальные переменные (пока заглушки)
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка здоровья сервиса"""
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
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        # TODO: Реализовать генерацию текста
        # Пока заглушка
        generated_text = f"Generated text based on: {request.prompt[:50]}..."

        generation_time = time.time() - start_time

        return GenerateResponse(
            text=generated_text,
            tokens_used=0,  # TODO: Посчитать реальные токены
            generation_time=generation_time
        )

    except Exception as e:
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
    return {
        "message": "PrometheusGPT Mini API",
        "author": "MagistrTheOne, Krasnodar, 2025",
        "status": "running",
        "model_params": model_config.total_params,
        "device": str(device)
    }


if __name__ == "__main__":
    print("=== PrometheusGPT Mini API ===")
    print(f"Author: MagistrTheOne, Krasnodar, 2025")
    print(f"Model parameters: ~{model_config.total_params / 1_000_000:.1f}M")
    print(f"Device: {device}")
    print("Starting server...")

    uvicorn.run(app, host="0.0.0.0", port=8000)
