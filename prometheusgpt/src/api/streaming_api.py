"""
PrometheusGPT Mini - Advanced Streaming API
Author: MagistrTheOne, Krasnodar, 2025

Продвинутый streaming API для длинных промптов с оптимизацией.
"""

import asyncio
import json
import time
import logging
from typing import AsyncGenerator, Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import torch
import torch.nn.functional as F

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Конфигурация для streaming генерации"""
    
    max_length: int = 512
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    stream_interval: int = 1
    buffer_size: int = 10
    timeout_seconds: int = 300
    enable_metrics: bool = True


class StreamingGenerator:
    """Генератор для streaming текста"""
    
    def __init__(self, model, tokenizer, config: StreamingConfig):
        """
        Args:
            model: модель для генерации
            tokenizer: токенизатор
            config: конфигурация streaming
        """
        
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
        
        # Метрики
        self.metrics = {
            'tokens_generated': 0,
            'generation_time': 0.0,
            'tokens_per_second': 0.0,
            'buffer_hits': 0,
            'buffer_misses': 0
        }
        
        # Буфер для кэширования
        self.generation_buffer = {}
        
    async def generate_stream(self, prompt: str, 
                            request_id: str = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Генерировать поток токенов
        
        Args:
            prompt: входной промпт
            request_id: ID запроса для отслеживания
        
        Yields:
            Словарь с информацией о сгенерированном токене
        """
        
        start_time = time.time()
        request_id = request_id or f"stream_{int(time.time())}"
        
        try:
            # Подготавливаем промпт
            author_prompt = self.tokenizer.prepare_text_with_author(prompt)
            input_tokens = self.tokenizer.encode(author_prompt, max_length=256)
            input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(self.device)
            
            # Проверяем буфер
            buffer_key = self._get_buffer_key(prompt)
            if buffer_key in self.generation_buffer:
                logger.info(f"Using cached generation for request {request_id}")
                self.metrics['buffer_hits'] += 1
                
                # Возвращаем кэшированный результат
                cached_tokens = self.generation_buffer[buffer_key]
                for i, token in enumerate(cached_tokens):
                    if i % self.config.stream_interval == 0:
                        token_text = self.tokenizer.decode([token])
                        yield {
                            'request_id': request_id,
                            'token': token_text,
                            'token_id': token,
                            'position': i,
                            'cached': True,
                            'timestamp': datetime.now().isoformat()
                        }
                        await asyncio.sleep(0.01)  # Небольшая задержка для streaming эффекта
                
                return
            
            self.metrics['buffer_misses'] += 1
            
            # Генерируем новые токены
            generated_tokens = []
            current_tokens = input_tensor.clone()
            eos_token_id = self.tokenizer.special_tokens.get('<eos>', 2)
            
            for i in range(self.config.max_length - input_tensor.size(1)):
                # Forward pass
                with torch.no_grad():
                    outputs = self.model(current_tokens)
                    next_token_logits = outputs[:, -1, :] / self.config.temperature
                    
                    # Top-k и top-p filtering
                    if self.config.top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, self.config.top_k)
                        next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                        next_token_logits[0, top_k_indices] = top_k_logits
                    
                    if self.config.top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > self.config.top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                # Добавляем токен
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
                generated_tokens.append(next_token.item())
                
                # Отправляем токен если нужно
                if i % self.config.stream_interval == 0:
                    token_text = self.tokenizer.decode([next_token.item()])
                    
                    yield {
                        'request_id': request_id,
                        'token': token_text,
                        'token_id': next_token.item(),
                        'position': i,
                        'cached': False,
                        'timestamp': datetime.now().isoformat()
                    }
                
                # Early stopping
                if next_token.item() == eos_token_id:
                    break
                
                # Небольшая задержка для streaming эффекта
                await asyncio.sleep(0.01)
            
            # Кэшируем результат
            if len(generated_tokens) > 0:
                self.generation_buffer[buffer_key] = generated_tokens
                
                # Ограничиваем размер буфера
                if len(self.generation_buffer) > self.config.buffer_size:
                    # Удаляем самый старый элемент
                    oldest_key = next(iter(self.generation_buffer))
                    del self.generation_buffer[oldest_key]
            
            # Обновляем метрики
            generation_time = time.time() - start_time
            self.metrics['tokens_generated'] += len(generated_tokens)
            self.metrics['generation_time'] += generation_time
            self.metrics['tokens_per_second'] = (
                self.metrics['tokens_generated'] / self.metrics['generation_time']
                if self.metrics['generation_time'] > 0 else 0
            )
            
            # Отправляем финальную статистику
            yield {
                'request_id': request_id,
                'type': 'completion',
                'total_tokens': len(generated_tokens),
                'generation_time': generation_time,
                'tokens_per_second': len(generated_tokens) / generation_time if generation_time > 0 else 0,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in streaming generation for request {request_id}: {e}")
            yield {
                'request_id': request_id,
                'type': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_buffer_key(self, prompt: str) -> str:
        """Получить ключ для буфера"""
        
        # Создаем хэш от промпта для кэширования
        import hashlib
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Получить метрики генератора"""
        
        return {
            'metrics': self.metrics.copy(),
            'buffer_size': len(self.generation_buffer),
            'buffer_hit_rate': (
                self.metrics['buffer_hits'] / 
                (self.metrics['buffer_hits'] + self.metrics['buffer_misses'])
                if (self.metrics['buffer_hits'] + self.metrics['buffer_misses']) > 0 else 0
            )
        }
    
    def clear_buffer(self):
        """Очистить буфер кэширования"""
        
        self.generation_buffer.clear()
        logger.info("Generation buffer cleared")


class StreamingAPI:
    """Streaming API для PrometheusGPT"""
    
    def __init__(self, model, tokenizer):
        """
        Args:
            model: модель для генерации
            tokenizer: токенизатор
        """
        
        self.model = model
        self.tokenizer = tokenizer
        self.generators = {}
        self.active_connections = {}
        
        # Создаем FastAPI приложение
        self.app = FastAPI(
            title="PrometheusGPT Streaming API",
            description="Advanced streaming API for long prompts",
            version="1.0.0"
        )
        
        # Настраиваем маршруты
        self._setup_routes()
    
    def _setup_routes(self):
        """Настроить маршруты API"""
        
        @self.app.post("/stream/generate")
        async def stream_generate(request: StreamingGenerateRequest):
            """HTTP streaming генерация"""
            
            config = StreamingConfig(
                max_length=request.max_length,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                stream_interval=request.stream_interval
            )
            
            generator = StreamingGenerator(self.model, self.tokenizer, config)
            
            async def generate():
                try:
                    async for token_data in generator.generate_stream(request.prompt):
                        yield f"data: {json.dumps(token_data)}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
            
            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        
        @self.app.websocket("/ws/stream")
        async def websocket_stream(websocket: WebSocket):
            """WebSocket streaming генерация"""
            
            await websocket.accept()
            connection_id = f"ws_{int(time.time())}"
            self.active_connections[connection_id] = websocket
            
            try:
                while True:
                    # Получаем запрос
                    data = await websocket.receive_text()
                    request_data = json.loads(data)
                    
                    # Создаем конфигурацию
                    config = StreamingConfig(
                        max_length=request_data.get('max_length', 512),
                        temperature=request_data.get('temperature', 0.8),
                        top_k=request_data.get('top_k', 50),
                        top_p=request_data.get('top_p', 0.9),
                        stream_interval=request_data.get('stream_interval', 1)
                    )
                    
                    # Создаем генератор
                    generator = StreamingGenerator(self.model, self.tokenizer, config)
                    
                    # Генерируем и отправляем токены
                    async for token_data in generator.generate_stream(
                        request_data['prompt'], 
                        request_data.get('request_id')
                    ):
                        await websocket.send_text(json.dumps(token_data))
                        
                        # Проверяем, не закрыто ли соединение
                        try:
                            await websocket.receive_text()
                        except:
                            break
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket connection {connection_id} disconnected")
            except Exception as e:
                logger.error(f"Error in WebSocket connection {connection_id}: {e}")
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'error': str(e)
                }))
            finally:
                if connection_id in self.active_connections:
                    del self.active_connections[connection_id]
        
        @self.app.get("/stream/metrics")
        async def get_streaming_metrics():
            """Получить метрики streaming"""
            
            all_metrics = {}
            for generator_id, generator in self.generators.items():
                all_metrics[generator_id] = generator.get_metrics()
            
            return {
                'active_connections': len(self.active_connections),
                'active_generators': len(self.generators),
                'generators_metrics': all_metrics
            }
        
        @self.app.post("/stream/clear_buffer")
        async def clear_buffer():
            """Очистить буфер кэширования"""
            
            for generator in self.generators.values():
                generator.clear_buffer()
            
            return {"message": "Buffer cleared"}
    
    def get_app(self) -> FastAPI:
        """Получить FastAPI приложение"""
        
        return self.app


# Pydantic модели для API
class StreamingGenerateRequest(BaseModel):
    """Запрос на streaming генерацию"""
    
    prompt: str = Field(..., min_length=1, max_length=2000, description="Промпт для генерации")
    max_length: int = Field(512, ge=1, le=2048, description="Максимальная длина генерации")
    temperature: float = Field(0.8, ge=0.1, le=2.0, description="Температура sampling")
    top_k: int = Field(50, ge=1, le=1000, description="Top-k sampling")
    top_p: float = Field(0.9, ge=0.1, le=1.0, description="Top-p sampling")
    stream_interval: int = Field(1, ge=1, le=10, description="Интервал между токенами")
    request_id: Optional[str] = Field(None, description="ID запроса")


class WebSocketStreamRequest(BaseModel):
    """WebSocket запрос на streaming"""
    
    prompt: str = Field(..., min_length=1, max_length=2000)
    max_length: int = Field(512, ge=1, le=2048)
    temperature: float = Field(0.8, ge=0.1, le=2.0)
    top_k: int = Field(50, ge=1, le=1000)
    top_p: float = Field(0.9, ge=0.1, le=1.0)
    stream_interval: int = Field(1, ge=1, le=10)
    request_id: Optional[str] = Field(None)


# Утилиты для интеграции с основным API
def create_streaming_endpoint(main_app: FastAPI, model, tokenizer):
    """Создать streaming эндпоинты в основном приложении"""
    
    streaming_api = StreamingAPI(model, tokenizer)
    
    # Добавляем маршруты к основному приложению
    main_app.include_router(streaming_api.app.router, prefix="/stream")
    
    return streaming_api
