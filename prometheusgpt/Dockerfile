# PrometheusGPT Mini - Dockerfile
# Author: MagistrTheOne, Krasnodar, 2025

FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создание пользователя
RUN useradd -m -s /bin/bash prometheusgpt
USER prometheusgpt

# Установка рабочей директории
WORKDIR /home/prometheusgpt/app

# Копирование файлов
COPY --chown=prometheusgpt:prometheusgpt requirements.txt .
COPY --chown=prometheusgpt:prometheusgpt src/ ./src/

# Создание виртуальной среды и установка зависимостей
RUN python3.10 -m venv venv && \
    ./venv/bin/pip install --no-cache-dir --upgrade pip && \
    ./venv/bin/pip install --no-cache-dir -r requirements.txt

# Создание директорий для данных
RUN mkdir -p data models logs

# Переменные окружения
ENV PYTHONPATH=/home/prometheusgpt/app
ENV CUDA_VISIBLE_DEVICES=0

# Порт
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Запуск API
CMD ["./venv/bin/python", "src/api/main.py"]
