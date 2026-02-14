# --- Stage 1: Builder (Сборка зависимостей) ---
FROM python:3.12-slim as builder

ENV PIP_NO_CACHE_DIR=off \
    POETRY_VIRTUALENVS_CREATE=false \
    UV_SYSTEM_PYTHON=1

# 1. Копируем uv (самый быстрый установщик пакетов)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 2. Ставим системные пакеты для сборки (gcc/g++ нужны для annoy/numpy)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./

RUN uv pip install --no-cache -r requirements.txt

# --- Stage 2: Final (Финальный образ) ---
FROM python:3.12-slim

WORKDIR /app

# 4. Копируем только установленные библиотеки из builder
# Это делает образ меньше и чище (без gcc, poetry и кэшей)
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 5. Копируем код проекта
COPY src/ src/
COPY data/ data/
COPY configs/ configs/
COPY embeddings/ embeddings/
COPY models/ models/
COPY service/ service/

# 6. Настройки рантайма
ENV OMP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

# Запуск (порт 8001, как ты хотел)
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "4"]