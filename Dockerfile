FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WHISPER_WEB_HOST=0.0.0.0 \
    WHISPER_WEB_PORT=7860 \
    WHISPER_WEB_MODEL=base

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN python -m pip install --upgrade pip \
    && python -m pip install . \
    && python -m pip install -r requirements-web.txt

EXPOSE 7860

CMD ["gunicorn", "--workers", "1", "--threads", "8", "--timeout", "0", "--bind", "0.0.0.0:7860", "webui:app"]
