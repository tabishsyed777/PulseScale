FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pulse_scale ./pulse_scale
COPY static ./static

EXPOSE 8000
USER 10001:10001
CMD ["uvicorn", "pulse_scale.api:app", "--host", "0.0.0.0", "--port", "8000"]
