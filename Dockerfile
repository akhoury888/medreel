FROM python:3.12-slim
ARG WITH_LARGE=0
ENV PYTHONUNBUFFERED=1 MEDREEL_MODELS=/models STORAGE_DIR=/data/storage PORT=8080
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl bzip2 libfribidi0 libharfbuzz0b libraqm0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY tools/setup_models.sh tools/setup_models.sh
RUN WITH_LARGE=$WITH_LARGE bash tools/setup_models.sh
COPY . .
RUN useradd -m medreel && mkdir -p /data && chown -R medreel /data /app
USER medreel
EXPOSE 8080
CMD ["sh", "-c", "gunicorn -w 2 -k gthread --threads 8 --timeout 600 -b 0.0.0.0:${PORT} wsgi:app"]
