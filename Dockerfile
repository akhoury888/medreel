FROM python:3.12-slim
# speech/voice models are NOT baked into the image: they download once into the /models volume on first start
# (tools/entrypoint.sh), so rebuilding the image never re-downloads ~4 GB. WITH_LARGE=1 adds Whisper large-v3.
ENV PYTHONUNBUFFERED=1 MEDREEL_MODELS=/models STORAGE_DIR=/data/storage PORT=8080 WITH_LARGE=0
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl bzip2 libfribidi0 libharfbuzz0b libraqm0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m medreel && mkdir -p /data /models && chown -R medreel /data /app /models
USER medreel
EXPOSE 8080
ENTRYPOINT ["sh", "/app/tools/entrypoint.sh"]
CMD ["sh", "-c", "gunicorn -w 2 -k gthread --threads 8 --timeout 600 -b 0.0.0.0:${PORT} wsgi:app"]
