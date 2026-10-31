# Smart Property AI — deployment image for Render.
#
# Two builds from one file, because the camera side is heavy:
#
#   docker build -t smartprop .                        # slim,  ~400 MB image
#   docker build -t smartprop --build-arg WITH_VISION=1 .   # full, ~1.5 GB
#
# Slim carries the sensor spine: the Isolation Forest, the physics rules, the
# forecaster, fusion and the advisor. That is components 1, 2, 3, 9 and the
# whole dashboard, and it runs inside 512 MB — Render's free instance.
#
# Full adds torch, ultralytics and the trained weights for components 5 and 7.
# Installed it is about 700 MB more, and with ResNet18 and YOLO both resident
# it wants roughly 1 GB at runtime, so it needs a 2 GB instance.
#
# Nothing breaks in the slim build. smart_property/registry.py loads each model
# only if its weights are present, and the dashboard reports which components
# are live, so a slim deployment says so on screen instead of pretending.

ARG WITH_VISION=0

FROM python:3.12-slim AS base
ARG WITH_VISION

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# libgomp is needed by scikit-learn and torch; curl is for the healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-base.txt .
RUN pip install -r requirements-base.txt

COPY requirements-vision.txt .
RUN if [ "$WITH_VISION" = "1" ]; then \
        pip install -r requirements-vision.txt; \
    else \
        echo "slim build: skipping torch, ultralytics and opencv"; \
    fi

COPY smart_property/ ./smart_property/
COPY server.py ./
COPY web/artifact/ ./web/artifact/

# Weights, by stage rather than by `if`.
#
# A COPY cannot be made conditional, and a later `rm` cannot shrink an earlier
# layer: copying models/ and then deleting it left 52 MB of unused weights in
# the slim image for good. So each variant is its own stage and WITH_VISION
# picks which one the build continues from.
FROM base AS weights-0
RUN mkdir -p models && echo "slim build: no model weights"

FROM base AS weights-1
# Named individually: models/ also holds 340 MB of training checkpoints and
# duplicate runs that the app never loads.
COPY models/bd3_grouped.pt models/yolov8n-crack-seg-best.pt \
     models/sequence_cnn.pt ./models/

FROM weights-${WITH_VISION} AS final
ARG WITH_VISION
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

# Render supplies PORT; 10000 is its default and a sane local fallback.
ENV PORT=10000
EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/api/config" > /dev/null || exit 1

# ONE worker, deliberately. server.py keeps the running flat in a module-level
# Engine with its own clock thread. A second worker would be a second flat with
# its own clock, and successive requests would land on different simulations,
# so the time on screen would jump about. Concurrency comes from threads
# instead, which share that one Engine.
CMD exec gunicorn server:app \
    --bind "0.0.0.0:${PORT}" \
    --workers 1 \
    --threads 8 \
    --timeout 120 \
    --graceful-timeout 30 \
    --access-logfile - \
    --error-logfile -
