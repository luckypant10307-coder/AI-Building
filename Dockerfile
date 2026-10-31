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

FROM python:3.12-slim AS base

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

ARG WITH_VISION=0
COPY requirements-vision.txt .
RUN if [ "$WITH_VISION" = "1" ]; then \
        pip install -r requirements-vision.txt; \
    else \
        echo "slim build: skipping torch, ultralytics and opencv"; \
    fi

COPY smart_property/ ./smart_property/
COPY server.py ./
COPY web/artifact/ ./web/artifact/

# Only the three weights the app actually loads, and only when they were asked
# for. models/ is gitignored, so a Render build from git will not have them:
# that build is slim by definition, which is why this step must not fail.
COPY models/ /tmp/models/
RUN mkdir -p models \
 && if [ "$WITH_VISION" = "1" ]; then \
        for w in bd3_grouped.pt yolov8n-crack-seg-best.pt sequence_cnn.pt; do \
            [ -f "/tmp/models/$w" ] && cp "/tmp/models/$w" models/ || true; \
        done; \
    fi \
 && rm -rf /tmp/models \
 && ls -la models/ || true

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
