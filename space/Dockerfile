FROM python:3.11-slim

WORKDIR /app

# git is needed for the qagent install from GitHub below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Light deps only (Gradio + numpy + matplotlib). The Space does not run live
# QAOA, so torch / pennylane are intentionally absent.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install qagent WITHOUT its declared deps (torch / pennylane / scipy): the
# Space only uses qagent.oracle (brute-force, greedy) and qagent.qaoa.encoding,
# all pure Python, plus the precomputed QAOA results JSON.
RUN pip install --no-cache-dir --no-deps \
    "qagent @ git+https://github.com/Quantum-Labor/qagent.git@main"

COPY app.py .
COPY safety.py .
COPY assets/ ./assets/
COPY precomputed/ ./precomputed/

# HuggingFace Spaces expect the app on 0.0.0.0:7860. No SYSTEM=spaces: this Space
# has no OAuth (it is fully open), so Gradio never takes the OAuth code path and
# the hot-reload watcher warning that SYSTEM=spaces triggers is avoided.
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    PYTHONUNBUFFERED=1
EXPOSE 7860

CMD ["python", "-u", "app.py"]
