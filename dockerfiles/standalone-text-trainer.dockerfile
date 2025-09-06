FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    vim \
    zip \
    tmux \
    iotop \
    nvtop \
    bmon \
    wget \
    nano \
    zsh \
    htop \
    redis-server \
    && rm -rf /var/lib/apt/lists/*
# Default dir
RUN mkdir -p /workspace
RUN mkdir -p /cache
RUN mkdir -p /workspace/scripts/datasets
RUN mkdir -p /app/checkpoints
WORKDIR /workspace/scripts
# Copy current folder to /workspace/auto_ml
COPY scripts /workspace/scripts
# Make entrypoint script executable
RUN chmod +x /workspace/scripts/entrypoint.sh
# Pytorch (Auto-selects backend https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection)

# Create a virtual environment for data generation
RUN python -m venv /workspace/axo_py
RUN bash -c "source /workspace/axo_py/bin/activate && \
    pip install uv && \
    pip install -U packaging==23.2 setuptools==75.8.0 wheel ninja && \
    uv pip install --no-build-isolation axolotl==0.9.1 && \
    pip install requests==2.32.3 && \
    deactivate"


# install the main dependencies
RUN pip install uv && \
    pip install -U packaging==23.2 setuptools==75.8.0 wheel ninja && \
    uv pip install -r /workspace/scripts/training_requirements.txt --system && \
    pip install hf_transfer==0.1.9 && \
    pip install tenacity==9.1.2 && \
    pip install tiktoken==0.9.0 && \
    pip install flash-attn==v2.7.4.post1 --no-build-isolation && \
    uv pip install vllm==0.8.3 --system && \
    pip install "fiber @ git+https://github.com/rayonlabs/fiber.git@2.4.0"

ENTRYPOINT ["./entrypoint.sh"]
