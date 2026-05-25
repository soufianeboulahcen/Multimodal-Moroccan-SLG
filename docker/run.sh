#!/usr/bin/env bash
# Wrapper to run a command inside the project's docker image with GPU + bind mount.
#
# Usage:
#   docker/run.sh                       # start interactive bash
#   docker/run.sh python --version
#   docker/run.sh python src/pose/extract_one.py /workspace/.../sample.mp4 ...
#
# Conventions:
#   - Project root is bind-mounted at /workspace/PFE-SOUFIAN inside the container.
#   - --gpus all exposes the GB10 to the container.
#   - --shm-size=8g avoids DataLoader IPC issues for later training phases.
#   - User mapping (-u) keeps file ownership consistent with the host user.
#   - --rm: containers are ephemeral; state lives in the bind mount only.
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMAGE="${PFE_IMAGE:-pfe-pose:latest}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image $IMAGE not found — building from docker/Dockerfile..." >&2
    docker build -t "$IMAGE" "$PROJECT_ROOT/docker"
fi

if [ $# -eq 0 ]; then
    set -- bash
fi

# Only allocate a TTY when stdout is one (avoids "input device is not a TTY"
# errors when invoked via ssh-without-pty, CI, or other non-interactive shells).
TTY_FLAG=()
[ -t 1 ] && TTY_FLAG=(-t)

exec docker run --rm -i "${TTY_FLAG[@]}" \
    --gpus all \
    --shm-size=8g \
    -u "$(id -u):$(id -g)" \
    -v "$PROJECT_ROOT:/workspace/PFE-SOUFIAN" \
    -w /workspace/PFE-SOUFIAN \
    "$IMAGE" \
    "$@"
