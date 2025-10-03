#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/WARP-evaluation}"
VENV_DIR="$REPO_DIR/.venv"
TOTAL_NODES="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-1}}"
JOB_ID="${SLURM_JOB_ID:-manual}"
DONE_FILE="$REPO_DIR/.ray_done_${JOB_ID}"

# Increase file descriptor limit (important for Ray / many open files)
if command -v ulimit >/dev/null 2>&1; then
  # best-effort; ignore failure if system limit lower
  ulimit -n 1048576 || ulimit -n 65535 || true
fi

# Temp dirs
mkdir -p /tmp/$USER "$REPO_DIR/tmp/ray"
export HF_HOME="/tmp/$USER"
export RAY_TMPDIR="$REPO_DIR/tmp/ray"

echo "[$(hostname)] starting (job=$JOB_ID) total_nodes=$TOTAL_NODES"

CUDA_MODULE_NAME="nvidia/cuda/12.0"
module load $CUDA_MODULE_NAME

if ! conda activate "$VENV_DIR" >/dev/null 2>&1; then
  echo "ERROR: failed to activate conda env at $VENV_DIR" >&2
  exit 1
fi

cd "$REPO_DIR"

# (Optional) install requirements if needed
if [ -f requirements.txt ]; then
  pip install -r requirements.txt >/dev/null
fi

# ===================== Cluster Role Determination =====================
HEAD_HOST="$(scontrol show hostnames "${SLURM_NODELIST:-}" 2>/dev/null | head -n1 || hostname -s)"
LOCAL_HOST_SHORT="$(hostname -s)"
ROLE="worker"
[ "$LOCAL_HOST_SHORT" = "$HEAD_HOST" ] && ROLE="head"

echo "[$(hostname)] role=$ROLE head_host=$HEAD_HOST"

trap 'ray stop --force >/dev/null 2>&1 || true' EXIT

# ===================== Head Node Logic =====================
if [ "$ROLE" = "head" ]; then
  echo "[head] starting ray head"
  ray start --head --port=6379 --disable-usage-stats --temp-dir "$RAY_TMPDIR" || true

  if [ "$TOTAL_NODES" -gt 1 ]; then
    echo "[head] waiting for $TOTAL_NODES nodes"
    TOTAL_NODES="$TOTAL_NODES" python - <<'PY'
import os, time, ray
target = int(os.environ.get("TOTAL_NODES", "1"))
ray.init(address="auto")
for _ in range(120):
    alive = [n for n in ray.nodes() if n.get("Alive")]
    if len(alive) >= target:
        break
    time.sleep(2)
PY
  fi

  python src/evaluator.py dannkoh/WARP-benchmark --model LLM360/K2-Think --instruct --batch_size 8 --pp 2
  touch "$DONE_FILE"
  sleep 5
else
  echo "[worker] starting ray worker (head=$HEAD_HOST)"
  ray start --address="$HEAD_HOST:6379" --disable-usage-stats --temp-dir "$RAY_TMPDIR" || true
  while [ ! -f "$DONE_FILE" ]; do
    sleep 5
  done
fi

echo "[$(hostname)] finished"
