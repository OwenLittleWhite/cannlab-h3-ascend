#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/workspace/h3-ascend
export PIP_CACHE_DIR="$ROOT/cache/pip"
export XDG_CACHE_HOME="$ROOT/cache"
export HF_HOME="$ROOT/cache/huggingface"
export TMPDIR="$ROOT/cache/tmp"
export ASCEND_PROCESS_LOG_PATH="$ROOT/logs/ascend"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export COMFYUI_MULTI_DEVICE_BACKEND=auto
# Conservative text encoder transport; diffusion still uses native HCCL.
export COMFYUI_TEXT_TP_BACKEND=host
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1
# The template Python links SQLite 3.17. Use the existing OS SQLite 3.31 only
# for this process; modern ComfyUI's asset schema needs boolean SQL literals.
export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libsqlite3.so.0:/lib/aarch64-linux-gnu/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
mkdir -p "$TMPDIR" "$ASCEND_PROCESS_LOG_PATH"
if [ -f /home/developer/Ascend/cann-9.0.0/set_env.sh ]; then
    set +u
    source /home/developer/Ascend/cann-9.0.0/set_env.sh
    set -u
fi
cd "$ROOT/ComfyUI"
exec "$ROOT/.venv/bin/python" main.py --listen 127.0.0.1 --port 8188 \
    --disable-auto-launch --cache-classic --preview-method none \
    --database-url "sqlite:///$ROOT/ComfyUI/user/comfyui-cannlab.db" "$@"
