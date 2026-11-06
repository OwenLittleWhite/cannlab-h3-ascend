#!/usr/bin/env bash
set -euo pipefail
API=/mnt/workspace/h3-ascend/api-gateway
export PYTHONPATH="$API/vendor/comfy-api-proxy/src:$API"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libsqlite3.so.0
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
cd "$API"
exec "$API/.venv/bin/python" serve.py --profile "$API/config/ascend.json" --state-dir "$API/runtime" --host 127.0.0.1
