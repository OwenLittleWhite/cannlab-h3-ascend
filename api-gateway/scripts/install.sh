#!/usr/bin/env bash
set -euo pipefail
API=/mnt/workspace/h3-ascend/api-gateway
export PIP_CACHE_DIR=/mnt/workspace/h3-ascend/cache/pip-api
export TMPDIR=/mnt/workspace/h3-ascend/cache/tmp
unset PYTHONPATH
export PYTHONNOUSERSITE=1
cd "$API"
python3 -m venv "$API/.venv"
"$API/.venv/bin/python" -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple --requirement requirements-lock.txt
"$API/.venv/bin/python" -m pip freeze > requirements-installed.txt
touch install.done
