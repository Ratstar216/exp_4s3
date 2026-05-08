#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

exec uv run python hello.py \
    --robot-ids 0 1 \
    --trajectory-thickness 24 \
    --trajectory-length 120000 \
    "$@"
