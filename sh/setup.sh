#!/usr/bin/env bash
set -e

export PYTHONPATH=$PYTHONPATH:/root/GPU-middleware/engine/modules

if [ -d "venv" ]; then
    . venv/bin/activate
else
    echo "venv not found"
    exit 1
fi

echo "Environment ready"