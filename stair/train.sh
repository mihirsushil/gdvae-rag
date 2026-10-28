#!/bin/zsh
# Overnight STAIR run: fine-tune, then score the untrained and fine-tuned models on the held-out set.
# caffeinate keeps the Mac awake until the script finishes; the log is written to data/stair/train.log.
set -euo pipefail
cd "${0:A:h}/.."
mkdir -p data/stair
{
  echo "started $(date)"
  .venv/bin/python stair/build_dataset.py
  /usr/bin/time -l .venv/bin/python -m mlx_lm lora --config stair/lora.yaml
  .venv/bin/python eval/run_heldout.py stair-base
  .venv/bin/python eval/run_heldout.py stair-finetuned
  echo "finished $(date)"
} 2>&1 | tee data/stair/train.log
