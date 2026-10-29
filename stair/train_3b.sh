#!/bin/zsh
# STAIR run 2 overnight: fine-tune the 3B model, pick the best checkpoint on synthetic valid questions,
# then score the untrained and fine-tuned 3B models on the held-out set. Needs data/stair/synth_more.jsonl
# (python stair/synth.py --more). Start it with: caffeinate -i nohup stair/train_3b.sh > /dev/null 2>&1 &
#
# The run sizes itself: a short pilot times one training step on this machine as it is tonight, then the
# real run gets as many steps as fit in TRAIN_HOURS (rounded down to 100 steps, at most MAX_EPOCHS).
set -euo pipefail
cd "${0:A:h}/.."
TRAIN_HOURS=7
MAX_EPOCHS=3
PILOT_STEPS=40
{
  echo "started $(date)"
  .venv/bin/python stair/build_dataset.py --more
  train_rows=$(wc -l < data/stair/sft_more/train.jsonl)

  pilot_start=$SECONDS
  .venv/bin/python -m mlx_lm lora --config stair/lora_3b.yaml --iters $PILOT_STEPS --val-batches 1 \
    --adapter-path data/stair/pilot_3b
  pilot_seconds=$(( SECONDS - pilot_start ))
  per_step=$(( pilot_seconds * 1.0 / PILOT_STEPS ))
  iters=$(( TRAIN_HOURS * 3600 * PILOT_STEPS / pilot_seconds / 100 * 100 ))
  if (( iters > MAX_EPOCHS * train_rows )); then iters=$(( MAX_EPOCHS * train_rows / 1000 * 1000 )); fi
  if (( iters < 1000 )); then echo "pilot too slow (${per_step}s/step); not starting"; exit 1; fi
  rm -rf data/stair/pilot_3b
  echo "pilot: ${per_step}s per step -> training for $iters steps ($(( iters * 1.0 / train_rows )) epochs)"

  rm -rf data/stair/adapters_3b  # a rerun must not mix in checkpoints from an earlier attempt
  /usr/bin/time -l .venv/bin/python -m mlx_lm lora --config stair/lora_3b.yaml --iters $iters
  echo "training finished $(date)"
  .venv/bin/python stair/select_checkpoint.py data/stair/adapters_3b data/stair/sft_more
  .venv/bin/python eval/run_heldout.py stair-base-3b
  .venv/bin/python eval/run_heldout.py stair-finetuned-3b
  echo "finished $(date)"
} 2>&1 | tee data/stair/train_3b.log
