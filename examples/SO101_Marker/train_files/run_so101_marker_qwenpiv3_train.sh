#!/usr/bin/env bash
# SO101_Marker — QwenPI_v3 (Qwen backbone + pi0-style flow-matching action head) launch script.
# Smoke-test profile by default (max_train_steps=20). Bump via --trainer.max_train_steps
# once the run is confirmed stable.
#
# Prereqs validated on this machine (see StarVLA_파이프라인_검증_결과.md for why):
#   - .venv has flash-attn + deepspeed installed via `uv pip`, not plain `pip`
#   - CUDA_HOME must point at a real nvcc (deepspeed import fails without it)
#   - QwenPI_v3's flow-matching DiT head is memory-heavy: even with the backbone frozen
#     and per_device_batch_size=1, it OOM'd at ~20GB on a 24GB GPU. Fixed by offloading
#     the DeepSpeed ZeRO-2 optimizer state to CPU (deepspeed_zero2_offload.yaml below).
#     If you have >=40GB VRAM you can likely switch back to deepspeed_zero2.yaml + raise
#     per_device_batch_size for more throughput.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # run from repo root (starVLA/)

###########################################################################################
# === Edit these for your run ===
Framework_name=QwenPI_v3
freeze_module_list='qwen_vl_interface'     # empty string '' to fine-tune the backbone too (needs more VRAM)
config_yaml=./examples/SO101_Marker/train_files/starvla_qwenpiv3_so101_marker.yaml
deepspeed_accelerate_config=./examples/SO101_Marker/train_files/deepspeed_zero2_offload.yaml
data_root_dir=/home/minsoo/.cache/huggingface/lerobot/mimiminsoo
data_mix=so101_marker_task
per_device_batch_size=1                    # keep at 1 on a 24GB GPU; see note above
run_root_dir=./results/Checkpoints
run_id=starvla_qwenpiv3_so101_marker_smoke
num_processes=1                            # number of GPUs
max_train_steps=20                         # smoke test; raise to 5000-30000 for a real run
###########################################################################################

export CUDA_HOME="${CUDA_HOME:-/usr/lib/nvidia-cuda-toolkit}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${run_root_dir}/${run_id}"
cp "$0" "${run_root_dir}/${run_id}/$(basename "$0")"
cp "${config_yaml}" "${run_root_dir}/${run_id}/$(basename "${config_yaml}")"

source .venv/bin/activate

accelerate launch \
  --config_file "${deepspeed_accelerate_config}" \
  --num_processes ${num_processes} \
  starVLA/training/train_starvla.py \
    --config_yaml "${config_yaml}" \
    --framework.name "${Framework_name}" \
    --datasets.vla_data.data_root_dir "${data_root_dir}" \
    --datasets.vla_data.data_mix "${data_mix}" \
    --datasets.vla_data.per_device_batch_size "${per_device_batch_size}" \
    --trainer.max_train_steps "${max_train_steps}" \
    --trainer.save_interval 1000 \
    --run_root_dir "${run_root_dir}" \
    --run_id "${run_id}" \
    ${freeze_module_list:+--trainer.freeze_modules "${freeze_module_list}"}
