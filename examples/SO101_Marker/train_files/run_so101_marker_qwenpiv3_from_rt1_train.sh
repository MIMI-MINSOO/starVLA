#!/usr/bin/env bash
# SO101_Marker — QwenPI_v3, backbone warm-started from the public
# StarVLA/Qwen3VL-PI_v3-Bridge-RT_1 checkpoint (Bridge+Fractal, WidowX/Google-robot,
# 7-dim EE-delta action space). Only `qwen_vl_interface` (the VLM backbone) is
# reloaded — action_dim differs (7 vs SO-101's 6) so action_model/project_layers
# are always freshly initialized regardless of the source checkpoint.
#
# Verified 2026-08-20: backbone loads cleanly (`load_pretrained_backbones`,
# strict=True on the `qwen_vl_interface` submodule only), 20-step smoke test
# loss 3.07 -> 1.49, checkpoint saved. See
# StarVLA_파이프라인_검증_결과.md §8 for the full writeup.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # run from repo root (starVLA/)

###########################################################################################
# === Edit these ===
Framework_name=QwenPI_v3
freeze_module_list='qwen_vl_interface'     # keep frozen on a 24GB GPU; see §2-4/§5 for why
config_yaml=./examples/SO101_Marker/train_files/starvla_qwenpiv3_so101_marker.yaml
deepspeed_accelerate_config=./examples/SO101_Marker/train_files/deepspeed_zero2_offload.yaml
data_root_dir=/home/minsoo/.cache/huggingface/lerobot/mimiminsoo
data_mix=so101_marker_task
per_device_batch_size=1
run_root_dir=./results/Checkpoints
run_id=starvla_qwenpiv3_so101_marker_from_rt1
num_processes=1
max_train_steps=20                         # smoke test; raise to 5000-30000 for a real run

pretrained_checkpoint=playground/Checkpoints/Qwen3VL-PI_v3-Bridge-RT_1/checkpoints/steps_50000_pytorch_model.pt
reload_modules='qwen_vl_interface'         # ONLY the backbone; action head always trains from scratch
###########################################################################################

export CUDA_HOME="${CUDA_HOME:-/usr/lib/nvidia-cuda-toolkit}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ ! -f "${pretrained_checkpoint}" ]; then
  echo "[1/1] Downloading StarVLA/Qwen3VL-PI_v3-Bridge-RT_1 backbone checkpoint ..."
  huggingface-cli download StarVLA/Qwen3VL-PI_v3-Bridge-RT_1 \
    checkpoints/steps_50000_pytorch_model.pt config.yaml config.full.yaml dataset_statistics.json README.md \
    --local-dir playground/Checkpoints/Qwen3VL-PI_v3-Bridge-RT_1
fi

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
    --trainer.pretrained_checkpoint "${pretrained_checkpoint}" \
    --trainer.reload_modules "${reload_modules}" \
    --run_root_dir "${run_root_dir}" \
    --run_id "${run_id}" \
    ${freeze_module_list:+--trainer.freeze_modules "${freeze_module_list}"}
