#!/usr/bin/env bash
# SO101_Marker — QwenPI_v3, RT-1 backbone warm-start, ~48h real run.
#
# Settings chosen from on-GPU timing tests on this machine (RTX 3090 24GB) on
# 2026-08-22, all with `qwen_vl_interface` frozen + ZeRO-2 CPU-offloaded optimizer:
#   bs=4 -> OOM immediately
#   bs=2 -> runs, but peaks at 23.6/24.5 GiB (only ~900MB headroom) -> too risky unattended
#   bs=1 -> peaks at 20.3 GiB (~4.3 GiB headroom), steady-state 1.50 s/it -> chosen
#
# max_train_steps=115000 * 1.50s/it ~= 172,500s ~= 47.9h.
# marker_100 has 31,291 frames, so 115000 steps at bs=1 ~= 3.7 epochs over the
# dataset -- reasonable, not excessive, for a single-dataset single-task run.
#
# save_interval=10000 -> ~11-12 checkpoints, ~11GB each (full state dict,
# includes the frozen backbone) -> ~130GB disk, well within the 941GB free
# at launch time. Increase save_interval if disk gets tight.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # run from repo root (starVLA/)

###########################################################################################
Framework_name=QwenPI_v3
freeze_module_list='qwen_vl_interface'
config_yaml=./examples/SO101_Marker/train_files/starvla_qwenpiv3_so101_marker.yaml
deepspeed_accelerate_config=./examples/SO101_Marker/train_files/deepspeed_zero2_offload.yaml
data_root_dir=/home/minsoo/.cache/huggingface/lerobot/mimiminsoo
data_mix=so101_marker_task
per_device_batch_size=1
run_root_dir=./results/Checkpoints
run_id=starvla_qwenpiv3_so101_marker_from_rt1_48h
num_processes=1
max_train_steps=115000
save_interval=10000
eval_interval=1000

pretrained_checkpoint=playground/Checkpoints/Qwen3VL-PI_v3-Bridge-RT_1/checkpoints/steps_50000_pytorch_model.pt
reload_modules='qwen_vl_interface'

wandb_project=starVLA_SO101_Marker
wandb_entity=mimiminsoo
###########################################################################################

export CUDA_HOME="${CUDA_HOME:-/usr/lib/nvidia-cuda-toolkit}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ ! -f "${pretrained_checkpoint}" ]; then
  echo "Downloading StarVLA/Qwen3VL-PI_v3-Bridge-RT_1 backbone checkpoint ..."
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
    --trainer.save_interval "${save_interval}" \
    --trainer.eval_interval "${eval_interval}" \
    --trainer.pretrained_checkpoint "${pretrained_checkpoint}" \
    --trainer.reload_modules "${reload_modules}" \
    --run_root_dir "${run_root_dir}" \
    --run_id "${run_id}" \
    --wandb_project "${wandb_project}" \
    --wandb_entity "${wandb_entity}" \
    ${freeze_module_list:+--trainer.freeze_modules "${freeze_module_list}"}
