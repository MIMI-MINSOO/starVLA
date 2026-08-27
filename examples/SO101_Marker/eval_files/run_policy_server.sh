#!/usr/bin/env bash
# SO101_Marker — download a trained checkpoint from HuggingFace and serve it
# over the GR00T N1.6 ZMQ protocol (byte-compatible with leisaac's
# Gr00t16ServicePolicyClient — see StarVLA_파이프라인_검증_결과.md §6).
#
# Run this on whichever machine will host the policy (needs the GPU + this
# .venv). The Isaac Sim / leisaac side connects to it over the network.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # run from repo root (starVLA/)

###########################################################################################
# === Edit these ===
hf_repo_id="mimiminsoo/starvla-qwenpiv3-so101-marker-rt1warmstart"
local_ckpt_dir="playground/Checkpoints/${hf_repo_id##*/}"   # download destination
port=5555
###########################################################################################

source .venv/bin/activate

echo "[1/2] Downloading ${hf_repo_id} -> ${local_ckpt_dir} (skips files already present) ..."
mkdir -p "${local_ckpt_dir}"
huggingface-cli download "${hf_repo_id}" --local-dir "${local_ckpt_dir}"

# Locate the actual .pt file: prefer final_model/pytorch_model.pt, else the
# highest-step checkpoints/steps_N_pytorch_model.pt. read_mode_config()
# (starVLA/model/framework/share_tools.py) requires config.yaml and
# dataset_statistics.json to sit two directories above the .pt file, i.e. in
# ${local_ckpt_dir} itself — that's exactly what push_model_to_hf.py uploads
# (the whole run directory), so no extra massaging should be needed.
if [ -f "${local_ckpt_dir}/final_model/pytorch_model.pt" ]; then
  ckpt_path="${local_ckpt_dir}/final_model/pytorch_model.pt"
else
  ckpt_path=$(ls -v "${local_ckpt_dir}"/checkpoints/steps_*_pytorch_model.pt 2>/dev/null | tail -n1)
fi
if [ -z "${ckpt_path:-}" ] || [ ! -f "${ckpt_path}" ]; then
  echo "ERROR: couldn't find a .pt checkpoint under ${local_ckpt_dir}/{final_model,checkpoints}/" >&2
  echo "        ls -R ${local_ckpt_dir} to see what was actually downloaded." >&2
  exit 1
fi
for required in config.yaml dataset_statistics.json; do
  if [ ! -f "${local_ckpt_dir}/${required}" ]; then
    echo "ERROR: ${local_ckpt_dir}/${required} missing — read_mode_config() needs it next to the run dir." >&2
    exit 1
  fi
done
echo "[2/2] Using checkpoint: ${ckpt_path}"

python deployment/model_server/server_policy_gr00t_zmq.py \
  --ckpt_path "${ckpt_path}" \
  --port "${port}" \
  --use_bf16
