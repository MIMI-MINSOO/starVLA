# RoboDojo

StarVLA supports [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo)
training and, through
[XPolicyLab](https://github.com/JinhuiYE/XPolicyLab/tree/fix/starvla-hf-robodojo-eval).
This directory provides the StarVLA-side data registration, reproducible
training recipes, and a thin launcher for evaluating the released Hugging Face
checkpoints. RoboDojo remains the source of the simulator, tasks, assets, and
official scoring; XPolicyLab maintains the policy adapter and evaluation
runtime.

The released policies share the same observation/action contract:

| Item | Value |
|---|---|
| Base VLM | `Qwen/Qwen3-VL-4B-Instruct` |
| Training data | RoboDojo LeRobot v2.1, 3,500 episodes, 35 training tasks |
| RGB observations | Head, left wrist, right wrist; resized to 224 x 224 |
| Robot state | Raw 14D ARX X5 absolute-joint state |
| Action | 14D absolute joint position (`abs_qpos`) |
| Normalization | Saved `arx_x5` q99 statistics, including continuous grippers |
| Predicted action horizon | 50 |
| Evaluation replanning interval | Execute 16 actions, then request a new chunk |

## Training

### Download the official dataset

The recipes read RoboDojo's official 64 GB LeRobot v2.1 export directly. No
StarVLA data conversion is required.

```bash
cd /absolute/path/to/RoboDojo

ROBO_DOJO_DATA_ROOT=/absolute/path/to/shared-datasets \
  bash scripts/RoboDojo/download_data.sh huggingface lerobot_v2.1
```

The downloader creates:

```text
/absolute/path/to/shared-datasets/RoboDojo_lerobot_v21_video
```

The checked-in
[`modality.json`](train_files/modality.json) matches the metadata shipped with
that physical dataset. The data registry selects the three RGB streams used by
the released policies and ignores additional cameras.

### Select a recipe

| Variant | Action head | Recipe | Released step |
|---|---|---|---:|
| QwenOFT | MLP with L1 action regression | `starvla_robodojo_v21_qwenoft_h50_q99.yaml` | 100,000 |
| QwenGR00T | 16-layer DiT-B flow-matching head | `starvla_robodojo_v21_qwengroot_h50_q99.yaml` | 130,000 |
| QwenPI_v3 | 36-layer LayerwiseFM head | `starvla_robodojo_v21_qwenpi_v3_h50_q99.yaml` | 100,000 |

All recipes train the VLM, VLM interface, and action head end to end. Shared
optimization settings are a per-GPU batch size of 16, AdamW with betas
`(0.9, 0.95)`, VLM learning rate `1e-5`, action-model learning rate
`1e-4`, 5,000 warmup steps, and a cosine schedule.

### Launch training

Run from the StarVLA repository root. For example, to train QwenPI_v3:

```bash
config_yaml=examples/simBenchmarks/RoboDojo/train_files/starvla_robodojo_v21_qwenpi_v3_h50_q99.yaml \
robodojo_data_root=/absolute/path/to/shared-datasets \
bash examples/simBenchmarks/RoboDojo/train_files/run_robodojo_train.sh
```

The launcher uses all visible GPUs by default. The following variables control
the Accelerate topology:

| Variable | Purpose |
|---|---|
| `NUM_PROCESSES` | Total number of Accelerate processes |
| `NUM_MACHINES` | Number of training machines; default 1 |
| `MACHINE_RANK` | Rank of the current machine |
| `MAIN_PROCESS_IP` | Main-node address for multi-machine training |
| `MAIN_PROCESS_PORT` | Main-node port; default 29500 |
| `DRY_RUN=1` | Validate paths and print the resolved command without training |

Arguments after the launcher are forwarded to `train_starvla.py`.

The benchmark registry is auto-discovered from
[`train_files/data_registry/data_config.py`](train_files/data_registry/data_config.py).
It registers the 35-task mixture as `robodojo_v21_all_h50_q99`, with a
50-action chunk and q99 transforms for all state/action dimensions.

The QwenPI_v3 recipe explicitly keeps:

```yaml
diffusion_model_cfg:
  interleave_self_attention: true
```

This is the canonical alternating self/cross-attention architecture used to
train the released RoboDojo PI-v3 checkpoint.

## Evaluation

### Install RoboDojo and XPolicyLab

Use separate policy and simulator environments:

```bash
git clone https://github.com/RoboDojo-Benchmark/RoboDojo.git
cd RoboDojo

git clone --single-branch \
  --branch fix/starvla-hf-robodojo-eval \
  https://github.com/JinhuiYE/XPolicyLab.git

bash scripts/init_assets.sh

cd XPolicyLab/policy/starVLA
bash install.sh
```

Export the checkout and environment paths:

```bash
export ROBODOJO_PATH=/absolute/path/to/RoboDojo
export STARVLA_ENV_PATH=/absolute/path/to/starvla-policy-env
export ROBODOJO_ENV_PATH=/absolute/path/to/robodojo-simulator-env
```

XPolicyLab must be located at `$ROBODOJO_PATH/XPolicyLab`. Absolute environment
prefixes are recommended for non-interactive cluster jobs.

### Evaluate a released checkpoint

The StarVLA launcher delegates checkpoint download, hash verification, policy
serving, simulator startup, and result collection to XPolicyLab:

```bash
ROBODOJO_PATH=/absolute/path/to/RoboDojo \
bash examples/simBenchmarks/RoboDojo/eval_files/start_eval.sh \
  <oft|groot|pi_v3> <task> <seed> <policy_gpu> <sim_gpu> \
  "$STARVLA_ENV_PATH" "$ROBODOJO_ENV_PATH" \
  <episode_count|native>
```

For example:

```bash
# QwenOFT
ROBODOJO_PATH="$ROBODOJO_PATH" \
bash examples/simBenchmarks/RoboDojo/eval_files/start_eval.sh \
  oft build_tower 0 0 1 "$STARVLA_ENV_PATH" "$ROBODOJO_ENV_PATH" native

# QwenGR00T
ROBODOJO_PATH="$ROBODOJO_PATH" \
bash examples/simBenchmarks/RoboDojo/eval_files/start_eval.sh \
  groot build_tower 0 0 1 "$STARVLA_ENV_PATH" "$ROBODOJO_ENV_PATH" native

# QwenPI_v3
ROBODOJO_PATH="$ROBODOJO_PATH" \
bash examples/simBenchmarks/RoboDojo/eval_files/start_eval.sh \
  pi_v3 build_tower 0 0 1 "$STARVLA_ENV_PATH" "$ROBODOJO_ENV_PATH" native
```

Use an integer episode count for a short wiring check and `native` for the
task's official count. A complete RoboDojo report evaluates all 42 tasks and
uses RoboDojo's official aggregation script.

The launcher prepares these public releases:

| Variant | Hugging Face checkpoint |
|---|---|
| `oft` | [StarVLA/Qwen3vl4b-OFT-RoboDojo](https://huggingface.co/StarVLA/Qwen3vl4b-OFT-RoboDojo) |
| `groot` | [StarVLA/Qwen3vl4b-GR00T-RoboDojo](https://huggingface.co/StarVLA/Qwen3vl4b-GR00T-RoboDojo) |
| `pi_v3` | [StarVLA/StarVLA-Qwen3vl4b-PIv3-RoboDojo](https://huggingface.co/StarVLA/StarVLA-Qwen3vl4b-PIv3-RoboDojo) |

Keep each checkpoint's `config.yaml`, `config.full.yaml`, and
`dataset_statistics.json` beside its `checkpoints/` directory. The dedicated
launcher preserves this layout automatically. Do not start a second
`deployment/model_server/server_policy.py` process: the launcher starts the
matching server and validates the RGB, state-normalization, action, horizon,
and PI-v3 forward contracts before simulation.

Useful evaluation controls:

| Variable | Purpose |
|---|---|
| `STARVLA_HF_ROOT` | Shared root for materialized Hugging Face run directories |
| `STARVLA_HF_LOCAL_FILES_ONLY=1` | Disable network access and use local HF files |
| `STARVLA_HF_VERIFY_ONLY=1` | Verify an already-materialized run directory |
| `STARVLA_BASE_VLM` | Local `Qwen3-VL-4B-Instruct` path for offline runs |
| `STARVLA_ROBODOJO_NUM_ENVS` | Number of parallel Isaac environments |

## Released checkpoint results

The tables below reproduce the results published in the three Hugging Face
model cards. The protocol contains 42 evaluation tasks, 50 episodes per task,
and 2,100 episodes per policy. Values are **success rate (%) / score**; higher
is better for both. The policies train on 35 tasks, while the complete
evaluation includes held-out and open tasks.

### Overall and category summary

| Policy | Average | Generalization | Precision | Long-Horizon | Memory | Open |
|---|---:|---:|---:|---:|---:|---:|
| QwenOFT | 4.86 / 8.01 | 4.33 / 6.42 | 11.75 / 17.54 | 5.50 / 12.95 | 1.67 / 1.77 | 0.50 / 0.60 |
| QwenGR00T | 3.81 / 7.35 | 3.50 / 6.52 | 5.75 / 10.09 | 6.50 / 15.46 | 3.33 / 4.37 | 0.00 / 0.00 |
| **QwenPI_v3** | **6.19 / 9.60** | **4.17 / 7.28** | **14.00 / 19.06** | **10.00 / 17.84** | **2.00 / 2.32** | **0.75 / 0.88** |

### Per-task results

| Group | Task | QwenOFT | QwenGR00T | QwenPI_v3 |
|---|---|---:|---:|---:|
| Generalization | `stack_bowls` | 18.00 / 21.00 | 10.00 / 14.80 | 14.00 / 16.70 |
| Generalization | `push_T` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| Generalization | `pack_objects_into_box` | 0.00 / 3.10 | 0.00 / 7.80 | 0.00 / 6.80 |
| Generalization | `fold_clothes` | 10.00 / 12.80 | 8.00 / 12.40 | 2.00 / 9.60 |
| Generalization | `hang_mugs` | 0.00 / 3.60 | 0.00 / 3.00 | 0.00 / 3.50 |
| Generalization | `sweep_blocks` | 0.00 / 0.00 | 0.00 / 0.00 | 2.00 / 2.00 |
| Generalization | `pour_liquid_into_cup` | 14.00 / 14.00 | 14.00 / 14.00 | 12.00 / 12.00 |
| Generalization | `make_toast` | 0.00 / 1.00 | 2.00 / 5.00 | 2.00 / 5.00 |
| Generalization | `arrange_largest_number` | 0.00 / 1.90 | 2.00 / 4.10 | 2.00 / 5.70 |
| Generalization | `sort_nesting_dolls_by_size` | 0.00 / 0.00 | 4.00 / 4.00 | 6.00 / 6.00 |
| Generalization | `store_laptop_and_headphones` | 4.00 / 11.20 | 0.00 / 7.20 | 2.00 / 8.40 |
| Generalization | `stack_blocks` | 6.00 / 8.40 | 2.00 / 5.90 | 8.00 / 11.60 |
| Precision | `fasten_screws` | 4.00 / 8.00 | 0.00 / 2.00 | 0.00 / 6.00 |
| Precision | `plug_in_charger` | 6.00 / 6.00 | 2.00 / 2.00 | 4.00 / 4.00 |
| Precision | `insert_tubes` | 40.00 / 51.60 | 28.00 / 40.40 | 44.00 / 56.80 |
| Precision | `pour_balls_into_vase` | 8.00 / 8.00 | 0.00 / 0.00 | 2.00 / 2.00 |
| Precision | `play_Xylophone` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| Precision | `deposit_coin` | 0.00 / 3.20 | 2.00 / 5.60 | 6.00 / 7.60 |
| Precision | `insert_key` | 0.00 / 12.90 | 0.00 / 9.90 | 0.00 / 11.10 |
| Precision | `build_tower` | 36.00 / 50.60 | 14.00 / 20.80 | 56.00 / 65.00 |
| Long-Horizon | `put_bottles_into_dustbin` | 22.00 / 40.90 | 26.00 / 44.40 | 64.00 / 73.60 |
| Long-Horizon | `fill_pen_holder` | 4.00 / 11.70 | 6.00 / 14.40 | 8.00 / 23.00 |
| Long-Horizon | `classify_objects` | 2.00 / 5.50 | 6.00 / 11.50 | 0.00 / 7.50 |
| Long-Horizon | `play_tic_tac_toe` | 0.00 / 12.40 | 2.00 / 16.40 | 0.00 / 6.80 |
| Long-Horizon | `fill_egg_holder` | 0.00 / 0.60 | 0.00 / 0.00 | 0.00 / 0.80 |
| Long-Horizon | `organize_table` | 0.00 / 16.50 | 0.00 / 25.00 | 4.00 / 27.00 |
| Long-Horizon | `make_kong` | 16.00 / 16.00 | 12.00 / 12.00 | 4.00 / 4.00 |
| Long-Horizon | `play_stacking_toy` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| Memory | `cover_blocks` | 0.00 / 0.60 | 6.00 / 12.10 | 0.00 / 1.50 |
| Memory | `match_and_pick_from_conveyor` | 10.00 / 10.00 | 14.00 / 14.00 | 12.00 / 12.00 |
| Memory | `swap_blocks` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| Memory | `swap_T` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| Memory | `press_by_number` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| Memory | `imitate_sorting_sequence` | 0.00 / 0.00 | 0.00 / 0.10 | 0.00 / 0.40 |
| Open | `align_blocks` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| Open | `general_pickup` | 4.00 / 4.00 | 0.00 / 0.00 | 6.00 / 6.00 |
| Open | `stack_blocks_by_language` | 0.00 / 0.80 | 0.00 / 0.00 | 0.00 / 0.80 |
| Open | `solve_equation` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| Open | `classify_objects_by_language` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.20 |
| Open | `pick_from_conveyor_by_image` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| Open | `store_tools_in_toolbox` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| Open | `pour_by_language` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |

## Improvement opportunities

### Scope of the released baseline

These three checkpoints intentionally measure a plain StarVLA baseline. They
start from the general-purpose `Qwen3-VL-4B-Instruct` VLM, but **do not start
from a robot/action-policy pretrained checkpoint**. The action heads are trained
from scratch on the RoboDojo mixture. The recipes also use one shared policy
without task-specific heads, task planners, reward shaping, per-task
fine-tuning, curriculum scheduling, or task-dependent inference rules.

This makes the released results a clean measurement of the StarVLA
architectures, but it also leaves several known sources of performance unused.

### What the leaderboard suggests

The official [RoboDojo leaderboard](https://robodojo-benchmark.com/leaderboard)
shows the following category profile for several leading entries. Values are
again **success rate (%) / score**. The leaderboard snapshot is dated
2026-08-04. The final row is the released QwenPI_v3 model-card result shown
above; it should not be confused with the separate `StarVLA` submission on the
leaderboard.

| Policy | Average | Generalization | Precision | Long-Horizon | Memory | Open |
|---|---:|---:|---:|---:|---:|---:|
| GalaxeaVLA (G0.5) | 14.88 / 20.23 | 12.83 / 18.46 | 20.42 / 28.25 | 32.25 / 44.12 | 7.33 / 8.61 | 1.58 / 1.73 |
| Xiaomi-Robotics-1 | 13.93 / 20.07 | 17.00 / 23.55 | 18.83 / 26.69 | 23.67 / 38.39 | 6.56 / 7.81 | 3.58 / 3.94 |
| Hy-Embodied-0.5-VLA | 8.80 / 13.07 | 8.39 / 11.77 | 8.00 / 13.81 | 14.92 / 25.74 | 12.11 / 13.37 | 0.58 / 0.65 |
| Spatial Forcing | 8.04 / 12.38 | 9.33 / 14.12 | 10.58 / 17.33 | 14.58 / 23.26 | 4.11 / 5.43 | 1.58 / 1.78 |
| Pi-05 | 6.91 / 11.41 | 8.17 / 13.37 | 5.50 / 12.40 | 14.67 / 23.54 | 4.56 / 5.78 | 1.67 / 1.98 |
| InternVLA-A1.5 | 7.14 / 11.15 | 6.83 / 10.36 | 10.17 / 15.23 | 13.75 / 23.80 | 3.56 / 4.93 | 1.42 / 1.43 |
| **StarVLA QwenPI_v3 (this release)** | **6.19 / 9.60** | **4.17 / 7.28** | **14.00 / 19.06** | **10.00 / 17.84** | **2.00 / 2.32** | **0.75 / 0.88** |

Based on the category gaps and the public training configurations, the most
promising next steps are:

1. **Add robot-policy pretraining before RoboDojo adaptation.** Several
   competitive public configurations initialize from robot-pretrained policies
   such as `G0Plus_3B-base`, `pi05_base`,
   `Hy-Embodied-0.5-VLA-UMI`, or `X-VLA-Pt`; the released StarVLA recipes
   initialize only from a vision-language model. A controlled comparison should
   initialize StarVLA from an OXE/generalist robot checkpoint, then fine-tune on
   the unchanged 35-task RoboDojo mixture.

2. **Provide explicit temporal memory.** QwenPI_v3 scores 2.32 in Memory,
   whereas Hy-Embodied-0.5-VLA scores 13.37. The latter's published
   configuration samples six historical images at 20-step intervals. StarVLA
   can test history-frame tokens, a compact recurrent state, or cached
   layer-wise features while keeping one task-agnostic policy.

3. **Improve long-horizon progress modeling.** The largest absolute category
   gap is Long-Horizon: 17.84 for QwenPI_v3 versus 44.12 for the leading entry.
   Useful ablations include task-phase or subgoal prediction, progress
   estimation, temporally abstract action chunks, and training losses that
   preserve consistency across consecutive replans.

4. **Strengthen spatial precision and multi-view fusion.** Precision is the
   strongest StarVLA category, but it still trails the leading score. Higher
   resolution wrist crops, multi-scale visual tokens, calibrated cross-view
   fusion, optional depth/3D features, and endpoint-aware action losses are
   direct candidates for insertion and assembly tasks.

5. **Rebalance difficult tasks without hard-coded solutions.** The per-task
   table exposes many zero-success tasks even when partial score is non-zero.
   Task-balanced sampling, failure-trajectory mining, stage-aware data
   augmentation, and curriculum scheduling can target this long tail while
   retaining a single generalist policy.

6. **Improve open-task language and goal grounding.** Open remains the weakest
   category for almost every leaderboard entry. Instruction paraphrasing,
   reference-image conditioning, object-relation supervision, and
   language-conditioned goal/progress prediction are preferable to
   task-specific inference rules.

These directions should be evaluated as controlled ablations under the same
42-task protocol, reporting both category averages and the full per-task table
rather than optimizing only `build_tower`.

## References

- [RoboDojo documentation](https://robodojo-benchmark.com/doc/)
- [RoboDojo leaderboard](https://robodojo-benchmark.com/leaderboard)
- [XPolicyLab StarVLA integration](https://github.com/JinhuiYE/XPolicyLab/tree/fix/starvla-hf-robodojo-eval/policy/starVLA)
- [QwenOFT RoboDojo checkpoint](https://huggingface.co/StarVLA/Qwen3vl4b-OFT-RoboDojo)
- [QwenGR00T RoboDojo checkpoint](https://huggingface.co/StarVLA/Qwen3vl4b-GR00T-RoboDojo)
- [QwenPI_v3 RoboDojo checkpoint](https://huggingface.co/StarVLA/StarVLA-Qwen3vl4b-PIv3-RoboDojo)
