# RoboDojo via XPolicyLab

StarVLA supports [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo), an Isaac Sim
benchmark for generalist robot-manipulation policies, through the companion
[XPolicyLab StarVLA integration](https://github.com/JinhuiYE/XPolicyLab/tree/fix/starvla-hf-robodojo-eval)
and the three released Hugging Face checkpoints.

As with the RoboTwin integration, the simulator is a separate checkout and its
environment is passed into the launcher as a path. Most executable integration
code intentionally lives outside this StarVLA repository:

| Repository | Responsibility |
|---|---|
| StarVLA | Data registration, training recipes, released checkpoint documentation, and a thin evaluation entry point. |
| XPolicyLab | StarVLA policy adapter, checkpoint preparation, policy-server launch, evaluation scripts, and runtime checks. |
| RoboDojo | Isaac Sim environment, tasks, assets, layouts, videos, and scoring. |

The companion launcher is the recommended way to evaluate the public
RoboDojo checkpoints. It pins the Hugging Face revisions, verifies the weight,
configuration, and normalization hashes, starts the matching vendored StarVLA
server, checks its metadata handshake, and only then starts RoboDojo. Do not
start a second `deployment/model_server/server_policy.py` process from an
arbitrary StarVLA checkout for these released checkpoints.

## Training data and released checkpoints

The three public policies were trained end to end from
`Qwen/Qwen3-VL-4B-Instruct` on the same 35-task RoboDojo LeRobot v2.1 mixture.
They use three RGB observations (head, left wrist, and right wrist), raw 14D
ARX X5 joint state, 14D absolute-joint actions, a 50-action prediction horizon,
and the saved `arx_x5` q99 normalization statistics.

| Variant | Training step | Action head | Released checkpoint |
|---|---:|---|---|
| QwenOFT | 100,000 | MLP/L1 regression | [Qwen3vl4b-OFT-RoboDojo](https://huggingface.co/StarVLA/Qwen3vl4b-OFT-RoboDojo) |
| QwenGR00T | 130,000 | 16-layer flow-matching DiT | [Qwen3vl4b-GR00T-RoboDojo](https://huggingface.co/StarVLA/Qwen3vl4b-GR00T-RoboDojo) |
| QwenPI_v3 | 100,000 | 36-layer LayerwiseFM | [StarVLA-Qwen3vl4b-PIv3-RoboDojo](https://huggingface.co/StarVLA/StarVLA-Qwen3vl4b-PIv3-RoboDojo) |

Each Hugging Face repository contains the evaluated weight plus
`config.yaml`, `config.full.yaml`, and `dataset_statistics.json`. Keep that
run-directory structure intact; the model server uses the sidecars to rebuild
the framework and unnormalize its actions.

### Train on the official RoboDojo dataset

Training uses RoboDojo's directly downloadable 64 GB LeRobot v2.1 export; no
conversion step is required. It contains 3,500 episodes from 35 tasks. Download
it from the RoboDojo checkout into a shared dataset directory:

```bash
cd /absolute/path/to/RoboDojo

ROBO_DOJO_DATA_ROOT=/absolute/path/to/shared-datasets \
  bash scripts/RoboDojo/download_data.sh huggingface lerobot_v2.1
```

This creates
`/absolute/path/to/shared-datasets/RoboDojo_lerobot_v21_video`. Select one of
the checked-in recipes:

| Variant | Training config | Released step |
|---|---|---:|
| QwenOFT | `starvla_robodojo_v21_qwenoft_h50_q99.yaml` | 100,000 |
| QwenGR00T | `starvla_robodojo_v21_qwengroot_h50_q99.yaml` | 130,000 |
| QwenPI_v3 | `starvla_robodojo_v21_qwenpi_v3_h50_q99.yaml` | 100,000 |

Then launch training from the StarVLA repository root. For example:

```bash
config_yaml=examples/simBenchmarks/RoboDojo/train_files/starvla_robodojo_v21_qwenpi_v3_h50_q99.yaml \
robodojo_data_root=/absolute/path/to/shared-datasets \
bash examples/simBenchmarks/RoboDojo/train_files/run_robodojo_train.sh
```

The launcher uses all locally visible GPUs by default. Use `NUM_PROCESSES`,
`NUM_MACHINES`, `MACHINE_RANK`, `MAIN_PROCESS_IP`, and `MAIN_PROCESS_PORT` for
explicit Accelerate topology, and `DRY_RUN=1` to inspect the resolved command.
Arguments after the script are forwarded to `train_starvla.py`.

The benchmark-specific data registry is auto-discovered from
[`train_files/data_registry/data_config.py`](train_files/data_registry/data_config.py).
It reads the official dataset in place and registers the three camera keys,
14D state/action layout, 50-step chunks, and continuous-gripper q99 transform.
The adjacent `modality.json` mirrors the metadata shipped with the physical
dataset; neither file performs a conversion.

The QwenPI_v3 recipe explicitly sets
`diffusion_model_cfg.interleave_self_attention: true`. This makes training use
the canonical alternating self/cross-attention DiT forward even though current
QwenPI_v3 defaults remain legacy-compatible for older checkpoints. The setting
must remain `true` for the released RoboDojo PI-v3 architecture.

## Environment setup

Use separate StarVLA policy and RoboDojo evaluation environments. Follow the
[RoboDojo installation guide](https://robodojo-benchmark.com/doc/usage/install-and-download/)
for Isaac Sim 5.1 and its simulator environment, then place XPolicyLab directly
inside the RoboDojo checkout:

```bash
git clone https://github.com/RoboDojo-Benchmark/RoboDojo.git
cd RoboDojo

git clone --single-branch \
  --branch fix/starvla-hf-robodojo-eval \
  https://github.com/JinhuiYE/XPolicyLab.git

# Install the official benchmark assets once.
bash scripts/init_assets.sh

# Install the StarVLA policy environment.
cd XPolicyLab/policy/starVLA
bash install.sh
```

Keep the two environment prefixes available for evaluation:

```bash
export ROBODOJO_PATH=/absolute/path/to/RoboDojo
export STARVLA_ENV_PATH=/absolute/path/to/starvla-policy-env
export ROBODOJO_ENV_PATH=/absolute/path/to/robodojo-sim-env
```

Conda environment names also work, but absolute prefixes are recommended for
non-interactive cluster shells. XPolicyLab remains at
`$ROBODOJO_PATH/XPolicyLab`; its scripts are the source of truth for the
evaluation implementation.

The commands and results below were verified with RoboDojo
`36bfcb7c580b149c6e39ed2eb77d60689152e570` and XPolicyLab
`65da3de0ac99898f3540e76d74998af92cf372b0`.

## Evaluate a released Hugging Face checkpoint

From the StarVLA repository root, use the thin wrapper in this example. Like the
RoboTwin launcher, it accepts the external benchmark checkout and environment
paths, then delegates to the scripts maintained with that integration. It
starts both the model server and simulator; no second terminal or manual
policy-server process is required.

```bash
ROBODOJO_PATH=/absolute/path/to/RoboDojo \
bash examples/simBenchmarks/RoboDojo/eval_files/start_eval.sh \
  pi_v3 build_tower 0 0 1 \
  "$STARVLA_ENV_PATH" \
  "$ROBODOJO_ENV_PATH" \
  10
```

Arguments after the script are:

```text
<oft|groot|pi_v3> <task> <seed> <policy_gpu> <sim_gpu>
<starvla_env> <robodojo_env> [episode_count|native]
```

Use `native` for the task's official episode count. A 10-episode run is a quick
contract check, not a replacement for the official 50-episode `build_tower`
protocol.

The wrapper above is equivalent to running the companion script directly:

```bash
cd "$ROBODOJO_PATH/XPolicyLab/policy/starVLA"

bash scripts/eval_hf_robodojo.sh \
  pi_v3 build_tower 0 0 1 \
  "$STARVLA_ENV_PATH" "$ROBODOJO_ENV_PATH" 10
```

QwenOFT and QwenGR00T commands, training entry points, and offline/cache
controls are maintained in
[XPolicyLab's StarVLA README](https://github.com/JinhuiYE/XPolicyLab/tree/fix/starvla-hf-robodojo-eval/policy/starVLA).

### What a valid model startup proves

Before RoboDojo moves the robot, the log must show that the downloaded files
match the release manifest, the policy server is listening, and the handshake
advertises the expected runtime contract:

- RGB images in head, left-wrist, right-wrist order;
- raw 14D ARX X5 state normalized exactly once by the server;
- unnormalized 14D absolute-joint actions returned to RoboDojo;
- a 50-action predicted chunk with 16 actions executed before replanning;
- for the pinned QwenPI_v3 release, `pi_v3_forward=canonical_interleaved`.

PI-v3 supports checkpoints with different forward semantics. The companion
manifest makes this requirement checkpoint-specific and rejects an incompatible
server before simulation, instead of accepting a loaded model that produces
invalid robot motion.

## Reproduced `build_tower` results

The public model cards report the official 50-episode references below. The
same pinned artifacts were also checked on ten seed-0 layouts through the
companion launcher.

| Released checkpoint | Official success / score | Reproduced success / score | Unstable |
|---|---:|---:|---:|
| QwenOFT | 36% / 50.60 | 4/10 / 58.00 | 0 |
| QwenGR00T | 14% / 20.80 | 1/10 / 20.00 | 0 |
| QwenPI_v3 | 56% / 65.00 | 5/10 / 57.00 | 0 |

As a final check of the StarVLA entry point shown above, QwenPI_v3 was rerun
on four seed-0 layouts: 2/4 succeeded, the score was 65.00, and none were
unstable.

All three public weights and sidecars passed full manifest SHA256 verification.
The model server reached its runtime handshake and RoboDojo reached
`Simulation App Startup Complete` before episodes began.

Near-zero success together with visibly abnormal arm motion is not treated as
small-sample noise. Check the startup contract above, the exact checkpoint
sidecars, and simulator health before interpreting the score.

## Useful launcher controls

| Variable | Purpose |
|---|---|
| `STARVLA_HF_ROOT` | Shared directory for downloaded checkpoint run directories. |
| `STARVLA_HF_LOCAL_FILES_ONLY=1` | Disable network access and use the local HF cache. |
| `STARVLA_HF_VERIFY_ONLY=1` | Verify an already-materialized run directory. |
| `STARVLA_BASE_VLM` | Use a local `Qwen3-VL-4B-Instruct` directory. |
| `STARVLA_ROBODOJO_NUM_ENVS` | Number of parallel Isaac environments; defaults to 1. |

For official task aggregation and leaderboard rules, follow the
[RoboDojo documentation](https://robodojo-benchmark.com/doc/) and
[leaderboard](https://robodojo-benchmark.com/leaderboard).
