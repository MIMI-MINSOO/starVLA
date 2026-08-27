"""SO101_Marker — data config, embodiment tags, and mixtures.

Tailored to the actual meta/modality.json already present in
~/.cache/huggingface/lerobot/mimiminsoo/marker_100 (single_arm + gripper
grouping, "front"/"wrist" cameras, flat "human.task_description" annotation
key) — NOT the per-joint SO101Config in examples/realRobots/Franka, whose
key names don't match this dataset's modality.json.
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


class SO101MarkerDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.front", "video.wrist"]
    state_keys = ["state.single_arm", "state.gripper"]
    action_keys = ["action.single_arm", "action.gripper"]
    # Per-key dims for PolicyNormProcessor (5+1 = 6-D total)
    action_key_dims = {"action.single_arm": 5, "action.gripper": 1}
    state_key_dims = {"state.single_arm": 5, "state.gripper": 1}
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))  # must equal trainer.action_horizon

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={k: "min_max" for k in self.state_keys},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={k: "min_max" for k in self.action_keys},
            ),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "so101_marker": SO101MarkerDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {}

DATASET_NAMED_MIXTURES = {
    "so101_marker_task": [
        ("marker_100", 1.0, "so101_marker"),
    ],
}
