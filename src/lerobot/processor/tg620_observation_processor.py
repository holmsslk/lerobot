#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="tg620_joint7_observation_processor")
class TG620Joint7ObservationProcessorStep(ObservationProcessorStep):
    """
    Convert raw TG620 observations into a standard LeRobot observation contract:

    - `observation.images.image`
    - `observation.images.image2`
    - `observation.state` shaped (7,)

    The input observation is expected to contain:

    - `joint1.pos` ... `joint6.pos`
    - `gripper.pos`
    - two image keys
    """

    image_key_1: str = "external_rgb"
    image_key_2: str = "ee_rgb"
    output_image_key_1: str = f"{OBS_IMAGES}.image"
    output_image_key_2: str = f"{OBS_IMAGES}.image2"
    state_joint_keys: tuple[str, ...] = (
        "joint1.pos",
        "joint2.pos",
        "joint3.pos",
        "joint4.pos",
        "joint5.pos",
        "joint6.pos",
        "gripper.pos",
    )
    drop_source_keys: bool = True

    def observation(self, observation):
        processed_obs = observation.copy()

        state = np.asarray([processed_obs[key] for key in self.state_joint_keys], dtype=np.float32)
        processed_obs[OBS_STATE] = state

        if self.image_key_1 in processed_obs:
            processed_obs[self.output_image_key_1] = processed_obs[self.image_key_1]
        if self.image_key_2 in processed_obs:
            processed_obs[self.output_image_key_2] = processed_obs[self.image_key_2]

        if self.drop_source_keys:
            for key in self.state_joint_keys:
                processed_obs.pop(key, None)
            processed_obs.pop(self.image_key_1, None)
            processed_obs.pop(self.image_key_2, None)

        return processed_obs

    def get_config(self) -> dict[str, Any]:
        return {
            "image_key_1": self.image_key_1,
            "image_key_2": self.image_key_2,
            "output_image_key_1": self.output_image_key_1,
            "output_image_key_2": self.output_image_key_2,
            "state_joint_keys": list(self.state_joint_keys),
            "drop_source_keys": self.drop_source_keys,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = {
            key: value.copy() for key, value in features.items()
        }
        obs_features = new_features[PipelineFeatureType.OBSERVATION]

        if self.drop_source_keys:
            for key in self.state_joint_keys:
                obs_features.pop(key, None)
            obs_features.pop(self.image_key_1, None)
            obs_features.pop(self.image_key_2, None)

        obs_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(len(self.state_joint_keys),))

        if self.image_key_1 in features[PipelineFeatureType.OBSERVATION]:
            obs_features[self.output_image_key_1] = features[PipelineFeatureType.OBSERVATION][self.image_key_1]
        if self.image_key_2 in features[PipelineFeatureType.OBSERVATION]:
            obs_features[self.output_image_key_2] = features[PipelineFeatureType.OBSERVATION][self.image_key_2]

        return new_features
