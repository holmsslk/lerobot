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

import json
import logging
import time
from functools import cached_property
from typing import TYPE_CHECKING

from lerobot.cameras import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _zmq_available, require_package

if TYPE_CHECKING or _zmq_available:
    import zmq
else:
    zmq = None

from ..robot import Robot
from .config_tg_arm620 import JOINT_ORDER, TGArm620Config

logger = logging.getLogger(__name__)


class TGArm620Follower(Robot):
    config_class = TGArm620Config
    name = "tg_arm620_follower"

    def __init__(self, config: TGArm620Config):
        require_package("pyzmq", extra="pyzmq-dep", import_name="zmq")
        if zmq is None:
            raise RuntimeError("pyzmq import is unavailable while constructing TGArm620Follower")

        self._zmq = zmq
        super().__init__(config)
        self.config = config

        self.cameras = make_cameras_from_configs(config.cameras)

        self._cmd_socket: zmq.Socket | None = None
        self._state_socket: zmq.Socket | None = None
        self._zmq_context: zmq.Context | None = None

        self._is_connected = False
        self._latest_joint_state = {joint: 0.0 for joint in JOINT_ORDER}
        self._latest_gripper = 0.0

    @cached_property
    def _state_ft(self) -> dict[str, type]:
        joint_ft = {f"{joint}.pos": float for joint in JOINT_ORDER}
        joint_ft["gripper.pos"] = float
        return joint_ft

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate

        try:
            zmq = self._zmq
            self._zmq_context = zmq.Context()

            self._cmd_socket = self._zmq_context.socket(zmq.PUSH)
            self._cmd_socket.setsockopt(zmq.CONFLATE, 1)
            self._cmd_socket.setsockopt(zmq.LINGER, 0)
            self._cmd_socket.connect(f"tcp://{self.config.remote_ip}:{self.config.cmd_port}")

            self._state_socket = self._zmq_context.socket(zmq.SUB)
            self._state_socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self._state_socket.setsockopt(zmq.CONFLATE, 1)
            self._state_socket.setsockopt(zmq.LINGER, 0)
            self._state_socket.connect(f"tcp://{self.config.remote_ip}:{self.config.state_port}")

            for cam in self.cameras.values():
                cam.connect()

            deadline = time.time() + self.config.connect_timeout_s
            while time.time() <= deadline:
                state = self._poll_latest_state(timeout_ms=100)
                if state is not None:
                    self._update_cached_state(state)
                    self._is_connected = True
                    logger.info("Connected to TG arm620 bridge and received initial state.")
                    return

            raise TimeoutError(
                f"Timed out after {self.config.connect_timeout_s}s waiting for TG arm620 state stream."
            )
        except Exception:
            self._disconnect_resources()
            raise

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def _poll_latest_state(self, timeout_ms: int) -> dict[str, float] | None:
        if self._state_socket is None:
            return None

        zmq = self._zmq
        poller = zmq.Poller()
        poller.register(self._state_socket, zmq.POLLIN)
        ready = dict(poller.poll(timeout_ms))
        if self._state_socket not in ready:
            return None

        latest_msg: str | None = None
        while True:
            try:
                latest_msg = self._state_socket.recv_string(zmq.NOBLOCK)
            except zmq.Again:
                break

        if latest_msg is None:
            return None

        return self._parse_state_message(latest_msg)

    def _parse_state_message(self, msg: str) -> dict[str, float] | None:
        try:
            payload = json.loads(msg)
        except json.JSONDecodeError:
            logger.warning("Received invalid state JSON from TG bridge. Ignoring packet.")
            return None

        if not isinstance(payload, dict):
            logger.warning("Received non-dict state packet from TG bridge. Ignoring packet.")
            return None

        state: dict[str, float] = {}
        for joint in JOINT_ORDER:
            key = f"{joint}.pos"
            fallback = self._latest_joint_state[joint]
            try:
                state[key] = float(payload.get(key, fallback))
            except (TypeError, ValueError):
                state[key] = fallback

        fallback_gripper = self._latest_gripper
        try:
            gripper = float(payload.get("gripper.pos", fallback_gripper))
        except (TypeError, ValueError):
            gripper = fallback_gripper
        state["gripper.pos"] = min(max(gripper, 0.0), 100.0)
        return state

    def _update_cached_state(self, state: dict[str, float]) -> None:
        for joint in JOINT_ORDER:
            self._latest_joint_state[joint] = state[f"{joint}.pos"]
        self._latest_gripper = state["gripper.pos"]

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        fresh_state = self._poll_latest_state(timeout_ms=self.config.polling_timeout_ms)
        if fresh_state is not None:
            self._update_cached_state(fresh_state)

        obs_dict: RobotObservation = {
            f"{joint}.pos": pos for joint, pos in self._latest_joint_state.items()
        }
        obs_dict["gripper.pos"] = self._latest_gripper

        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.read_latest()

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if self._cmd_socket is None:
            raise RuntimeError("TG arm620 command socket was not initialized")

        action_sent: dict[str, float] = {}
        for joint in JOINT_ORDER:
            key = f"{joint}.pos"
            fallback = self._latest_joint_state[joint]
            try:
                target = float(action.get(key, fallback))
            except (TypeError, ValueError):
                target = fallback
            low, high = self.config.joint_limits_rad[joint]
            action_sent[key] = min(max(target, low), high)

        try:
            gripper = float(action.get("gripper.pos", self._latest_gripper))
        except (TypeError, ValueError):
            gripper = self._latest_gripper
        action_sent["gripper.pos"] = min(max(gripper, 0.0), 100.0)

        payload = {**action_sent, "timestamp": time.time()}
        try:
            self._cmd_socket.send_string(json.dumps(payload), flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            logger.warning("TG arm620 command socket busy; dropping outdated action packet.")

        return action_sent

    def _disconnect_resources(self) -> None:
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()

        if self._state_socket is not None:
            self._state_socket.close()
            self._state_socket = None

        if self._cmd_socket is not None:
            self._cmd_socket.close()
            self._cmd_socket = None

        if self._zmq_context is not None:
            self._zmq_context.term()
            self._zmq_context = None

        self._is_connected = False

    @check_if_not_connected
    def disconnect(self) -> None:
        self._disconnect_resources()
