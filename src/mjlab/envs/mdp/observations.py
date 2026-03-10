"""Useful methods for MDP observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor

from mjlab.managers.manager_term_config import (
    ObservationTermCfg,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


##
# Root state.
##


class base_lin_vel:
    def __init__(
        self,
        cfg: ObservationTermCfg,
        env: ManagerBasedRlEnv,
    ):
        self.cfg = cfg
        self.env = env

    def __call__(
        self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        return asset.data.root_link_lin_vel_b

    def apply_symmetry(self, obs: torch.Tensor) -> None:
        obs[1] *= -1


class base_ang_vel:
    def __init__(
        self,
        cfg: ObservationTermCfg,
        env: ManagerBasedRlEnv,
    ):
        self.cfg = cfg
        self.env = env

    def __call__(
        self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        return asset.data.root_link_ang_vel_b

    def apply_symmetry(self, obs: torch.Tensor) -> None:
        obs[1] *= -1


class projected_gravity:
    def __init__(
        self,
        cfg: ObservationTermCfg,
        env: ManagerBasedRlEnv,
    ):
        self.cfg = cfg
        self.env = env

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        return asset.data.projected_gravity_b

    def apply_symmetry(self, obs: torch.Tensor) -> None:
        obs[1] *= -1


##
# Joint state.
##


class joint_pos_rel:
    def __init__(
        self,
        cfg: ObservationTermCfg,
        env: ManagerBasedRlEnv,
    ):
        self.cfg = cfg
        self.env = env

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        assert self.cfg is not None
        asset: Entity = env.scene[asset_cfg.name]
        default_joint_pos = asset.data.default_joint_pos
        assert default_joint_pos is not None
        jnt_ids = asset_cfg.joint_ids
        return asset.data.joint_pos[:, jnt_ids] - default_joint_pos[:, jnt_ids]

    def apply_symmetry(
        self,
        obs: torch.Tensor,
    ) -> None:
        robot = self.env.scene.entities["robot"]

        if "asset_cfg" in self.cfg.params.keys():
            joint_nb = len(self.cfg.params["asset_cfg"].joint_ids)
            actuator_nb = len(robot.actuator_names)
            offset = actuator_nb - joint_nb
        else:
            offset = 0

        joint_ids, _ = robot.find_joints_by_actuator_names(self.cfg.symmetry_regex)

        inversed_indexes = (
            torch.tensor(joint_ids, device=self.env.device, dtype=torch.int) - offset
        )

        obs[inversed_indexes] *= -1


class joint_vel_rel:
    def __init__(
        self,
        cfg: ObservationTermCfg,
        env: ManagerBasedRlEnv,
    ):
        self.cfg = cfg
        self.env = env

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        default_joint_vel = asset.data.default_joint_vel
        assert default_joint_vel is not None
        jnt_ids = asset_cfg.joint_ids
        return asset.data.joint_vel[:, jnt_ids] - default_joint_vel[:, jnt_ids]

    def apply_symmetry(
        self,
        obs: torch.Tensor,
    ) -> None:
        robot = self.env.scene.entities["robot"]

        if "asset_cfg" in self.cfg.params.keys():
            joint_nb = len(self.cfg.params["asset_cfg"].joint_ids)
            actuator_nb = len(robot.actuator_names)
            offset = actuator_nb - joint_nb
        else:
            offset = 0

        joint_ids, _ = robot.find_joints_by_actuator_names(self.cfg.symmetry_regex)

        inversed_indexes = (
            torch.tensor(joint_ids, device=self.env.device, dtype=torch.int) - offset
        )

        obs[inversed_indexes] *= -1


##
# Actions.
##


class last_action:
    def __init__(
        self,
        cfg: ObservationTermCfg,
        env: ManagerBasedRlEnv,
    ):
        self.cfg = cfg
        self.env = env

    def __call__(
        self, env: ManagerBasedRlEnv, action_name: str | None = None
    ) -> torch.Tensor:
        if action_name is None:
            return env.action_manager.action
        return env.action_manager.get_term(action_name).raw_action

    def apply_symmetry(self, obs: torch.Tensor):
        robot = self.env.scene.entities["robot"]

        joint_ids, _ = robot.find_joints_by_actuator_names(self.cfg.symmetry_regex)

        offset = len(robot.actuator_names) - len(joint_ids)

        inversed_indexes = (
            torch.tensor(joint_ids, device=self.env.device, dtype=torch.int) - offset
        )

        obs[inversed_indexes] *= -1


##
# Commands.
##


class generated_commands:
    def __init__(
        self,
        cfg: ObservationTermCfg,
        env: ManagerBasedRlEnv,
    ):
        self.cfg = cfg
        self.env = env

    def __call__(self, env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
        command = env.command_manager.get_command(command_name)
        assert command is not None
        return command

    def apply_symmetry(self, obs: torch.Tensor) -> None:
        robot = self.env.scene.entities["robot"]

        if "asset_cfg" in self.cfg.params.keys():
            joint_nb = len(self.cfg.params["asset_cfg"].joint_ids)
            actuator_nb = len(robot.actuator_names)
            offset = actuator_nb - joint_nb
        else:
            offset = 0

        joint_ids, _ = robot.find_joints_by_actuator_names(self.cfg.symmetry_regex)

        inversed_indexes = (
            torch.tensor(joint_ids, device=self.env.device, dtype=torch.int) - offset
        )

        obs[inversed_indexes] *= -1


##
# Sensors.
##


class builtin_sensor:
    def __init__(
        self,
        cfg: ObservationTermCfg,
        env: ManagerBasedRlEnv,
    ):
        self.cfg = cfg
        self.env = env

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
    ) -> torch.Tensor:
        """Get observation from a built-in sensor by name."""
        sensor = self.env.scene[sensor_name]
        assert isinstance(sensor, BuiltinSensor)
        return sensor.data

    def apply_symmetry(self, obs: torch.Tensor) -> None:
        obs[1] *= -1
