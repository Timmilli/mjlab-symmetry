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

from copy import deepcopy

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class ObservationFunc:
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    cfg: ObservationTermCfg | None,
    symmetry_regex: str | None = None,
    dofs_filter: str | None = None,
  ):
    self.env: ManagerBasedRlEnv = env
    if cfg is not None:
      self.cfg: ObservationTermCfg = cfg

    if symmetry_regex is None:
      self.symmetry_regex: str = self.cfg.symmetry_regex
    else:
      self.symmetry_regex = symmetry_regex
    if dofs_filter is None:
      self.dofs_filter: str | None = self.cfg.dofs_filter
    else:
      self.dofs_filter = dofs_filter

    self.robot: Entity = self.env.scene.entities["robot"]

    # Get all joint ids and names of the robot
    self.joint_ids: list[int]
    self.joint_ids, joint_names = self.robot.find_joints_by_actuator_names(r".*")
    self.inversed_joint_ids: list[int] = deepcopy(self.joint_ids)

    # For each left joint, find its right and switch the indexes
    for joint_id, joint_name in zip(self.joint_ids, joint_names):
      if "left" in joint_name.lower():
        opposite_joint_name = joint_name.lower().replace("left", "right")
        opposite_joint_id, _ = self.robot.find_joints_by_actuator_names(
          r"(?i)" + opposite_joint_name
        )

        tmp = self.inversed_joint_ids[joint_id]
        self.inversed_joint_ids[joint_id] = self.inversed_joint_ids[
          opposite_joint_id[0]
        ]
        self.inversed_joint_ids[opposite_joint_id[0]] = tmp

    # If the joint should be ignored by dofs_filter, ignores it
    if self.dofs_filter is None:
      removed_position_ids = []
    else:
      removed_position_ids, _ = self._get_removed_joints(self.dofs_filter)

    position_joint_ids = deepcopy(self.joint_ids)
    position_inversed_joint_ids = deepcopy(self.inversed_joint_ids)

    for joint_id in removed_position_ids:
      position_joint_ids.remove(joint_id)
      position_inversed_joint_ids.remove(joint_id)

    # If the joint should be included by the symmetry_regex, includes it
    removed_sign_ids, _ = self._get_removed_joints(self.symmetry_regex)

    sign_joint_ids = deepcopy(position_joint_ids)

    for joint_id in removed_sign_ids:
      if joint_id not in removed_position_ids:
        sign_joint_ids.remove(joint_id)

    # Transform everything into torch.Tensor for easier manipulation later
    # and fixes the indexes according to how many joints have been removed
    self.position_joint_ids: torch.Tensor = torch.tensor(
      position_joint_ids, device=self.env.device, dtype=torch.int
    ) - len(removed_position_ids)  # Only because Head is first in the joint list
    self.position_inversed_joint_ids: torch.Tensor = torch.tensor(
      position_inversed_joint_ids, device=self.env.device, dtype=torch.int
    ) - len(removed_position_ids)  # Only because Head is first in the joint list

    self.sign_joint_ids: torch.Tensor = torch.tensor(
      sign_joint_ids, device=self.env.device, dtype=torch.int
    ) - len(removed_position_ids)  # Only because Head is first in the joint list

  def _get_removed_joints(self, regex: str) -> tuple[list[int], list[str]]:
    regex_flag = ""
    if "(?i)" in regex:
      regex_flag = "(?i)"
    return self.robot.find_joints_by_actuator_names(
      rf"{regex_flag}" + r"^(.(?!(" + regex.replace("(?i)", "") + r")))*$"
    )

  def apply_joint_symmetry(
    self,
    obs: torch.Tensor,  # [N, nb_joints]
  ) -> None:
    # Switch between left and right
    obs[:, self.position_joint_ids] = obs[:, self.position_inversed_joint_ids]
    # Flip the signs according to sagittal plan
    obs[:, self.sign_joint_ids] = -1 * obs[:, self.sign_joint_ids]

  def apply_xyz_symmetry(
    self,
    obs: torch.Tensor,  # [N, 3]
  ) -> None:
    obs[:, 1] *= -1

  def apply_rpy_symmetry(
    self,
    obs: torch.Tensor,  # [N, 3]
  ) -> None:
    obs[:, [0, 2]] *= -1

  def apply_foot_symmetry(
    self,
    obs: torch.Tensor,  # [N, 2]
  ) -> None:
    obs[:, [0, 1]] = obs[:, [1, 0]]


##
# Root state.
##


class base_lin_vel(ObservationFunc):
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    cfg: ObservationTermCfg,
  ):
    super().__init__(env, cfg)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b

  def apply_symmetry(self, obs: torch.Tensor) -> None:
    self.apply_xyz_symmetry(obs)


class base_ang_vel(ObservationFunc):
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    cfg: ObservationTermCfg,
  ):
    super().__init__(env, cfg)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_ang_vel_b

  def apply_symmetry(self, obs: torch.Tensor) -> None:
    self.apply_rpy_symmetry(obs)


class projected_gravity(ObservationFunc):
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    cfg: ObservationTermCfg,
  ):
    super().__init__(env, cfg)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b

  def apply_symmetry(self, obs: torch.Tensor) -> None:
    self.apply_xyz_symmetry(obs)


##
# Joint state.
##


class joint_pos_rel(ObservationFunc):
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    cfg: ObservationTermCfg,
  ):
    super().__init__(env, cfg)

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
    self.apply_joint_symmetry(obs)


class joint_vel_rel(ObservationFunc):
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    cfg: ObservationTermCfg,
  ):
    super().__init__(env, cfg)

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
    self.apply_joint_symmetry(obs)


##
# Actions.
##


class last_action(ObservationFunc):
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    cfg: ObservationTermCfg,
  ):
    super().__init__(env, cfg)

  def __call__(
    self, env: ManagerBasedRlEnv, action_name: str | None = None
  ) -> torch.Tensor:
    if action_name is None:
      return env.action_manager.action
    return env.action_manager.get_term(action_name).raw_action

  def apply_symmetry(self, obs: torch.Tensor):
    self.apply_joint_symmetry(obs)


##
# Commands.
##


class generated_commands(ObservationFunc):
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    cfg: ObservationTermCfg,
  ):
    super().__init__(env, cfg)

  def __call__(self, env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    assert command is not None
    return command

  def apply_symmetry(self, obs: torch.Tensor) -> None:
    self.apply_joint_symmetry(obs)


##
# Sensors.
##


class builtin_sensor(ObservationFunc):
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    cfg: ObservationTermCfg,
  ):
    super().__init__(env, cfg)

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
    if "ang_vel" in self.cfg.params["sensor_name"]:
      self.apply_rpy_symmetry(obs)
    elif "lin_vel" in self.cfg.params["sensor_name"]:
      self.apply_xyz_symmetry(obs)
