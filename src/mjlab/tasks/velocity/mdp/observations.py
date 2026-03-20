from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from mjlab.managers.manager_term_config import (
  ObservationTermCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

from mjlab.envs.mdp.observations import ObservationFunc

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class foot_height(ObservationFunc):
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
    return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)

  def apply_symmetry(
    self,
    obs: torch.Tensor,  # [N, 2]
  ):
    self.apply_foot_symmetry(obs)


class foot_air_time(ObservationFunc):
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
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    current_air_time = sensor_data.current_air_time
    assert current_air_time is not None
    return current_air_time

  def apply_symmetry(
    self,
    obs: torch.Tensor,  # [N, 2]
  ):
    self.apply_foot_symmetry(obs)


class foot_contact(ObservationFunc):
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
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    assert sensor_data.found is not None
    return (sensor_data.found > 0).float()

  def apply_symmetry(
    self,
    obs: torch.Tensor,  # [N, 2]
  ):
    self.apply_foot_symmetry(obs)


class foot_contact_forces(ObservationFunc):
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
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    assert sensor_data.force is not None
    forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
    return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))

  def apply_symmetry(
    self,
    obs: torch.Tensor,  # [N, 6]
  ) -> None:
    self.apply_xyz_symmetry(obs[:, :3])
    self.apply_xyz_symmetry(obs[:, 3:])
