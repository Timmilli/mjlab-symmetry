from __future__ import annotations

import torch
from tensordict import TensorDict
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.entity.entity import Entity
import inspect

from mjlab.envs.mdp.observations import ObservationFunc
from mjlab.envs.mdp.actions.actions_config import JointPositionActionCfg


def compute_symmetric_states(
    env: ManagerBasedRlEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Augments the given observations and actions by applying symmetry transformations.

    This function creates augmented versions of the provided observations and actions by applying
    four symmetrical transformations: original, left-right, front-back, and diagonal. The symmetry
    transformations are beneficial for reinforcement learning tasks by providing additional
    diverse data without requiring additional data collection.

    Args:
        env: The environment instance.
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """
    obs_aug = None
    actions_aug = None

    if obs is None and actions is None:
        return obs_aug, actions_aug

    env = env.unwrapped

    if obs is not None:
        assert isinstance(obs, TensorDict)

        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        assert isinstance(obs_aug, TensorDict)

        active_terms = env.observation_manager.active_terms

        for group in active_terms.keys():
            term_cfgs = env.observation_manager._group_obs_term_cfgs[group]
            for index in range(len(active_terms[group])):
                term_cfg = term_cfgs[index]
                observation_func = term_cfg.func
                if hasattr(observation_func, "apply_symmetry"):
                    term_dim = env.observation_manager.group_obs_term_dim[group]
                    offset = sum([k[0] for k in term_dim[:index]])
                    observation_func.apply_symmetry(
                        obs_aug[group][
                            batch_size:,
                            offset : offset + term_dim[index][0],
                        ],
                    )

    if actions is not None:
        assert isinstance(actions, torch.Tensor)
        batch_size = actions.shape[0]
        actions_aug = actions.repeat(2, 1)
        robot: Entity = env.scene.entities["robot"]
        offset = robot.data.joint_pos_target.shape[1] - actions_aug.shape[1]
        action_keys = [k for k in env.cfg.actions.keys()]
        action = env.cfg.actions[action_keys[0]]
        assert isinstance(action, JointPositionActionCfg)
        dofs_filter = action.actuator_names[0]
        symmetry_regex = r"(?i).*_(roll|yaw)$"
        ObservationFunc(
            env, None, symmetry_regex=symmetry_regex, dofs_filter=dofs_filter
        ).apply_joint_symmetry(
            actions_aug[batch_size:]
        )  # TODO: change that so it doesn't re-initialize for each env
        import ipdb

        ipdb.set_trace()

    return obs_aug, actions_aug
