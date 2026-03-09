"""Symmetry configuration"""

from __future__ import annotations

from dataclasses import dataclass
from types import FunctionType


@dataclass
class RslRlSymmetryCfg:
  """From IsaacLab:

  Configuration for the symmetry-augmentation in the training.

  When :meth:`use_data_augmentation` is True, the :meth:`data_augmentation_func` is used to generate
  augmented observations and actions. These are then used to train the model.

  When :meth:`use_mirror_loss` is True, the :meth:`mirror_loss_coeff` is used to weight the
  symmetry-mirror loss. This loss is directly added to the agent's loss function.

  If both :meth:`use_data_augmentation` and :meth:`use_mirror_loss` are False, then no symmetry-based
  training is enabled. However, the :meth:`data_augmentation_func` is called to compute and log
  symmetry metrics. This is useful for performing ablations.

  For more information, please check the work from :cite:`mittal2024symmetry`.
  """

  def __init__(
    self,
    use_data_augmentation: bool = False,
    use_mirror_loss: bool = False,
    data_augmentation_func: FunctionType | None = None,
    mirror_loss_coeff: float = 0.0,
  ) -> None:
    """ "use_data_augmentation": bool,
    Whether to use symmetry-based data augmentation. Default is False.

    "use_mirror_loss": bool,
    Whether to use the symmetry-augmentation loss. Default is False.

    "data_augmentation_func": FunctionType,
    The symmetry data augmentation function.

      The function signature should be as follows:

      Args:

          env (VecEnv): The environment object. This is used to access the environment's properties.
          obs (tensordict.TensorDict | None): The observation tensor dictionary. If None, the observation is not used.
          action (torch.Tensor | None): The action tensor. If None, the action is not used.

      Returns:
          A tuple containing the augmented observation dictionary and action tensors. The tensors can be None,
          if their respective inputs are None.


    "mirror_loss_coeff": Float,
    The weight for the symmetry-mirror loss. Default is 0.0."""

    self.cfg: dict[str, bool | FunctionType | float | None] = {
      "use_data_augmentation": use_data_augmentation,
      "use_mirror_loss": use_mirror_loss,
      "data_augmentation_func": data_augmentation_func,
      "mirror_loss_coeff": mirror_loss_coeff,
    }
