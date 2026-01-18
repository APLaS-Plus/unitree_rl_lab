# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Symmetry data augmentation functions for quadruped robot locomotion training.

This module provides functions to generate symmetric observations and actions
for quadruped robots (e.g., Unitree Go2) to improve sample efficiency via
morphological symmetry exploitation.

Go2 Joint Configuration (SDK order):
    Index 0-2:  FR_hip, FR_thigh, FR_calf  (Front Right)
    Index 3-5:  FL_hip, FL_thigh, FL_calf  (Front Left)
    Index 6-8:  RR_hip, RR_thigh, RR_calf  (Rear Right)
    Index 9-11: RL_hip, RL_thigh, RL_calf  (Rear Left)

Symmetry Swap: FR <-> FL, RR <-> RL
"""

from __future__ import annotations

import torch
from tensordict import TensorDict


# ============================================================================
# Permutation Indices for Joint Symmetry
# ============================================================================

# Joint index mapping for left-right swap (12 joints)
# FR (0,1,2) <-> FL (3,4,5), RR (6,7,8) <-> RL (9,10,11)
JOINT_SWAP_INDICES = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

# Indices of hip joints that need sign flip (hip abduction/adduction)
HIP_JOINT_INDICES = [0, 3, 6, 9]


def _switch_go2_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the joint data tensor.

    Go2 Joint Ordering:
        FR: [0, 1, 2] (hip, thigh, calf)
        FL: [3, 4, 5]
        RR: [6, 7, 8]
        RL: [9, 10, 11]
    """
    joint_data_switched = torch.zeros_like(joint_data)
    # left <-- right: FL <- FR, RL <- RR
    joint_data_switched[..., [3, 4, 5, 9, 10, 11]] = joint_data[..., [0, 1, 2, 6, 7, 8]]
    # right <-- left: FR <- FL, RR <- RL
    joint_data_switched[..., [0, 1, 2, 6, 7, 8]] = joint_data[..., [3, 4, 5, 9, 10, 11]]

    # Flip the sign of the hip joints (abduction/adduction reverses under left-right mirror)
    joint_data_switched[..., [0, 3, 6, 9]] *= -1.0

    return joint_data_switched


@torch.no_grad()
def compute_symmetric_states(
    env,
    obs: torch.Tensor | TensorDict | None = None,
    actions: torch.Tensor | None = None,
    obs_type: str = "policy",
) -> tuple[torch.Tensor | TensorDict | None, torch.Tensor | None]:
    """Augments the given observations and actions by applying symmetry transformations.

    This function creates augmented versions of the provided observations and actions by applying
    left-right symmetrical transformation. The symmetry transformation is beneficial for
    reinforcement learning tasks by providing additional diverse data without requiring
    additional data collection.

    Args:
        env: The environment instance.
        obs: The original observation tensor or TensorDict. Defaults to None.
        actions: The original actions tensor. Defaults to None.
        obs_type: The type of observation to augment. Defaults to "policy".

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """
    # Handle TensorDict input
    if obs is not None and hasattr(obs, "batch_size"):
        # obs is a TensorDict
        is_tensordict = True
        original_obs = obs
        # Extract the observation tensor from TensorDict
        if "obs" in obs.keys():
            obs_tensor = obs["obs"]
        elif "policy" in obs.keys():
            obs_tensor = obs["policy"]
        else:
            obs_tensor = next(iter(obs.values()))
    elif obs is not None:
        is_tensordict = False
        obs_tensor = obs
    else:
        is_tensordict = False
        obs_tensor = None

    # observations
    if obs_tensor is not None:
        num_envs = obs_tensor.shape[0]
        obs_dim = obs_tensor.shape[1]
        device = obs_tensor.device

        # since we have 2 different symmetries (original + left-right), augment batch size by 2
        obs_aug = torch.zeros(num_envs * 2, obs_dim, device=device)

        # -- original
        obs_aug[:num_envs] = obs_tensor[:]

        # -- left-right symmetric
        sym_obs = obs_tensor.clone()

        # [0:3] base_ang_vel: [Roll, Pitch, Yaw]
        # Flip Roll (X) and Yaw (Z). Keep Pitch (Y) unchanged.
        sym_obs[:, 0] = -sym_obs[:, 0]  # Roll rate -> Flip
        sym_obs[:, 2] = -sym_obs[:, 2]  # Yaw rate -> Flip

        # [3:6] projected_gravity: flip Y component
        sym_obs[:, 4] = -sym_obs[:, 4]

        # [6:9] velocity_commands: flip Y (vy) and Z (wz)
        sym_obs[:, 7] = -sym_obs[:, 7]  # vy
        sym_obs[:, 8] = -sym_obs[:, 8]  # wz

        # [9:21] joint_pos_rel: swap L/R, flip hip signs
        sym_obs[:, 9:21] = _switch_go2_joints_left_right(sym_obs[:, 9:21])

        # [21:33] joint_vel_rel: swap L/R, flip hip signs
        sym_obs[:, 21:33] = _switch_go2_joints_left_right(sym_obs[:, 21:33])

        # [33:45] last_action: swap L/R, flip hip signs
        sym_obs[:, 33:45] = _switch_go2_joints_left_right(sym_obs[:, 33:45])

        # Privileged observations (if present)
        if obs_dim > 45:
            # [45:57] joint_effort: swap L/R, flip hip signs
            if obs_dim >= 57:
                sym_obs[:, 45:57] = _switch_go2_joints_left_right(sym_obs[:, 45:57])

            # [57:60] base_lin_vel: flip Y component
            if obs_dim >= 60:
                sym_obs[:, 58] = -sym_obs[:, 58]

            # friction & base_mass: no change needed

        obs_aug[num_envs : 2 * num_envs] = sym_obs

        # If input was TensorDict, return TensorDict
        if is_tensordict:
            # Construct augmented TensorDict
            obs_aug_dict = TensorDict(
                {"obs": obs_aug}, batch_size=[num_envs * 2], device=device
            )
        else:
            obs_aug_dict = obs_aug
    else:
        obs_aug_dict = None

    # actions
    if actions is not None:
        num_envs = actions.shape[0]
        action_dim = actions.shape[1]
        device = actions.device

        # since we have 2 different symmetries, augment batch size by 2
        actions_aug = torch.zeros(num_envs * 2, action_dim, device=device)

        # -- original
        actions_aug[:num_envs] = actions[:]

        # -- left-right
        actions_aug[num_envs : 2 * num_envs] = _switch_go2_joints_left_right(actions[:])
    else:
        actions_aug = None

    return obs_aug_dict, actions_aug
