# Copyright (c) 2024-2025, Adaptation Module Authors
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play (evaluate) the Student (Adaptation) policy in simulation.

This script runs the combined Student-Teacher policy where:
- Student (Adaptation Net) predicts latent features from observation history
- Teacher's remaining layers produce actions from the predicted latent
"""

import gymnasium as gym
import pathlib
import sys

sys.path.insert(0, f"{pathlib.Path(__file__).parent}")
from list_envs import import_packages  # noqa: F401

tasks = []
for task_spec in gym.registry.values():
    if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
        tasks.append(task_spec.id)

import argparse
import argcomplete

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play Student (Adaptation) Policy.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument(
    "--task",
    type=str,
    default="Unitree-Go2-Velocity",
    choices=tasks,
    help="Name of the task.",
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)
parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps to run.")

# Model checkpoints
parser.add_argument(
    "--student_ckpt",
    type=str,
    required=True,
    help="Path to trained student (adaptation) model checkpoint.",
)
parser.add_argument(
    "--teacher_ckpt",
    type=str,
    default=None,
    help="Path to teacher policy checkpoint. If not provided, uses default from agent_cfg.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time
import torch
import torch.nn as nn
from importlib.metadata import version

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

# Local model import
from models import MultiHeadAdaptationNet, StudentTeacherPolicy


class StudentTeacherInference(nn.Module):
    """Combined Student-Teacher policy for RMA-style inference.

    RMA推理流程:
        1. student_obs (历史本体感知) -> Student Encoder -> predicted_latent
        2. current_proprioception + predicted_latent -> Teacher Actor -> actions
    """

    def __init__(
        self,
        student: MultiHeadAdaptationNet,
        teacher_actor: nn.Module,
        proprioception_dim: int,
    ):
        super().__init__()
        self.student = student
        self.teacher_actor = teacher_actor
        self.proprioception_dim = proprioception_dim

        # Freeze all parameters
        self.student.eval()
        for p in self.student.parameters():
            p.requires_grad = False
        for p in self.teacher_actor.parameters():
            p.requires_grad = False

    def forward(
        self, student_obs: torch.Tensor, current_proprioception: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            student_obs: Student observation (flattened history), shape (batch, obs_dim * history_len).
            current_proprioception: Current proprioception, shape (batch, proprioception_dim).

        Returns:
            actions: Action output.
        """
        # Student predicts latent
        latent, _ = self.student(student_obs)

        # Concatenate proprioception + predicted latent
        actor_input = torch.cat([current_proprioception, latent], dim=-1)

        # Pass through Teacher actor
        actions = self.teacher_actor(actor_input)
        return actions


def load_student_model(ckpt_path: str, device: str) -> MultiHeadAdaptationNet:
    """Load trained student model from checkpoint."""
    print(f"[INFO] Loading student model from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    config = ckpt["config"]
    model = MultiHeadAdaptationNet(
        obs_dim=config["obs_dim"],
        history_length=config["history_length"],
        latent_dim=config["latent_dim"],
        phys_dim=config["phys_dim"],
        conv_channels=config["conv_channels"],
        kernel_size=config.get("kernel_size", 5),
        activation="elu",
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"[INFO] Student model loaded. Config: {config}")
    return model


def main():
    """Play Student Policy."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(
        args_cli.task, args_cli
    )

    # Load Teacher checkpoint
    if args_cli.teacher_ckpt:
        teacher_resume_path = retrieve_file_path(args_cli.teacher_ckpt)
    else:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        teacher_resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )

    log_dir = os.path.dirname(teacher_resume_path)
    print(f"[INFO] Loading teacher checkpoint from: {teacher_resume_path}")

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_student"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load teacher runner to get actor
    device = agent_cfg.device
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(teacher_resume_path)

    try:
        teacher_actor = runner.alg.policy.actor
    except AttributeError:
        teacher_actor = runner.alg.actor_critic.actor

    print(f"[INFO] Teacher actor loaded: {type(teacher_actor)}")

    # Load student model
    student_model = load_student_model(args_cli.student_ckpt, device)

    # Get proprioception dimension from observation config
    # PolicyCfg维度: base_ang_vel(3) + projected_gravity(3) + velocity_commands(3) +
    #               joint_pos_rel(12) + joint_vel_rel(12) + last_action(12) = 45
    raw_obs_sample = env.unwrapped.observation_manager.compute()
    if "policy" in raw_obs_sample:
        proprioception_dim = raw_obs_sample["policy"].shape[-1]
    else:
        proprioception_dim = 45  # 默认值
    print(f"[INFO] Proprioception dimension: {proprioception_dim}")

    # Create combined policy with RMA-style inference
    combined_policy = StudentTeacherInference(
        student=student_model,
        teacher_actor=teacher_actor,
        proprioception_dim=proprioception_dim,
    )
    combined_policy.to(device)
    combined_policy.eval()

    dt = env.unwrapped.step_dt

    # reset environment
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    else:
        obs = env.get_observations()

    # simulate environment
    print(f"[INFO] Running Student policy for {args_cli.max_steps} steps...")
    for step in range(args_cli.max_steps):
        start_time = time.time()

        with torch.inference_mode():
            # Get student observation (history)
            raw_obs = env.unwrapped.observation_manager.compute()

            if "student_policy" in raw_obs:
                student_obs = raw_obs["student_policy"]
            else:
                # Fallback: use policy obs (will likely fail if dimensions don't match)
                print(
                    "[WARNING] 'student_policy' not found in observations. Using 'policy'."
                )
                student_obs = raw_obs["policy"]

            # Get current proprioception (single step, without history)
            if "policy" in raw_obs:
                current_proprioception = raw_obs["policy"]
            else:
                # 如果没有单独的policy组，从student_policy取最后一步
                obs_dim = student_obs.shape[-1] // 50  # 假设history_length=50
                current_proprioception = student_obs[:, -obs_dim:]

            # Student-Teacher forward (RMA-style: proprioception + predicted latent)
            actions = combined_policy(student_obs, current_proprioception)

            # env stepping
            obs, _, _, _ = env.step(actions)

        if args_cli.video and step == args_cli.video_length:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

        if (step + 1) % 100 == 0:
            print(f"  Step {step + 1}/{args_cli.max_steps}")

    # close the simulator
    env.close()
    print("[INFO] Student policy evaluation complete.")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
