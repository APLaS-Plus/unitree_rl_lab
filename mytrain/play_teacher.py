# Copyright (c) 2024-2025, Adaptation Module Authors
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play (evaluate) the Teacher policy in simulation.

This script runs the teacher policy and can optionally collect data
for training the adaptation module.
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
parser = argparse.ArgumentParser(
    description="Play Teacher Policy and optionally collect data."
)
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

# Data collection arguments
parser.add_argument(
    "--collect_data",
    action="store_true",
    default=False,
    help="Collect data for adaptation training.",
)
parser.add_argument(
    "--data_save_path",
    type=str,
    default="dataset/teacher_data.pt",
    help="Path to save collected data.",
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


def main():
    """Play Teacher Policy."""
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

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )

    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

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
            "video_folder": os.path.join(log_dir, "videos", "play_teacher"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner and load policy
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Get teacher actor for latent extraction (first layer hook)
    try:
        teacher_actor = runner.alg.policy.actor
    except AttributeError:
        teacher_actor = runner.alg.actor_critic.actor

    dt = env.unwrapped.step_dt

    # Data collection buffers
    if args_cli.collect_data:
        student_obs_list = []
        teacher_latent_list = []
        phys_params_list = []

        # 检查是否使用RMAActorCritic (带encoder)
        policy_module = (
            runner.alg.policy
            if hasattr(runner.alg, "policy")
            else runner.alg.actor_critic
        )
        has_rma_encoder = hasattr(policy_module, "encoder")

        if has_rma_encoder:
            print("[INFO] RMAActorCritic detected. Will use encoder output for latent.")
        else:
            # 兼容旧版: 使用hook捕获第一层输出
            print("[INFO] Standard ActorCritic detected. Using first layer hook.")
            first_layer_output = {}

            def hook_fn(module, input, output):
                first_layer_output["latent"] = output.detach()

            first_layer = list(teacher_actor.children())[0]
            hook_handle = first_layer.register_forward_hook(hook_fn)
            print(f"[INFO] Hook attached to: {first_layer}")

    # reset environment
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    else:
        obs = env.get_observations()

    # simulate environment
    print(f"[INFO] Running for {args_cli.max_steps} steps...")
    for step in range(args_cli.max_steps):
        start_time = time.time()

        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            # Collect data if enabled
            if args_cli.collect_data:
                # Get observations from underlying env
                raw_obs = env.unwrapped.observation_manager.compute()

                # Student obs (proprioception with history)
                if "student_policy" in raw_obs:
                    student_obs_list.append(raw_obs["student_policy"].clone().cpu())

                # Teacher latent (from encoder or hook)
                if has_rma_encoder:
                    # RMAActorCritic: 直接调用encoder获取隐向量
                    from tensordict import TensorDict

                    obs_td = TensorDict(
                        raw_obs, batch_size=[raw_obs["policy"].shape[0]]
                    )
                    latent = policy_module.get_encoder_output(obs_td)
                    teacher_latent_list.append(latent.clone().cpu())
                elif "latent" in first_layer_output:
                    # 标准ActorCritic: 使用hook捕获的输出
                    teacher_latent_list.append(
                        first_layer_output["latent"].clone().cpu()
                    )

                # Physical params (privileged info from new observation group)
                # 新版观测配置使用独立的 'privileged' 组
                # privileged维度: joint_effort(12) + base_lin_vel(3) + friction(2) + base_mass(1) + feet_heights(4) = 22
                if "privileged" in raw_obs:
                    phys_params_list.append(raw_obs["privileged"].clone().cpu())
                else:
                    # 兼容旧格式: 从policy末尾提取特权信息
                    policy_obs = raw_obs["policy"]
                    phys_start_idx = -18  # 旧版不含feet_heights
                    phys_params_list.append(
                        policy_obs[:, phys_start_idx:].clone().cpu()
                    )

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

    # Save collected data
    if args_cli.collect_data:
        # 清理hook (如果使用了)
        if not has_rma_encoder and "hook_handle" in dir():
            hook_handle.remove()

        os.makedirs(os.path.dirname(args_cli.data_save_path), exist_ok=True)

        data = {
            "student_obs": torch.cat(student_obs_list, dim=0),
            "teacher_latent": torch.cat(teacher_latent_list, dim=0),
            "phys_params": torch.cat(phys_params_list, dim=0),
        }

        torch.save(data, args_cli.data_save_path)
        print(f"[INFO] Data saved to: {args_cli.data_save_path}")
        print(f"  - student_obs shape: {data['student_obs'].shape}")
        print(f"  - teacher_latent shape: {data['teacher_latent'].shape}")
        print(f"  - phys_params shape: {data['phys_params'].shape}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
