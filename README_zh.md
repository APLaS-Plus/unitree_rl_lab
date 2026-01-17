# Unitree RL Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?style=flat&logo=Discord&logoColor=white)](https://discord.gg/ZwcVwxv5rq)

## 概览 (Overview)

本项目提供了一套基于 [IsaacLab](https://github.com/isaac-sim/IsaacLab) 构建的宇树 (Unitree) 机器人强化学习环境。

目前支持 Unitree **Go2**, **H1** 和 **G1-29dof** 机器人。

<div align="center">

| <div align="center"> Isaac Lab </div>                                                                                          | <div align="center"> Mujoco </div>                                                                                                | <div align="center"> 实物 (Physical) </div>                                                                                     |
| ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| [<img src="https://oss-global-cdn.unitree.com/static/d879adac250648c587d3681e90658b49_480x397.gif" width="240px">](g1_sim.gif) | [<img src="https://oss-global-cdn.unitree.com/static/3c88e045ab124c3ab9c761a99cb5e71f_480x397.gif" width="240px">](g1_mujoco.gif) | [<img src="https://oss-global-cdn.unitree.com/static/6c17c6cf52ec4e26bbfab1fbf591adb2_480x270.gif" width="240px">](g1_real.gif) |

</div>

## 安装 (Installation)

- 按照 [安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 安装 Isaac Lab。
- 安装 Unitree RL IsaacLab 独立环境。
  - 将此仓库克隆或复制到 Isaac Lab 安装目录之外（即 `IsaacLab` 目录之外）：

    ```bash
    git clone https://github.com/unitreerobotics/unitree_rl_lab.git
    ```

  - 使用已安装 Isaac Lab 的 python 解释器，以编辑模式安装该库：

    ```bash
    conda activate env_isaaclab
    ./unitree_rl_lab.sh -i
    # 重启 shell 以激活环境更改。
    ```

- 下载宇树机器人描述文件

  _方法 1: 使用 USD 文件_
  - 从 [unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model/tree/main) 下载 unitree usd 文件，保持文件夹结构
    ```bash
    git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
    ```
  - 在 `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py` 中配置 `UNITREE_MODEL_DIR`。

    ```bash
    UNITREE_MODEL_DIR = "</home/user/projects/unitree_usd>"
    ```

  _方法 2: 使用 URDF 文件 [推荐]_ 仅适用于 Isaacsim >= 5.0
  - 从 [unitree_ros](https://github.com/unitreerobotics/unitree_ros) 下载 unitree 机器人 urdf 文件
    ```
    git clone https://github.com/unitreerobotics/unitree_ros.git
    ```
  - 在 `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py` 中配置 `UNITREE_ROS_DIR`。
    ```bash
    UNITREE_ROS_DIR = "</home/user/projects/unitree_ros/unitree_ros>"
    ```
  - [可选]: 如果你想使用 urdf 文件，请更改 _robot_cfg.spawn_

- 通过以下方式验证环境是否正确安装：
  - 列出可用任务：

    ```bash
    ./unitree_rl_lab.sh -l # 这是一个比 isaaclab 更快的版本
    ```

  - 运行任务：

    ```bash
    ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity # 支持任务名称自动补全
    # 等同于
    python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity
    ```

  - 使用训练好的智能体进行推理（演示）：

    ```bash
    ./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity # 支持任务名称自动补全
    # 等同于
    python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity
    ```

## 部署 (Deploy)

模型训练完成后，我们需要在 Mujoco 中对训练好的策略执行 sim2sim (仿真到仿真) 测试，以测试模型性能。
然后进行 sim2real (仿真到真机) 部署。

### 设置 (Setup)

```bash
# 安装依赖
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
# 安装 unitree_sdk2
git clone git@github.com:unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF # 安装到 /usr/local 目录
sudo make install
# 编译 robot_controller
cd unitree_rl_lab/deploy/robots/g1_29dof # 或其他机器人
mkdir build && cd build
cmake .. && make
```

### Sim2Sim

安装 [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco?tab=readme-ov-file#installation)。

- 将 `/simulate/config.yaml` 中的 `robot` 设置为 g1
- 将 `domain_id` 设置为 0
- 将 `enable_elastic_hand` 设置为 1
- 将 `use_joystck` 设置为 1

```bash
# 启动仿真
cd unitree_mujoco/simulate/build
./unitree_mujoco
# ./unitree_mujoco -i 0 -n eth0 -r g1 -s scene_29dof.xml # 替代方案
```

```bash
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl
# 1. 按 [L2 + Up] 让机器人站起来
# 2. 点击 mujoco 窗口，然后按 8 让机器人脚接触地面
# 3. 按 [R1 + X] 运行策略
# 4. 点击 mujoco 窗口，然后按 9 禁用弹性带
```

### Sim2Real

你可以使用此程序直接控制机器人，但请确保板载控制程序已关闭。

```bash
./g1_ctrl --network eth0 # eth0 是网络接口名称
```

## 致谢 (Acknowledgements)

本仓库建立在以下开源项目的支持和贡献之上。特别感谢：

- [IsaacLab](https://github.com/isaac-sim/IsaacLab): 训练和运代码的基础。
- [mujoco](https://github.com/google-deepmind/mujoco.git): 提供强大的仿真功能。
- [robot_lab](https://github.com/fan-ziqi/robot_lab): 参考了项目结构和部分实现。
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking): 通用的人形机器人运动跟踪控制框架。

### 详细文件结构分析 (Detailed File Structure Analysis)

以下是 `unitree_rl_lab` 目录的详细文件结构及其说明，特别是源代码部分的深入解析：

```
unitree_rl_lab/
├── deploy/                  # 部署相关的 C++ 代码
│   ├── include/             # 头文件目录
│   ├── robots/              # 不同机器人的部署代码
│   │   ├── b2/              # B2 机器人
│   │   ├── g1_23dof/        # G1 (23自由度) 机器人
│   │   ├── g1_29dof/        # G1 (29自由度) 机器人
│   │   ├── go2/             # Go2 机器人
│   │   ├── go2w/            # Go2W 机器人
│   │   ├── h1/              # H1 机器人
│   │   └── h1_2/            # H1_2 机器人
│   └── thirdparty/          # 第三方依赖库 (json, lcm 等)
│
├── doc/                     # 文档资料
├── docker/                  # Docker 构建相关文件
├── scripts/                 # Python 脚本，主要用于训练和运行
│   ├── mimic/               # 模仿学习数据处理脚本
│   │   ├── csv_to_npz.py    # 将 CSV 格式的动捕数据转换为 NPZ 格式
│   │   └── replay_npz.py    # 回放 NPZ 格式的数据以进行验证
│   ├── rsl_rl/              # 基于 RSL-RL 库的训练和推理脚本
│   │   ├── train.py         # 启动强化学习训练
│   │   └── play.py          # 加载训练好的模型进行推理演示
│   └── list_envs.py         # 列出所有可用环境/任务的脚本
│
├── source/                  # Python 源代码目录
│   └── unitree_rl_lab/      # 主要的 python 包
│       ├── unitree_rl_lab/  # 包的具体实现逻辑
│       │   ├── assets/      # 资产定义
│       │   │   └── robots/  # 机器人定义 (unitree.py) 和执行器参数 (unitree_actuators.py)
│       │   ├── tasks/       # 强化学习任务定义的核心目录
│       │   │   ├── locomotion/ # 通用移动任务 (Locomotion)
│       │   │   │   ├── agents/ # 代理配置
│       │   │   │   ├── mdp/    # 马尔可夫决策过程相关 (观测、奖励等)
│       │   │   │   └── robots/ # 针对特定机器人的任务配置 (g1, go2, h1)
│       │   │   └── mimic/      # 模仿学习任务 (Mimic)
│       │   │       └── robots/ # 针对特定机器人的模仿任务配置 (主要是 g1_29dof)
│       │   └── utils/       # 工具函数
│       │       ├── export_deploy_cfg.py # 导出部署配置
│       │       └── parser_cfg.py        # 配置文件解析
│       ├── setup.py         # Python 包安装脚本
│       └── pyproject.toml   # 项目配置文件
│
├── unitree_rl_lab.sh        # 项目管理脚本 (安装、运行任务等)
├── mapexample.py            # 地图示例脚本
├── requirements.txt         # Python 依赖列表
└── README.md                # 原始英文说明文档
```

### 关键目录与功能说明

1.  **`deploy/` (部署层)**
    - 这是将训练好的策略应用到实际机器人或高性能仿真 (MuJoCo C++) 的关键部分。
    - 结构按机器人型号划分 (如 `g1_29dof`, `go2`)，每个目录下通常包含独立的 CMake 构建配置和控制源码。

2.  **`source/unitree_rl_lab/unitree_rl_lab/` (核心逻辑层)**
    - **`tasks/`**: 所有的环境逻辑都在这里定义。
      - **`locomotion/`**: 标准的运动控制任务，包含不同机器人的配置文件 (如 `tasks/locomotion/robots/go2/config.py` 等)。
      - **`mimic/`**: 涉及模仿学习的任务，通常需要根据参考数据 (Reference Motion) 训练。
    - **`assets/`**: 定义了机器人的物理属性、URDF/USD 路径引用以及电机 (Actuator) 的 PD 参数等。

3.  **`scripts/` (执行层)**
    - **`rsl_rl/`**: 是用户最常交互的目录，包含 `train.py` 和 `play.py`。使用 `task` 参数来指定在 `source/.../tasks` 中定义的任务名称。
    - **`mimic/`**: 如果你进行模仿学习研究，需要先用这里的脚本处理动捕数据。
