# TG620 完整流程 README

这份文档面向下面这个目标场景：

- 使用 `/home/holms/Projects/Arm_Project` 提供的遥操作链路
- 在当前 `lerobot` 仓库中完成 TG620 的
  - 数据采集
  - XVLA 训练
  - 在线推理部署
- 数据契约固定为：
  - `2` 路图像
  - `7` 维状态
  - `7` 维动作

当前这条链路在本仓库里的默认语义是：

- `observation.images.image`
- `observation.images.image2`
- `observation.state = [joint1, joint2, joint3, joint4, joint5, joint6, gripper]`
- `action = [joint1, joint2, joint3, joint4, joint5, joint6, gripper]`

也就是说，当前默认是“关节空间 + 夹爪”，不是“末端位姿 + 夹爪”。

---

## 1. 当前已经接好的内容

本仓库已经补好了 TG620 相关的几块能力：

- `tg_arm620_follower`
  - TG620 从臂机器人适配
  - 通过 ZMQ 与桥接程序通信
- `tg_arm620_ros2`
  - 从 ROS2 `/joint_target` 订阅遥操作目标
  - 直接作为 `lerobot-record` 的 teleoperator 使用
- TG620 观测标准化
  - 录制时自动把原始观测转成 LeRobot 标准格式
  - 自动生成：
    - `observation.images.image`
    - `observation.images.image2`
    - `observation.state`
- TG620 XVLA 推理脚本
  - 可直接把训练出来的 XVLA checkpoint 部署到 TG620

对应文件：

- 机器人：
  - [src/lerobot/robots/tg_arm620/config_tg_arm620.py](/home/holms/Projects/lerobot/lerobot/src/lerobot/robots/tg_arm620/config_tg_arm620.py:1)
  - [src/lerobot/robots/tg_arm620/tg_arm620.py](/home/holms/Projects/lerobot/lerobot/src/lerobot/robots/tg_arm620/tg_arm620.py:1)
- ROS2 teleop：
  - [src/lerobot/teleoperators/tg_arm620_ros2/config_tg_arm620_ros2.py](/home/holms/Projects/lerobot/lerobot/src/lerobot/teleoperators/tg_arm620_ros2/config_tg_arm620_ros2.py:1)
  - [src/lerobot/teleoperators/tg_arm620_ros2/tg_arm620_ros2.py](/home/holms/Projects/lerobot/lerobot/src/lerobot/teleoperators/tg_arm620_ros2/tg_arm620_ros2.py:1)
- 观测处理：
  - [src/lerobot/processor/tg620_observation_processor.py](/home/holms/Projects/lerobot/lerobot/src/lerobot/processor/tg620_observation_processor.py:1)
- XVLA 训练模板：
  - [examples/tg620/train_config_tg620_xvla_joint7.json](/home/holms/Projects/lerobot/lerobot/examples/tg620/train_config_tg620_xvla_joint7.json:1)
- XVLA 在线推理：
  - [examples/tg620/run_xvla_inference_tg620.py](/home/holms/Projects/lerobot/lerobot/examples/tg620/run_xvla_inference_tg620.py:1)
- ROS2 到 TG620 的独立桥接脚本：
  - [examples/tg620/ros2_joint_target_to_lerobot.py](/home/holms/Projects/lerobot/lerobot/examples/tg620/ros2_joint_target_to_lerobot.py:1)

---

## 2. 系统结构

整条链路可以理解成下面三段：

### 2.1 数据采集

`Arm_Project` 发布遥操作指令：

- ROS2 topic: `/joint_target`

LeRobot 侧完成：

- `tg_arm620_ros2` 读取遥操作指令
- `tg_arm620_follower` 读取 TG620 实际状态和相机
- `lerobot-record` 保存为 LeRobot 标准数据集

### 2.2 训练

使用采集好的 LeRobot dataset，按 XVLA 的配置训练：

- 输入：
  - 两路图像
  - 7 维状态
- 输出：
  - 7 维动作

### 2.3 部署

在线部署时：

- `tg_arm620_follower` 读取实时观测
- `run_xvla_inference_tg620.py` 调用 XVLA 预测
- 预测出的 7 维动作直接发送给 TG620

---

## 3. 依赖说明

这部分很重要。TG620 这条链路依赖不只是 Python 包，还包含 ROS2、相机、以及 TG620 外部桥接服务。

## 3.1 Python / LeRobot 基础依赖

建议环境：

- Ubuntu 22.04
- Python 3.12
- `uv`
- `git-lfs`

在本仓库根目录执行：

```bash
uv sync --locked --extra dev --extra test
git lfs install
git lfs pull
```

如果你希望更完整一些，也可以直接：

```bash
uv sync --locked --extra all
```

## 3.2 TG620 相关 Python 依赖

TG620 适配和推理这条链路会用到：

- `pyzmq`
- `opencv-python`
- `numpy`
- `torch`
- `draccus`
- `Pillow`

其中：

- `pyzmq`
  - TG620 follower 与桥接服务通信需要
- `opencv-python`
  - 使用 OpenCV 相机配置时需要
- `Pillow`
  - 如果你跑离线最小 XVLA 脚本并加载图片，需要它

如果当前环境没装齐，可以补：

```bash
uv add pyzmq opencv-python pillow
```

## 3.3 ROS2 依赖

如果你要直接读取 `Arm_Project` 的 `/joint_target`，需要 ROS2 Python 环境可用。

推荐：

- ROS2 Humble

至少要保证下面这些 Python 包来自 ROS2 环境：

- `rclpy`
- `sensor_msgs`

运行前需要先 source：

```bash
source /opt/ros/humble/setup.bash
```

如果 `Arm_Project` 自己有 workspace，也通常还需要：

```bash
source /home/holms/Projects/Arm_Project/<your_ros_ws>/install/setup.bash
```

如果你不知道具体 workspace，就先只 source `/opt/ros/humble/setup.bash`，然后用 `ros2 topic list` 检查 `/joint_target` 是否存在。

## 3.4 相机依赖

本 README 默认使用两路相机：

- `external_rgb`
- `ee_rgb`

支持的方式取决于你的相机配置。当前最容易直接跑通的是：

- OpenCV 摄像头

如果你使用的是 Realsense 或其他相机，需要其对应运行库正常可用。

## 3.5 TG620 外部桥接服务

`tg_arm620_follower` 并不是直接控制机械臂底层驱动，它是通过一个外部桥接服务进行通信：

- 状态订阅端口：默认 `6002`
- 指令发送端口：默认 `6001`

也就是说，在使用 `lerobot-record` 或推理脚本前，你必须已经有一个外部服务在负责：

- 向 LeRobot 发布 TG620 当前关节/夹爪状态
- 接收 LeRobot 发出的目标动作

如果这个桥接服务没启动，TG620 机器人连接会超时。

---

## 4. 关键数据契约

TG620 当前这条链路的数据契约如下。

## 4.1 观测

录制和推理统一使用：

- `observation.images.image`
- `observation.images.image2`
- `observation.state`

其中 `observation.state` 顺序固定为：

1. `joint1.pos`
2. `joint2.pos`
3. `joint3.pos`
4. `joint4.pos`
5. `joint5.pos`
6. `joint6.pos`
7. `gripper.pos`

## 4.2 动作

动作也是 7 维，顺序固定为：

1. `joint1.pos`
2. `joint2.pos`
3. `joint3.pos`
4. `joint4.pos`
5. `joint5.pos`
6. `joint6.pos`
7. `gripper.pos`

## 4.3 图像命名

建议机器人侧相机键名固定为：

- `external_rgb`
- `ee_rgb`

录制/推理时会自动映射成：

- `external_rgb -> observation.images.image`
- `ee_rgb -> observation.images.image2`

---

## 5. TG620 机器人配置

建议先准备一个 TG620 机器人 JSON 配置文件，例如 `examples/tg620/tg620_robot.json`。

你可以在本地任意路径创建，内容如下：

```json
{
  "type": "tg_arm620_follower",
  "id": "tg620",
  "remote_ip": "127.0.0.1",
  "cmd_port": 6001,
  "state_port": 6002,
  "control_hz": 30.0,
  "polling_timeout_ms": 300,
  "connect_timeout_s": 5.0,
  "max_latency_ms": 300,
  "max_vel_rad_s": 2.61799,
  "max_acc_rad_s2": 2.61799,
  "gripper_type": 1,
  "gripper_speed": 40.0,
  "gripper_effort": 100.0,
  "cameras": {
    "external_rgb": {
      "type": "opencv",
      "index_or_path": 0,
      "width": 640,
      "height": 480,
      "fps": 30
    },
    "ee_rgb": {
      "type": "opencv",
      "index_or_path": 1,
      "width": 640,
      "height": 480,
      "fps": 30
    }
  }
}
```

字段含义：

- `remote_ip`
  - TG620 ZMQ 桥接服务所在机器 IP
- `cmd_port`
  - LeRobot 向桥接服务发送动作的端口
- `state_port`
  - LeRobot 从桥接服务接收状态的端口
- `control_hz`
  - TG620 控制频率上限
- `max_vel_rad_s`
  - 关节速度限制
- `max_acc_rad_s2`
  - 关节加速度限制

如果桥接服务不在本机，请把 `127.0.0.1` 改成真实 IP。

---

## 6. ROS2 遥操作输入说明

LeRobot 当前读取 `Arm_Project` 的方式是：

- topic: `/joint_target`
- message type: `sensor_msgs/msg/JointState`

默认约定：

- 前 6 维：6 个关节目标
- 第 7 维：夹爪

当前 teleoperator 默认会做下面的关节符号翻转：

- `joint3`
- `joint4`
- `joint6`

这是为了和你现有 `Arm_Project` 的映射方式保持一致。

夹爪约定：

- 如果收到的是 `[-1,1]` 或 `[0,1]` 量级
  - 会按 `gripper_scale=100.0` 自动缩放到 `0..100`
- 如果本来就是大于 `1` 的数
  - 直接按 `0..100` 的指令处理

---

## 7. 正式开始前的检查

在开始录制前，建议按下面顺序检查。

## 7.1 检查 ROS2 topic

```bash
source /opt/ros/humble/setup.bash
ros2 topic list | grep joint_target
```

如果能看到 `/joint_target`，说明 ROS2 遥操作话题已发布。

你还可以看一下数据：

```bash
ros2 topic echo /joint_target
```

## 7.2 检查 TG620 ZMQ 桥接服务

确认负责 TG620 的外部桥接程序已经启动，并且：

- `6001` 能接收动作
- `6002` 能发送状态

如果桥接没启动，`tg_arm620_follower.connect()` 会超时。

## 7.3 检查相机索引

如果使用 OpenCV 相机，先确认相机编号：

```bash
ls /dev/video*
```

如果 `0` 和 `1` 不是你要的两路相机，要改 robot config 里的 `index_or_path`。

---

## 8. 数据采集

这里给你两种方法。

## 8.1 方法 A：直接用 `lerobot-record` 录标准格式

这是推荐方式，因为会直接录成 LeRobot 训练可用的数据结构。

```bash
source /opt/ros/humble/setup.bash

uv run lerobot-record \
  --robot.type=tg_arm620_follower \
  --robot.remote_ip=127.0.0.1 \
  --robot.cmd_port=6001 \
  --robot.state_port=6002 \
  --robot.id=tg620 \
  --robot.cameras='{
    external_rgb: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30},
    ee_rgb: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}
  }' \
  --teleop.type=tg_arm620_ros2 \
  --teleop.topic=/joint_target \
  --teleop.node_name=lerobot_tg620_ros2_teleop \
  --dataset.repo_id=<your_user>/tg620_joint7 \
  --dataset.single_task="TG620 teleop task" \
  --dataset.num_episodes=20 \
  --dataset.fps=30 \
  --display_data=true
```

这条命令会做这些事：

- 连接 TG620 follower
- 读取两路相机
- 从 ROS2 `/joint_target` 读取遥操作动作
- 自动把观测转成 LeRobot 标准格式
- 直接写入 LeRobot dataset

## 8.2 方法 B：先用独立桥接脚本验证遥操作链路

如果你想先验证 ROS2 -> TG620 这段动作桥接是否正确，可以先跑：

```bash
source /opt/ros/humble/setup.bash

uv run examples/tg620/ros2_joint_target_to_lerobot.py \
  --robot-config /path/to/tg620_robot.json \
  --joint-topic /joint_target \
  --ros-node-name lerobot_tg620_joint_bridge \
  --control-hz 30 \
  --invert-joints joint3,joint4,joint6 \
  --gripper-scale 100 \
  --dry-run
```

先用 `--dry-run` 看转换后的动作是否正确。

确认没问题后再去掉 `--dry-run`，它就会真的向 TG620 发送动作。

这个脚本不负责录数据，它只用于验证桥接和动作映射。

---

## 9. 录制后的数据预期

如果录制成功，TG620 数据集应满足：

- 两路图像：
  - `observation.images.image`
  - `observation.images.image2`
- 一个 7 维状态：
  - `observation.state`
- 一个 7 维动作：
  - `action`

这是后面 XVLA 训练模板默认要求的格式。

---

## 10. XVLA 训练

## 10.1 训练配置模板

本仓库已经提供了 TG620 的 XVLA 配置模板：

- [examples/tg620/train_config_tg620_xvla_joint7.json](/home/holms/Projects/lerobot/lerobot/examples/tg620/train_config_tg620_xvla_joint7.json:1)

它的核心设定是：

- 2 路真实图像
- 1 个空相机槽位
- 7 维 state
- 7 维 action

其中这两个字段非常关键：

- `policy.num_image_views = 3`
- `policy.empty_cameras = 1`

原因是：

- `xvla-base` 预训练底座按 3 路视觉设计
- 你的 TG620 实际只有 2 路图像
- 所以这里保留 3 视角接口，但空出 1 路由模型内部处理

## 10.2 开始训练前要改的字段

至少修改这些字段：

- `dataset.repo_id`
- `policy.repo_id`
- `output_dir`

例如：

```json
{
  "dataset": {
    "repo_id": "YOUR_USERNAME/tg620_joint7"
  },
  "policy": {
    "repo_id": "YOUR_USERNAME/xvla-tg620-joint7"
  },
  "output_dir": "outputs/train/tg620_xvla_joint7"
}
```

## 10.3 开始训练

```bash
uv run lerobot-train \
  --config_path=examples/tg620/train_config_tg620_xvla_joint7.json \
  --policy.path=lerobot/xvla-base
```

## 10.4 常用训练覆盖命令

```bash
uv run lerobot-train \
  --config_path=examples/tg620/train_config_tg620_xvla_joint7.json \
  --policy.path=lerobot/xvla-base \
  --dataset.repo_id=YOUR_USERNAME/tg620_joint7 \
  --policy.repo_id=YOUR_USERNAME/xvla-tg620-joint7 \
  --output_dir=outputs/train/tg620_xvla_joint7 \
  --steps=30000 \
  --batch_size=16 \
  --policy.device=cuda
```

## 10.5 训练输出目录

训练后的 checkpoint 通常会长这样：

```text
outputs/train/tg620_xvla_joint7/
  checkpoints/
    5000/
      pretrained_model/
      training_state/
    10000/
      pretrained_model/
      training_state/
    ...
```

部署时要使用的模型路径通常是：

```text
outputs/train/tg620_xvla_joint7/checkpoints/<step>/pretrained_model
```

---

## 11. XVLA checkpoint 要求

在线推理时，`--model-path` 必须指向一个 `pretrained_model` 目录，至少包含：

- `config.json`
- `model.safetensors`
- `policy_preprocessor.json`
- `policy_postprocessor.json`

你的目录如果像这样：

```text
checkpoints/100000/
  pretrained_model/
    config.json
    model.safetensors
    policy_preprocessor.json
    policy_postprocessor.json
    train_config.json
  training_state/
    ...
```

那么正确的推理路径是：

```text
checkpoints/100000/pretrained_model
```

不是 `checkpoints/100000` 本身。

---

## 12. 在线推理部署

## 12.1 先 dry-run

强烈建议先 dry-run，不要一上来就发真实动作。

```bash
uv run examples/tg620/run_xvla_inference_tg620.py \
  --model-path /path/to/checkpoints/100000/pretrained_model \
  --robot-config /path/to/tg620_robot.json \
  --task "TG620 teleop task" \
  --fps 10 \
  --print-contract \
  --dry-run
```

这一步会：

- 连接 TG620 机器人桥接
- 读取两路相机
- 读取 7 维状态
- 执行 XVLA 推理
- 打印预测出的 7 维动作
- 不真正下发动作

## 12.2 正式部署

确认 dry-run 输出正常后，再去掉 `--dry-run`：

```bash
uv run examples/tg620/run_xvla_inference_tg620.py \
  --model-path /path/to/checkpoints/100000/pretrained_model \
  --robot-config /path/to/tg620_robot.json \
  --task "TG620 teleop task" \
  --fps 10
```

## 12.3 推理时脚本会做什么

推理脚本会自动：

- 读取原始机器人观测
- 通过 TG620 observation processor 变成标准 observation contract
- 校验 checkpoint 的输入输出与机器人契约是否匹配
- 调用 preprocessor 和 postprocessor
- 生成 7 维动作
- 把动作映射回：
  - `joint1.pos`
  - `joint2.pos`
  - `joint3.pos`
  - `joint4.pos`
  - `joint5.pos`
  - `joint6.pos`
  - `gripper.pos`

---

## 13. 动作语义说明

当前这套 TG620 XVLA 训练/推理链路里，输出的 7 维动作向量默认解释为：

1. `joint1.pos`
2. `joint2.pos`
3. `joint3.pos`
4. `joint4.pos`
5. `joint5.pos`
6. `joint6.pos`
7. `gripper.pos`

也就是说：

- 这是“各轴关节目标 + 夹爪”
- 不是“末端位姿 + 夹爪”

如果以后你要切成“末端位姿 + 夹爪”，那会是另一条数据契约，需要一起改：

- 录制 state 定义
- 训练配置
- 推理解释方式
- TG620 控制侧的 IK / 末端控制逻辑

当前不建议在这条已跑通的 joint-space 链路上混用。

---

## 14. 常见问题

## 14.1 `ROS2 Python packages are required`

说明当前 shell 没有 ROS2 Python 环境。

先执行：

```bash
source /opt/ros/humble/setup.bash
```

如果 `Arm_Project` 有自己的 ROS workspace，也继续 source 它的 `install/setup.bash`。

## 14.2 连接 TG620 超时

通常说明：

- TG620 外部桥接服务没启动
- `remote_ip` 不对
- `cmd_port` / `state_port` 不对
- 网络不通

优先检查：

- 目标 IP
- ZMQ 桥接进程
- 端口号

## 14.3 相机打不开

通常检查：

- `index_or_path` 是否正确
- 相机有没有被别的程序占用
- OpenCV 是否能正常访问该设备

## 14.4 XVLA 推理报 feature mismatch

说明 checkpoint 和当前运行时观测不一致。

最常见原因：

- checkpoint 训练时用的是不同相机键
- state 维度不一致
- action 维度不一致
- 模型按 3 路图像训练，但当前配置没对上

先用：

```bash
uv run examples/tg620/run_xvla_inference_tg620.py \
  --model-path /path/to/checkpoints/100000/pretrained_model \
  --robot-config /path/to/tg620_robot.json \
  --print-contract \
  --dry-run
```

看打印出来的 contract 是否和训练配置一致。

## 14.5 录制能跑，推理不稳定

优先排查：

- 相机帧率跟不上
- 推理频率太高
- GPU 不够
- TG620 桥接延迟过大

可以先把推理频率降到：

```bash
--fps 5
```

确认稳定后再提高。

---

## 15. 推荐操作顺序

如果你现在要从零开始，我建议按这个顺序做：

1. 先确认 ROS2 `/joint_target` 在发数据
2. 确认 TG620 外部桥接服务能正常收发
3. 用 `ros2_joint_target_to_lerobot.py --dry-run` 验证动作映射
4. 用 `lerobot-record` 录制少量样本
5. 检查 dataset 契约是否是 `2 图像 + 7 state + 7 action`
6. 用 `train_config_tg620_xvla_joint7.json` 启动 XVLA 训练
7. 用 `run_xvla_inference_tg620.py --dry-run` 验证 checkpoint
8. 最后再去掉 `--dry-run` 上真实机械臂

---

## 16. 最小命令清单

### 环境准备

```bash
uv sync --locked --extra all
git lfs install
git lfs pull
source /opt/ros/humble/setup.bash
```

### 录制

```bash
uv run lerobot-record \
  --robot.type=tg_arm620_follower \
  --robot.remote_ip=127.0.0.1 \
  --robot.cmd_port=6001 \
  --robot.state_port=6002 \
  --robot.id=tg620 \
  --robot.cameras='{
    external_rgb: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30},
    ee_rgb: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}
  }' \
  --teleop.type=tg_arm620_ros2 \
  --teleop.topic=/joint_target \
  --dataset.repo_id=<your_user>/tg620_joint7 \
  --dataset.single_task="TG620 teleop task" \
  --dataset.num_episodes=20 \
  --dataset.fps=30
```

### 训练

```bash
uv run lerobot-train \
  --config_path=examples/tg620/train_config_tg620_xvla_joint7.json \
  --policy.path=lerobot/xvla-base \
  --dataset.repo_id=YOUR_USERNAME/tg620_joint7 \
  --policy.repo_id=YOUR_USERNAME/xvla-tg620-joint7 \
  --output_dir=outputs/train/tg620_xvla_joint7
```

### 推理 dry-run

```bash
uv run examples/tg620/run_xvla_inference_tg620.py \
  --model-path outputs/train/tg620_xvla_joint7/checkpoints/100000/pretrained_model \
  --robot-config /path/to/tg620_robot.json \
  --task "TG620 teleop task" \
  --fps 10 \
  --print-contract \
  --dry-run
```

### 正式部署

```bash
uv run examples/tg620/run_xvla_inference_tg620.py \
  --model-path outputs/train/tg620_xvla_joint7/checkpoints/100000/pretrained_model \
  --robot-config /path/to/tg620_robot.json \
  --task "TG620 teleop task" \
  --fps 10
```

---

## 17. 当前实现边界

这份 README 对应的是当前已经落地的 joint-space 方案：

- 支持：
  - `2` 路图像
  - `7` 维 joint/gripper state
  - `7` 维 joint/gripper action
  - 采集、训练、部署整链路
- 暂不直接支持：
  - `EE pose + gripper` 作为主训练 state
  - 末端位姿动作直接控制

如果你后面要切到“末端位姿 + 夹爪”，建议单独开一版 TG620 EE-state pipeline，不要直接覆盖当前 joint7 版本。
