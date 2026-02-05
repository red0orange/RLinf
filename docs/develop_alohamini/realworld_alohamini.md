## 方案总览（推荐）：把 LeRobot 当“底盘驱动”，RLinf 通过 ROS2 跟它交互
### 组件分层
- **树莓派 Host（必须先做校准）**
  - 运行：`python -m lerobot.robots.alohamini.lekiwi_host`
  - 首次会交互式校准（README 里那套流程），这一步建议做成**一次性人工流程**，不要在 RLinf 里做。
- **上位机 ROS2 Bridge（长期常驻）**
  - 运行：`ros2 run alohamini_ros2_bridge bridge_node --ros-args -p remote_ip:=<pi_ip>`
  - 作用（来自 `bridge_node.py` / README）：
    - 从 LeKiwiClient 拉取 obs（ZMQ），发布：
      - `joint_states`（注意：单位是 LeRobot motor space，不保证是 rad）
      - `camera/<cam_name>/image_raw`（`bgr8`）
      - `state/state_vec`（LeRobot 内部 state_order 的向量）
    - 订阅控制命令：
      - `cmd_joint`（`sensor_msgs/JointState`，把 position 当作 `<name>.pos` 目标值发回 Host）
      - 还有 `cmd_vel/cmd_json/cmd_lift_m` 等（RLinf 不需要可以不管）
- **RLinf（Env 节点）**
  - 实现一个 **RLinf RealWorld 风格的“ROS2 env”**：订阅 `joint_states` + `camera/.../image_raw`，发布 `cmd_joint`。
  - 策略动作空间用你想要的 **关节增量**：在 env 内部把 delta 积分成“关节目标值”，再发 `cmd_joint`。

这样 RLinf 不需要直接 import/运行 LeRobot，也不需要在 Ray worker 里跑 ZMQ/电机驱动；只做 ROS2 pub/sub，稳定性最好。

---

## 你需要实现哪些部分（落到 RLinf 侧的最小改造清单）

### 1) 新增一个“ROS2 AlohaMini 环境任务”（gym id）
做法类似 Franka 的 `PegInsertionEnv-v1`：
- 新建 `rlinf/envs/realworld/alohamini/tasks/<task>.py`
  - 内部用 `rclpy` 订阅：
    - `joint_states`
    - `camera/<cam>/image_raw`（你作为主视角的那个 cam）
  - 发布：
    - `cmd_joint`（JointState.name 填关节名，position 填“目标值”）
  - 提供：
    - `reset()`：把内部“目标关节值”初始化为当前关节值（或一个预设 home pose），并发一次 `cmd_joint`（如果你需要回中）
    - `step(delta_q)`：更新目标值 `q_target = q_target + delta_q * scale`，clip 后发 `cmd_joint`
    - `observation` 输出结构要兼容 RLinf 的 RealWorld 打包逻辑（见下一条）

### 2) 观测结构必须对齐 RLinf 的 RealWorld 数据管道
RLinf 的 `RealWorldEnv._wrap_obs()` 期望 raw_obs 里有：
- `raw_obs["state"]`: dict（按 key 排序拼成 `states` 向量）
- `raw_obs["frames"]`: dict（按 `main_image_key` 取出 `main_images`）

所以你 env 的 raw_obs 最省事的做法是：
- `state`：直接提供一组稳定 key（比如 `arm_left_shoulder_pan.pos` 这类，或者你自己定义），值是 `np.ndarray` 标量/小向量
- `frames`：至少提供一个 key（例如 `wrist_1`），值是 `uint8(H,W,3)`  
（bridge 发布的是 `bgr8`，RLinf 模型侧一般不关心 BGR/RGB，但你要保持一致；必要时在 env 里转 RGB）

> 更简单的捷径：直接订阅 `state/state_vec`（Float32MultiArray）作为你的 `states`，再在 env 里映射到固定维度。这比拼 dict 更稳。

### 3) **必须处理：RLinf 现有 RealWorld wrapper 链不适合“关节增量”**
现在 `RealWorldEnv` 会固定套 `RelativeFrame/Quat2Euler/GripperClose` 这类“末端 6DoF”假设的 wrapper。你的动作是关节增量，这些 wrapper 会不匹配。

推荐做法（二选一）：
- **A（推荐）**：给 `RealWorldEnv` 增加可配置 wrapper pipeline（joint 模式下不套 RelativeFrame/Quat2Euler/GripperClose）
- **B**：新增一个 env_type（例如 `realworld_ros2_joint`）并在 `get_env_cls()` 返回你自己的 env 类，绕开 `RealWorldEnv` 这套固定 wrapper

如果你想最快落地，我建议走 **B**（侵入最小、最不容易影响 Franka）。

### 4) 动作单位与安全边界（很关键）
桥接层明确写了：`joint_states` 的单位**不保证是 rad**，而是 “LeRobot motor space”。

所以你要做两件事：
- **统一单位**：RLinf policy 的输出 delta 也用同一单位（motor space delta）
- **安全 clip**：用校准文件里的 `range_min/range_max` 做 clip  
  - LeRobot 的校准文件是 json（`lerobot/robots/robot.py` 会在 `~/.cache/.../lerobot/...` 下读写），你可以在 RLinf env 启动时读取它，给每个 joint 一个 min/max，防止策略把目标推爆。

---

## 如何单独测试“机器人本体适配”是否完整（不依赖训练）
按下面顺序测，能快速定位是 Host、Bridge 还是 RLinf env 的问题。

### 测试 0：Host 校准与联通性（人工一次性）
- 在树莓派跑 `python -m lerobot.robots.alohamini.lekiwi_host`
- 按 README 完成校准（必要）
- 观察 host 能持续输出观测（哪怕没有 client）

### 测试 1：ROS2 Bridge 自检（只跑 bridge，不跑 RLinf）
- 启动 bridge：`ros2 run alohamini_ros2_bridge bridge_node ...`
- 用 `ros2 topic echo /joint_states` 确认关节状态在刷新
- 用 `ros2 topic echo /camera/<cam>/image_raw`（或 rqt_image_view）确认图像刷新
- 手动发一条 `cmd_joint`（小改动）后，观察 `joint_states` 是否变化  
  这一步能把 LeRobot<->硬件链路问题排除掉。

### 测试 2：RLinf env 单进程冒烟（reset/step contract）
- `reset()` 能拿到一帧图像 + 一次 state 向量
- 连续 `step(0)` 100 次不 hang
- 连续 `step(小delta)` 观察 `joint_states` 跟随

### 测试 3：RLinf EnvManager 多进程启停（模拟真实运行）
- 反复启动/停止 env 进程（如果你用 offload/多进程），确保 rclpy 不会重复 init 卡死、subscriber 不会失效

---

## 针对你引用的校准段落（136-154）的“落地建议”
- **不要把校准塞进 RLinf**：校准是强交互、耗时且可能需要现场操作。建议：
  - 在树莓派以 systemd/service 形式常驻 `lekiwi_host`
  - 校准只在部署时做一次，校准文件持久化
- RLinf 只做两件事：
  - 检查 “bridge 话题是否在更新”（超时就报错提示“先确认 host/bridge 在跑、是否已校准”）
  - 发送 joint 命令（delta→target→cmd_joint）