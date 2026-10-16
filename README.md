# Zhian RDK X5 Robotic Arm

基于地瓜机器人 RDK X5 的边缘智能机械臂系统，面向实验室场景，构建"感知 → 决策 → 执行 → 校验"闭环。

## 项目简介

本项目以 RDK X5 作为主控与端侧 AI 推理平台，通过 BPU 加速 YOLO 视觉模型，结合 ArUco 桌面标记完成实验器材定位，调用小米 MIMO 大模型将中文指令转为结构化任务，驱动六自由度机械臂执行抓取、放置、倾倒废液等操作。

## 核心模块

```
src/
├── control/     运动控制 — V7 串口协议、轨迹回放、闭环跟踪
├── perception/  视觉感知 — YOLO 目标检测 + ArUco 桌面标记定位
├── planning/    任务规划 — MIMO 大模型指令解析、坐标解析、区域注册
├── bridge/      HTTP 桥接 — 演示页面后端服务
├── safety/      安全模块 — (待补充)
├── simulation/  仿真环境 — (待补充，基于 MuJoCo)
├── training/    RL 训练 — (待补充，基于 Stable-Baselines3)
└── utils/       工具函数 — (待补充)
```

### control — 运动控制

| 文件 | 功能 |
|------|------|
| `serial_protocol_v7.py` | V7 串口通信协议（文本命令 + RL 二进制帧） |
| `serial_pick_and_place_v7.py` | 抓放主控，A/B/C 三段轨迹 + 闭环跟踪 |
| `serial_replay_v7_trajectory.py` | 轨迹 JSON 加载与回放 |
| `serial_closed_loop_tracker_v7.py` | 实时闭环关节跟踪 |
| `serial_fixed_rate_replay_v7.py` | 定频轨迹回放 |
| `run_pick_place_task_v7.py` | 抓放任务执行入口 |

### perception — 视觉感知

| 文件 | 功能 |
|------|------|
| `vision_table_yolo_locator.py` | YOLO + ArUco 四角标记桌面目标定位 |
| `vision_yolo_bridge.py` | YOLO 推理桥接（相机采集 → 检测结果） |
| `vision_realtime_yolo.py` | 实时 YOLO 检测 |
| `vision_pick_place_run_v8.py` | 视觉抓取执行入口（视觉 → 任务 → 执行） |
| `vision_to_pick_place_task.py` | 视觉检测结果 → 抓放任务 JSON 转换 |

### planning — 任务规划

| 文件 | 功能 |
|------|------|
| `mimo_client.py` | 小米 MIMO API 客户端（OpenAI 兼容协议） |
| `task_planner.py` | LLM 任务规划（中文指令 → 结构化 JSON） |
| `task_schema.py` | 任务 JSON Schema 定义 |
| `location_resolver.py` | 视觉坐标解析（像素 → 机器人基座坐标） |
| `area_registry.py` | 桌面区域坐标注册表（配置台、置物台、废液桶等） |
| `templates.py` | 提示词模板 |
| `text_experiment_to_v8.py` | 文本指令 → 视觉定位 → 任务执行完整管线 |
| `text_experiment_assistant.py` | 实验助手（简化版指令入口） |

### bridge — HTTP 桥接

| 文件 | 功能 |
|------|------|
| `ai_demo_bridge.py` | 演示页面后端 HTTP 服务，连接浏览器前端与任务执行 |

## 环境变量

复制 `.env.example` 为 `.env` 并填入实际值：

```bash
cp .env.example .env
```

| 变量名 | 用途 | 默认值 |
|--------|------|--------|
| `YOLO_MODEL_PATH` | YOLO 权重文件路径 | `models/weights/best.pt` |
| `YOLO_PYTHON` | YOLO 推理用的 Python 解释器 | `python` |
| `YOLO_ROOT` | YOLO 训练项目根目录 | `.` |
| `RL_SNAPSHOT_DIR` | RL 模型快照目录 | `models/weights/rl_snapshot` |
| `MIMO_API_KEY` | 小米 MIMO API Key | (无，必须设置) |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备模型权重

YOLO 模型权重不包含在本仓库中。请将训练好的 `best.pt` 放置到：

```
models/weights/best.pt
```

或通过环境变量指定路径：

```bash
export YOLO_MODEL_PATH=/path/to/your/best.pt
```

### 3. 运行视觉检测

```bash
python scripts/run_vision.py --camera-index 2
```

### 4. 运行文本指令任务

```bash
export MIMO_API_KEY=your_key_here
python scripts/run_demo.py --instruction "把左边的烧杯放到置物台上层"
```

### 5. 启动演示 HTTP 服务

```bash
export MIMO_API_KEY=your_key_here
python scripts/run_bridge.py --port 8766
```

## 执行确认与离线测试

执行确认规则、完全离线测试与剩余边界见 [执行授权说明](docs/execution_authorization_cn.md)。
`POST /api/assistant/execute` 现在要求请求顶层 `confirm: true`；`/plan` 始终仅规划。
任务中的 `confirm_before_run` 必须是布尔值 `true`，不能充当执行授权。

从仓库根目录执行 `make test`，只运行使用模拟接口的离线授权测试。

## 硬件需求

| 组件 | 型号 | 角色 |
|------|------|------|
| 主控板 | 地瓜机器人 RDK X5 | 端侧 AI 推理 + 主控 |
| 机械臂 | 零一造物 ZERO 6-DOF | Feetech 舵机，串口通信 |
| 相机 | RGB / RGB-D（Orbbec 等） | 桌面视觉感知 |
| 标记 | ArUco tag36h11 × 4 | 桌面坐标系标定 |

## 模型权重说明

模型权重（`.pt`、`.onnx`、`.bin` 等）不提交到 Git 仓库。

获取方式：
- **YOLO 模型**：使用 Ultralytics YOLOv8 在自定义数据集上训练
- **RL 模型**：使用 Stable-Baselines3 在 MuJoCo 环境中训练
- **BPU 模型**：将 ONNX 模型通过地瓜工具链转换为 `.bin` 格式

详见 [models/README.md](models/README.md)

## 项目结构

```
zhian-rdk-x5-robotic-arm/
├── src/              # 源代码
├── configs/          # 相机内参、区域坐标等配置文件
├── scripts/          # 启动脚本（run_vision / run_demo / run_bridge）
├── web/              # 浏览器 3D 可视化前端（待迁移）
├── models/           # 模型权重（不入库，见 README）
├── docs/             # 文档
├── assets/           # 图片和演示素材
├── tests/            # 完全离线的执行授权测试
├── .env.example      # 环境变量模板
├── requirements.txt  # Python 依赖
├── Makefile          # 常用命令
└── LICENSE           # MIT License
```

### configs — 配置文件

| 文件 | 说明 |
|------|------|
| `camera_gripper.yaml` | 夹爪相机内参（USB 免驱相机 640×480，60° FOV） |
| `camera_top.yaml` | 顶部全局相机内参（奥比中光 Astra Pro 1280×720） |
| `default_areas.json` | 工作区域坐标定义（配置台、置物台上下层、废液桶） |

### scripts — 启动脚本

| 文件 | 说明 |
|------|------|
| `run_vision.py` | 启动实时 YOLO 视觉检测 |
| `run_demo.py` | 启动文本指令演示（MIMO → 视觉 → 抓放） |
| `run_bridge.py` | 启动 HTTP 桥接服务（浏览器演示页面后端） |

## 当前状态

核心代码已迁移：control（6 文件）、perception（5 文件）、planning（8 文件）、bridge（1 文件）。
配置文件已就位：camera_gripper.yaml、camera_top.yaml、default_areas.json。
启动脚本已创建：run_vision.py、run_demo.py、run_bridge.py。

待补充：simulation（MuJoCo 环境）、training（RL 训练脚本）、web 前端。

## License

[MIT License](LICENSE) — Copyright (c) 2026 Chen Zeyu
