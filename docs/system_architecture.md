# System Architecture

> TODO: Will be updated after source code migration.

## Overview

The system follows a "Perception → Decision → Execution → Verification" closed-loop architecture.

```
[Camera] → [YOLO+ArUco Perception] → [LLM Task Planner (MIMO)]
                                            ↓
[Serial Protocol v7] ← [Trajectory Builder] ← [Planned Task JSON]
        ↓
[Mechanical Arm Execution] → [Closed-loop Tracker] → [Vision Re-check]
```

## Modules

- **Perception**: `src/perception/` — YOLO object detection, ArUco marker localization
- **Planning**: `src/planning/` — MIMO LLM task planning, location resolution
- **Control**: `src/control/` — Serial protocol, trajectory replay, closed-loop tracking
- **Simulation**: `src/simulation/` — MuJoCo RL environment
- **Training**: `src/training/` — Stable-Baselines3 RL training
- **Bridge**: `src/bridge/` — HTTP bridge for browser demo
- **Safety**: `src/safety/` — Collision detection, joint limits
