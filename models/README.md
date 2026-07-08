# Models

Model weights are **not** included in this repository due to file size constraints.

## Directory Structure

```
models/
├── README.md          # This file
└── weights/           # Model weights go here (not in repo)
    ├── best.pt        # YOLOv8 trained weights
    └── ...
```

## How to Obtain Models

### YOLO Object Detection Model

The YOLO model was trained using Ultralytics YOLOv8 on a custom dataset of laboratory equipment
(beakers, brown bottles, transparent bottles).

**Option 1: Download from Release**

> TODO: Upload model weights to GitHub Releases.

**Option 2: Train from Scratch**

> TODO: Provide training instructions and dataset preparation guide.

**Option 3: BPU Model (RDK X5)**

> TODO: Provide BPU-converted .bin model for RDK X5 inference.

## Model Details

| Model | Framework | Input | Purpose |
|-------|-----------|-------|---------|
| YOLOv8n (best.pt) | PyTorch / Ultralytics | RGB image | Tabletop object detection |
| BPU (.bin) | D-Robotics BPU Runtime | NV12 (6-head NHWC) | RDK X5 edge inference |

## Notes

- `.pt`, `.pth`, `.onnx`, `.bin` files are excluded by `.gitignore`
- Place downloaded weights in `models/weights/`
