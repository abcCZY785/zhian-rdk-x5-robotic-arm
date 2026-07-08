# Model Deployment

> TODO: Will be updated after source code migration.

## YOLO Model

- Trained with Ultralytics YOLOv8
- Target: tabletop experiment equipment (beakers, bottles, etc.)
- Model path: see `models/README.md` for download instructions

## BPU Deployment (RDK X5)

> TODO: Document BPU conversion process (ONNX → .bin).

## ArUco Markers

- Type: tag36h11
- Four markers define the tabletop coordinate system
- Marker generation: see `scripts/generate_aruco_table_markers.py`
