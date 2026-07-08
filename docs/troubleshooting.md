# Troubleshooting

> TODO: Will be updated after source code migration.

## Common Issues

### Serial port not found

- Check USB connection to RDK X5 / STM32
- Verify port permissions: `sudo chmod 666 /dev/ttyUSB0`

### Camera not detected

- Check USB connection
- Verify camera index: `ls /dev/video*`

### YOLO model not loading

- Ensure model weights are in `models/weights/`
- Check Python environment has `ultralytics` installed

### ArUco markers not detected

- Ensure markers are printed at correct size
- Check lighting conditions
- Verify marker IDs match configuration
