#!/usr/bin/env python3
# YOLOv8 Training Script
# Install: pip install ultralytics

from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data='/home/zorro/Desktop/Projects/SynchroCity/Yolo Object Detection/YOLODataset/data.yaml',
    epochs=10,
    imgsz=640,
    batch=16,
    name='vehicle_detection',
    patience=50,
    save=True,
    device="cpu"  # Use GPU 0, set to 'cpu' for CPU training
)

# Validate the model
metrics = model.val()

# Export the model
model.export(format='onnx')

print("Training complete!")
print(f"Best model: runs/detect/vehicle_detection/weights/best.pt")
