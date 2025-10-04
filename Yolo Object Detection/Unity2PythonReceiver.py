import socket
import base64
import os
import threading
from datetime import datetime
import yaml
import json
import random
import shutil
import cv2
import numpy as np

class TCPImageReceiverYOLO:
    def __init__(self, host="localhost", port=25002, yolo_dataset_folder="YOLODataset"):
        self.host = host
        self.port = port
        self.yolo_dataset_folder = yolo_dataset_folder
        self.running = False
        self.server_socket = None
        self.classes = ["vehicle"]  # Will be auto-populated
        self.train_split = 0.8
        self.stats = {
            'total_received': 0,
            'with_ground_truth': 0,
            'empty_ground_truth': 0
        }
        
        # Create YOLO dataset structure
        self.create_yolo_structure()
        
        print(f"TCP Image Receiver initialized on {host}:{port}")
        print(f"YOLO dataset will be created at: {os.path.abspath(self.yolo_dataset_folder)}")

    def create_yolo_structure(self):
        """Create YOLO dataset folder structure"""
        folders = [
            os.path.join(self.yolo_dataset_folder, "images", "train"),
            os.path.join(self.yolo_dataset_folder, "images", "val"),
            os.path.join(self.yolo_dataset_folder, "labels", "train"),
            os.path.join(self.yolo_dataset_folder, "labels", "val"),
            os.path.join(self.yolo_dataset_folder, "visualizations"),
        ]
        
        for folder in folders:
            os.makedirs(folder, exist_ok=True)

    def start_server(self):
        """Start the TCP server to receive images"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True

            print(f"Server listening on {self.host}:{self.port}")
            print("Receiving images and preparing YOLO dataset on-the-fly...")
            print("Press Ctrl+C to stop and finalize dataset\n")

            while self.running:
                try:
                    self.server_socket.settimeout(1.0)
                    client_socket, client_address = self.server_socket.accept()
                    print(f"Connection established with {client_address}")

                    client_thread = threading.Thread(
                        target=self.handle_client, args=(client_socket, client_address)
                    )
                    client_thread.daemon = True
                    client_thread.start()

                except socket.timeout:
                    continue
                except socket.error as e:
                    if self.running:
                        print(f"Socket error: {e}")

        except Exception as e:
            print(f"Error starting server: {e}")
        finally:
            self.stop_server()

    def handle_client(self, client_socket, client_address):
        """Handle incoming client connection"""
        try:
            client_socket.settimeout(30.0)

            while self.running:
                try:
                    size_data = self.receive_exact(client_socket, 4)
                    if not size_data:
                        break

                    data_size = int.from_bytes(size_data, byteorder="big")
                    message_data = self.receive_exact(client_socket, data_size)
                    if not message_data:
                        break

                    success = self.process_message(message_data.decode("utf-8"), client_address)

                    if success:
                        client_socket.send(b"ACK")
                    else:
                        client_socket.send(b"ERR")

                except socket.timeout:
                    print(f"Client {client_address} timed out")
                    break
                except Exception as e:
                    print(f"Error handling client {client_address}: {e}")
                    break

        except Exception as e:
            print(f"Error in client handler for {client_address}: {e}")
        finally:
            client_socket.close()
            print(f"Connection with {client_address} closed")

    def receive_exact(self, sock, num_bytes):
        """Receive exactly num_bytes from socket"""
        data = b""
        while len(data) < num_bytes:
            try:
                chunk = sock.recv(num_bytes - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.error as e:
                print(f"Socket receive error: {e}")
                return None
        return data

    def process_message(self, message, client_address):
        """Process the received JSON message and save directly to YOLO format"""
        try:
            data = json.loads(message)
            camera_id = data.get("camera_id", "unknown")
            base64_data = data.get("image_data", "")
            ground_truth = data.get("ground_truth", [])

            self.stats['total_received'] += 1

            if not base64_data:
                print(f"Empty image data received from camera '{camera_id}'")
                return False

            # Decode image
            image_bytes = base64.b64decode(base64_data)

            # Skip if no ground truth
            if not isinstance(ground_truth, list) or len(ground_truth) == 0:
                self.stats['empty_ground_truth'] += 1
                return True

            self.stats['with_ground_truth'] += 1

            # Decide train or val split randomly
            split = "train" if random.random() < self.train_split else "val"

            # Create unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{camera_id}_{timestamp}"

            # Save image
            img_path = os.path.join(self.yolo_dataset_folder, "images", split, f"{filename}.png")
            with open(img_path, "wb") as f:
                f.write(image_bytes)

            # Convert and save labels in YOLO format
            yolo_annotations = []
            for item in ground_truth:
                if len(item) >= 6:
                    class_label = item[0]
                    # vehicle_name = item[1]  # Not needed for YOLO
                    cx, cy, w, h = float(item[2]), float(item[3]), float(item[4]), float(item[5])

                    # Get or add class
                    if class_label not in self.classes:
                        self.classes.append(class_label)
                    class_id = self.classes.index(class_label)

                    yolo_annotations.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            # Save labels
            label_path = os.path.join(self.yolo_dataset_folder, "labels", split, f"{filename}.txt")
            with open(label_path, 'w') as f:
                for annotation in yolo_annotations:
                    f.write(annotation + '\n')

            # Visualize every 10th image
            if self.stats['with_ground_truth'] % 10 == 0:
                self.visualize_image(img_path, label_path, filename)

            # Print progress
            if self.stats['with_ground_truth'] % 50 == 0:
                print(f"\n{'='*60}")
                print(f"Progress: {self.stats['with_ground_truth']} images with annotations")
                print(f"Train images: ~{int(self.stats['with_ground_truth'] * self.train_split)}")
                print(f"Val images: ~{int(self.stats['with_ground_truth'] * (1 - self.train_split))}")
                print(f"Classes detected: {self.classes}")
                print(f"{'='*60}\n")
            else:
                print(f"✓ {camera_id}: {len(ground_truth)} objects detected -> {split}")

            return True

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON message: {e}")
            return False
        except Exception as e:
            print(f"Error processing message: {e}")
            import traceback
            traceback.print_exc()
            return False

    def visualize_image(self, img_path, label_path, filename):
        """Create visualization with bounding boxes"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                return

            h, w = img.shape[:2]

            # Read and draw annotations
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        # YOLO format: center_x, center_y, width, height (all normalized 0-1)
                        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                        # Convert normalized coordinates to pixel coordinates
                        center_x_px = cx * w
                        center_y_px = cy * h
                        width_px = bw * w
                        height_px = bh * h
                        
                        # Calculate corner coordinates
                        x1 = int(center_x_px - width_px / 2)
                        y1 = int(center_y_px - height_px / 2)
                        x2 = int(center_x_px + width_px / 2)
                        y2 = int(center_y_px + height_px / 2)
                        
                        # Clamp to image boundaries
                        x1 = max(0, min(x1, w))
                        y1 = max(0, min(y1, h))
                        x2 = max(0, min(x2, w))
                        y2 = max(0, min(y2, h))

                        # Draw bounding box
                        color = (0, 255, 0)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                        # Draw label with background
                        label = self.classes[class_id] if class_id < len(self.classes) else f"class_{class_id}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Save visualization
            vis_path = os.path.join(self.yolo_dataset_folder, "visualizations", f"{filename}.png")
            cv2.imwrite(vis_path, img)

        except Exception as e:
            print(f"Error creating visualization: {e}")

    def finalize_dataset(self):
        """Create YAML config and training script"""
        print("\n" + "="*60)
        print("FINALIZING DATASET")
        print("="*60)

        # Create YAML config
        config = {
            'path': os.path.abspath(self.yolo_dataset_folder),
            'train': 'images/train',
            'val': 'images/val',
            'names': {i: name for i, name in enumerate(self.classes)}
        }

        yaml_path = os.path.join(self.yolo_dataset_folder, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"✓ Created YAML config: {yaml_path}")

        # Count final images
        train_images = len([f for f in os.listdir(os.path.join(self.yolo_dataset_folder, "images", "train")) if f.endswith('.png')])
        val_images = len([f for f in os.listdir(os.path.join(self.yolo_dataset_folder, "images", "val")) if f.endswith('.png')])

        # Generate training script
        script_content = f"""#!/usr/bin/env python3
# YOLOv8 Training Script
# Install: pip install ultralytics

from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data='{os.path.abspath(self.yolo_dataset_folder)}/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='vehicle_detection',
    patience=50,
    save=True,
    device=0  # Use GPU 0, set to 'cpu' for CPU training
)

# Validate the model
metrics = model.val()

# Export the model
model.export(format='onnx')

print("Training complete!")
print(f"Best model: runs/detect/vehicle_detection/weights/best.pt")
"""

        script_path = os.path.join(self.yolo_dataset_folder, 'train_yolo.py')
        with open(script_path, 'w') as f:
            f.write(script_content)

        print(f"✓ Generated training script: {script_path}")

        # Print summary
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        print(f"Total images received: {self.stats['total_received']}")
        print(f"Images with annotations: {self.stats['with_ground_truth']}")
        print(f"Images without annotations: {self.stats['empty_ground_truth']}")
        print(f"Train images: {train_images}")
        print(f"Val images: {val_images}")
        print(f"Classes: {self.classes}")
        print(f"Dataset location: {os.path.abspath(self.yolo_dataset_folder)}")
        print("="*60)

        print("\nNEXT STEPS:")
        print("1. Install YOLOv8: pip install ultralytics")
        print("2. Review visualizations in: YOLODataset/visualizations/")
        print(f"3. Train model: python {script_path}")
        print("4. Monitor training: tensorboard --logdir runs/detect")
        print("="*60 + "\n")

    def stop_server(self):
        """Stop the TCP server and finalize dataset"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        
        if self.stats['with_ground_truth'] > 0:
            self.finalize_dataset()
        
        print("Server stopped")


def load_config():
    """Load port configuration"""
    try:
        config_paths = [
            "configs/port_config.yaml",
            "../configs/port_config.yaml",
            "../../configs/port_config.yaml",
        ]

        for path in config_paths:
            if os.path.exists(path):
                with open(path, "r") as f:
                    config = yaml.safe_load(f)
                print(f"Loaded config from: {path}")
                return config

        return {}
    except Exception as e:
        print(f"Could not load config: {e}")
        return {}


def main():
    config = load_config()
    port = config.get("unity_image_tcp", 25002)

    receiver = TCPImageReceiverYOLO(port=port)

    try:
        receiver.start_server()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        receiver.stop_server()


if __name__ == "__main__":
    main()