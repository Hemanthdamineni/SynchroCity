# Unity Traffic Simulation with YOLO Object Detection and RL Neural Network

This project creates a complete pipeline for training and deploying a reinforcement learning system that uses YOLO object detection on Unity-generated traffic simulation data. It integrates with your existing `NN_Training.py` reinforcement learning system for intelligent traffic light control.

## 🎯 Project Overview

The system consists of four main components:

1. **Unity Traffic Simulator** - Generates realistic traffic scenarios with ground truth annotations
2. **YOLO Object Detection** - Trained on Unity data to detect vehicles, pedestrians, traffic lights, etc.
3. **RL Neural Network** - Uses YOLO detections to make traffic management decisions
4. **Integrated RL System** - Combines baseline neural network with DQN agent for adaptive control

## 🏗️ Architecture

```
Unity Simulation → ZeroMQ → YOLO Detection → ZeroMQ → Integrated RL → Traffic Control
      ↓                        ↓                        ↓              ↓
Ground Truth Data    Real-time Detection    Baseline NN + DQN    Adaptive Control
      ↓                        ↓                        ↓              ↓
Dataset Creation     Performance Monitor    RL Training      Learning & Optimization
```

## 🚀 Quick Start

### 1. Initial Setup

```bash
# Install dependencies and setup environment
python setup.py
```

### 2. Unity Configuration

1. Open Unity project: `Unity City Project/Latest`
2. Ensure `Unity2Python.cs` is attached to a GameObject in the scene
3. Configure camera assignments and render textures
4. Install NetMQ package in Unity (Window → Package Manager)

### 3. Run Complete Pipeline

Run streaming test, dataset capture, YOLO training, RL training, and demo:

```bash
cd "Yolo Object Detection"
# Test Unity → Python streaming (Unity must be in Play mode)
python test_unity_feed.py --address tcp://127.0.0.1:5555 --duration 15

# Capture dataset from Unity feed
python capture_dataset.py --address tcp://127.0.0.1:5555 --out dataset --seconds 120

# Train YOLO on captured dataset
python train_yolo.py --dataset dataset --model yolov8n.pt --epochs 10

# Train RL agent (optional)
python train_rl.py --timesteps 20000

# Run end-to-end demo
python run_demo.py
```

### 4. Testing

```bash
cd "Yolo Object Detection"
python -m unittest discover -s tests -p "test_*.py"
```

## 📁 Project Structure

```
Traffic Simulator/
├── Unity City Project/Latest/          # Unity project
│   └── Assets/Unity2Python.cs         # Unity→Python communication
├── RL Neural Network/                 # Your existing RL system
│   └── NN_Training.py                # Original traffic light RL controller
├── Yolo Object Detection/             # Main Python components  
│   ├── unity_zmq_receiver.py          # Receives Unity frames
│   ├── dataset_preparation.py         # Creates YOLO training dataset
│   ├── yolo_trainer.py               # Trains YOLO model
│   ├── yolo_integration.py           # Real-time YOLO detection
│   ├── traffic_rl_system.py          # RL neural network
│   ├── integrated_rl_system.py       # Combines baseline NN + DQN
│   ├── training_pipeline.py          # Complete training orchestration
│   ├── integration_test.py           # End-to-end testing
│   ├── complete_pipeline_test.py     # Complete system test
│   ├── setup.py                      # Environment setup
│   └── requirements_updated.txt      # Python dependencies
├── processed_dataset/                 # YOLO training data
├── trained_models/                   # Saved models
└── training_results/                 # Training logs and metrics
```

## 🔧 Component Details

### Unity2Python Communication (ZeroMQ)

The updated Unity script uses ZeroMQ PUB-SUB pattern for reliable, high-performance communication:

- **Publisher**: Unity broadcasts camera frames with ground truth data
- **Subscriber**: Python components receive and process frames
- **Topics**: `camera_1`, `camera_2`, etc. for different camera feeds

**Key Features**:
- Real-time frame streaming (up to 30 FPS)
- Ground truth bounding box detection
- Automatic object classification (vehicles, pedestrians, traffic lights, signs, bicycles)
- Rate limiting and memory management

### YOLO Object Detection

Custom YOLO training pipeline specifically designed for Unity traffic data:

**Training Process**:
1. Converts Unity ground truth to YOLO format
2. Creates train/val/test splits
3. Trains YOLOv8 model with data augmentation
4. Validates performance on test set
5. Exports optimized model formats (ONNX, TensorRT)

**Features**:
- Support for traffic-specific object classes
- Real-time inference integration
- Performance monitoring and statistics
- Model versioning and comparison

### Integrated RL System

The new `integrated_rl_system.py` combines your existing neural network from `NN_Training.py` with advanced RL capabilities:

**Architecture**:
- **Baseline Neural Network**: Multi-output network for signal duration and anomaly detection
- **DQN Agent**: Deep Q-Network for learning optimal traffic control policies
- **State Representation**: Enhanced features from YOLO detections (12-dimensional state)
- **Action Space**: 8 discrete actions for traffic management

**Key Features**:
- Combines supervised learning (baseline NN) with reinforcement learning (DQN)
- Real-time processing of YOLO detection data
- Adaptive traffic light control based on current conditions
- Experience replay and target networks for stable training
- Comprehensive reward system for traffic efficiency and safety

**Integration with Your System**:
Your existing `NN_Training.py` provides the foundation:
- `TrafficLightNeuralNetwork` for baseline predictions
- `DQNTrafficLightAgent` for reinforcement learning
- `TrafficLightRLController` for combining both approaches
- SUMO data integration for realistic traffic simulation

## 📊 Data Flow

### Phase 1: Data Collection
```
Unity Simulation → Unity2Python.cs → ZeroMQ Publisher → Python Receiver → Dataset Storage
```

### Phase 2: YOLO Training
```
Raw Dataset → Preprocessing → YOLO Format → Training → Validation → Model Export
```

### Phase 3: Real-time Operation
```
Unity Frames → YOLO Detection → RL Processing → Control Actions → Unity Feedback
```

## ⚙️ Configuration

Edit `pipeline_config.yaml` to customize:

```yaml
# YOLO training parameters
yolo:
  epochs: 100
  batch_size: 16
  conf_threshold: 0.25

# RL parameters  
rl:
  episodes: 1000
  learning_rate: 0.0001
  batch_size: 64

# Communication ports
communication:
  unity_publisher_port: 5555
  yolo_publisher_port: 5556
  rl_publisher_port: 5557
```

## 🎮 Unity Setup Instructions

### 1. Unity2Python Script Configuration

1. Attach `Unity2Python.cs` to a GameObject in your scene
2. Assign cameras to the `cameras` array
3. Create and assign RenderTextures for each camera
4. Configure ground truth detection layers:
   - Vehicle Layer Mask
   - Pedestrian Layer Mask  
   - Traffic Light Layer Mask

### 2. Scene Requirements

- Multiple cameras positioned at intersections
- Traffic simulation with vehicles and pedestrians
- Traffic lights with controllable states
- Proper object tagging for ground truth detection

### 3. Performance Optimization

- Use appropriate render texture resolutions (640x480 recommended)
- Enable frame rate limiting (30 FPS max)
- Configure garbage collection settings
- Monitor memory usage during long runs

## 📈 Performance Monitoring

The system includes comprehensive monitoring:

### Real-time Metrics
- Unity frame rate and transmission
- YOLO detection accuracy and speed
- RL action frequency and rewards
- Pipeline latency measurements

### Training Progress
- Episode rewards and loss curves
- Model validation metrics
- Performance trend analysis
- Hyperparameter sensitivity

### System Health
- Memory usage and cleanup
- Network connection status
- Error rates and recovery
- Resource utilization

## 🔄 System Integration

### Your Existing RL System (`NN_Training.py`)
Your original system provides:
- Multi-output neural network for signal duration and anomaly detection
- DQN agent with experience replay
- SUMO data integration
- Real-time traffic light control

### New Integration (`integrated_rl_system.py`) 
The new system enhances your work by:
- Connecting to YOLO detection pipeline
- Enhanced state representation from visual data
- Improved action space for traffic control
- Real-time ZeroMQ communication
- Integration with Unity simulation

### Communication Flow
```
Unity → ZeroMQ (5555) → YOLO → ZeroMQ (5556) → Integrated RL → ZeroMQ (5557) → Unity Control
```

### Running Both Systems
```bash
# Option 1: Use integrated system (recommended)
python integrated_rl_system.py

# Option 2: Use your original system with SUMO
cd "RL Neural Network" && python NN_Training.py

# Option 3: Run both in parallel for comparison
# Terminal 1:
python integrated_rl_system.py --control-address tcp://*:5558
# Terminal 2: 
cd "RL Neural Network" && python NN_Training.py
```

### Testing Your Complete System

```bash
# Test the complete integration
python complete_pipeline_test.py --auto-find-model --duration 300

# Test individual components
python yolo_integration.py --model yolov8n.pt
python integrated_rl_system.py --yolo-address tcp://127.0.0.1:5556

# Run your original RL system
cd "RL Neural Network"
python NN_Training.py
```

### Performance Benchmarking
```bash
# Benchmark YOLO inference
python yolo_integration.py --model <model.pt> --benchmark

# Benchmark RL performance  
python traffic_rl_system.py --evaluation-mode
```

## 🐛 Troubleshooting

### Common Issues

**Unity Connection Problems**:
- Verify ZeroMQ ports are not blocked
- Check Unity2Python script configuration
- Ensure NetMQ package is installed in Unity

**YOLO Training Issues**:
- Verify dataset format and annotations
- Check GPU memory availability
- Validate image paths and labels

**RL Training Problems**:
- Monitor reward convergence
- Check action space configuration
- Verify state representation

### Debug Mode

Enable detailed logging:
```bash
python unity_zmq_receiver.py --verbose
python yolo_integration.py --debug
python traffic_rl_system.py --log-level debug
```

## 📚 Advanced Usage

### Custom Object Classes

Add new object types by:
1. Updating Unity ground truth detection
2. Modifying YOLO class mappings
3. Adjusting RL state representation

### Multi-Camera Training

Scale to multiple cameras:
1. Configure additional cameras in Unity
2. Update dataset preparation for multi-view
3. Modify RL agent for multi-camera inputs

### Transfer Learning

Use pre-trained models:
1. Load existing YOLO weights
2. Fine-tune on Unity data
3. Transfer RL knowledge between scenarios

## 📡 TCP Image Transmission

This project now includes a TCP-based image transmission system that allows Unity to send camera frames directly to Python for processing and storage.

### Components

1. **Unity Image Sender** (`ImageSender.cs`) - Captures camera frames and sends them via TCP
2. **Python TCP Receiver** (`tcp_image_receiver.py`) - Receives and stores images locally
3. **Configuration** (`port_config.yaml`) - Defines the TCP port (25002) for image transmission

### How It Works

1. Unity captures camera frames at a configurable rate (default 30 FPS)
2. Images are encoded as base64 strings and sent via TCP to the Python receiver
3. Python receiver decodes the base64 data and saves images to the `received_images` folder
4. Each image is timestamped and numbered for easy identification
5. Multi-camera support allows capturing from multiple camera angles simultaneously

### Setup

1. Attach the `ImageSender.cs` script to a GameObject in Unity
2. Assign multiple cameras to the `captureCameras` array
3. Optionally name each camera in the `cameraNames` array
4. Configure the IP address and port to match your Python receiver settings
5. Run the Python receiver: `python tcp_image_receiver.py`
6. Start the Unity scene in Play mode

### Configuration Options

In Unity:
- `captureCameras`: Array of Camera components to capture from
- `cameraNames`: Array of names for each camera (used for folder organization)
- `serverIP`: IP address of the Python receiver (default: 127.0.0.1)
- `serverPort`: Port number (default: 25002, configurable in port_config.yaml)
- `captureWidth/captureHeight`: Image resolution (default: 640x480)
- `captureRate`: Frames per second (default: 30)
- `encodeToJPG`: Whether to encode as JPG (true) or PNG (false)
- `jpgQuality`: JPG quality if encoding to JPG (10-100, default: 75)

### Multi-Camera Support

The system now supports multiple cameras:
- Images from each camera are saved in separate subfolders within `received_images`
- Camera names are used as folder names for organization
- Cameras are cycled through at the specified capture rate
- Automatic camera detection is available if no cameras are manually assigned

Example folder structure:
```
received_images/
├── front_camera/
│   ├── front_camera_image_20230101_120000_0001.png
│   └── front_camera_image_20230101_120000_0002.png
├── rear_camera/
│   ├── rear_camera_image_20230101_120000_0001.png
│   └── rear_camera_image_20230101_120000_0002.png
└── side_camera/
    ├── side_camera_image_20230101_120000_0001.png
    └── side_camera_image_20230101_120000_0002.png
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branches
3. Add tests for new functionality
4. Submit pull requests with detailed descriptions

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- Unity Technologies for the simulation platform
- Ultralytics for the YOLO implementation
- PyTorch team for the deep learning framework
- ZeroMQ community for reliable messaging

## 📞 Support

For questions and support:
1. Check the troubleshooting section
2. Review component documentation
3. Create GitHub issues for bugs
4. Join community discussions

---

**🎯 Ready to train your traffic management AI? Start with `python setup.py` and follow the quick start guide!**