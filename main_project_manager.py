#!/usr/bin/env python3
"""
Main Project Manager for Unity Traffic Simulation with YOLO and RL
Manages the complete pipeline and provides a unified interface
"""

import os
import sys
import subprocess
import argparse
import time
import threading
import json
from pathlib import Path
from datetime import datetime

class TrafficSimulationManager:
    def __init__(self):
        """Initialize the project manager"""
        self.project_root = Path(__file__).parent
        self.unity_project = self.project_root / "Unity City Project" / "Latest"
        self.yolo_dir = self.project_root / "Yolo Object Detection"
        self.rl_dir = self.project_root / "RL Neural Network"
        
        # Active processes
        self.processes = {}
        self.running = False
        
        print("[MANAGER] Traffic Simulation Project Manager")
        print(f"[INFO] Project root: {self.project_root}")
        
    def check_environment(self):
        """Check if environment is properly set up"""
        print("[INFO] Checking environment...")
        
        # Check directories
        required_dirs = [self.unity_project, self.yolo_dir, self.rl_dir]
        for directory in required_dirs:
            if not directory.exists():
                print(f"[ERROR] Missing directory: {directory}")
                return False
            print(f"[SUCCESS] Found: {directory.name}")
        
        # Check key files
        key_files = [
            self.unity_project / "Assets" / "Unity2Python.cs",
            self.yolo_dir / "integrated_rl_system.py",
            self.yolo_dir / "yolo_integration.py",
            self.rl_dir / "NN_Training.py"
        ]
        
        for file_path in key_files:
            if not file_path.exists():
                print(f"[ERROR] Missing file: {file_path}")
                return False
            print(f"[SUCCESS] Found: {file_path.name}")
        
        return True
    
    def setup_project(self):
        """Setup the project environment"""
        print("[INFO] Setting up project...")
        
        os.chdir(self.yolo_dir)
        
        try:
            # Run setup script with encoding fix
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            result = subprocess.run([sys.executable, "setup.py"], 
                                  capture_output=True, text=True, 
                                  shell=True, env=env)
            if result.returncode == 0:
                print("[SUCCESS] Project setup completed")
                return True
            else:
                print(f"[ERROR] Setup failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"[ERROR] Setup error: {e}")
            return False
    
    def start_component(self, component_name, script_path, args=None):
        """Start a project component"""
        if args is None:
            args = []
        
        cmd = [sys.executable, str(script_path)] + args
        print(f"[START] Starting {component_name}...")
        
        try:
            # Use PowerShell format as per memory
            process = subprocess.Popen(cmd, cwd=script_path.parent, shell=True)
            self.processes[component_name] = process
            time.sleep(2)  # Wait for startup
            
            if process.poll() is None:
                print(f"[SUCCESS] {component_name} started successfully")
                return True
            else:
                print(f"[ERROR] {component_name} failed to start")
                return False
        except Exception as e:
            print(f"[ERROR] Error starting {component_name}: {e}")
            return False
    
    def run_data_collection(self, duration=30):
        """Run data collection from Unity"""
        print(f"[DATA] Starting data collection for {duration} minutes...")
        
        success = self.start_component(
            "Unity Data Receiver",
            self.yolo_dir / "unity_zmq_receiver.py",
            ["--save-dataset", "--dataset-path", "raw_unity_data"]
        )
        
        if success:
            print("[UNITY] Please start Unity simulation now...")
            time.sleep(duration * 60)  # Wait for specified duration
            self.stop_component("Unity Data Receiver")
        
        return success
    
    def train_yolo(self):
        """Train YOLO model"""
        print("[TRAIN] Training YOLO model...")
        
        # Prepare dataset first
        print("[DATA] Preparing dataset...")
        os.chdir(self.yolo_dir)
        
        result = subprocess.run([
            sys.executable, "dataset_preparation.py",
            "--raw-data", "raw_unity_data",
            "--output", "processed_dataset"
        ], shell=True)
        
        if result.returncode != 0:
            print("[ERROR] Dataset preparation failed")
            return False
        
        # Train YOLO
        print("[TRAIN] Starting YOLO training...")
        result = subprocess.run([
            sys.executable, "yolo_trainer.py",
            "--dataset-config", "processed_dataset/dataset.yaml",
            "--epochs", "50"
        ], shell=True)
        
        return result.returncode == 0
    
    def run_complete_pipeline(self):
        """Run the complete pipeline"""
        print("[PIPELINE] Starting complete pipeline...")
        
        # Clean up any existing processes that might hold ports
        print("[CLEANUP] Checking for existing Python processes...")
        try:
            result = subprocess.run(["taskkill", "/F", "/IM", "python.exe"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("[CLEANUP] Cleaned up existing Python processes")
                time.sleep(2)  # Wait for cleanup
        except:
            pass  # No existing processes to clean
        
        # Start YOLO integration first (it binds to port 5556)
        yolo_success = self.start_component(
            "YOLO Integration",
            self.yolo_dir / "yolo_integration.py",
            ["--model", "yolov8n.pt"]
        )
        
        if not yolo_success:
            return False
        
        # Wait for YOLO integration to fully initialize and bind ports
        print("[INFO] Waiting for YOLO Integration to initialize...")
        time.sleep(5)
        
        # Start integrated RL system (it connects to port 5556, binds to port 5557)
        rl_success = self.start_component(
            "Integrated RL System",
            self.yolo_dir / "integrated_rl_system.py",
            ["--control-address", "tcp://*:5557"]
        )
        
        if not rl_success:
            self.stop_component("YOLO Integration")
            return False
        
        print("[SUCCESS] Complete pipeline started successfully!")
        print("[UNITY] Please start Unity simulation...")
        print("[STOP] Press Ctrl+C to stop")
        
        self.running = True
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[STOP] Stopping pipeline...")
            self.stop_all_components()
        
        return True
    
    def run_original_rl(self):
        """Run your original RL system"""
        print("[RL] Starting original RL system...")
        
        os.chdir(self.rl_dir)
        
        try:
            subprocess.run([sys.executable, "NN_Training.py"], shell=True)
            return True
        except Exception as e:
            print(f"[ERROR] Error running original RL: {e}")
            return False
    
    def test_pipeline(self):
        """Test the complete pipeline"""
        print("[TEST] Testing complete pipeline...")
        
        os.chdir(self.yolo_dir)
        
        result = subprocess.run([
            sys.executable, "complete_pipeline_test.py",
            "--auto-find-model", "--duration", "60"
        ], shell=True)
        
        return result.returncode == 0
    
    def stop_component(self, component_name):
        """Stop a specific component"""
        if component_name in self.processes:
            process = self.processes[component_name]
            if process.poll() is None:
                process.terminate()
                process.wait()
            del self.processes[component_name]
            print(f"[STOP] Stopped {component_name}")
    
    def stop_all_components(self):
        """Stop all running components"""
        self.running = False
        for name in list(self.processes.keys()):
            self.stop_component(name)
    
    def cleanup_old_files(self):
        """Remove duplicate and old files"""
        print("🧹 Cleaning up old/duplicate files...")
        
        # Files to remove (duplicates/old versions)
        files_to_remove = [
            self.yolo_dir / "Unity2Python.py",  # Old version, replaced by unity_zmq_receiver.py
            self.yolo_dir / "web_server.py",    # Old web server
            self.yolo_dir / "requirements.txt", # Old requirements, use requirements_updated.txt
            self.yolo_dir / "infer_yolo.py",    # Old inference script
            self.yolo_dir / "train_yolo.py",    # Old training script
            self.yolo_dir / "rl_env.py",        # Old RL environment
            self.yolo_dir / "traffic_rl_system.py"  # Replaced by integrated_rl_system.py
        ]
        
        removed_count = 0
        for file_path in files_to_remove:
            if file_path.exists():
                try:
                    file_path.unlink()
                    print(f"[REMOVED] Removed: {file_path.name}")
                    removed_count += 1
                except Exception as e:
                    print(f"[WARNING] Could not remove {file_path.name}: {e}")
        
        print(f"[SUCCESS] Cleaned up {removed_count} old files")
        
        # Create backup of important files
        backup_dir = self.project_root / "backup"
        backup_dir.mkdir(exist_ok=True)
        
        important_files = [
            self.rl_dir / "NN_Training.py",
            self.unity_project / "Assets" / "Unity2Python.cs"
        ]
        
        for file_path in important_files:
            if file_path.exists():
                backup_path = backup_dir / f"{file_path.name}.backup"
                try:
                    backup_path.write_text(file_path.read_text())
                    print(f"[BACKUP] Backed up: {file_path.name}")
                except Exception as e:
                    print(f"[WARNING] Could not backup {file_path.name}: {e}")
    
    def show_status(self):
        """Show project status"""
        print("\n[STATUS] Project Status:")
        print(f"   Project root: {self.project_root}")
        print(f"   Unity project: {'[OK]' if self.unity_project.exists() else '[MISSING]'}")
        print(f"   YOLO directory: {'[OK]' if self.yolo_dir.exists() else '[MISSING]'}")
        print(f"   RL directory: {'[OK]' if self.rl_dir.exists() else '[MISSING]'}")
        
        print(f"\n[PROCESSES] Running processes:")
        for name, process in self.processes.items():
            status = "Running" if process.poll() is None else "Stopped"
            print(f"   {name}: {status}")
        
        # Check for trained models
        models_dir = self.yolo_dir / "trained_models"
        if models_dir.exists():
            models = list(models_dir.glob("*.pt"))
            print(f"\n[MODELS] Trained models: {len(models)}")
            for model in models[-3:]:  # Show last 3 models
                print(f"   {model.name}")

def main():
    parser = argparse.ArgumentParser(description='Traffic Simulation Project Manager')
    parser.add_argument('action', choices=[
        'setup', 'collect-data', 'train-yolo', 'run-pipeline', 
        'run-original-rl', 'test', 'cleanup', 'status'
    ], help='Action to perform')
    parser.add_argument('--duration', type=int, default=30, 
                       help='Duration for data collection (minutes)')
    
    args = parser.parse_args()
    
    manager = TrafficSimulationManager()
    
    if not manager.check_environment():
        print("[ERROR] Environment check failed. Please ensure all directories and files are present.")
        sys.exit(1)
    
    if args.action == 'setup':
        success = manager.setup_project()
    elif args.action == 'collect-data':
        success = manager.run_data_collection(args.duration)
    elif args.action == 'train-yolo':
        success = manager.train_yolo()
    elif args.action == 'run-pipeline':
        manager.running = True
        success = manager.run_complete_pipeline()
    elif args.action == 'run-original-rl':
        success = manager.run_original_rl()
    elif args.action == 'test':
        success = manager.test_pipeline()
    elif args.action == 'cleanup':
        manager.cleanup_old_files()
        success = True
    elif args.action == 'status':
        manager.show_status()
        success = True
    else:
        print(f"[ERROR] Unknown action: {args.action}")
        success = False
    
    if success:
        print(f"\n[SUCCESS] Action '{args.action}' completed successfully!")
    else:
        print(f"\n[ERROR] Action '{args.action}' failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()