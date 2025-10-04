"""
Multi-Output Neural Network for Traffic Light Control
====================================================

A PyTorch implementation of a multi-output neural network designed for intelligent
traffic light control systems. This model processes features from YOLO vehicle detection
and SUMO simulation data to predict optimal signal duration and detect traffic anomalies.

Input Features:
- vehicle_count: Number of vehicles detected by YOLO
- car_ratio: Ratio of cars to total vehicles (0-1)
- truck_ratio: Ratio of trucks to total vehicles (0-1)
- avg_speed: Average speed from SUMO simulation (m/s)
- density: Traffic density from SUMO simulation (0-1)

Outputs:
1. Signal Duration: Continuous regression output (seconds)
2. Anomaly Detection: Binary classification (0=normal, 1=anomaly)

Author: PyTorch Expert
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from typing import Tuple, Dict, Optional, List
import numpy as np
import zmq
import json
import math
import time
import threading
import random
from collections import deque
import pickle


class TrafficLightNeuralNetwork(nn.Module):
    """
    Multi-output neural network for traffic light control.
    
    Architecture:
    - Input Layer: 5 features (vehicle_count, car_ratio, truck_ratio, avg_speed, density)
    - Shared Hidden Layers: Fully connected layers with ReLU activation and dropout
    - Output Head 1: Signal Duration (regression) - continuous value in seconds
    - Output Head 2: Anomaly Detection (binary classification) - 0=normal, 1=anomaly
    
    Args:
        input_size (int): Number of input features (default: 5)
        hidden_sizes (list): List of hidden layer sizes (default: [128, 64, 32])
        dropout_rate (float): Dropout rate for regularization (default: 0.2)
        signal_min_duration (float): Minimum signal duration in seconds (default: 10.0)
        signal_max_duration (float): Maximum signal duration in seconds (default: 120.0)
    """
    
    def __init__(
        self,
        input_size: int = 5,
        hidden_sizes: list = [128, 64, 32],
        dropout_rate: float = 0.2,
        signal_min_duration: float = 10.0,
        signal_max_duration: float = 120.0
    ):
        super(TrafficLightNeuralNetwork, self).__init__()
        
        # Store configuration parameters
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.signal_min_duration = signal_min_duration
        self.signal_max_duration = signal_max_duration
        
        # Build shared hidden layers
        self.shared_layers = nn.ModuleList()
        
        # Create shared hidden layers with ReLU activations
        prev_size = input_size
        for hidden_size in hidden_sizes:
            # Linear layer
            self.shared_layers.append(nn.Linear(prev_size, hidden_size))
            # ReLU activation
            self.shared_layers.append(nn.ReLU())
            # Dropout for regularization
            self.shared_layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output Head 1: Signal Duration (Regression)
        # Maps shared features to a single continuous value representing signal duration
        self.signal_duration_head = nn.Sequential(
            nn.Linear(hidden_sizes[-1], 16),  # Additional layer for signal duration
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),  # Reduced dropout for output layer
            nn.Linear(16, 1),  # Single output for duration
            nn.Sigmoid()  # Sigmoid to constrain output to [0, 1] range
        )
        
        # Output Head 2: Anomaly Detection (Binary Classification)
        # Maps shared features to anomaly probability
        self.anomaly_detection_head = nn.Sequential(
            nn.Linear(hidden_sizes[-1], 16),  # Additional layer for anomaly detection
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),  # Reduced dropout for output layer
            nn.Linear(16, 1),  # Single output for anomaly probability
            nn.Sigmoid()  # Sigmoid for binary classification probability
        )
        
        # Initialize weights using Xavier/Glorot initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize network weights using Xavier/Glorot initialization for better training stability.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the neural network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
                             Features: [vehicle_count, car_ratio, truck_ratio, avg_speed, density]
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'signal_duration': Predicted signal duration in seconds (batch_size, 1)
                - 'anomaly_probability': Anomaly detection probability (batch_size, 1)
                - 'anomaly_prediction': Binary anomaly prediction (batch_size, 1)
        """
        # Input validation
        if x.dim() != 2 or x.size(1) != self.input_size:
            raise ValueError(f"Expected input shape (batch_size, {self.input_size}), got {x.shape}")
        
        # Forward pass through shared hidden layers
        shared_features = x
        for layer in self.shared_layers:
            shared_features = layer(shared_features)
        
        # Output Head 1: Signal Duration (Regression)
        # Get raw output from sigmoid [0, 1]
        signal_duration_raw = self.signal_duration_head(shared_features)
        # Scale to actual duration range [min_duration, max_duration]
        signal_duration = self.signal_min_duration + signal_duration_raw * (self.signal_max_duration - self.signal_min_duration)
        
        # Output Head 2: Anomaly Detection (Binary Classification)
        # Get anomaly probability from sigmoid [0, 1]
        anomaly_probability = self.anomaly_detection_head(shared_features)
        # Convert probability to binary prediction using 0.5 threshold
        anomaly_prediction = (anomaly_probability > 0.5).float()
        
        return {
            'signal_duration': signal_duration,
            'anomaly_probability': anomaly_probability,
            'anomaly_prediction': anomaly_prediction
        }
    
    def predict_signal_duration(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method to get only signal duration predictions.
        
        Args:
            x (torch.Tensor): Input features
            
        Returns:
            torch.Tensor: Signal duration predictions in seconds
        """
        with torch.no_grad():
            outputs = self.forward(x)
            return outputs['signal_duration']
    
    def detect_anomaly(self, x: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method to get anomaly detection results.
        
        Args:
            x (torch.Tensor): Input features
            threshold (float): Decision threshold for anomaly detection
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (probabilities, binary_predictions)
        """
        with torch.no_grad():
            outputs = self.forward(x)
            probabilities = outputs['anomaly_probability']
            predictions = (probabilities > threshold).float()
            return probabilities, predictions
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get model architecture information.
        
        Returns:
            Dict[str, any]: Model configuration and parameter count
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'dropout_rate': self.dropout_rate,
            'signal_duration_range': (self.signal_min_duration, self.signal_max_duration),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


class ExperienceReplayBuffer:
    """
    Experience replay buffer for storing and sampling past experiences in RL training.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize experience replay buffer.
        
        Args:
            capacity (int): Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """
        Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """
        Sample a batch of experiences from buffer.
        
        Args:
            batch_size (int): Number of experiences to sample
            
        Returns:
            Tuple of batched experiences
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class DQNTrafficLightAgent(nn.Module):
    """
    Deep Q-Network (DQN) agent for traffic light control using reinforcement learning.
    
    This agent learns to control traffic lights by optimizing signal durations and
    phase changes based on traffic conditions and rewards.
    """
    
    def __init__(
        self,
        state_size: int = 8,  # Enhanced state representation
        action_size: int = 10,  # Discrete actions for signal control
        hidden_sizes: List[int] = [256, 128, 64],
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 100,
        device: str = 'cpu'
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_size (int): Size of state representation
            action_size (int): Number of possible actions
            hidden_sizes (List[int]): Hidden layer sizes
            learning_rate (float): Learning rate for optimizer
            gamma (float): Discount factor for future rewards
            epsilon (float): Initial exploration rate
            epsilon_min (float): Minimum exploration rate
            epsilon_decay (float): Epsilon decay rate
            target_update_freq (int): Frequency to update target network
            device (str): Device to run on ('cpu' or 'cuda')
        """
        super(DQNTrafficLightAgent, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.device = device
        
        # Main Q-network
        self.q_network = self._build_network(hidden_sizes)
        
        # Target Q-network (for stability)
        self.target_network = self._build_network(hidden_sizes)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.memory = ExperienceReplayBuffer(capacity=10000)
        
        # Training statistics
        self.training_step = 0
        self.episode_rewards = []
        self.losses = []
    
    def _build_network(self, hidden_sizes: List[int]) -> nn.Module:
        """Build the Q-network architecture."""
        layers = []
        prev_size = self.state_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # Output layer for Q-values
        layers.append(nn.Linear(prev_size, self.action_size))
        
        return nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Q-network.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            torch.Tensor: Q-values for all actions
        """
        return self.q_network(state)
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state (np.ndarray): Current state
            training (bool): Whether in training mode
            
        Returns:
            int: Selected action
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.forward(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self, batch_size: int = 32) -> float:
        """
        Train the agent on a batch of experiences.
        
        Args:
            batch_size (int): Batch size for training
            
        Returns:
            float: Training loss
        """
        if len(self.memory) < batch_size:
            return 0.0
        
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.losses.append(loss.item())
        return loss.item()
    
    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_rewards': self.episode_rewards,
            'losses': self.losses
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.episode_rewards = checkpoint['episode_rewards']
        self.losses = checkpoint['losses']


class TrafficLightRLController:
    """
    RL-based traffic light controller that combines the original neural network
    with reinforcement learning for adaptive traffic management.
    """
    
    def __init__(
        self,
        use_rl: bool = True,
        rl_learning_rate: float = 0.001,
        device: str = 'cpu'
    ):
        """
        Initialize RL traffic light controller.
        
        Args:
            use_rl (bool): Whether to use RL for control decisions
            rl_learning_rate (float): Learning rate for RL agent
            device (str): Device to run on
        """
        self.use_rl = use_rl
        self.device = device
        
        # Original neural network (for baseline predictions)
        self.baseline_nn = TrafficLightNeuralNetwork()
        self.baseline_nn.eval()
        
        # RL agent
        if self.use_rl:
            self.rl_agent = DQNTrafficLightAgent(
                state_size=8,  # Enhanced state representation
                action_size=10,  # 10 discrete actions for signal control
                learning_rate=rl_learning_rate,
                device=device
            )
        
        # State tracking
        self.current_state = None
        self.previous_state = None
        self.current_action = None
        self.episode_reward = 0.0
        self.episode_count = 0
        
        # Traffic light state
        self.current_phase = 0  # 0: Red, 1: Yellow, 2: Green
        self.phase_duration = 0.0
        self.total_waiting_time = 0.0
        self.vehicle_count_history = deque(maxlen=10)
        
        # Performance metrics
        self.performance_metrics = {
            'total_reward': 0.0,
            'average_waiting_time': 0.0,
            'throughput': 0.0,
            'efficiency': 0.0
        }
    
    def _create_state_representation(
        self,
        yolo_data: Dict[str, float],
        sumo_data: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Create enhanced state representation for RL agent.
        
        Args:
            yolo_data: YOLO detection data
            sumo_data: SUMO simulation data
            
        Returns:
            np.ndarray: State vector
        """
        # Base features from original NN
        vehicle_count = yolo_data.get('vehicle_count', 0)
        car_ratio = yolo_data.get('car_ratio', 0.0)
        truck_ratio = yolo_data.get('truck_ratio', 0.0)
        
        if sumo_data:
            avg_speed = sumo_data.get('avg_speed', 0.0)
            density = sumo_data.get('density', 0.0)
        else:
            avg_speed = 15.0  # Default values
            density = 0.5
        
        # Enhanced features for RL
        current_phase = self.current_phase / 2.0  # Normalize to [0, 1]
        phase_duration_norm = min(self.phase_duration / 120.0, 1.0)  # Normalize to [0, 1]
        
        # Historical features
        avg_vehicle_count = np.mean(self.vehicle_count_history) if self.vehicle_count_history else vehicle_count
        vehicle_count_trend = (vehicle_count - avg_vehicle_count) / max(avg_vehicle_count, 1.0)
        
        # Create state vector
        state = np.array([
            vehicle_count / 100.0,  # Normalized vehicle count
            car_ratio,
            truck_ratio,
            avg_speed / 50.0,  # Normalized speed
            density,
            current_phase,
            phase_duration_norm,
            vehicle_count_trend
        ], dtype=np.float32)
        
        return state
    
    def _calculate_reward(
        self,
        yolo_data: Dict[str, float],
        sumo_data: Optional[Dict[str, float]] = None,
        action: int = None
    ) -> float:
        """
        Calculate reward based on traffic efficiency and safety.
        
        Args:
            yolo_data: YOLO detection data
            sumo_data: SUMO simulation data
            action: Action taken by RL agent
            
        Returns:
            float: Reward value
        """
        if not sumo_data:
            return 0.0
        
        # Base reward components
        vehicle_count = yolo_data.get('vehicle_count', 0)
        avg_speed = sumo_data.get('avg_speed', 0.0)
        density = sumo_data.get('density', 0.0)
        
        # Efficiency reward (higher speed, lower density is better)
        efficiency_reward = (avg_speed / 50.0) * (1.0 - density) * 10.0
        
        # Throughput reward (more vehicles processed is better)
        throughput_reward = min(vehicle_count / 50.0, 1.0) * 5.0
        
        # Safety reward (avoid extreme densities)
        safety_reward = 0.0
        if density > 0.9:  # High density penalty
            safety_reward = -10.0
        elif density < 0.1:  # Very low density (inefficient)
            safety_reward = -2.0
        else:
            safety_reward = 5.0
        
        # Phase change penalty (encourage stability)
        phase_change_penalty = -1.0 if action and action != 0 else 0.0
        
        # Waiting time penalty
        waiting_penalty = -min(vehicle_count * 0.1, 5.0)
        
        total_reward = efficiency_reward + throughput_reward + safety_reward + phase_change_penalty + waiting_penalty
        
        return total_reward
    
    def _action_to_signal_duration(self, action: int) -> float:
        """
        Convert RL action to signal duration.
        
        Args:
            action (int): RL action (0-9)
            
        Returns:
            float: Signal duration in seconds
        """
        # Map actions to signal durations
        duration_mapping = {
            0: 10.0,   # Very short
            1: 15.0,   # Short
            2: 20.0,   # Short-medium
            3: 25.0,   # Medium
            4: 30.0,   # Medium-long
            5: 40.0,   # Long
            6: 50.0,   # Very long
            7: 60.0,   # Extended
            8: 80.0,   # Very extended
            9: 120.0   # Maximum
        }
        return duration_mapping.get(action, 30.0)
    
    def control_traffic_light(
        self,
        yolo_data: Dict[str, float],
        sumo_data: Optional[Dict[str, float]] = None,
        training: bool = True
    ) -> Dict[str, float]:
        """
        Control traffic light using RL agent or baseline NN.
        
        Args:
            yolo_data: YOLO detection data
            sumo_data: SUMO simulation data
            training: Whether in training mode
            
        Returns:
            Dict[str, float]: Control decisions and predictions
        """
        # Create state representation
        state = self._create_state_representation(yolo_data, sumo_data)
        
        if self.use_rl and training:
            # Use RL agent for control
            action = self.rl_agent.act(state, training=True)
            signal_duration = self._action_to_signal_duration(action)
            
            # Calculate reward if we have previous state
            if self.previous_state is not None:
                reward = self._calculate_reward(yolo_data, sumo_data, action)
                self.rl_agent.remember(
                    self.previous_state,
                    self.current_action,
                    reward,
                    state,
                    False  # Not done
                )
                self.episode_reward += reward
                
                # Train agent
                loss = self.rl_agent.replay()
                if loss > 0:
                    self.performance_metrics['total_reward'] += reward
            
            # Update state tracking
            self.previous_state = self.current_state.copy() if self.current_state is not None else None
            self.current_state = state.copy()
            self.current_action = action
            
        else:
            # Use baseline neural network
            with torch.no_grad():
                features = torch.FloatTensor(state[:5]).unsqueeze(0)  # Use first 5 features for baseline
                baseline_outputs = self.baseline_nn(features)
                signal_duration = baseline_outputs['signal_duration'].item()
                action = 0  # No RL action
        
        # Update traffic light state
        self.phase_duration = signal_duration
        self.vehicle_count_history.append(yolo_data.get('vehicle_count', 0))
        
        # Get anomaly detection from baseline NN (default to No for training)
        with torch.no_grad():
            features = torch.FloatTensor(state[:5]).unsqueeze(0)
            baseline_outputs = self.baseline_nn(features)
            # Default anomaly detection to No for training purposes
            anomaly_probability = 0.0  # Always low probability
            anomaly_prediction = 0.0   # Always No
        
        # Update performance metrics
        self._update_performance_metrics(yolo_data, sumo_data)
        
        return {
            'signal_duration': signal_duration,
            'anomaly_probability': anomaly_probability,
            'anomaly_prediction': anomaly_prediction,
            'rl_action': action if self.use_rl else None,
            'rl_epsilon': self.rl_agent.epsilon if self.use_rl else None,
            'episode_reward': self.episode_reward if self.use_rl else None
        }
    
    def _update_performance_metrics(
        self,
        yolo_data: Dict[str, float],
        sumo_data: Optional[Dict[str, float]] = None
    ):
        """Update performance metrics."""
        if sumo_data:
            self.performance_metrics['throughput'] = yolo_data.get('vehicle_count', 0)
            self.performance_metrics['efficiency'] = sumo_data.get('avg_speed', 0.0) / 50.0
    
    def start_new_episode(self):
        """Start a new RL episode."""
        if self.use_rl:
            self.episode_count += 1
            self.rl_agent.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0.0
            self.previous_state = None
            self.current_state = None
            self.current_action = None
    
    def save_model(self, filepath: str):
        """Save RL model."""
        if self.use_rl:
            self.rl_agent.save(filepath)
    
    def load_model(self, filepath: str):
        """Load RL model."""
        if self.use_rl:
            self.rl_agent.load(filepath)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        metrics = self.performance_metrics.copy()
        if self.use_rl and self.rl_agent.episode_rewards:
            metrics['average_episode_reward'] = np.mean(self.rl_agent.episode_rewards[-100:])
            metrics['epsilon'] = self.rl_agent.epsilon
        return metrics






class TrafficDataPreprocessor:
    """
    Utility class for preprocessing traffic data from YOLO and SUMO sources.
    """
    
    def __init__(
        self,
        max_vehicle_count: float = 100.0,
        max_speed: float = 50.0,  # m/s
        max_density: float = 1.0
    ):
        """
        Initialize preprocessor with normalization parameters.
        
        Args:
            max_vehicle_count: Maximum expected vehicle count for normalization
            max_speed: Maximum expected speed for normalization (m/s)
            max_density: Maximum expected density for normalization
        """
        self.max_vehicle_count = max_vehicle_count
        self.max_speed = max_speed
        self.max_density = max_density
    
    def normalize_features(
        self,
        vehicle_count: float,
        car_ratio: float,
        truck_ratio: float,
        avg_speed: float,
        density: float
    ) -> torch.Tensor:
        """
        Normalize input features to [0, 1] range for better training stability.
        
        Args:
            vehicle_count: Number of vehicles detected by YOLO
            car_ratio: Ratio of cars to total vehicles (0-1)
            truck_ratio: Ratio of trucks to total vehicles (0-1)
            avg_speed: Average speed from SUMO simulation (m/s)
            density: Traffic density from SUMO simulation (0-1)
            
        Returns:
            torch.Tensor: Normalized feature vector with batch dimension
        """
        # Normalize each feature to [0, 1] range
        norm_vehicle_count = min(vehicle_count / self.max_vehicle_count, 1.0)
        norm_car_ratio = max(0.0, min(car_ratio, 1.0))  # Clamp to [0, 1]
        norm_truck_ratio = max(0.0, min(truck_ratio, 1.0))  # Clamp to [0, 1]
        norm_avg_speed = min(avg_speed / self.max_speed, 1.0)
        norm_density = max(0.0, min(density / self.max_density, 1.0))
        
        # Create normalized feature tensor
        features = torch.tensor([
            norm_vehicle_count,
            norm_car_ratio,
            norm_truck_ratio,
            norm_avg_speed,
            norm_density
        ], dtype=torch.float32)
        
        return features.unsqueeze(0)  # Add batch dimension for single sample
    
    def batch_normalize_features(self, feature_batch: np.ndarray) -> torch.Tensor:
        """
        Normalize a batch of features.
        
        Args:
            feature_batch: Array of shape (batch_size, 5) with features
            
        Returns:
            torch.Tensor: Normalized feature batch
        """
        batch_size = feature_batch.shape[0]
        normalized_batch = np.zeros_like(feature_batch)
        
        # Normalize each feature in the batch
        normalized_batch[:, 0] = np.minimum(feature_batch[:, 0] / self.max_vehicle_count, 1.0)
        normalized_batch[:, 1] = np.clip(feature_batch[:, 1], 0.0, 1.0)
        normalized_batch[:, 2] = np.clip(feature_batch[:, 2], 0.0, 1.0)
        normalized_batch[:, 3] = np.minimum(feature_batch[:, 3] / self.max_speed, 1.0)
        normalized_batch[:, 4] = np.clip(feature_batch[:, 4], 0.0, 1.0)
        
        return torch.tensor(normalized_batch, dtype=torch.float32)


class SUMODataReceiver:
    """
    ZeroMQ subscriber to receive real-time SUMO vehicle data from Sumo2Unity.py.
    """
    
    def __init__(self, port: int = 5556, timeout: int = 1000):
        """
        Initialize SUMO data receiver.
        
        Args:
            port (int): ZeroMQ port to connect to (default: 5556)
            timeout (int): Socket timeout in milliseconds (default: 1000)
        """
        self.port = port
        self.timeout = timeout
        self.context = None
        self.socket = None
        self.connected = False
        self.latest_data = None
        self.lock = threading.Lock()
        
    def connect(self):
        """Connect to SUMO ZeroMQ publisher."""
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.SUB)
            self.socket.connect(f"tcp://localhost:{self.port}")
            self.socket.setsockopt(zmq.SUBSCRIBE, b"")
            self.socket.setsockopt(zmq.RCVHWM, 1000)
            self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)
            self.connected = True
            print(f"✅ Connected to SUMO via ZeroMQ (port {self.port})")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to SUMO: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from SUMO ZeroMQ publisher."""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        self.connected = False
        print("🔌 Disconnected from SUMO")
    
    def wait_for_data(self, max_retries: int = -1) -> bool:
        """
        Wait for SUMO data with retry logic.
        
        Args:
            max_retries (int): Maximum number of retries (-1 for infinite)
            
        Returns:
            bool: True if data received successfully
        """
        retry_count = 0
        
        while max_retries == -1 or retry_count < max_retries:
            if not self.connected:
                if not self.connect():
                    time.sleep(1)
                    retry_count += 1
                    continue
            
            try:
                # Try to receive data
                sumo_data = self.socket.recv_string()
                data = json.loads(sumo_data)
                
                if "vehicles" in data and len(data["vehicles"]) > 0:
                    # Extract vehicle data
                    vehicles = data["vehicles"]
                    
                    # Calculate average speed
                    speeds = []
                    for vehicle in vehicles:
                        long_speed = vehicle.get("long_speed", 0.0)
                        lat_speed = vehicle.get("lat_speed", 0.0)
                        vert_speed = vehicle.get("vert_speed", 0.0)
                        speed = math.sqrt(long_speed**2 + lat_speed**2 + vert_speed**2)
                        speeds.append(speed)
                    
                    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
                    
                    # Calculate density (total vehicles / road capacity)
                    total_vehicles = len(vehicles)
                    road_capacity = 100.0  # Assume road capacity = 100 vehicles
                    density = min(total_vehicles / road_capacity, 1.0)
                    
                    # Store latest data
                    with self.lock:
                        self.latest_data = {
                            "avg_speed": avg_speed,
                            "density": density,
                            "total_vehicles": total_vehicles,
                            "timestamp": time.time()
                        }
                    
                    print(f"📊 Received SUMO data: {total_vehicles} vehicles, avg_speed={avg_speed:.2f} m/s, density={density:.3f}")
                    return True
                    
            except zmq.Again:
                # Timeout occurred, retry
                retry_count += 1
                if retry_count % 10 == 0:  # Print status every 10 retries
                    print(f"⏳ Waiting for SUMO data... (retry {retry_count})")
                continue
            except json.JSONDecodeError:
                print("⚠ Invalid JSON received from SUMO, retrying...")
                retry_count += 1
                continue
            except Exception as e:
                print(f"❌ Error receiving SUMO data: {e}")
                self.connected = False
                retry_count += 1
                time.sleep(1)
                continue
        
        return False
    
    def get_latest_data(self) -> Optional[Dict[str, float]]:
        """
        Get the latest SUMO data.
        
        Returns:
            Optional[Dict[str, float]]: Latest SUMO data or None if not available
        """
        with self.lock:
            return self.latest_data.copy() if self.latest_data else None


def run_inference(yolo_input: Dict[str, float], sumo_data: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Run inference on the traffic light neural network with YOLO and SUMO inputs.
    
    This function combines YOLO vehicle detection data with SUMO simulation data,
    normalizes the features, and returns predictions for signal duration and anomaly detection.
    
    Args:
        yolo_input (Dict[str, float]): YOLO detection data containing:
            - "vehicle_count": Number of vehicles detected (int)
            - "car_ratio": Ratio of cars to total vehicles (float, 0-1)
            - "truck_ratio": Ratio of trucks to total vehicles (float, 0-1)
        
        sumo_data (Optional[Dict[str, float]]): SUMO simulation data containing:
            - "avg_speed": Average speed of vehicles (float, m/s)
            - "density": Traffic density (float, 0-1)
            If None, will use default values for testing.
    
    Returns:
        Dict[str, float]: Dictionary containing:
            - "signal_duration": Predicted signal duration in seconds
            - "anomaly_probability": Anomaly detection probability (0-1)
            - "anomaly_prediction": Binary anomaly prediction (0 or 1)
    
    Example:
        yolo_data = {"vehicle_count": 30, "car_ratio": 0.8, "truck_ratio": 0.2}
        sumo_data = {"avg_speed": 12.5, "density": 0.7}
        outputs = run_inference(yolo_data, sumo_data)
        print(f"Signal Duration: {outputs['signal_duration']:.1f} seconds")
    """
    # Initialize model and preprocessor (singleton pattern for efficiency)
    if not hasattr(run_inference, '_model'):
        run_inference._model = TrafficLightNeuralNetwork(
            input_size=5,
            hidden_sizes=[128, 64, 32],
            dropout_rate=0.2,
            signal_min_duration=10.0,
            signal_max_duration=120.0
        )
        run_inference._model.eval()  # Set to evaluation mode
    
    if not hasattr(run_inference, '_preprocessor'):
        run_inference._preprocessor = TrafficDataPreprocessor()
    
    # Validate YOLO input data
    required_yolo_keys = ["vehicle_count", "car_ratio", "truck_ratio"]
    
    for key in required_yolo_keys:
        if key not in yolo_input:
            raise ValueError(f"Missing required YOLO input key: {key}")
    
    # Validate YOLO data types and ranges
    if not isinstance(yolo_input["vehicle_count"], (int, float)) or yolo_input["vehicle_count"] < 0:
        raise ValueError("vehicle_count must be a non-negative number")
    
    for ratio_key in ["car_ratio", "truck_ratio"]:
        if not isinstance(yolo_input[ratio_key], (int, float)) or not (0 <= yolo_input[ratio_key] <= 1):
            raise ValueError(f"{ratio_key} must be a number between 0 and 1")
    
    # Handle SUMO data (use defaults if not provided)
    if sumo_data is None:
        # Use default values for testing when SUMO data is not available
        avg_speed = 15.0  # Default average speed
        density = 0.5    # Default density
        print("⚠ Using default SUMO values for testing")
    else:
        # Validate SUMO data
        if "avg_speed" not in sumo_data or "density" not in sumo_data:
            raise ValueError("SUMO data must contain 'avg_speed' and 'density' keys")
        
        avg_speed = float(sumo_data["avg_speed"])
        density = float(sumo_data["density"])
        
        if avg_speed < 0:
            raise ValueError("avg_speed must be a non-negative number")
        if not (0 <= density <= 1):
            raise ValueError("density must be a number between 0 and 1")
    
    # Normalize features using the preprocessor
    features = run_inference._preprocessor.normalize_features(
        vehicle_count=float(yolo_input["vehicle_count"]),
        car_ratio=float(yolo_input["car_ratio"]),
        truck_ratio=float(yolo_input["truck_ratio"]),
        avg_speed=avg_speed,
        density=density
    )
    
    # Run inference
    with torch.no_grad():
        outputs = run_inference._model(features)
    
    # Extract scalar values from tensors (default anomaly to No for training)
    result = {
        "signal_duration": outputs['signal_duration'].item(),
        "anomaly_probability": 0.0,  # Default to low probability
        "anomaly_prediction": 0.0    # Default to No
    }
    
    return result


def get_model_info() -> Dict[str, any]:
    """
    Get information about the neural network model architecture.
    
    Returns:
        Dict[str, any]: Model configuration and parameter information
    """
    model = TrafficLightNeuralNetwork()
    return model.get_model_info()


def main():
    """
    Main function for real-time RL traffic light control.
    """
    print("🚦 Real-Time Traffic Light Control System with Reinforcement Learning")
    print("=" * 80)
    
    # Display model information
    model_info = get_model_info()
    print(f"Neural Network Architecture:")
    print(f"- Input Size: {model_info['input_size']}")
    print(f"- Hidden Layers: {model_info['hidden_sizes']}")
    print(f"- Signal Duration Range: {model_info['signal_duration_range']} seconds")
    print(f"- Total Parameters: {model_info['total_parameters']:,}")
    
    # Start real-time RL control
    run_realtime_rl_control()


def run_realtime_rl_control():
    """Run real-time RL control with SUMO connection."""
    print("\n" + "=" * 80)
    print("REAL-TIME RL TRAFFIC LIGHT CONTROL")
    print("=" * 80)
    
    # Initialize RL controller
    rl_controller = TrafficLightRLController(
        use_rl=True,
        rl_learning_rate=0.001,
        device='cpu'
    )
    
    print("✅ RL Controller initialized with DQN agent")
    print(f"   - State Size: 8 (enhanced representation)")
    print(f"   - Action Size: 10 (discrete signal durations)")
    print(f"   - Learning Rate: 0.001")
    print(f"   - Epsilon: {rl_controller.rl_agent.epsilon:.3f}")
    
    print("\n" + "=" * 80)
    print("INITIALIZING SUMO CONNECTION")
    print("=" * 80)
    
    # Initialize SUMO data receiver
    sumo_receiver = SUMODataReceiver(port=5556, timeout=1000)
    
    # Wait for SUMO connection and first data
    print("⏳ Waiting for Sumo2Unity.py to start and send data...")
    print("   Make sure Sumo2Unity.py is running and publishing on port 5556")
    
    if not sumo_receiver.wait_for_data(max_retries=-1):
        print("❌ Failed to receive SUMO data. Exiting.")
        return
    
    print("\n" + "=" * 80)
    print("STARTING REAL-TIME RL CONTROL")
    print("=" * 80)
    print("Starting continuous RL inference loop...")
    print("Press Ctrl+C to stop")
    print()
    
    # Define YOLO input
    yolo_data = {
        "vehicle_count": 25,
        "car_ratio": 0.8,
        "truck_ratio": 0.2
    }
    
    print(f"YOLO Input Configuration:")
    print(f"  - Vehicle Count: {yolo_data['vehicle_count']}")
    print(f"  - Car Ratio: {yolo_data['car_ratio']:.2f}")
    print(f"  - Truck Ratio: {yolo_data['truck_ratio']:.2f}")
    print()
    
    cycle_count = 0
    episode_count = 0
    last_episode_time = time.time()
    
    try:
        while True:
            cycle_count += 1
            
            # Get latest SUMO data
            sumo_data = sumo_receiver.get_latest_data()
            
            if sumo_data is None:
                print(f"⚠ Cycle {cycle_count}: No SUMO data available, skipping...")
                time.sleep(1)
                continue
            
            # Run RL-controlled inference
            try:
                predictions = rl_controller.control_traffic_light(
                    yolo_data, 
                    sumo_data, 
                    training=True
                )
                
                # Display results (simplified)
                if cycle_count % 10 == 0:  # Only show every 10th cycle
                    print(f"🔄 Cycle {cycle_count} - {time.strftime('%H:%M:%S')}")
                    print(f"   SUMO: {sumo_data['total_vehicles']} vehicles, "
                          f"speed={sumo_data['avg_speed']:.1f} m/s, "
                          f"density={sumo_data['density']:.2f}")
                    print(f"   🚦 Signal: {predictions['signal_duration']:.1f}s, "
                          f"Action: {predictions['rl_action']}, "
                          f"ε: {predictions['rl_epsilon']:.3f}")
                    print(f"   ⚠  Anomaly: No (training mode)")
                    if predictions['episode_reward'] is not None:
                        print(f"   🎯 Reward: {predictions['episode_reward']:.2f}")
                    print()
                
                # Start new episode every 50 cycles or 2 minutes (reduced frequency)
                current_time = time.time()
                if cycle_count % 50 == 0 or (current_time - last_episode_time) > 120:
                    rl_controller.start_new_episode()
                    episode_count += 1
                    last_episode_time = current_time
                    
                    # Display performance metrics (simplified)
                    metrics = rl_controller.get_performance_metrics()
                    print(f"\n📈 Episode {episode_count} - Reward: {metrics['total_reward']:.2f}, ε: {metrics.get('epsilon', 0):.3f}")
                
            except Exception as e:
                print(f"❌ Cycle {cycle_count}: Error in RL inference - {e}")
                continue
            
            # Wait for next SUMO data update (increased delay to reduce cycles)
            time.sleep(2.0)  # 2 seconds between cycles instead of 0.5
            
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("🛑 STOPPING RL TRAFFIC LIGHT CONTROL SYSTEM")
        print("=" * 80)
        print("Received interrupt signal. Shutting down gracefully...")
        
        # Save RL model
        try:
            rl_controller.save_model("traffic_light_rl_model.pth")
            print("✅ RL model saved to 'traffic_light_rl_model.pth'")
        except Exception as e:
            print(f"⚠ Could not save RL model: {e}")
        
        # Display final performance metrics
        metrics = rl_controller.get_performance_metrics()
        print(f"\n📊 Final Performance Summary:")
        print(f"   - Total Episodes: {episode_count}")
        print(f"   - Total Reward: {metrics['total_reward']:.2f}")
        print(f"   - Average Episode Reward: {metrics.get('average_episode_reward', 0):.2f}")
        print(f"   - Final Epsilon: {metrics.get('epsilon', 0):.3f}")
        
    finally:
        # Cleanup
        sumo_receiver.disconnect()
        print("✅ System shutdown complete.")




# Real-time traffic light control system with RL
if __name__ == "__main__":
    main()