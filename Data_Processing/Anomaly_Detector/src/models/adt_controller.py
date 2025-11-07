"""
ADT Controller - Agent-based Dynamic Thresholding using Deep Q-Network
Based on: Yang et al. "ADT: Agent-based Dynamic Thresholding for Anomaly Detection" (2023)
Paper: https://arxiv.org/pdf/2312.01488

Learns optimal threshold adjustments from human feedback (correct/false positive labels).
Integrated with SPOT+ADWIN for per-entity threshold optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
import random


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for threshold adjustment policy.
    
    Architecture (from Yang et al. 2023):
    - Input: 3 features (current_delta, precision, alert_rate)
    - Hidden: 64 → 32 neurons with ReLU
    - Output: 5 Q-values (one per action)
    """
    
    def __init__(self, state_dim: int = 3, hidden_dim1: int = 64, 
                 hidden_dim2: int = 32, action_dim: int = 5):
        """
        Args:
            state_dim: State vector dimension (default: 3)
            hidden_dim1: First hidden layer size (default: 64)
            hidden_dim2: Second hidden layer size (default: 32)
            action_dim: Number of actions (default: 5)
        """
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, action_dim)
        
        # Xavier initialization for stable training
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: state → Q-values
        
        Args:
            state: State tensor (batch_size, 3)
        
        Returns:
            Q-values for each action (batch_size, 5)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class ExperienceReplay:
    """
    Experience replay buffer for DQN training.
    Stores (state, action, reward, next_state) transitions.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Args:
            capacity: Maximum buffer size
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool = False):
        """
        Add experience to buffer.
        
        Args:
            state: Current state (3,)
            action: Action taken (0-4)
            reward: Reward received
            next_state: Next state (3,)
            done: Episode termination flag
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample random batch from buffer.
        
        Args:
            batch_size: Number of samples
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor(np.array([x[0] for x in batch]))
        actions = torch.LongTensor([x[1] for x in batch])
        rewards = torch.FloatTensor([x[2] for x in batch])
        next_states = torch.FloatTensor(np.array([x[3] for x in batch]))
        dones = torch.FloatTensor([x[4] for x in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)


class ADTController:
    """
    Agent-based Dynamic Thresholding Controller using DQN.
    
    Learns to adjust SPOT thresholds based on human feedback.
    
    State: [current_delta, precision, alert_rate]
    Actions: [-0.02, -0.01, 0, +0.01, +0.02] (threshold adjustments)
    Reward: F1-score - volume_penalty
    
    Usage:
        controller = ADTController(entity_id='AGY_45200')
        
        # During detection
        state = controller.get_state()
        action = controller.select_action(state, epsilon=0.1)
        adjusted_threshold = base_threshold * (1 + controller.actions[action])
        
        # After human feedback
        reward = controller.compute_reward(precision=0.85, recall=0.78, alert_volume=120)
        controller.update(state, action, reward, next_state)
    """
    
    def __init__(
        self,
        entity_id: str,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 0.1,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        target_update_freq: int = 100,
        device: str = 'cpu'
    ):
        """
        Args:
            entity_id: Entity identifier
            learning_rate: Adam optimizer learning rate
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate per update
            batch_size: Replay buffer batch size
            target_update_freq: Steps between target network updates
            device: 'cpu' or 'cuda'
        """
        self.entity_id = entity_id
        self.device = torch.device(device)
        
        # Action space: threshold adjustment deltas
        self.actions = [-0.02, -0.01, 0.0, 0.01, 0.02]
        self.n_actions = len(self.actions)
        
        # DQN networks
        self.policy_net = DQNNetwork(state_dim=3, action_dim=self.n_actions).to(self.device)
        self.target_net = DQNNetwork(state_dim=3, action_dim=self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is not trained
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Experience replay
        self.replay_buffer = ExperienceReplay(capacity=10000)
        
        # Training hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # State tracking
        self.current_delta = 0.0  # Current threshold adjustment
        self.precision_history = deque(maxlen=100)  # Last 100 labels
        self.alert_history = deque(maxlen=100)  # Last 100 detection runs
        
        # Training stats
        self.update_count = 0
        self.total_reward = 0.0
    
    def get_state(self) -> np.ndarray:
        """
        Get current state vector.
        
        Returns:
            State array [current_delta, precision, alert_rate]
        """
        # Precision from last 100 human labels (with NaN handling)
        if len(self.precision_history) > 0:
            # Use nanmean to ignore NaN values
            precision = np.nanmean(self.precision_history)
            # If all values are NaN, use default
            if np.isnan(precision):
                precision = 0.5
        else:
            precision = 0.5  # Default: assume 50% precision initially
        
        # Alert rate from last 100 detection runs (with NaN handling)
        if len(self.alert_history) > 0:
            alert_rate = np.nanmean(self.alert_history)
            if np.isnan(alert_rate):
                alert_rate = 0.05
        else:
            alert_rate = 0.05  # Default: assume 5% alert rate
        
        # Sanitize current_delta (prevent NaN/Inf)
        current_delta = self.current_delta
        if not np.isfinite(current_delta):
            current_delta = 0.0
            self.current_delta = 0.0  # Reset if corrupted
        
        state = np.array([
            current_delta,
            precision,
            alert_rate
        ], dtype=np.float32)
        
        return state
    
    def select_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state (3,)
            epsilon: Exploration rate (uses self.epsilon if None)
        
        Returns:
            Action index (0-4)
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # Epsilon-greedy: explore with probability epsilon
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        
        # Exploit: select action with highest Q-value
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def compute_reward(
        self,
        precision: float,
        recall: float = None,
        alert_volume: int = 0
    ) -> float:
        """
        Compute reward from detection performance.
        
        Reward = F1-score - volume_penalty
        
        Args:
            precision: Precision (correct / total alerts)
            recall: Recall (if None, assumes 0.85 as default)
            alert_volume: Number of alerts in batch
        
        Returns:
            Reward value
        """
        # Validate inputs (prevent NaN/Inf propagation)
        if not np.isfinite(precision) or precision < 0 or precision > 1:
            precision = 0.5  # Default to neutral
        
        # Default recall if not provided
        if recall is None:
            recall = 0.85
        
        if not np.isfinite(recall) or recall < 0 or recall > 1:
            recall = 0.85  # Default
        
        # F1-score as primary reward
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        # Validate F1-score
        if not np.isfinite(f1_score):
            f1_score = 0.0
        
        # Penalty for too many alerts (target: ~50 per batch)
        target_volume = 50
        if alert_volume > target_volume:
            volume_penalty = (alert_volume - target_volume) * 0.001
        else:
            volume_penalty = 0.0
        
        reward = f1_score - volume_penalty
        
        # Final sanity check
        if not np.isfinite(reward):
            reward = 0.0
        
        return reward
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False
    ):
        """
        Update DQN from experience.
        
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: New state
            done: Episode termination flag
        """
        # Validate inputs before adding to buffer (prevent corruption)
        if not np.all(np.isfinite(state)):
            print(f"[WARNING] ADT {self.entity_id}: Non-finite state, skipping update")
            return
        
        if not np.all(np.isfinite(next_state)):
            print(f"[WARNING] ADT {self.entity_id}: Non-finite next_state, skipping update")
            return
        
        if not np.isfinite(reward):
            print(f"[WARNING] ADT {self.entity_id}: Non-finite reward, skipping update")
            return
        
        # Add to replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        # Only train if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute Q-values for current states
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values (from target network)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss (MSE between current and target Q-values)
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Validate loss (prevent NaN propagation)
        if not torch.isfinite(loss):
            print(f"[WARNING] ADT {self.entity_id}: Non-finite loss detected, skipping update")
            return
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update epsilon (decay exploration)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Track total reward
        self.total_reward += reward
    
    def update_from_feedback(
        self,
        feedback_batch: List[Dict],
        alert_rate: float
    ):
        """
        Update ADT from batch of human feedback.
        
        Args:
            feedback_batch: List of {is_correct: bool, ...}
            alert_rate: Current alert rate (alerts / samples)
        """
        # Compute precision from feedback
        if len(feedback_batch) == 0:
            return
        
        n_correct = sum(1 for f in feedback_batch if f['is_correct'])
        precision = n_correct / len(feedback_batch)
        
        # Validate alert_rate (prevent corrupted history)
        if not np.isfinite(alert_rate) or alert_rate < 0 or alert_rate > 1:
            alert_rate = 0.05  # Default to 5% if invalid
        
        # Update history
        self.precision_history.append(precision)
        self.alert_history.append(alert_rate)
        
        # Get current state
        state = self.get_state()
        
        # Select action (current policy)
        action = self.select_action(state, epsilon=0.0)  # No exploration during update
        
        # Apply action to update delta
        delta_change = self.actions[action]
        new_delta = np.clip(self.current_delta + delta_change, -0.1, 0.1)
        self.current_delta = new_delta
        
        # Compute reward
        reward = self.compute_reward(precision, alert_volume=len(feedback_batch))
        
        # Get next state
        next_state = self.get_state()
        
        # Update DQN
        self.update(state, action, reward, next_state, done=False)
    
    def get_threshold_adjustment(self) -> float:
        """
        Get current threshold adjustment factor.
        
        Returns:
            Multiplier for SPOT threshold (e.g., 1.02 = +2% increase)
        """
        return 1.0 + self.current_delta
    
    def get_state_dict(self) -> Dict:
        """
        Get controller state for serialization.
        
        Returns:
            State dictionary
        """
        return {
            'entity_id': self.entity_id,
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_delta': self.current_delta,
            'precision_history': list(self.precision_history),
            'alert_history': list(self.alert_history),
            'epsilon': self.epsilon,
            'update_count': self.update_count,
            'total_reward': self.total_reward
        }
    
    @classmethod
    def from_state_dict(cls, state_dict: Dict, device: str = 'cpu') -> 'ADTController':
        """
        Load controller from state dictionary.
        
        Args:
            state_dict: State dictionary from get_state_dict()
            device: 'cpu' or 'cuda'
        
        Returns:
            Loaded ADTController
        """
        controller = cls(entity_id=state_dict['entity_id'], device=device)
        
        controller.policy_net.load_state_dict(state_dict['policy_net'])
        controller.target_net.load_state_dict(state_dict['target_net'])
        
        # Load optimizer state dict
        optimizer_state = state_dict['optimizer']
        controller.optimizer.load_state_dict(optimizer_state)
        
        # BUG FIX #57: Move optimizer state tensors to correct device
        # Optimizer state may contain CUDA tensors if saved on GPU, but we're loading on CPU (or vice versa)
        for state in controller.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(controller.device)
        
        controller.current_delta = state_dict['current_delta']
        controller.precision_history = deque(state_dict['precision_history'], maxlen=100)
        controller.alert_history = deque(state_dict['alert_history'], maxlen=100)
        controller.epsilon = state_dict['epsilon']
        controller.update_count = state_dict['update_count']
        controller.total_reward = state_dict['total_reward']
        
        return controller


if __name__ == "__main__":
    # Test ADT controller
    print("Testing ADT Controller...")
    
    controller = ADTController(entity_id='TEST_001')
    
    # Simulate detection cycles
    for episode in range(10):
        state = controller.get_state()
        action = controller.select_action(state)
        
        # Simulate feedback (random for test)
        precision = np.random.uniform(0.7, 0.95)
        alert_volume = np.random.randint(20, 100)
        
        reward = controller.compute_reward(precision, alert_volume=alert_volume)
        
        # Update
        feedback_batch = [{'is_correct': np.random.random() > 0.2} for _ in range(alert_volume)]
        controller.update_from_feedback(feedback_batch, alert_rate=0.05)
        
        print(f"Episode {episode+1}: Action={controller.actions[action]:.2f}, "
              f"Precision={precision:.2%}, Reward={reward:.3f}, "
              f"Delta={controller.current_delta:.3f}, Epsilon={controller.epsilon:.3f}")
    
    print("\n[OK] ADT Controller test complete!")

