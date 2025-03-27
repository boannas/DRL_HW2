from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType

class MC(BaseAlgorithm):
    def __init__(
            self,
            num_of_action: int,
            action_range: list,
            discretize_state_weight: list,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float,
    ) -> None:
        """
        Initialize the Monte Carlo algorithm.

        Args:
            num_of_action (int): Number of possible actions.
            action_range (list): Scaling factor for actions.
            discretize_state_weight (list): Scaling factor for discretizing states.
            learning_rate (float): Learning rate for Q-value updates.
            initial_epsilon (float): Initial value for epsilon in epsilon-greedy policy.
            epsilon_decay (float): Rate at which epsilon decays.
            final_epsilon (float): Minimum value for epsilon.
            discount_factor (float): Discount factor for future rewards.
        """
        super().__init__(
            control_type=ControlType.MONTE_CARLO,
            num_of_action=num_of_action,
            action_range=action_range,
            discretize_state_weight=discretize_state_weight,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
            
        )
        
    def update(
        self, 
        
    ):
        """
        Update Q-values using Monte Carlo.

        This method applies the Monte Carlo update rule to improve policy decisions by updating the Q-table.
        """
        G = 0  # Initialize return
        
        # Track visited states to ensure first-visit update
        visited_states = set()
        
        # Traverse the episode forward
        for t in range(len(self.obs_hist)):
            state = self.obs_hist[t]
            action = self.action_hist[t]
            reward = self.reward_hist[t]
            
            if (state, action) not in visited_states:
                visited_states.add((state, action))
                G = 0  # Reset G for first visit
                
                # Compute return from time step t onward
                for k in range(t, len(self.reward_hist)):
                    G += (self.discount_factor ** (k - t)) * self.reward_hist[k]
                
                # Increment visit count
                self.n_values[state][action] += 1
                
                # Update Q-value using incremental mean
                alpha = 1 / self.n_values[state][action]  # Learning rate as inverse of visits
                self.q_values[state][action] += alpha * (G - self.q_values[state][action])
        
        # Clear history after episode update
        self.obs_hist.clear()
        self.action_hist.clear()
        self.reward_hist.clear()
        
        # Decay epsilon for exploration-exploitation balance
        self.decay_epsilon()
    
    def store_experience(self, state, action, reward):
        """
        Store experience for Monte Carlo episode update.
        """
        self.obs_hist.append(state)
        self.action_hist.append(action)
        self.reward_hist.append(reward)