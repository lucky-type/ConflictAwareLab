"""Safety Shield Wrapper for Runtime Collision Prevention.

This module implements a safety filter that intercepts agent actions
and prevents dangerous ones based on LIDAR sensor readings.

The shield operates independently of the learned policy and provides
hard safety guarantees by blocking actions that would lead to collision.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from typing import Tuple, Optional


class SafetyShieldWrapper(gym.Wrapper):
    """
    Safety Shield that intercepts and modifies dangerous actions.
    
    Uses LIDAR readings from observation to detect imminent collisions
    and replaces dangerous actions with safe fallback actions.
    
    Place this wrapper before ResidualWrapper so it sees the combined action:
    DroneSwarmEnv -> SafetyShieldWrapper -> ResidualWrapper -> AppendLambdaWrapper
    
    This ensures the Shield sees the combined action (base + K * residual),
    not the raw residual action from the policy.
    """
    
    def __init__(
        self,
        env: gym.Env,
        threshold: float = 0.3,
        wall_threshold: float = 0.1,
        fallback_strength: float = 0.8,
        lidar_start_idx: int = 8,
        num_lidar_rays: int = 72,
        verbose: int = 1,
    ):
        """
        Initialize the Safety Shield.
        
        Args:
            env: Base environment (should have LIDAR in observation)
            threshold: Minimum normalized LIDAR distance (0-1) to trigger shield.
                      Lower values = more aggressive shielding
            fallback_strength: Strength of retreat action (0-1)
            lidar_start_idx: Index where LIDAR readings start in observation.
                            Default is 8 for: goal_dir(3) + dist(1) + vel(3) + lambda(1)
            num_lidar_rays: Number of LIDAR rays
            verbose: Verbosity level (0=silent, 1=summary, 2=detailed)
        """
        super().__init__(env)
        
        self.threshold = threshold  # Drone threshold
        self.wall_threshold = wall_threshold  # Wall threshold (more conservative)
        self.fallback_strength = fallback_strength
        self.lidar_start_idx = lidar_start_idx
        self.num_lidar_rays = num_lidar_rays
        self.verbose = verbose
        
        # Tracking metrics
        self.intervention_count = 0
        self.total_steps = 0
        self.episode_interventions = 0
        
        # Store last observation and info for LIDAR extraction and hit type detection
        self._last_obs = None
        self._last_info = None
        
        # Precompute LIDAR direction vectors (for fallback action calculation)
        # Assumes Fibonacci sphere distribution matching DroneSwarmEnv
        self._lidar_directions = self._generate_lidar_directions(num_lidar_rays)
        
        if verbose >= 1:
            print(f"\n[SafetyShield] Initialized:")
            print(f"  Drone threshold: {threshold} (shield triggers when LIDAR < {threshold})")
            print(f"  Wall threshold: {wall_threshold} (shield triggers when LIDAR < {wall_threshold})")
            print(f"  Fallback strength: {fallback_strength}")
            print(f"  LIDAR start index: {lidar_start_idx}")
            print(f"  Number of rays: {num_lidar_rays}")
    
    def _generate_lidar_directions(self, num_rays: int) -> np.ndarray:
        """Generate LIDAR direction vectors using Fibonacci sphere."""
        directions = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle
        
        for i in range(num_rays):
            y = 1 - (i / float(num_rays - 1)) * 2  # y: -1 to 1
            radius = np.sqrt(1 - y * y)
            theta = phi * i
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            directions.append(np.array([x, y, z]))
        
        return np.array(directions)
    
    def _extract_lidar(self, obs: np.ndarray) -> np.ndarray:
        """Extract LIDAR readings from observation."""
        end_idx = self.lidar_start_idx + self.num_lidar_rays
        if len(obs) >= end_idx:
            return obs[self.lidar_start_idx:end_idx]
        else:
            # Fallback: assume LIDAR is at end of observation
            return obs[-self.num_lidar_rays:]
    
    def _is_dangerous(self, lidar: np.ndarray, action: np.ndarray) -> Tuple[bool, int]:
        """
        Check if current situation is dangerous.
        
        Returns:
            (is_dangerous, danger_direction_idx)
        """
        # Find minimum LIDAR reading
        min_idx = np.argmin(lidar)
        min_distance = lidar[min_idx]
        
        if min_distance < self.threshold:
            return True, min_idx
        
        return False, -1
    
    def _compute_safe_action(self, lidar: np.ndarray, danger_idx: int) -> np.ndarray:
        """
        Compute safe fallback action (retreat from danger).
        
        Strategy: Move in opposite direction of the closest obstacle.
        """
        # Get direction TO the obstacle
        danger_direction = self._lidar_directions[danger_idx]
        
        # Safe action = move AWAY from obstacle
        # For holonomic drone: action is [vx, vy, vz] normalized to [-1, 1]
        safe_direction = -danger_direction  # Opposite direction
        
        # Scale by fallback strength
        safe_action = safe_direction * self.fallback_strength
        
        # Clip to action space bounds
        safe_action = np.clip(safe_action, -1.0, 1.0).astype(np.float32)
        
        return safe_action
    
    def step(self, action: np.ndarray):
        """
        Execute step with safety shield protection.
        
        If danger is detected, the action is replaced with a safe fallback.
        Uses separate thresholds for drones vs walls based on LIDAR hit type.
        """
        self.total_steps += 1
        
        # Get LIDAR and hit info from last observation
        if self._last_obs is not None and self._last_info is not None:
            lidar = self._extract_lidar(self._last_obs)
            lidar_hit_info = self._last_info.get('lidar_hit_info', [])
            
            # Find most dangerous obstacle using type-specific thresholds
            max_danger_ratio = 0.0
            danger_idx = -1
            is_dangerous = False
            danger_type = "unknown"
            
            for i in range(len(lidar)):
                if i >= len(lidar_hit_info):
                    break
                
                distance_m, hit_type, obj_id, direction = lidar_hit_info[i]
                lidar_norm = lidar[i]  # Already normalized [0, 1]
                
                # Apply different thresholds based on hit type
                if hit_type == "drone":
                    threshold = self.threshold  # 0.3
                elif hit_type == "wall":
                    threshold = self.wall_threshold  # 0.1
                else:
                    continue  # Ignore "none" or "self"
                
                # Calculate danger ratio (how much threshold is exceeded)
                if lidar_norm < threshold:
                    danger_ratio = (threshold - lidar_norm) / threshold
                    if danger_ratio > max_danger_ratio:
                        max_danger_ratio = danger_ratio
                        danger_idx = i
                        is_dangerous = True
                        danger_type = hit_type
            
            if is_dangerous:
                # Compute safe fallback action
                original_action = action.copy()
                action = self._compute_safe_action(lidar, danger_idx)
                
                self.intervention_count += 1
                self.episode_interventions += 1
                
                if self.verbose >= 2:
                    min_dist = lidar[danger_idx]
                    used_threshold = self.threshold if danger_type == "drone" else self.wall_threshold
                    print(f"[Shield] INTERVENTION! Type={danger_type}, LIDAR={min_dist:.3f} < {used_threshold:.3f}")
                    print(f"  Original action: {original_action}")
                    print(f"  Safe action: {action}")
            elif self.verbose >= 2 and self.total_steps % 100 == 0:
                # Periodic status even when no danger
                min_dist = np.min(lidar) if len(lidar) > 0 else 1.0
                print(f"[Shield] Step {self.total_steps}: No danger detected (min_lidar={min_dist:.3f})")
        
        # Execute action
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Store observation AND info for next step
        self._last_obs = obs
        self._last_info = info
        
        # Add shield metrics to info
        # Report if an intervention happened THIS step (not just "any this episode")
        info['shield_intervention'] = is_dangerous if 'is_dangerous' in locals() else False
        info['shield_episode_interventions'] = self.episode_interventions  # Total this episode
        info['shield_total_interventions'] = self.intervention_count  # Total across all episodes
        
        # On episode end, log summary
        if (terminated or truncated) and self.verbose >= 1:
            intervention_rate = (self.episode_interventions / max(1, self.total_steps)) * 100
            if self.episode_interventions > 0:
                print(f"[Shield] Episode ended: {self.episode_interventions} interventions ({intervention_rate:.1f}%)")
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset environment and shield state."""
        obs, info = self.env.reset(**kwargs)
        
        # Store initial observation AND info
        self._last_obs = obs
        self._last_info = info
        
        # Reset episode-level counters
        self.episode_interventions = 0
        
        return obs, info
    
    def get_metrics(self) -> dict:
        """Get shield performance metrics."""
        intervention_rate = (self.intervention_count / max(1, self.total_steps)) * 100
        return {
            'shield_total_interventions': self.intervention_count,
            'shield_total_steps': self.total_steps,
            'shield_intervention_rate': intervention_rate,
        }
