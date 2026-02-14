"""Training callback for metrics broadcasting and trajectory recording."""
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pybullet as p
from stable_baselines3.common.callbacks import BaseCallback

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from .. import models


class ExperimentCallback(BaseCallback):
    """Custom callback for Stable-Baselines3 to broadcast metrics during training."""

    def __init__(
        self,
        experiment_id: int,
        broadcast_func,
        loop,
        save_interval: int = 100,
        snapshot_freq: int = 10000,
        total_steps: int = 100000,
        model_save_func=None,
        db_session_factory=None,
        environment=None,
        lagrange_state=None,
        safety_config=None,
        lagrange_callback=None,
        run_without_simulation: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.experiment_id = experiment_id
        self.broadcast_func = broadcast_func
        self.loop = loop
        self.save_interval = save_interval
        self.snapshot_freq = snapshot_freq
        self.total_steps = total_steps
        self.model_save_func = model_save_func
        self.db_session_factory = db_session_factory
        self.environment = environment
        self.lagrange_state = lagrange_state  # For tracking λ
        self.safety_config = safety_config  # For tracking ε
        self.lagrange_callback = lagrange_callback  # For tracking warmup state
        self.run_without_simulation = run_without_simulation
        
        # Episode tracking
        self.episode_count = 0
        self.trajectory = []  # Store trajectory for replay
        self.current_episode_frames = []  # Frames for current episode
        
        # Performance metrics (A)
        self.episode_rewards_raw = []  # Raw rewards (before shaping)
        self.episode_rewards_shaped = []  # Shaped rewards (r' = r - λ*cost)
        self.episode_lengths = []
        self.episode_successes = []  # Boolean success per episode
        self.episode_crashes = []  # Boolean crash per episode
        self.episode_timeouts = []  # Boolean timeout per episode
        
        # Safety/Risk metrics (B)
        self.episode_costs = []  # CUMULATIVE cost per episode (sum of all costs)
        self.episode_near_misses = []  # Count of near-misses per episode
        self.episode_danger_time = []  # Steps in danger zone per episode
        
        # Current episode accumulators
        self.current_episode_reward_raw = 0.0
        self.current_episode_reward_shaped = 0.0
        self.current_episode_cost = 0.0  # CUMULATIVE cost for current episode
        self.current_episode_near_misses = 0
        self.current_episode_danger_time = 0
        self.current_episode_length = 0
        
        # Residual learning metrics
        self.residual_magnitudes = []
        self.base_magnitudes = []
        
        # NEW: Advanced residual learning metrics (Group 3)
        self.cosine_similarities = []  # Cosine similarity between base and residual actions
        self.intervention_flags = []  # Boolean flags for significant interventions
        self.effective_k_values = []  # Adaptive K-factor values
        
        # NEW: CARS K components for charting
        self.k_conf_values = []    # K_conf (conflict-aware)
        self.k_risk_values = []    # K_risk (λ-based)
        self.action_clipping_flags = []  # Action saturation tracking

        # Cache for algorithm-specific metrics (captured during training)
        self.last_train_metrics = {}
        
        # Vectorized environment support: per-env cumulative cost buffers
        # Will be initialized in _on_training_start when we know num_envs
        self.episode_cum_costs = None  # List of cumulative costs, one per env
        self.episode_cum_rewards_raw = None  # Per-env reward accumulators
        self.episode_cum_rewards_shaped = None  # Per-env shaped reward accumulators
        self.episode_lengths_per_env = None  # Per-env step counters
        
        # Best model tracking (for catastrophic forgetting protection)
        self.best_success_rate = 0.0
        self.last_avg_cost = 0.0  # For smooth graphs (avoid spiky zeros)
    
    def _on_training_start(self) -> None:
        """Called when training starts - hook into logger"""
        # Initialize per-env cumulative cost buffers for vectorized envs
        if hasattr(self.model, 'env') and hasattr(self.model.env, 'num_envs'):
            num_envs = self.model.env.num_envs
        else:
            num_envs = 1  # Single environment
        
        # Initialize per-env buffers only once so chunked training doesn't reset mid-episode stats.
        # SB3 calls _on_training_start at the beginning of every model.learn() chunk; resetting here
        # would wipe accumulated costs and underreport safety violations.
        if self.episode_cum_costs is None:
            self.episode_cum_costs = [0.0] * num_envs
            self.episode_cum_rewards_raw = [0.0] * num_envs
            self.episode_cum_rewards_shaped = [0.0] * num_envs
            self.episode_lengths_per_env = [0] * num_envs
            # New: Per-env tracking for near-misses and danger time
            self.episode_cum_near_misses = [0] * num_envs
            self.episode_cum_danger = [0] * num_envs
            print(f"[Callback] Initialized {num_envs} environment(s) with per-env tracking buffers")
        else:
            print(f"[Callback] Preserving existing buffers across training chunk (num_envs={num_envs})")
        
        # Store original logger record method
        if hasattr(self.model, 'logger') and self.model.logger:
            original_record = self.model.logger.record
            callback_self = self
            
            def wrapped_record(key: str, value, exclude=None):
                # Capture training metrics as they're logged
                if key.startswith("train/"):
                    callback_self.last_train_metrics[key] = value
                return original_record(key, value, exclude)
            
            # Replace logger's record method
            self.model.logger.record = wrapped_record
            print("[Callback] Hooked into SB3 logger to capture training metrics")

    def _on_step(self) -> bool:
        """Called after each environment step."""
        
        # Debug: Print every 100 steps
        if self.n_calls % 100 == 0:
            print(f"[Callback] Step {self.n_calls}, trajectory size: {len(self.trajectory)}, current episode frames: {len(self.current_episode_frames)}")
        
        # Capture trajectory data for replay
        try:
            # Get current observation and info from the environment
            # Handle vectorized environments - extract first env's data
            new_obs_raw = self.locals.get("new_obs")
            rewards_raw = self.locals.get("rewards")
            dones_raw = self.locals.get("dones")
            infos_raw = self.locals.get("infos")
            
            # Extract scalars from arrays
            if new_obs_raw is not None and len(new_obs_raw) > 0:
                obs = new_obs_raw[0]
            else:
                obs = None
                
            if rewards_raw is not None and len(rewards_raw) > 0:
                reward = float(rewards_raw[0])
            else:
                reward = 0.0
                
            if dones_raw is not None and len(dones_raw) > 0:
                done = bool(dones_raw[0])
            else:
                done = False
            
            # Extract info
            info = {}
            if infos_raw is not None and len(infos_raw) > 0:
                info = infos_raw[0]
            
            # Collect residual learning metrics if available
            if "residual_info" in info:
                res_info = info["residual_info"]
                self.residual_magnitudes.append(res_info.get("residual_magnitude", 0.0))
                self.base_magnitudes.append(res_info.get("base_magnitude", 0.0))
                
                # Use pre-computed cos_sim from ResidualActionWrapper (more accurate)
                if "cos_sim" in res_info:
                    self.cosine_similarities.append(res_info["cos_sim"])
                elif "base_action" in res_info and "residual_action" in res_info:
                    # Fallback: calculate if not provided
                    base_action = np.array(res_info["base_action"])
                    residual_action = np.array(res_info["residual_action"])
                    base_norm = np.linalg.norm(base_action)
                    res_norm = np.linalg.norm(residual_action)
                    if base_norm > 1e-8 and res_norm > 1e-8:
                        cos_sim = float(np.dot(base_action, residual_action) / (base_norm * res_norm))
                    else:
                        cos_sim = 0.0
                    self.cosine_similarities.append(cos_sim)
                
                # Collect effective_k for adaptive K-factor tracking
                if "effective_k" in res_info:
                    self.effective_k_values.append(res_info["effective_k"])
                
                # NEW: Collect CARS K components
                if "K_conf" in res_info:
                    self.k_conf_values.append(res_info["K_conf"])
                if "K_risk" in res_info:
                    self.k_risk_values.append(res_info["K_risk"])
                if "action_clipped" in res_info:
                    self.action_clipping_flags.append(res_info["action_clipped"])

                # Check for significant intervention
                res_norm = res_info.get("residual_magnitude", 0.0)
                intervention_threshold = 0.05
                is_intervention = res_norm > intervention_threshold
                self.intervention_flags.append(is_intervention)
            
            # ===== METRIC TRACKING =====
            # Track metrics per environment for vectorized runs (num_envs > 1).
            
            # Iterate through all environments to accumulate costs AND handle episode completions
            for env_idx in range(len(infos_raw) if infos_raw else 1):
                env_info = infos_raw[env_idx] if infos_raw and env_idx < len(infos_raw) else {}
                env_done = dones_raw[env_idx] if dones_raw and env_idx < len(dones_raw) else False
                env_reward = rewards_raw[env_idx] if rewards_raw and env_idx < len(rewards_raw) else 0.0
                
                cost_this_step = env_info.get('cost', 0.0)
                
                # Accumulate metrics for this specific environment
                if (self.episode_cum_costs is not None and env_idx < len(self.episode_cum_costs)):
                    self.episode_cum_costs[env_idx] += cost_this_step
                    self.episode_lengths_per_env[env_idx] += 1
                    
                    # Track rewards (reconstruct raw reward if Lagrangian shaping is active)
                    if self.lagrange_state:
                        reward_shaped = env_reward
                        reward_raw = reward_shaped + (self.lagrange_state.lam * cost_this_step)
                    else:
                        reward_raw = env_reward
                        reward_shaped = env_reward
                    
                    self.episode_cum_rewards_raw[env_idx] += reward_raw
                    self.episode_cum_rewards_shaped[env_idx] += reward_shaped
                    
                    # Track near-misses and danger time for THIS environment
                    min_drone_dist = env_info.get('min_drone_distance')
                    if min_drone_dist is not None and min_drone_dist != float('inf'):
                        # Get thresholds (default to standard values if env attribute missing)
                        near_miss_thresh = 1.5
                        if self.environment and hasattr(self.environment, 'near_miss_threshold'):
                            near_miss_thresh = self.environment.near_miss_threshold
                        
                        danger_thresh = near_miss_thresh * 0.5
                        
                        if min_drone_dist < near_miss_thresh and not env_info.get('crashed', False):
                            self.episode_cum_near_misses[env_idx] += 1
                        
                        if min_drone_dist < danger_thresh:
                            self.episode_cum_danger[env_idx] += 1
                    
                    # Record metrics when each environment finishes to capture per-env episodes.
                    if env_done:
                        self.episode_count += 1
                        ep_len = max(1, self.episode_lengths_per_env[env_idx])
                        cumulative_cost = self.episode_cum_costs[env_idx]
                        
                        # Record this episode's metrics to main lists
                        self.episode_rewards_raw.append(self.episode_cum_rewards_raw[env_idx])
                        self.episode_rewards_shaped.append(self.episode_cum_rewards_shaped[env_idx])
                        self.episode_lengths.append(ep_len)
                        self.episode_successes.append(env_info.get('success', False))
                        self.episode_crashes.append(env_info.get('crashed', False))
                        
                        is_timeout = env_info.get('timeout', False) and not env_info.get('success', False) and not env_info.get('crashed', False)
                        self.episode_timeouts.append(is_timeout)
                        
                        # Safety metrics - compare cumulative_cost with epsilon (per-episode budget)
                        # NOTE: epsilon is the per-episode budget (e.g., 0.3 means max 0.3 cumulative cost)
                        epsilon_value = self.safety_config.get('risk_budget', 0.3) if self.safety_config else 0.3
                        # avg_cost_per_step = cumulative_cost / ep_len  # OLD INCORRECT LOGIC
                        # violation = avg_cost_per_step > epsilon_value # OLD INCORRECT LOGIC
                        
                        # Fix: Check cumulative cost against budget
                        violation = cumulative_cost > epsilon_value
                        
                        if self.verbose >= 1 or env_idx == 0:  # Always log env 0, optionally others
                            print(f"[EXPERIMENT_SAFETY] Env {env_idx} Episode {self.episode_count}: "
                                  f"eps={epsilon_value:.3f} len={ep_len} cost_cum={cumulative_cost:.3f} "
                                  f"viol={violation}")
                        
                        self.episode_costs.append(cumulative_cost)
                        
                        # Save accumulated metrics
                        self.episode_near_misses.append(float(self.episode_cum_near_misses[env_idx]))
                        self.episode_danger_time.append(float(self.episode_cum_danger[env_idx]))
                        
                        # Reset this environment's accumulators
                        self.episode_cum_costs[env_idx] = 0.0
                        self.episode_cum_rewards_raw[env_idx] = 0.0
                        self.episode_cum_rewards_shaped[env_idx] = 0.0
                        self.episode_lengths_per_env[env_idx] = 0
                        self.episode_cum_near_misses[env_idx] = 0
                        self.episode_cum_danger[env_idx] = 0
            
            # For backward compatibility, ALSO track primary environment (env 0) in legacy accumulators
            # This supports visualization/trajectory recording which currently expects env 0 only
            cost_this_step = info.get('cost', 0.0)
            self.current_episode_cost += cost_this_step
            self.current_episode_length += 1
            
            # Track raw reward (before Lagrangian shaping) for env 0
            if self.lagrange_state:
                reward_shaped = reward
                reward_raw = reward_shaped + (self.lagrange_state.lam * cost_this_step)
            else:
                reward_raw = reward
                reward_shaped = reward
            
            self.current_episode_reward_raw += reward_raw
            self.current_episode_reward_shaped += reward_shaped
            
            # Track near-misses and danger time from LIDAR
            # Only treat inter-drone proximity as near-miss; ignore walls for this metric.
            min_drone_distance_m = info.get('min_drone_distance')
            
            if min_drone_distance_m is not None and min_drone_distance_m != float('inf'):
                # Near-miss threshold from environment (in meters)
                near_miss_threshold_m = 1.5  # Default in meters
                if self.environment and hasattr(self.environment, 'near_miss_threshold'):
                    near_miss_threshold_m = self.environment.near_miss_threshold
                
                # Check if this step is a near-miss to a DRONE (not wall)
                if min_drone_distance_m < near_miss_threshold_m and not info.get('crashed', False):
                    self.current_episode_near_misses += 1
                
                # Check if in danger zone (HALF of near-miss threshold = stricter/closer)
                danger_threshold_m = near_miss_threshold_m * 0.5
                if min_drone_distance_m < danger_threshold_m:
                    self.current_episode_danger_time += 1
            
            # Check if obs exists (handle both None and numpy arrays)
            has_obs = obs is not None
            if isinstance(obs, np.ndarray):
                has_obs = obs.size > 0
            
            if not self.run_without_simulation and has_obs and self.environment:
                # Map observation to position based on drone environment
                env_length = self.environment.length if self.environment else 100.0
                env_width = self.environment.width if self.environment else 20.0
                env_height = getattr(self.environment, 'height', 20.0)
                
                obs_array = np.array(obs) if not isinstance(obs, np.ndarray) else obs
                obs_len = len(obs_array)
                
                # New observation structure (simplified):
                # [0-2]: goal direction (normalized to [0,1])
                # [3]: goal distance (normalized)
                # [4-6]: velocity (normalized to [0,1])
                # [7-20]: LIDAR readings (14 directions)
                
                # Get actual position from PyBullet environment instead of observation
                x_pos, y_pos, z_pos = 0.0, env_height / 2.0, env_length / 2.0
                velocity = [0.0, 0.0, 0.0]
                orientation = [0.0, 0.0, 0.0]
                
                try:
                    # Try to get the actual environment (unwrap if vectorized)
                    env_to_check = None
                    if hasattr(self, 'model') and hasattr(self.model, 'env'):
                        training_env = self.model.env
                        # Check if it's a VecEnv (vectorized)
                        if hasattr(training_env, 'envs'):
                            env_to_check = training_env.envs[0]
                        else:
                            env_to_check = training_env
                        
                        # Unwrap if needed
                        if hasattr(env_to_check, 'unwrapped'):
                            env_to_check = env_to_check.unwrapped
                        
                        # Get actual position from PyBullet
                        if env_to_check and hasattr(env_to_check, 'drone_id'):
                            physics_client = getattr(env_to_check, 'physics_client', 0)
                            pos, quat = p.getBasePositionAndOrientation(env_to_check.drone_id, physicsClientId=physics_client)
                            x_pos, y_pos, z_pos = float(pos[0]), float(pos[1]), float(pos[2])
                            
                            # Get velocity
                            vel, ang_vel = p.getBaseVelocity(env_to_check.drone_id, physicsClientId=physics_client)
                            velocity = [float(vel[0]), float(vel[1]), float(vel[2])]
                            
                            # Convert quaternion to euler angles
                            qx, qy, qz, qw = quat
                            pitch = np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
                            roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
                            yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                            orientation = [float(roll), float(pitch), float(yaw)]
                            
                except Exception as e:
                    if self.n_calls % 500 == 0:  # Log occasionally
                        print(f"[Callback] Could not get position from PyBullet: {e}")
                
                # Get goal position directly from environment instead of reconstructing from observation
                try:
                    # Access environment - handle both vectorized and non-vectorized envs
                    env_for_goal = None
                    if hasattr(self, 'model') and hasattr(self.model, 'env'):
                        training_env = self.model.env
                        # Check if it's a VecEnv (vectorized)
                        if hasattr(training_env, 'envs'):
                            env_for_goal = training_env.envs[0]
                        else:
                            env_for_goal = training_env
                        
                        # Unwrap if needed
                        if hasattr(env_for_goal, 'unwrapped'):
                            env_for_goal = env_for_goal.unwrapped
                    
                    # Try to get goal_position directly from environment
                    if env_for_goal and hasattr(env_for_goal, 'goal_position') and env_for_goal.goal_position is not None:
                        goal_x = float(env_for_goal.goal_position[0])
                        goal_y = float(env_for_goal.goal_position[1])
                        goal_z = float(env_for_goal.goal_position[2])
                        if self.n_calls % 500 == 0:
                            print(f"[Training Callback {self.n_calls}] Using env.goal_position: [{goal_x:.2f}, {goal_y:.2f}, {goal_z:.2f}]")
                    else:
                        # Fallback: reconstruct from observation (but with correct formula - obs is already in [-1,1])
                        goal_dir_x = float(obs_array[0]) if obs_len > 0 else 0.0
                        goal_dir_y = float(obs_array[1]) if obs_len > 1 else 0.0
                        goal_dir_z = float(obs_array[2]) if obs_len > 2 else 0.0
                        goal_distance_norm = float(obs_array[3]) if obs_len > 3 else 0.5
                        
                        max_possible_distance = np.linalg.norm([env_length, env_width, env_height])
                        goal_distance = goal_distance_norm * max_possible_distance
                        goal_x = x_pos + goal_dir_x * goal_distance
                        goal_y = y_pos + goal_dir_y * goal_distance
                        goal_z = z_pos + goal_dir_z * goal_distance
                        if self.n_calls % 500 == 0:
                            print(f"[Training Callback {self.n_calls}] WARNING: Using reconstructed goal: [{goal_x:.2f}, {goal_y:.2f}, {goal_z:.2f}]")
                except Exception as e:
                    # Ultimate fallback - use observation (with correct formula)
                    goal_dir_x = float(obs_array[0]) if obs_len > 0 else 0.0
                    goal_dir_y = float(obs_array[1]) if obs_len > 1 else 0.0
                    goal_dir_z = float(obs_array[2]) if obs_len > 2 else 0.0
                    goal_distance_norm = float(obs_array[3]) if obs_len > 3 else 0.5
                    
                    max_possible_distance = np.linalg.norm([env_length, env_width, env_height])
                    goal_distance = goal_distance_norm * max_possible_distance
                    goal_x = x_pos + goal_dir_x * goal_distance
                    goal_y = y_pos + goal_dir_y * goal_distance
                    goal_z = z_pos + goal_dir_z * goal_distance
                    if self.n_calls % 500 == 0:
                        print(f"[Training Callback {self.n_calls}] ERROR getting goal: {e}, using fallback")
                
                # Create frame data
                frame_data = {
                    "experiment_id": self.experiment_id,
                    "frame": self.n_calls,
                    "episode": self.episode_count,
                    "timestamp": time.time(),
                    "agent_position": {
                        "x": float(x_pos),
                        "y": float(y_pos),
                        "z": float(z_pos),
                    },
                    "agent_velocity": {
                        "x": float(velocity[0]),
                        "y": float(velocity[1]),
                        "z": float(velocity[2]),
                    },
                    "agent_orientation": {
                        "roll": float(orientation[0]),
                        "pitch": float(orientation[1]),
                        "yaw": float(orientation[2]),
                    },
                    "target_position": {
                        "x": float(goal_x),
                        "y": float(goal_y),
                        "z": float(goal_z),
                    },
                    "obstacles": [],
                    "lidar_readings": [],
                }
                
                # Add LIDAR readings from info if available
                if 'lidar_readings' in info:
                    frame_data["lidar_readings"] = [float(x) for x in info['lidar_readings']]
                
                # Add LIDAR hit information for visualization
                if 'lidar_hit_info' in info:
                    lidar_hits = []
                    for hit in info['lidar_hit_info']:
                        distance, hit_type, obj_id, direction_vec = hit
                        lidar_hits.append({
                            "distance": float(distance),
                            "type": hit_type,
                            "direction": [float(direction_vec[0]), float(direction_vec[1]), float(direction_vec[2])]
                        })
                    frame_data["lidar_hit_info"] = lidar_hits
                
                # Add swarm drone positions if available
                try:
                    # Access environment - try multiple paths
                    env_to_check = None
                    
                    # First try: stored environment reference
                    if self.environment and hasattr(self.environment, 'swarm_drones'):
                        env_to_check = self.environment
                    # Second try: from model's env
                    elif hasattr(self, 'model') and hasattr(self.model, 'env'):
                        training_env = self.model.env
                        # Unwrap VecEnv wrapper
                        if hasattr(training_env, 'venv'):
                            training_env = training_env.venv
                        # Get first env from DummyVecEnv
                        if hasattr(training_env, 'envs') and len(training_env.envs) > 0:
                            env_to_check = training_env.envs[0]
                        else:
                            env_to_check = training_env
                        
                        # Unwrap Monitor wrapper
                        while hasattr(env_to_check, 'env'):
                            env_to_check = env_to_check.env
                    
                    if env_to_check and hasattr(env_to_check, 'swarm_drones'):
                        swarm_drones = env_to_check.swarm_drones
                        physics_client = getattr(env_to_check, 'physics_client', 0)
                        obstacles_data = []
                        
                        for drone in swarm_drones:
                            try:
                                pos, _ = p.getBasePositionAndOrientation(drone.id, physicsClientId=physics_client)
                                vel, _ = p.getBaseVelocity(drone.id, physicsClientId=physics_client)
                                obstacles_data.append({
                                    "x": float(pos[0]),
                                    "y": float(pos[1]),
                                    "z": float(pos[2]),
                                    "diameter": float(drone.diameter),
                                    "speed": float(drone.max_speed),
                                    "strategy": drone.strategy,
                                    "chaos": float(drone.chaos),
                                    "vx": float(vel[0]),
                                    "vy": float(vel[1]),
                                    "vz": float(vel[2])
                                })
                            except Exception as drone_err:
                                # Skip individual drone if error
                                continue
                        
                        if len(obstacles_data) > 0:
                            frame_data["obstacles"] = obstacles_data
                        
                        # Debug log first time we successfully capture swarm
                        if self.n_calls == 1 and len(obstacles_data) > 0:
                            print(f"[Callback] ✅ Successfully captured {len(obstacles_data)} swarm drones")
                except Exception as e:
                    # Log error only on first call to avoid spam
                    if self.n_calls == 1:
                        print(f"[Callback] ⚠️ Could not get swarm drone positions: {e}")
                    pass
                
                frame_data["reward"] = float(reward)
                frame_data["done"] = bool(done)
                frame_data["success"] = bool(info.get('success', False))
                frame_data["crashed"] = bool(info.get('crashed', False))
                frame_data["crash_type"] = str(info.get('crash_type', 'none'))
                
                self.current_episode_frames.append(frame_data)
                
                # Broadcast live frame every 50 steps for visualization during training
                if self.n_calls % 50 == 0:
                    try:
                        asyncio.run_coroutine_threadsafe(
                            self.broadcast_func(
                                self.experiment_id,
                                {
                                    "type": "simulation_frame",
                                    "data": frame_data,
                                },
                            ),
                            self.loop
                        )
                    except Exception as e:
                        print(f"[Callback] Frame broadcast error: {e}")
                
                # If episode ended, save frames to trajectory (sample every Nth frame to limit size)
                if done:
                    # Episode metrics are now handled in the per-environment loop above
                    # This block only handles env 0's trajectory/visualization data
                    
                    # Reset env 0 legacy accumulators for next episode
                    self.current_episode_reward_raw = 0.0
                    self.current_episode_reward_shaped = 0.0
                    self.current_episode_cost = 0.0
                    self.current_episode_near_misses = 0
                    self.current_episode_danger_time = 0
                    self.current_episode_length = 0
                    
                    # Sample frames (e.g., every 5th frame) to avoid huge trajectories
                    sampled_frames = self.current_episode_frames[::5] if len(self.current_episode_frames) > 100 else self.current_episode_frames
                    self.trajectory.extend(sampled_frames)
                    self.current_episode_frames = []
                    
                    # Limit total trajectory size to avoid memory issues
                    max_trajectory_size = 10000
                    if len(self.trajectory) > max_trajectory_size:
                        # Keep most recent frames
                        self.trajectory = self.trajectory[-max_trajectory_size:]
                    
                    print(f"[Callback] Episode {self.episode_count} ended. Total trajectory frames: {len(self.trajectory)}")
            elif done and self.run_without_simulation:
                # In fast mode we do not record frames, but reset legacy accumulators.
                self.current_episode_reward_raw = 0.0
                self.current_episode_reward_shaped = 0.0
                self.current_episode_cost = 0.0
                self.current_episode_near_misses = 0
                self.current_episode_danger_time = 0
                self.current_episode_length = 0
                self.current_episode_frames = []
        except Exception as e:
            print(f"[Callback] Error capturing trajectory: {e}")
        
        # Broadcast progress every save_interval steps
        if self.n_calls % self.save_interval == 0:
            print(f"[Callback] Step {self.n_calls}/{self.total_steps} for experiment {self.experiment_id}")
            
            # Calculate comprehensive metrics
            metrics = self._calculate_metrics()

            # Save metrics to database
            if self.db_session_factory:
                self._save_metrics_to_db(metrics)

            # Broadcast via WebSocket from worker thread
            self._broadcast_metrics(metrics)

        # Save snapshot at intervals
        if self.snapshot_freq and self.n_calls % self.snapshot_freq == 0:
            if self.model_save_func:
                try:
                    # Calculate metrics for snapshot metadata
                    current_metrics = self._calculate_metrics()
                    self.model_save_func(self.n_calls, metrics=current_metrics)
                    print(f"[Callback] Triggered snapshot save at step {self.n_calls}")
                except Exception as e:
                    print(f"[Callback] Snapshot save error: {e}")

        return True  # Continue training

    def _calculate_metrics(self) -> dict:
        """Calculate comprehensive metrics for broadcasting."""
        metrics = {
            "step": self.n_calls,
            "episode_count": self.episode_count,
        }
        
        # Use a window for recent metrics (last 50 episodes or all if less)
        window_size = 50
        
        # A) Performance Metrics
        if len(self.episode_rewards_raw) > 0:
            recent_rewards_raw = self.episode_rewards_raw[-window_size:]
            recent_rewards_shaped = self.episode_rewards_shaped[-window_size:]
            recent_lengths = self.episode_lengths[-window_size:]
            recent_successes = self.episode_successes[-window_size:]
            recent_crashes = self.episode_crashes[-window_size:]
            recent_timeouts = self.episode_timeouts[-window_size:]
            
            metrics["reward_raw_mean"] = float(np.mean(recent_rewards_raw))
            metrics["reward_raw_std"] = float(np.std(recent_rewards_raw))
            metrics["reward_shaped_mean"] = float(np.mean(recent_rewards_shaped))
            # Raw (unrounded) values for statistical analysis
            metrics["success_rate_raw"] = float(np.mean(recent_successes)) * 100.0
            metrics["crash_rate_raw"] = float(np.mean(recent_crashes)) * 100.0
            metrics["timeout_rate_raw"] = float(np.mean(recent_timeouts)) * 100.0
            # Rounded to 0.5% precision for UI display
            metrics["success_rate"] = round(metrics["success_rate_raw"] * 2) / 2
            metrics["crash_rate"] = round(metrics["crash_rate_raw"] * 2) / 2
            metrics["timeout_rate"] = round(metrics["timeout_rate_raw"] * 2) / 2
            metrics["mean_ep_length"] = float(np.mean(recent_lengths))
            metrics["ep_length_std"] = float(np.std(recent_lengths))
        else:
            metrics.update({
                "reward_raw_mean": 0.0,
                "reward_raw_std": 0.0,
                "reward_shaped_mean": 0.0,
                "success_rate": 0.0,
                "crash_rate": 0.0,
                "timeout_rate": 0.0,
                "mean_ep_length": 0.0,
                "ep_length_std": 0.0
            })
        
        # B) Safety/Risk Metrics
        self._add_safety_metrics(metrics, window_size)
        
        # C) Lagrangian Dynamics
        if self.lagrange_state:
            metrics["lambda"] = float(self.lagrange_state.lam)
        else:
            metrics["lambda"] = 0.0
        
        # Add warmup metrics if available
        self._add_warmup_metrics(metrics)
        
        if self.safety_config:
            metrics["epsilon"] = float(self.safety_config.get('risk_budget', 0.3))
        else:
            metrics["epsilon"] = 0.0
        
        # D) Also include SB3's buffer metrics for compatibility
        ep_info_buffer = self.model.ep_info_buffer
        if len(ep_info_buffer) > 0:
            metrics["mean_reward"] = float(np.mean([ep_info["r"] for ep_info in ep_info_buffer]))
        else:
            metrics["mean_reward"] = metrics.get("reward_raw_mean", 0.0)
        
        # Best model checkpoint tracking
        self._check_best_model(metrics)

        # E) Add algorithm-specific metrics if available
        self._add_algorithm_metrics(metrics)
        
        # F) Residual Dynamics Logging
        self._add_residual_metrics(metrics)
        
        return metrics

    def _add_safety_metrics(self, metrics: dict, window_size: int):
        """Add safety-related metrics to the metrics dict."""
        if len(self.episode_costs) > 0:
            recent_costs = self.episode_costs[-window_size:]
            recent_near_misses = self.episode_near_misses[-window_size:]
            recent_danger_time = self.episode_danger_time[-window_size:]
            
            avg_cost = float(np.mean(recent_costs))
            metrics["cost_mean"] = avg_cost
            metrics["cost_std"] = float(np.std(recent_costs))
            metrics["near_miss_mean"] = float(np.mean(recent_near_misses))
            metrics["danger_time_mean"] = float(np.mean(recent_danger_time))
            
            # Cache for next interval (prevents spiky zeros in graphs)
            self.last_avg_cost = avg_cost
            
            # Normalize cumulative costs by episode length when comparing to per-step epsilon.
            if self.safety_config:
                epsilon = self.safety_config.get('risk_budget', 0.3)
                
                # Get corresponding episode lengths for normalization
                recent_lengths = self.episode_lengths[-window_size:]
                
                # Ensure we have matching lengths (should always be true if logged correctly)
                if len(recent_costs) == len(recent_lengths):
                    # Calculate violations: cumulative_cost > epsilon
                    violations = [cost > epsilon for cost in recent_costs]
                    violation_count = sum(violations)
                    violation_rate_value = float(np.mean(violations)) * 100.0
                    metrics["violation_rate"] = violation_rate_value
                    
                    # Log violation rate calculation
                    if self.n_calls % self.save_interval == 0:
                        print(f"[VIOLATION_CALC] {violation_count}/{len(recent_costs)} episodes violated "
                              f"(eps={epsilon:.5f}), rate={violation_rate_value:.1f}%")
                else:
                    # Fallback if lengths don't match (shouldn't happen)
                    print(f"[VIOLATION_CALC] Warning: Cost/length mismatch ({len(recent_costs)} vs {len(recent_lengths)})")
                    metrics["violation_rate"] = 0.0
            else:
                metrics["violation_rate"] = 0.0
        else:
            # Use last known value instead of 0.0 to prevent spiky graphs
            # when save_interval is small and no episodes finish in this interval
            metrics["cost_mean"] = self.last_avg_cost
            metrics["cost_std"] = 0.0
            metrics["near_miss_mean"] = 0.0
            metrics["danger_time_mean"] = 0.0
            metrics["violation_rate"] = 0.0
        
    def _add_warmup_metrics(self, metrics: dict):
        """Add warmup-related metrics if available."""
        if self.lagrange_callback:
            # Check if warmup is active
            if (self.lagrange_callback.target_lambda is not None and 
                self.lagrange_callback.warmup_episodes > 0 and 
                self.lagrange_callback.episode_count < self.lagrange_callback.warmup_episodes):
                
                # Warmup is active - add target and progress to metrics
                metrics["warmup_target_lambda"] = float(self.lagrange_callback.target_lambda)
                metrics["warmup_progress"] = float(self.lagrange_callback.episode_count / self.lagrange_callback.warmup_episodes)
            else:
                # Warmup inactive or finished
                metrics["warmup_target_lambda"] = None
                metrics["warmup_progress"] = None
        else:
            metrics["warmup_target_lambda"] = None
            metrics["warmup_progress"] = None

    def _check_best_model(self, metrics: dict):
        """Check if current model is the best and save if so."""
        if len(self.episode_successes) > 0:
            current_success_rate = metrics.get("success_rate", 0.0)
            if current_success_rate > self.best_success_rate and self.model_save_func:
                self.best_success_rate = current_success_rate
                print(f"\n[BEST_MODEL] New high score! Success rate: {current_success_rate:.1f}% (was {self.best_success_rate:.1f}%)")
                # Save with _best suffix
                try:
                    model_path = f"models/experiment_{self.experiment_id}_final_model"
                    self.model.save(f"{model_path}_best")
                    print(f"[BEST_MODEL] Saved to {model_path}_best.zip")
                except Exception as e:
                    print(f"[BEST_MODEL] Failed to save: {e}")

    def _add_algorithm_metrics(self, metrics: dict):
        """Add algorithm-specific metrics from captured training logs."""
        try:
            if len(self.last_train_metrics) > 0:
                # Debug once to see what's available
                if not hasattr(self, '_logged_train_keys'):
                    self._logged_train_keys = True
                    print(f"[Callback] First captured training metrics at step {self.n_calls}")
                    print(f"[Callback] Available keys: {list(self.last_train_metrics.keys())}")
                
                # On-policy algorithms
                # PPO: train/policy_gradient_loss, train/value_loss, train/entropy_loss
                # A2C: train/policy_loss, train/value_loss, train/entropy_loss
                if "train/policy_gradient_loss" in self.last_train_metrics:
                    metrics["policy_loss"] = float(self.last_train_metrics["train/policy_gradient_loss"])
                elif "train/policy_loss" in self.last_train_metrics:
                    metrics["policy_loss"] = float(self.last_train_metrics["train/policy_loss"])
                
                if "train/value_loss" in self.last_train_metrics:
                    metrics["value_loss"] = float(self.last_train_metrics["train/value_loss"])
                
                # Off-policy algorithms
                # SAC: train/actor_loss, train/critic_loss, train/ent_coef
                # TD3: train/actor_loss, train/critic_loss
                # DDPG: train/actor_loss, train/critic_loss (inherits from TD3)
                if "train/actor_loss" in self.last_train_metrics:
                    metrics["actor_loss"] = float(self.last_train_metrics["train/actor_loss"])
                if "train/critic_loss" in self.last_train_metrics:
                    metrics["critic_loss"] = float(self.last_train_metrics["train/critic_loss"])
                
                # Entropy loss (PPO, A2C)
                if "train/entropy_loss" in self.last_train_metrics:
                    metrics["entropy"] = float(self.last_train_metrics["train/entropy_loss"])
                
                # Learning rate (most algorithms have this)
                if "train/learning_rate" in self.last_train_metrics:
                    metrics["learning_rate"] = float(self.last_train_metrics["train/learning_rate"])
        except Exception as e:
            # Silently fail
            pass

    def _add_residual_metrics(self, metrics: dict):
        """Add residual learning metrics if available."""
        if len(self.residual_magnitudes) > 0:
            mean_res_mag = np.mean(self.residual_magnitudes)
            mean_base_mag = np.mean(self.base_magnitudes)
            
            # Add to metrics dict for websocket broadcast
            metrics["residual_correction_magnitude"] = float(mean_res_mag)
            metrics["residual_base_magnitude"] = float(mean_base_mag)
            
            # Calculate intervention ratio (how much we are changing the base)
            if mean_base_mag > 1e-6:
                ratio = mean_res_mag / mean_base_mag
                metrics["residual_intervention_ratio"] = float(ratio)
            else:
                metrics["residual_intervention_ratio"] = 0.0
            
            # NEW: Residual Contribution Ratio = ||a_res|| / (||a_base|| + ||a_res|| + eps)
            # Calculate per-step then average for accuracy
            if len(self.base_magnitudes) == len(self.residual_magnitudes):
                contribution_ratios = []
                for res_mag, base_mag in zip(self.residual_magnitudes, self.base_magnitudes):
                    ratio = res_mag / (base_mag + res_mag + 1e-8)
                    contribution_ratios.append(ratio)
                metrics["residual_contribution_mean"] = float(np.mean(contribution_ratios))
            else:
                # Fallback: use aggregated magnitudes
                metrics["residual_contribution_mean"] = mean_res_mag / (mean_base_mag + mean_res_mag + 1e-8)
            
            # TensorBoard logging
            self.logger.record("residual/correction_magnitude", mean_res_mag)
            self.logger.record("residual/base_action_magnitude", mean_base_mag)
            self.logger.record("residual/intervention_ratio", metrics["residual_intervention_ratio"])
            self.logger.record("residual/contribution_ratio", metrics["residual_contribution_mean"])
            
            # Clear buffers
            self.residual_magnitudes = []
            self.base_magnitudes = []
        else:
            # Set defaults if no residual data
            metrics["residual_correction_magnitude"] = 0.0
            metrics["residual_base_magnitude"] = 0.0
            metrics["residual_intervention_ratio"] = 0.0
            metrics["residual_contribution_mean"] = 0.0
        
        # NEW: Policy Conflict (Cosine Similarity) metrics
        if len(self.cosine_similarities) > 0:
            metrics["conflict_mean"] = float(np.mean(self.cosine_similarities))
            metrics["conflict_std"] = float(np.std(self.cosine_similarities))
            metrics["conflict_negative_rate"] = float(np.mean([cs < 0 for cs in self.cosine_similarities])) * 100.0

            # TensorBoard logging
            self.logger.record("residual/policy_conflict_mean", metrics["conflict_mean"])
            self.logger.record("residual/policy_conflict_std", metrics["conflict_std"])

            # Clear buffer
            self.cosine_similarities = []
        else:
            metrics["conflict_mean"] = 0.0
            metrics["conflict_std"] = 0.0
            metrics["conflict_negative_rate"] = 0.0
        
        # NEW: Intervention Rate (% of steps where residual > threshold)
        if len(self.intervention_flags) > 0:
            metrics["intervention_rate"] = float(np.mean(self.intervention_flags)) * 100.0
            
            # TensorBoard logging
            self.logger.record("residual/intervention_rate", metrics["intervention_rate"])
            
            # Clear buffer
            self.intervention_flags = []
        else:
            metrics["intervention_rate"] = 0.0
        
        # NEW: Adaptive K-Factor metrics
        if len(self.effective_k_values) > 0:
            metrics["effective_k_mean"] = float(np.mean(self.effective_k_values))
            metrics["effective_k_min"] = float(np.min(self.effective_k_values))
            metrics["effective_k_max"] = float(np.max(self.effective_k_values))
            
            # TensorBoard logging
            self.logger.record("residual/effective_k_mean", metrics["effective_k_mean"])
            self.logger.record("residual/effective_k_range", metrics["effective_k_max"] - metrics["effective_k_min"])
            
            # Clear buffer
            self.effective_k_values = []
        else:
            metrics["effective_k_mean"] = 0.0
            metrics["effective_k_min"] = 0.0
            metrics["effective_k_max"] = 0.0
        
        # NEW: CARS K Components metrics (for separate charts)
        if len(self.k_conf_values) > 0:
            metrics["k_conf_mean"] = float(np.mean(self.k_conf_values))
            self.logger.record("cars/k_conf", metrics["k_conf_mean"])
            self.k_conf_values = []
        else:
            metrics["k_conf_mean"] = 1.0  # Default when CARS disabled
        
        if len(self.k_risk_values) > 0:
            metrics["k_risk_mean"] = float(np.mean(self.k_risk_values))
            self.logger.record("cars/k_risk", metrics["k_risk_mean"])

            # Risk activation rate (% of steps where K_risk > 1.0)
            metrics["risk_activation_rate"] = float(np.mean([kr > 1.0 for kr in self.k_risk_values])) * 100.0

            self.k_risk_values = []
        else:
            metrics["k_risk_mean"] = 1.0  # Default when no λ
            metrics["risk_activation_rate"] = 0.0

        # Action clipping rate (% of steps where action saturated at ±1)
        if len(self.action_clipping_flags) > 0:
            metrics["action_clipping_rate"] = float(np.mean(self.action_clipping_flags)) * 100.0
            self.action_clipping_flags = []
        else:
            metrics["action_clipping_rate"] = 0.0

    def _save_metrics_to_db(self, metrics: dict):
        """Save metrics to database."""
        try:
            from .. import crud
            import datetime
            
            db = self.db_session_factory()
            try:
                crud.create_metric(
                    db,
                    experiment_id=self.experiment_id,
                    step=self.n_calls,
                    values=metrics,
                    logs=None  # No logs needed
                )
                print(f"[Callback] Saved metrics to database for step {self.n_calls}")
            finally:
                db.close()
        except Exception as e:
            print(f"[Callback] Database save error: {e}")

    def _broadcast_metrics(self, metrics: dict):
        """Broadcast metrics via WebSocket."""
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.broadcast_func(
                    self.experiment_id,
                    {
                        "type": "progress",
                        "data": {
                            "experiment_id": self.experiment_id,
                            "step": self.n_calls,
                            "total_steps": self.total_steps,
                            "metrics": metrics,
                            "status": "In Progress",
                            "timestamp": time.time(),
                        },
                    },
                ),
                self.loop
            )
            # Wait briefly to ensure broadcast completes
            future.result(timeout=1.0)
        except Exception as e:
            print(f"[Callback] Broadcast error: {e}")
