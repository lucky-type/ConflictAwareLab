"""Simulation-related functionality for background experiments."""

import asyncio
import time
import datetime
import threading
from typing import Dict, Any
import numpy as np
import pybullet as p
from sqlalchemy.orm import Session

from stable_baselines3 import PPO, SAC


class SimulationMixin:
    """Mixin class for simulation-related experiment functionality."""
    
    async def _send_initial_simulation_frame(
        self,
        experiment_id: int,
        env,
        environment,
        obs,
        broadcast_func,
    ):
        """Send initial simulation frame with environment setup."""
        try:
            # Get environment dimensions
            env_length = environment.length if environment else 100.0
            env_width = environment.width if environment else 20.0
            env_height = getattr(environment, 'height', 20.0)
            
            # Default values
            x_pos, y_pos, z_pos = 0.0, env_width / 2.0, env_length / 2.0
            velocity = [0.0, 0.0, 0.0]
            orientation = [0.0, 0.0, 0.0]
            
            # Get initial drone position from PyBullet
            try:
                # Unwrap to get to the base DroneEnv
                base_env = env
                while hasattr(base_env, 'env'):
                    base_env = base_env.env
                
                if hasattr(base_env, 'drone_id'):
                    physics_client = getattr(base_env, 'physics_client', 0)
                    pos, quat = p.getBasePositionAndOrientation(base_env.drone_id, physicsClientId=physics_client)
                    x_pos, y_pos, z_pos = float(pos[0]), float(pos[1]), float(pos[2])
                    
                    # Convert quaternion to euler angles
                    qx, qy, qz, qw = quat
                    pitch = np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
                    roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
                    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                    orientation = [float(roll), float(pitch), float(yaw)]
            except Exception as e:
                print(f"[Simulation] Could not get initial position from PyBullet: {e}")
            
            # Extract goal position from observation
            obs_array = np.array(obs) if not isinstance(obs, np.ndarray) else obs
            obs_len = len(obs_array)
            
            # Goal direction is in observation [0-2], distance in [3]
            goal_dir_x = float(obs_array[0]) if obs_len > 0 else 0.0
            goal_dir_y = float(obs_array[1]) if obs_len > 1 else 0.0
            goal_dir_z = float(obs_array[2]) if obs_len > 2 else 0.0
            goal_distance_norm = float(obs_array[3]) if obs_len > 3 else 0.5
            
            # Use actual goal position from environment
            base_env = env
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            
            if hasattr(base_env, 'goal_position') and base_env.goal_position is not None:
                goal_x = float(base_env.goal_position[0])
                goal_y = float(base_env.goal_position[1])
                goal_z = float(base_env.goal_position[2])
            else:
                # Fallback: reconstruct from observation
                max_possible_distance = np.linalg.norm([env_length, env_width, env_height])
                goal_distance = goal_distance_norm * max_possible_distance
                goal_x = x_pos + goal_dir_x * goal_distance
                goal_y = y_pos + goal_dir_y * goal_distance
                goal_z = z_pos + goal_dir_z * goal_distance
            
            # Get swarm drone positions for obstacles
            obstacles_data = []
            try:
                base_env = env
                while hasattr(base_env, 'env'):
                    base_env = base_env.env
                
                if hasattr(base_env, 'swarm_drones'):
                    physics_client = getattr(base_env, 'physics_client', 0)
                    for drone in base_env.swarm_drones:
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
                    print(f"[Simulation] Initial frame: captured {len(obstacles_data)} swarm drones")
            except Exception as e:
                print(f"[Simulation] Could not get initial swarm positions: {e}")
            
            # Initial frame data
            initial_frame = {
                "experiment_id": experiment_id,
                "frame": 0,
                "timestamp": time.time(),
                "agent_position": {
                    "x": x_pos,
                    "y": y_pos,
                    "z": z_pos,
                },
                "agent_velocity": {
                    "x": velocity[0],
                    "y": velocity[1],
                    "z": velocity[2],
                },
                "agent_orientation": {
                    "roll": orientation[0],
                    "pitch": orientation[1],
                    "yaw": orientation[2],
                },
                "target_position": {
                    "x": goal_x,
                    "y": goal_y,
                    "z": goal_z,
                },
                "obstacles": obstacles_data,
                "lidar_readings": obs_array[7:21].tolist() if obs_len > 20 else [],
                "reward": 0.0,
                "done": False,
                "success": False,
                "crashed": False,
                "crash_type": "none",
            }
            
            # Broadcast initial frame
            await broadcast_func(
                experiment_id,
                {
                    "type": "simulation_frame",
                    "data": initial_frame,
                },
            )
            
        except Exception as e:
            print(f"[Simulation] Error sending initial frame: {e}")

    async def start_simulation(
        self,
        experiment: "models.Experiment",
        db: Session,
        broadcast_func,
    ):
        """Start a simulation experiment in the background."""
        experiment_id = experiment.id

        if experiment_id in self.running_tasks:
            raise RuntimeError(f"Experiment {experiment_id} is already running")

        # Create cancellation event for this experiment
        cancel_event = threading.Event()
        self.cancel_events[experiment_id] = cancel_event

        task = asyncio.create_task(
            self._run_simulation_wrapper(experiment, db, broadcast_func, cancel_event)
        )
        self.running_tasks[experiment_id] = task
        self.paused_experiments[experiment_id] = False

        from .. import crud

        crud.update_experiment_status(db, experiment_id, "In Progress")

    async def _run_simulation_wrapper(
        self,
        experiment: "models.Experiment",
        db: Session,
        broadcast_func,
        cancel_event: threading.Event,
    ):
        """Wrapper that runs simulation loop."""
        experiment_id = experiment.id
        
        try:
            await self._run_simulation(experiment, db, broadcast_func, cancel_event)
        except Exception as e:
            print(f"\n!!! Simulation error for experiment {experiment_id} !!!")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            from .. import crud
            crud.update_experiment_status(db, experiment_id, "Cancelled")
        finally:
            # Cleanup
            if experiment_id in self.running_tasks:
                del self.running_tasks[experiment_id]
            if experiment_id in self.paused_experiments:
                del self.paused_experiments[experiment_id]
            if experiment_id in self.cancel_events:
                del self.cancel_events[experiment_id]
            if experiment_id in self.envs_cache:
                try:
                    self.envs_cache[experiment_id].close()
                except:
                    pass
                del self.envs_cache[experiment_id]

    async def _run_simulation(
        self,
        experiment: "models.Experiment",
        db: Session,
        broadcast_func,
        cancel_event: threading.Event,
    ):
        """Internal simulation loop with trajectory recording."""
        from .. import crud, models as db_models

        experiment_id = experiment.id
        print(f"\n=== Starting simulation for experiment {experiment_id} ===")

        try:
            # Load trained model from snapshot
            if not experiment.model_snapshot_id:
                raise ValueError("Simulation requires a model_snapshot_id")

            snapshot = crud.get_entity(
                db, db_models.ModelSnapshot, experiment.model_snapshot_id
            )
            if not snapshot:
                raise ValueError("Model snapshot not found")

            environment = crud.get_entity(
                db, db_models.Environment, experiment.env_id
            )
            
            agent = crud.get_entity(
                db, db_models.Agent, experiment.agent_id
            )
            
            reward_function = crud.get_entity(
                db, db_models.RewardFunction, experiment.reward_id
            )
            
            print(f"[Simulation] Using snapshot {snapshot.id} from {snapshot.file_path}")
            print(f"[Simulation] Environment: {environment.name if environment else 'Unknown'}")
            print(f"[Simulation] Agent: {agent.name if agent else 'Unknown'}")
            print(f"[Simulation] Reward function: {reward_function.name if reward_function else 'Unknown'}")

            # Create DroneSwarmEnv (same as training does)
            from ..drone_env import DroneSwarmEnv
            
            # Get swarm configuration from environment obstacles
            obstacles = environment.obstacles if environment.obstacles else []
            num_swarm_drones = len(obstacles)
            
            print(f"[Simulation] Creating environment with {num_swarm_drones} obstacles")
            
            # Prepare reward function parameters
            reward_params = {
                'w_progress': reward_function.w_progress,
                'w_time': reward_function.w_time,
                'w_jerk': reward_function.w_jerk,
                'success_reward': reward_function.success_reward,
                'crash_reward': reward_function.crash_reward
            }
            
            # Create DroneSwarmEnv with unique physics client for simulation
            env = DroneSwarmEnv(
                corridor_length=environment.length,
                corridor_width=environment.width,
                corridor_height=environment.height,
                buffer_start=environment.buffer_start,
                buffer_end=environment.buffer_end,
                agent_diameter=agent.agent_diameter,
                agent_max_speed=agent.agent_max_speed,
                agent_max_acceleration=getattr(agent, 'agent_max_acceleration', 5.0),
                num_neighbors=5,
                num_swarm_drones=num_swarm_drones,
                swarm_obstacles=obstacles,
                reward_params=reward_params,
                lidar_range=agent.lidar_range,
                lidar_rays=agent.lidar_rays,
                target_diameter=environment.target_diameter,
                max_ep_length=experiment.max_ep_length,
                kinematic_type=agent.kinematic_type if hasattr(agent, 'kinematic_type') else "holonomic",
                # Predicted mode for reproducible scenarios
                predicted_seed=int(environment.created_at.timestamp()) if getattr(environment, 'is_predicted', False) and environment.created_at else None
            )
            
            if getattr(environment, 'is_predicted', False) and environment.created_at:
                print(f"[Simulation] Predicted mode ENABLED: seed={int(environment.created_at.timestamp())}")
            
            print(f"[Simulation] Created DroneSwarmEnv with observation space: {env.observation_space}")
            print(f"[Simulation] Kinematic type: {agent.kinematic_type if hasattr(agent, 'kinematic_type') else 'holonomic'}")
            print(f"[Simulation] Lidar: range={agent.lidar_range}m, rays={agent.lidar_rays}")
            print(f"[Simulation] Action space: {env.action_space}")
            print(f"[Simulation] PyBullet client ID: {env.physics_client}")
            
            # Don't cache simulation environments - always create fresh ones
            # This prevents conflicts with training environments
            # self.envs_cache[experiment_id] = env

            # Get the agent configuration from the snapshot metadata
            # This agent was used during training, so we recreate env with its exact config
            snapshot_agent_id = snapshot.metrics_at_save.get('agent_id') if snapshot.metrics_at_save else None
            
            if snapshot_agent_id:
                print(f"[Simulation] Loading agent config from snapshot (agent_id={snapshot_agent_id})")
                trained_agent = crud.get_entity(db, db_models.Agent, snapshot_agent_id)
                
                if trained_agent:
                    print(f"[Simulation] Found trained agent: {trained_agent.name}")
                    print(f"[Simulation] Config: kinematic={trained_agent.kinematic_type if hasattr(trained_agent, 'kinematic_type') else 'holonomic'}, lidar_rays={trained_agent.lidar_rays}")
                    
                    # Check if we need to recreate the environment
                    if (trained_agent.lidar_rays != agent.lidar_rays or 
                        getattr(trained_agent, 'kinematic_type', 'holonomic') != getattr(agent, 'kinematic_type', 'holonomic')):
                        
                        print(f"[Simulation] Agent config changed since training, recreating environment...")
                        env.close()
                        
                        env = DroneSwarmEnv(
                            corridor_length=environment.length,
                            corridor_width=environment.width,
                            corridor_height=environment.height,
                            buffer_start=environment.buffer_start,
                            buffer_end=environment.buffer_end,
                            agent_diameter=trained_agent.agent_diameter,
                            agent_max_speed=trained_agent.agent_max_speed,
                            agent_max_acceleration=getattr(trained_agent, 'agent_max_acceleration', 5.0),
                            num_neighbors=5,
                            num_swarm_drones=num_swarm_drones,
                            swarm_obstacles=obstacles,
                            reward_params=reward_params,
                            lidar_range=trained_agent.lidar_range,
                            lidar_rays=trained_agent.lidar_rays,
                            target_diameter=environment.target_diameter,
                            max_ep_length=experiment.max_ep_length,
                            kinematic_type=getattr(trained_agent, 'kinematic_type', 'holonomic'),
                            # Predicted mode for reproducible scenarios
                            predicted_seed=int(environment.created_at.timestamp()) if getattr(environment, 'is_predicted', False) and environment.created_at else None
                        )
                        
                        print(f"[Simulation] Recreated env with trained agent config: {env.observation_space}")
                    else:
                        print(f"[Simulation] Agent config matches, using current environment")
                else:
                    print(f"[Simulation] Warning: Agent {snapshot_agent_id} not found, using current agent")
            else:
                print(f"[Simulation] Warning: No agent_id in snapshot metadata, using current agent config")

            # Wrap environment with AppendLambdaWrapper so observation space matches training.
            # Load lambda value from snapshot metadata (value at time of model saving)
            from ..lagrangian_safety import LagrangeState, AppendLambdaWrapper
            
            stored_lambda = 0.0
            if snapshot.metrics_at_save and 'lambda' in snapshot.metrics_at_save:
                stored_lambda = float(snapshot.metrics_at_save['lambda'])
                print(f"[Simulation] Using λ={stored_lambda:.4f} from model snapshot")
            else:
                print(f"[Simulation] No λ in snapshot metadata, using λ=0.0")
            
            lagrange_state = LagrangeState(lam=stored_lambda)
            env = AppendLambdaWrapper(env, lagrange_state)
            print(f"[Simulation] Wrapped environment with AppendLambdaWrapper (λ={stored_lambda:.4f})")
            print(f"[Simulation] Final observation space: {env.observation_space}")

            # Load model with correct environment
            try:
                model = PPO.load(snapshot.file_path, env=env)
                model_algo = "PPO"
                print(f"[Simulation] Loaded {model_algo} model")
            except Exception as e:
                try:
                    model = SAC.load(snapshot.file_path, env=env)
                    model_algo = "SAC"
                    print(f"[Simulation] Loaded {model_algo} model")
                except Exception as e2:
                    print(f"[Simulation] Failed to load model: PPO error: {e}, SAC error: {e2}")
                    raise ValueError("Failed to load model snapshot")

            # Reset environment to get initial state
            obs, _ = env.reset()
            
            # Send initial frame before starting simulation
            await self._send_initial_simulation_frame(
                experiment_id, env, environment, obs, broadcast_func
            )
            
            print(f"[Simulation] Sent initial frame, starting simulation...")

            # Trajectory storage for replay
            trajectory = []
            
            # Run simulation episodes
            episode = 0
            max_episodes = 1  # Single episode for now
            frame_counter = 0
            target_fps = 30  # Target 30 FPS for faster execution
            frame_duration = 1.0 / target_fps

            while episode < max_episodes:
                # Check if cancelled first (before pause check)
                if cancel_event.is_set():
                    print(f"[Simulation] Cancelled at frame {frame_counter}")
                    break
                    
                # Check if paused
                while self.paused_experiments.get(experiment_id, False):
                    # Also check cancel during pause
                    if cancel_event.is_set():
                        print(f"[Simulation] Cancelled while paused")
                        break
                    await asyncio.sleep(0.5)

                # Check if cancelled again after pause
                if cancel_event.is_set():
                    print(f"[Simulation] Cancelled at frame {frame_counter}")
                    break

                # Reset for new episode (obs already reset before loop if first episode)
                if episode > 0:
                    obs, _ = env.reset()
                done = False
                truncated = False
                step = 0
                total_reward = 0
                episode_start_time = time.time()

                print(f"[Simulation] Starting episode {episode + 1}/{max_episodes} (max_ep_length={experiment.max_ep_length})")

                while not (done or truncated):
                    # Check cancel first
                    if cancel_event.is_set():
                        print(f"[Simulation] Cancelled during episode at step {step}")
                        done = True
                        break
                        
                    # Check pause/cancel in the loop
                    while self.paused_experiments.get(experiment_id, False):
                        # Check cancel during pause
                        if cancel_event.is_set():
                            print(f"[Simulation] Cancelled while paused at step {step}")
                            done = True
                            break
                        await asyncio.sleep(0.5)
                    
                    # Check again after pause
                    if cancel_event.is_set():
                        print(f"[Simulation] Cancelled during episode at step {step}")
                        done = True
                        break

                    frame_start = time.time()
                    
                    # Predict action using trained model
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(action)
                    
                    # Simulation ends on both success (reaching target) and crashes
                    total_reward += reward
                    step += 1
                    frame_counter += 1

                    # Get actual drone position from PyBullet environment
                    env_length = environment.length if environment else 100.0
                    env_width = environment.width if environment else 20.0
                    env_height = getattr(environment, 'height', 20.0)
                    
                    # Default values
                    x_pos, y_pos, z_pos = 0.0, env_width / 2.0, env_length / 2.0
                    velocity = [0.0, 0.0, 0.0]
                    orientation = [0.0, 0.0, 0.0]
                    
                    try:
                        # Get drone position directly from PyBullet
                        # Unwrap to get to the base DroneEnv
                        base_env = env
                        while hasattr(base_env, 'env'):
                            base_env = base_env.env
                        
                        if hasattr(base_env, 'drone_id'):
                            physics_client = getattr(base_env, 'physics_client', 0)
                            pos, quat = p.getBasePositionAndOrientation(base_env.drone_id, physicsClientId=physics_client)
                            x_pos, y_pos, z_pos = float(pos[0]), float(pos[1]), float(pos[2])
                            
                            # Get velocity
                            vel, ang_vel = p.getBaseVelocity(base_env.drone_id, physicsClientId=physics_client)
                            velocity = [float(vel[0]), float(vel[1]), float(vel[2])]
                            
                            # Convert quaternion to euler angles
                            qx, qy, qz, qw = quat
                            pitch = np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
                            roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
                            yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                            orientation = [float(roll), float(pitch), float(yaw)]
                    except Exception as e:
                        if frame_counter % 100 == 0:
                            print(f"[Simulation] Could not get position from PyBullet: {e}")
                    
                    # Extract goal position from observation
                    obs_array = np.array(obs) if not isinstance(obs, np.ndarray) else obs
                    obs_len = len(obs_array)
                    
                    # Goal direction is in observation [0-2], distance in [3]
                    goal_dir_x = float(obs_array[0]) if obs_len > 0 else 0.0
                    goal_dir_y = float(obs_array[1]) if obs_len > 1 else 0.0
                    goal_dir_z = float(obs_array[2]) if obs_len > 2 else 0.0
                    goal_distance_norm = float(obs_array[3]) if obs_len > 3 else 0.5
                    
                    # Use actual goal position from environment (not reconstructed from observation)
                    # Unwrap to get to the base DroneEnv
                    base_env = env
                    while hasattr(base_env, 'env'):
                        base_env = base_env.env
                    
                    if hasattr(base_env, 'goal_position') and base_env.goal_position is not None:
                        goal_x = float(base_env.goal_position[0])
                        goal_y = float(base_env.goal_position[1])
                        goal_z = float(base_env.goal_position[2])
                        if frame_counter % 100 == 0:
                            print(f"[Simulation Frame {frame_counter}] Using env.goal_position: [{goal_x:.2f}, {goal_y:.2f}, {goal_z:.2f}]")
                    else:
                        # Fallback: reconstruct from observation (less accurate)
                        max_possible_distance = np.linalg.norm([env_length, env_width, env_height])
                        goal_distance = goal_distance_norm * max_possible_distance
                        goal_x = x_pos + goal_dir_x * goal_distance
                        goal_y = y_pos + goal_dir_y * goal_distance
                        goal_z = z_pos + goal_dir_z * goal_distance
                        if frame_counter % 100 == 0:
                            print(f"[Simulation Frame {frame_counter}] WARNING: Using reconstructed goal: [{goal_x:.2f}, {goal_y:.2f}, {goal_z:.2f}]")

                    # Frame data for broadcast
                    frame_data = {
                        "experiment_id": experiment_id,
                        "frame": frame_counter,
                        "timestamp": time.time(),
                        "agent_position": {
                            "x": x_pos,
                            "y": y_pos,
                            "z": z_pos,
                        },
                        "agent_velocity": {
                            "x": velocity[0],
                            "y": velocity[1],
                            "z": velocity[2],
                        },
                        "agent_orientation": {
                            "roll": orientation[0],
                            "pitch": orientation[1],
                            "yaw": orientation[2],
                        },
                        "target_position": {
                            "x": goal_x,
                            "y": goal_y,
                            "z": goal_z,
                        },
                        "obstacles": [],
                        "lidar_readings": obs_array[7:21].tolist() if obs_len > 20 else [],
                        "reward": float(reward),
                        "done": bool(done or truncated),
                        "success": bool(info.get('success', False)),
                        "crashed": bool(info.get('crashed', False)),
                        "crash_type": str(info.get('crash_type', 'none')),
                    }
                    
                    # Add swarm drone positions to obstacles
                    try:
                        # Unwrap to get to the base DroneEnv
                        base_env = env
                        while hasattr(base_env, 'env'):
                            base_env = base_env.env
                        
                        if hasattr(base_env, 'swarm_drones'):
                            physics_client = getattr(base_env, 'physics_client', 0)
                            obstacles_data = []
                            for drone in base_env.swarm_drones:
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
                            frame_data["obstacles"] = obstacles_data
                            if frame_counter == 1 and len(obstacles_data) > 0:
                                print(f"[Simulation] ✅ Successfully captured {len(obstacles_data)} swarm drones")
                    except Exception as e:
                        if frame_counter % 100 == 0:
                            print(f"[Simulation] Could not get swarm positions: {e}")

                    # Store in trajectory for replay
                    trajectory.append(frame_data)

                    # Broadcast simulation frame via WebSocket (every frame for real-time visualization)
                    await broadcast_func(
                        experiment_id,
                        {
                            "type": "simulation_frame",
                            "data": frame_data,
                        },
                    )

                    # Increment frame counter
                    frame_counter += 1

                    # Save metrics and partial trajectory to database periodically (every 20 frames)
                    if frame_counter % 20 == 0:
                        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                        log_message = f"[{timestamp}] Frame {frame_counter} - Position: ({x_pos:.1f}, {y_pos:.1f}, {z_pos:.1f}) - Reward: {total_reward:.2f}"
                        
                        try:
                            # Save metrics
                            crud.create_metric(
                                db,
                                experiment_id=experiment_id,
                                step=frame_counter,
                                values={
                                    "reward": float(reward),
                                    "total_reward": float(total_reward),
                                    "position_x": x_pos,
                                    "position_y": y_pos,
                                    "position_z": z_pos,
                                },
                                logs=log_message
                            )
                            
                            # Update trajectory in database periodically for live replay
                            experiment = crud.get_entity(db, db_models.Experiment, experiment_id)
                            if experiment:
                                experiment.trajectory_data = trajectory.copy()
                                experiment.current_step = frame_counter
                            
                            db.commit()
                        except Exception as e:
                            print(f"[Simulation] Failed to save data: {e}")

                    # Control frame rate - don't throttle too aggressively
                    elapsed = time.time() - frame_start
                    sleep_time = max(0, frame_duration - elapsed)
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)

                episode += 1
                print(f"[Simulation] Episode {episode} completed: {step} steps, reward={total_reward:.2f}")

            # Check final status
            if cancel_event.is_set():
                final_status = "Cancelled"
            else:
                final_status = "Completed"

            # Save trajectory to experiment for replay
            experiment = crud.get_entity(db, db_models.Experiment, experiment_id)
            if experiment:
                experiment.trajectory_data = trajectory
                experiment.current_step = frame_counter
                experiment.status = final_status
                db.commit()
                print(f"[Simulation] Saved {len(trajectory)} frames to trajectory")

            # Broadcast final status
            await broadcast_func(
                experiment_id,
                {
                    "type": "progress",
                    "data": {
                        "experiment_id": experiment_id,
                        "status": final_status,
                        "frames": frame_counter,
                        "timestamp": time.time(),
                    },
                },
            )

        except Exception as e:
            print(f"[Simulation] Error: {e}")
            import traceback
            traceback.print_exc()
            crud.update_experiment_status(db, experiment_id, "Cancelled")
            raise
        finally:
            # Always close the environment to release PyBullet connection
            try:
                if 'env' in locals():
                    env.close()
                    print(f"[Simulation] Closed PyBullet environment")
            except Exception as e:
                print(f"[Simulation] Error closing environment: {e}")
