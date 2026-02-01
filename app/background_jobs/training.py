"""Training-specific methods for the ExperimentRunner."""
from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from .callback import ExperimentCallback

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from .. import models


class TrainingMixin:
    """Mixin class containing training-specific methods."""

    async def start_training(
        self,
        experiment: models.Experiment,
        db: Session,
        broadcast_func,
    ):
        """Start a training experiment in the background."""
        experiment_id = experiment.id

        # Check if already running
        if experiment_id in self.running_tasks:
            raise RuntimeError(f"Experiment {experiment_id} is already running")

        # Check if we have capacity for a new job
        if not self.can_start_new_job():
            raise RuntimeError(f"Maximum concurrent jobs ({self.max_concurrent_jobs}) reached. Please wait for a job to complete.")

        # Create cancellation event for this experiment
        cancel_event = threading.Event()
        self.cancel_events[experiment_id] = cancel_event

        # Create and start the background task that runs training in executor
        task = asyncio.create_task(
            self._run_training_wrapper(experiment, db, broadcast_func, cancel_event)
        )
        self.running_tasks[experiment_id] = task
        self.paused_experiments[experiment_id] = False

        # Update experiment status
        from .. import crud

        crud.update_experiment_status(db, experiment_id, "In Progress")

    async def _run_training_wrapper(
        self,
        experiment: models.Experiment,
        db: Session,
        broadcast_func,
        cancel_event: threading.Event,
    ):
        """Wrapper that runs training in executor thread."""
        experiment_id = experiment.id
        
        try:
            # Run the entire training in a thread pool
            # Note: Don't pass db session to thread - it will create its own
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._run_training_sync,
                experiment.id,  # Pass ID instead of object
                broadcast_func,
                loop,
                cancel_event
            )
        except Exception as e:
            print(f"\n!!! Training error for experiment {experiment_id} !!!")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            from .. import crud
            crud.update_experiment_status(db, experiment_id, "Cancelled")
        finally:
            # Close the database session
            db.close()
            # Cleanup task tracking
            self.cleanup_experiment(experiment_id)

    def _run_training_sync(
        self,
        experiment_id: int,
        broadcast_func,
        loop,
        cancel_event: threading.Event,
    ):
        """Synchronous training function that runs in a thread."""
        from .. import crud, models as db_models
        from ..database import SessionLocal

        # Create a new database session for this thread
        db = SessionLocal()
        
        print(f"\n=== Starting training for experiment {experiment_id} ===")

        try:
            # Load the experiment and related entities
            experiment, agent, reward_function, environment = self._load_training_entities(
                db, experiment_id
            )

            # Set random seed for reproducibility if specified
            experiment_seed = getattr(experiment, 'seed', None)
            if experiment_seed is not None:
                self._set_random_seeds(experiment_seed)
                print(f"[Training] Random seed set to: {experiment_seed}")
            else:
                print(f"[Training] No seed specified, using random initialization")

            # Select algorithm class
            AlgorithmClass = self._get_algorithm_class(agent.type)

            # Create and configure environment
            env, lagrange_state, safety_config, training_mode = self._create_training_environment(
                db, experiment, agent, reward_function, environment
            )

            # Create or load model
            model = self._create_or_load_model(
                db, experiment, agent, AlgorithmClass, env, lagrange_state,
                training_mode, safety_config, experiment_seed
            )

            # Define snapshot save function
            save_snapshot = self._create_snapshot_saver(
                db, experiment_id, experiment, agent, reward_function, lagrange_state, 
                training_mode, model
            )

            # Create callback for metric broadcasting
            callback_list, experiment_callback = self._create_training_callback(
                experiment_id, broadcast_func, loop, experiment, environment,
                lagrange_state, safety_config, save_snapshot
            )

            # Execute training loop
            self._execute_training_loop(
                db, experiment, model, callback_list, cancel_event, broadcast_func, loop
            )

        except Exception as e:
            self._handle_training_error(db, experiment_id, e)
        finally:
            db.close()
            self.cleanup_experiment(experiment_id)

    def _load_training_entities(self, db, experiment_id):
        """Load experiment and related entities."""
        from .. import crud, models as db_models

        experiment = crud.get_entity(db, db_models.Experiment, experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Get related entities
        print(f"Loading agent with ID: {experiment.agent_id}")
        agent = crud.get_entity(db, db_models.Agent, experiment.agent_id)
        print(f"Agent loaded: {agent.name if agent else 'None'}")
        
        print(f"Loading reward function with ID: {experiment.reward_id}")
        reward_function = crud.get_entity(
            db, db_models.RewardFunction, experiment.reward_id
        )
        print(f"Reward function loaded: {reward_function.name if reward_function else 'None'}")
        
        print(f"Loading environment with ID: {experiment.env_id}")
        environment = crud.get_entity(
            db, db_models.Environment, experiment.env_id
        )
        print(f"Environment loaded: {environment.name if environment else 'None'}")

        if not agent or not reward_function or not environment:
            raise ValueError("Missing required experiment dependencies")

        return experiment, agent, reward_function, environment

    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility across all components."""
        import random
        import torch

        # Set Python random seed
        random.seed(seed)

        # Set NumPy random seed
        np.random.seed(seed)

        # Set PyTorch random seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # For additional reproducibility (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print(f"[Training] Seeds set - Python: {seed}, NumPy: {seed}, PyTorch: {seed}")

    def _get_algorithm_class(self, algorithm_type):
        """Get the appropriate algorithm class."""
        algorithm_class_map = {
            "PPO": PPO,
            "SAC": SAC,
            "TD3": TD3,
            "A2C": A2C,
            "DDPG": DDPG,
        }

        print(f"Algorithm type: {algorithm_type}")
        AlgorithmClass = algorithm_class_map.get(algorithm_type)
        if not AlgorithmClass:
            raise ValueError(f"Unsupported algorithm type: {algorithm_type}")

        return AlgorithmClass

    def _create_training_environment(self, db, experiment, agent, reward_function, environment):
        """Create and configure the training environment."""
        from ..drone_env import DroneSwarmEnv
        from ..lagrangian_safety import LagrangeState, AppendLambdaWrapper, LagrangianRewardVecWrapper
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor

        # Get swarm configuration from environment obstacles
        obstacles = environment.obstacles if environment.obstacles else []
        num_swarm_drones = len(obstacles)
        
        print(f"Environment '{environment.name}' has {num_swarm_drones} obstacles configured")
        print(f"Obstacles data: {obstacles}")
        
        # Prepare reward function parameters
        reward_params = {
            'w_progress': reward_function.w_progress,
            'w_time': reward_function.w_time,
            'w_jerk': reward_function.w_jerk,
            'success_reward': reward_function.success_reward,
            'crash_reward': reward_function.crash_reward
        }
        
        # Create environment
        env_kwargs = {
            'corridor_length': environment.length,
            'corridor_width': environment.width,
            'corridor_height': environment.height,
            'buffer_start': environment.buffer_start,
            'buffer_end': environment.buffer_end,
            'agent_diameter': agent.agent_diameter,
            'agent_max_speed': agent.agent_max_speed,
            'agent_max_acceleration': getattr(agent, 'agent_max_acceleration', 5.0),
            'num_neighbors': 5,
            'num_swarm_drones': num_swarm_drones,
            'swarm_obstacles': obstacles,
            'reward_params': reward_params,
            'lidar_range': agent.lidar_range,
            'lidar_rays': agent.lidar_rays,
            'target_diameter': environment.target_diameter,
            'max_ep_length': experiment.max_ep_length,
            'kinematic_type': agent.kinematic_type if hasattr(agent, 'kinematic_type') else "holonomic"
        }
        
        # Add predicted_seed for reproducible environments
        if getattr(environment, 'is_predicted', False) and environment.created_at:
            predicted_seed = int(environment.created_at.timestamp())
            env_kwargs['predicted_seed'] = predicted_seed
            print(f"[Training] Predicted mode ENABLED: seed={predicted_seed} (from {environment.created_at})")
        
        print(f"Creating DroneSwarmEnv...")
        env = DroneSwarmEnv(**env_kwargs)
        
        # Configure safety constraint
        safety_config = experiment.safety_constraint
        safety_enabled = safety_config and isinstance(safety_config, dict) and safety_config.get('enabled', False)
        
        if safety_enabled:
            print(f"\n[Training] Safety Constraint ENABLED")
            env.configure_safety_constraint(
                enabled=True,
                collision_weight=safety_config['collision_weight'],
                near_miss_weight=safety_config['near_miss_weight'],
                danger_zone_weight=safety_config['danger_zone_weight'],
                near_miss_threshold_multiplier=safety_config['near_miss_threshold'],
                ignore_walls=safety_config['ignore_walls'],
                wall_near_miss_weight=safety_config['wall_near_miss_weight'],
                wall_danger_zone_weight=safety_config['wall_danger_zone_weight'],
                wall_near_miss_threshold=safety_config['wall_near_miss_threshold']
            )
            lagrange_state = LagrangeState(lam=safety_config.get('initial_lambda', 0.0))
        else:
            print(f"\n[Training] Safety Constraint DISABLED")
            lagrange_state = LagrangeState(lam=0.0)
        
        # SHIELDING COMMENTED OUT
        # Wrapper order keeps the Shield ahead of residual so it sees the combined action.
        # Order: DroneSwarmEnv → SafetyShieldWrapper → ResidualActionWrapper → AppendLambdaWrapper
        # This ensures Shield intercepts the final action before it reaches the environment.

        # Add Safety Shield FIRST if enabled (before ResidualActionWrapper)
        # This way Shield sees the combined action, not raw residual actions
        # shield_enabled = safety_config and isinstance(safety_config, dict) and safety_config.get('shield_enabled', False)
        shield_wrapper = None  # Store reference for CARS K_shield (always None now)
        # if shield_enabled:
        #     from ..safety_shield import SafetyShieldWrapper
        #
        #     # Shield wraps DroneSwarmEnv directly, so obs has NO lambda appended yet
        #     # Observation structure: [goal_dir(3), dist(1), vel(3), LIDAR(N)]
        #     base_features = 7  # goal_dir(3) + dist(1) + vel(3)
        #     lidar_start = base_features  # No lambda at this point
        #     num_lidar = agent.lidar_rays if hasattr(agent, 'lidar_rays') else 72
        #
        #     shield_wrapper = SafetyShieldWrapper(
        #         env,
        #         threshold=safety_config.get('shield_threshold', 0.3),
        #         wall_threshold=safety_config.get('shield_wall_threshold', 0.1),
        #         fallback_strength=safety_config.get('shield_fallback_strength', 0.8),
        #         lidar_start_idx=lidar_start,
        #         num_lidar_rays=num_lidar,
        #         verbose=1
        #     )
        #     env = shield_wrapper  # Also update env chain
        #     print(f"[Training] SafetyShield ENABLED (drone_threshold={safety_config.get('shield_threshold', 0.3)}, wall_threshold={safety_config.get('shield_wall_threshold', 0.1)}, lidar_start={lidar_start})")
        #     print(f"[Training] SafetyShield wraps DroneSwarmEnv BEFORE ResidualActionWrapper (sees combined action)")
        
        # Handle residual learning mode (AFTER Shield, so Shield sees combined action)
        training_mode = getattr(experiment, 'training_mode', 'standard')
        env = self._handle_residual_mode(db, experiment, env, lagrange_state, training_mode, shield_wrapper)
        
        # Always append λ to observations for consistent observation space (LAST wrapper before Monitor)
        env = AppendLambdaWrapper(env, lagrange_state)
        print(f"  Observation space (after AppendLambdaWrapper): {env.observation_space}")
        
        # Create monitored and vectorized environment
        def make_env(base=env):
            return Monitor(base)
        
        vec_env = DummyVecEnv([make_env])
        
        # Add Lagrangian reward shaping if safety is enabled
        if safety_enabled:
            env = LagrangianRewardVecWrapper(vec_env, lagrange_state, cost_key="cost")
            print(f"[Training] Environment wrapped with AppendLambda + Monitor + LagrangianRewardVecWrapper")
        else:
            env = vec_env
            print(f"[Training] Environment wrapped with AppendLambda + Monitor (no reward shaping)")

        return env, lagrange_state, safety_config, training_mode

    def _handle_residual_mode(self, db, experiment, env, lagrange_state, training_mode, shield_wrapper=None):
        """Handle residual learning mode configuration with optional CARS shield integration."""
        if training_mode != 'residual':
            return env

        from .. import crud, models as db_models
        
        residual_base_model_id = getattr(experiment, 'residual_base_model_id', None)
        if not residual_base_model_id:
            raise ValueError("Residual mode requires residual_base_model_id")
        
        print(f"\n[Training] 🎯 RESIDUAL LEARNING MODE ENABLED")
        
        # Load the base model snapshot
        base_snapshot = crud.get_entity(db, db_models.ModelSnapshot, residual_base_model_id)
        if not base_snapshot:
            raise ValueError(f"Base model snapshot {residual_base_model_id} not found")
        
        print(f"[Training] Base model: {base_snapshot.file_path}")
        
        # Get base model algorithm from snapshot metadata
        base_model_agent_id = base_snapshot.metrics_at_save.get('agent_id') if base_snapshot.metrics_at_save else None
        if base_model_agent_id:
            base_model_agent = crud.get_entity(db, db_models.Agent, base_model_agent_id)
            base_algorithm = base_model_agent.type if base_model_agent else "PPO"
        else:
            base_algorithm = "PPO"  # Default
        
        print(f"[Training] Base model algorithm: {base_algorithm}")
        
        # Get base model's frozen lambda from snapshot
        base_model_lambda = 0.0
        if base_snapshot.metrics_at_save and 'lambda' in base_snapshot.metrics_at_save:
            base_model_lambda = float(base_snapshot.metrics_at_save['lambda'])
            print(f"[Training] Base model λ (frozen): {base_model_lambda:.4f}")
        else:
            print(f"[Training] Base model λ not found in snapshot, using 0.0")
        
        # Get k_factor and adaptive_k from ResidualConnector
        residual_connector_id = getattr(experiment, 'residual_connector_id', None)
        k_factor = 0.15  # Default
        adaptive_k = False  # Default
        enable_k_conf = True  # Default
        enable_k_risk = True  # Default
        # SHIELDING COMMENTED OUT
        # enable_k_shield = True  # Default
        if residual_connector_id:
            residual_connector = crud.get_entity(db, db_models.ResidualConnector, residual_connector_id)
            if residual_connector:
                k_factor = getattr(residual_connector, 'k_factor', 0.15)
                adaptive_k = getattr(residual_connector, 'adaptive_k', False)
                enable_k_conf = getattr(residual_connector, 'enable_k_conf', True)
                enable_k_risk = getattr(residual_connector, 'enable_k_risk', True)
                # SHIELDING COMMENTED OUT
                # enable_k_shield = getattr(residual_connector, 'enable_k_shield', True)
                print(f"[Training] K-factor: {k_factor} (from ResidualConnector)")
                print(f"[Training] Adaptive K: {adaptive_k}")
                if adaptive_k:
                    # SHIELDING COMMENTED OUT - removed K_shield from log
                    print(f"[Training] CARS Ablation: K_conf={enable_k_conf}, K_risk={enable_k_risk}")

        # Wrap environment with ResidualActionWrapper (with CARS integration)
        from ..wrappers import ResidualActionWrapper
        env = ResidualActionWrapper(
            env=env,
            base_model_path=base_snapshot.file_path,
            base_algorithm=base_algorithm,
            lagrange_state=lagrange_state,  # Adaptive lambda for residual policy (for K_risk)
            base_lambda_frozen=base_model_lambda,  # Frozen lambda for base model
            k_factor=k_factor,
            adaptive_k=adaptive_k,
            # SHIELDING COMMENTED OUT - CARS: Shield integration for K_shield
            # shield_wrapper=shield_wrapper,
            # CARS: Ablation control
            enable_k_conf=enable_k_conf,
            enable_k_risk=enable_k_risk,
            # SHIELDING COMMENTED OUT
            # enable_k_shield=enable_k_shield,
        )
        if adaptive_k:
            # SHIELDING COMMENTED OUT - removed shield from log
            print(f"[Training] CARS enabled")
        print(f"[Training] Environment wrapped with ResidualActionWrapper")
        
        return env

    def _create_or_load_model(self, db, experiment, agent, AlgorithmClass, env,
                             lagrange_state, training_mode, safety_config, seed=None):
        """Create a new model or load an existing one based on experiment type."""
        # Parse parameters
        params = agent.parameters if isinstance(agent.parameters, dict) else {}

        # Add seed to params if specified
        if seed is not None:
            params['seed'] = seed

        # Check if this is a Fine-Tuning experiment
        is_fine_tuning = experiment.type == "Fine-Tuning"

        if is_fine_tuning and experiment.base_model_snapshot_id:
            return self._load_fine_tuning_model(db, experiment, AlgorithmClass, env)
        elif training_mode == 'residual':
            return self._create_residual_model(db, experiment, agent, AlgorithmClass, env, params, seed)
        else:
            # Regular Training: Create new model
            print(f"Creating new {agent.type} model...")
            model = AlgorithmClass(
                "MlpPolicy",
                env,
                verbose=0,
                **params,
            )
            print(f"Model created successfully")
            return model

    def _load_fine_tuning_model(self, db, experiment, AlgorithmClass, env):
        """Load a model for fine-tuning."""
        from .. import crud, models as db_models
        
        print(f"Fine-Tuning mode detected - loading base model from snapshot {experiment.base_model_snapshot_id}")
        base_snapshot = crud.get_entity(db, db_models.ModelSnapshot, experiment.base_model_snapshot_id)
        
        if not base_snapshot:
            raise ValueError(f"Base model snapshot {experiment.base_model_snapshot_id} not found")
        
        print(f"Loading base model from: {base_snapshot.file_path}")
        model = AlgorithmClass.load(base_snapshot.file_path, env=env)
        print(f"Base model loaded successfully, continuing training from step {model.num_timesteps}")
        
        # Apply fine-tuning strategy
        from ..fine_tuning_strategies import apply_fine_tuning_strategy
        strategy = experiment.fine_tuning_strategy or "full_finetune"
        apply_fine_tuning_strategy(model, strategy)
        
        return model

    def _create_residual_model(self, db, experiment, agent, AlgorithmClass, env, params, seed=None):
        """Create a residual learning model."""
        from .. import crud, models as db_models

        print(f"\n[Training] Creating lightweight residual policy...")

        # Copy defaults from agent first
        residual_params = params.copy()

        # Ensure seed is passed to residual model
        if seed is not None:
            residual_params['seed'] = seed
        
        # Load configuration from ResidualConnector
        residual_algorithm = agent.type  # Default to agent's algorithm
        
        if experiment.residual_connector_id:
            print(f"[Training] Loading configuration from ResidualConnector ID {experiment.residual_connector_id}...")
            connector = crud.get_entity(db, db_models.ResidualConnector, experiment.residual_connector_id)
            
            if connector:
                if hasattr(connector, 'algorithm') and connector.algorithm:
                    residual_algorithm = connector.algorithm
                    print(f"[Training] Using algorithm from Connector: {residual_algorithm}")
                
                if connector.parameters:
                    print(f"[Training] Overriding parameters from Connector: {connector.parameters}")
                    residual_params.update(connector.parameters)
                
                if connector.net_arch:
                    print(f"[Training] Using architecture from Connector: {connector.net_arch}")
                    residual_params['policy_kwargs'] = residual_params.get('policy_kwargs', {})
                    residual_params['policy_kwargs']['net_arch'] = connector.net_arch
                else:
                    residual_params['policy_kwargs'] = residual_params.get('policy_kwargs', {})
                    residual_params['policy_kwargs']['net_arch'] = [64, 64]
            else:
                print("[Training] ⚠️ Connector not found, using Agent defaults with [64, 64] arch")
                residual_params['policy_kwargs'] = residual_params.get('policy_kwargs', {})
                residual_params['policy_kwargs']['net_arch'] = [64, 64]
        else:
            print("[Training] No Connector ID provided, using Agent defaults with [64, 64] arch")
            residual_params['policy_kwargs'] = residual_params.get('policy_kwargs', {})
            residual_params['policy_kwargs']['net_arch'] = [64, 64]
        
        # Select the appropriate Algorithm Class for residual model
        residual_algorithm_map = {
            "PPO": PPO,
            "SAC": SAC,
            "TD3": TD3,
            "A2C": A2C,
            "DDPG": DDPG,
        }
        ResidualAlgorithmClass = residual_algorithm_map.get(residual_algorithm)
        if not ResidualAlgorithmClass:
            raise ValueError(f"Unsupported residual algorithm type: {residual_algorithm}")
        
        print(f"[Training] Creating new {residual_algorithm} model for residual learning...")
        
        model = ResidualAlgorithmClass(
            "MlpPolicy",
            env,
            verbose=0,
            **residual_params,
        )
        print(f"[Training] Residual policy model created")
        
        # Zero-initialize the residual policy weights
        from ..wrappers import initialize_residual_policy_weights
        initialize_residual_policy_weights(model, initial_log_std=-2.0)
        
        # Verify residual outputs near-zero
        test_obs = np.zeros(env.observation_space.shape, dtype=np.float32)
        test_action, _ = model.predict(test_obs, deterministic=True)
        action_mag = np.linalg.norm(test_action)
        if action_mag > 0.05:
            print(f"[Training] ⚠️ WARNING: Residual not properly zeroed! Mag={action_mag:.4f}")
        else:
            print(f"[Training] ✅ Residual verified near-zero (mag={action_mag:.6f})")
        
        return model

    def _create_snapshot_saver(self, db, experiment_id, experiment, agent, reward_function, 
                              lagrange_state, training_mode, model):
        """Create a function to save model snapshots."""
        from .. import crud
        
        def save_snapshot(step: int, metrics: dict = None, filename: str = None):
            if filename:
                snapshot_path = filename
            else:
                snapshot_path = f"models/experiment_{experiment_id}_step_{step}.zip"
            
            model.save(snapshot_path)
            print(f"[Training] Saved snapshot at step {step} to {snapshot_path}")
            
            # Save snapshot to database with metadata
            snapshot_metrics = {
                "step": step, 
                "mean_reward": 0.0, 
                "agent_id": agent.id, 
                "reward_id": reward_function.id,
                "lambda": lagrange_state.lam if lagrange_state else 0.0,
                "training_mode": training_mode,
                "residual_base_model_id": getattr(experiment, 'residual_base_model_id', None) if training_mode == 'residual' else None,
                "residual_connector_id": getattr(experiment, 'residual_connector_id', None) if training_mode == 'residual' else None,
            }
            
            # Add performance metrics if provided
            if metrics:
                # Add success/failure rates
                if "success_rate" in metrics:
                    snapshot_metrics["success_rate"] = metrics["success_rate"]
                if "crash_rate" in metrics:
                    snapshot_metrics["failure_rate"] = metrics["crash_rate"]
                if "reward_shaped_mean" in metrics:
                    snapshot_metrics["reward"] = metrics["reward_shaped_mean"]
                elif "mean_reward" in metrics:
                    snapshot_metrics["reward"] = metrics["mean_reward"]
                
                # Copy other useful metrics
                for key in ["violation_rate", "cost_mean", "lambda", "epsilon", "warmup_progress"]:
                    if key in metrics:
                        snapshot_metrics[key] = metrics[key]
                
                if "cancelled" in metrics:
                    snapshot_metrics["cancelled"] = True
                if "completed" in metrics:
                    snapshot_metrics["completed"] = True
            
            crud.create_experiment_snapshot(
                db,
                experiment_id=experiment_id,
                iteration=step,
                file_path=snapshot_path,
                metrics=snapshot_metrics,
            )
            print(f"[Training] Snapshot record created in database for step {step} (λ={snapshot_metrics['lambda']:.4f})")
        
        return save_snapshot

    def _create_training_callback(self, experiment_id, broadcast_func, loop, experiment, 
                                 environment, lagrange_state, safety_config, save_snapshot):
        """Create the training callback."""
        from ..database import SessionLocal
        
        # Add Lagrangian callback ONLY if safety constraint is enabled
        lagrange_callback = None
        safety_enabled = safety_config and isinstance(safety_config, dict) and safety_config.get('enabled', False)
        
        if safety_enabled:
            from ..lagrangian_safety import LagrangianCallback
            lagrange_callback = LagrangianCallback(
                state=lagrange_state,
                epsilon=safety_config.get('risk_budget', 0.3),
                alpha=safety_config.get('lambda_learning_rate', 0.02),
                update_every_n_episodes=safety_config.get('update_frequency', 50),
                target_lambda=safety_config.get('target_lambda'),
                warmup_episodes=safety_config.get('warmup_episodes', 0),
                warmup_schedule=safety_config.get('warmup_schedule', 'exponential'),
                verbose=1
            )
            print(f"[Training] Created LagrangianCallback for warmup tracking")
        
        callback = ExperimentCallback(
            experiment_id=experiment_id,
            broadcast_func=broadcast_func,
            loop=loop,
            save_interval=100,
            snapshot_freq=experiment.snapshot_freq or 10000,
            total_steps=experiment.total_steps or 100000,
            model_save_func=save_snapshot,
            db_session_factory=SessionLocal,
            environment=environment,
            lagrange_state=lagrange_state,
            safety_config=safety_config,
            lagrange_callback=lagrange_callback,
        )
        
        # Create callback list
        from stable_baselines3.common.callbacks import CallbackList
        callbacks = [callback]
        
        if lagrange_callback:
            callbacks.append(lagrange_callback)
            print(f"[Training] Added LagrangianCallback to training")
        
        return CallbackList(callbacks), callback

    def _execute_training_loop(self, db, experiment, model, callback_list, cancel_event, 
                              broadcast_func, loop):
        """Execute the main training loop."""
        from .. import crud, models as db_models
        
        # Store model in cache
        self.models_cache[experiment.id] = model
        self.envs_cache[experiment.id] = model.env

        # Training loop with pause support
        total_steps = experiment.total_steps or 100000
        chunk_size = 1000
        print(f"[Training Loop] Starting with total_steps={total_steps}, chunk_size={chunk_size}")

        # Get the actual ExperimentCallback for trajectory access
        experiment_callback = None
        if hasattr(callback_list, 'callbacks'):
            # callback_list is a CallbackList, find the ExperimentCallback
            for cb in callback_list.callbacks:
                if hasattr(cb, 'experiment_id'):  # ExperimentCallback has experiment_id
                    experiment_callback = cb
                    break
        elif hasattr(callback_list, 'experiment_id'):
            # callback_list is directly the ExperimentCallback  
            experiment_callback = callback_list

        # Use model.num_timesteps to track actual progress
        while model.num_timesteps < total_steps:
            # Check if paused
            while self.paused_experiments.get(experiment.id, False):
                time.sleep(0.5)

            # Check if cancelled via event
            if cancel_event.is_set():
                print(f"[Training] Cancellation detected at step {model.num_timesteps}")
                break

            # Train a chunk
            steps_to_train = min(chunk_size, total_steps - model.num_timesteps)
            print(f"[Training] Starting chunk: {steps_to_train} steps (total: {model.num_timesteps}/{total_steps})")
            
            model.learn(
                total_timesteps=steps_to_train,
                callback=callback_list,
                reset_num_timesteps=False,
            )
            
            print(f"[Training] Completed chunk. Total steps: {model.num_timesteps}/{total_steps}")
            
            # Update current_step in database after each chunk
            exp = crud.get_entity(db, db_models.Experiment, experiment.id)
            if exp:
                exp.current_step = model.num_timesteps
                db.commit()

        # Handle training completion or cancellation
        self._finalize_training(
            db, experiment, model, experiment_callback, cancel_event, broadcast_func, loop, total_steps
        )

    def _finalize_training(self, db, experiment, model, callback, cancel_event, 
                          broadcast_func, loop, total_steps):
        """Handle training completion or cancellation."""
        from .. import crud, models as db_models
        
        experiment_id = experiment.id
        
        if cancel_event.is_set():
            # Training was cancelled
            actual_steps = model.num_timesteps
            print(f"[Training] Experiment {experiment_id} was cancelled at step {actual_steps}")
            
            # Calculate metrics if callback available
            metrics = {"step": actual_steps, "cancelled": True}
            if callback:
                try:
                    latest_metrics = callback._calculate_metrics()
                    if latest_metrics:
                        metrics.update(latest_metrics)
                except Exception as e:
                    print(f"[Training] Error calculating final metrics: {e}")
            
            # Save final model at cancellation point using the callback's save function if available
            # This ensures all metadata (agent_id, reward, etc.) is preserved correctly
            saved = False
            final_path = f"models/experiment_{experiment_id}_cancelled_step_{actual_steps}.zip"
            
            if callback and callback.model_save_func:
                try:
                    callback.model_save_func(actual_steps, metrics=metrics, filename=final_path)
                    saved = True
                except TypeError:
                    # Fallback for save func without filename support (if signature wasn't updated)
                    callback.model_save_func(actual_steps, metrics=metrics)
                    saved = True
                except Exception as e:
                    print(f"[Training] Error using model_save_func: {e}")
            
            if not saved:
                # Fallback to manual save if no callback
                model.save(final_path)
                crud.create_experiment_snapshot(
                    db,
                    experiment_id=experiment_id,
                    iteration=actual_steps,
                    file_path=final_path,
                    metrics=metrics,
                )
            
            # Update experiment status
            exp = crud.get_entity(db, db_models.Experiment, experiment_id)
            if exp:
                exp.current_step = actual_steps
                exp.status = "Cancelled"
                if hasattr(callback, 'trajectory'):
                    exp.trajectory_data = callback.trajectory.copy()
                db.commit()
            
            # Broadcast cancellation
            asyncio.run_coroutine_threadsafe(
                broadcast_func(
                    experiment_id,
                    {
                        "type": "progress",
                        "data": {
                            "experiment_id": experiment_id,
                            "step": actual_steps,
                            "total_steps": total_steps,
                            "metrics": metrics,
                            "status": "Cancelled",
                            "timestamp": time.time(),
                        },
                    },
                ),
                loop
            )
        else:
            # Training completed successfully
            print(f"[Training] Experiment {experiment_id} completed successfully")
            
            # Calculate metrics
            metrics = {"step": total_steps, "completed": True}
            if callback:
                try:
                    latest_metrics = callback._calculate_metrics()
                    if latest_metrics:
                        metrics.update(latest_metrics)
                except Exception as e:
                    print(f"[Training] Error calculating final metrics: {e}")

            # Save final model using callback's save function
            saved = False
            final_path = f"models/experiment_{experiment_id}_final.zip"
            
            if callback and callback.model_save_func:
                try:
                    callback.model_save_func(total_steps, metrics=metrics, filename=final_path)
                    saved = True
                except TypeError:
                    callback.model_save_func(total_steps, metrics=metrics)
                    saved = True
                except Exception as e:
                    print(f"[Training] Error using model_save_func: {e}")
            
            if not saved:
                model.save(final_path)
                crud.create_experiment_snapshot(
                    db,
                    experiment_id=experiment_id,
                    iteration=total_steps,
                    file_path=final_path,
                    metrics=metrics,
                )

            # Update status and progress to completed
            exp = crud.get_entity(db, db_models.Experiment, experiment_id)
            if exp:
                exp.current_step = total_steps
                exp.status = "Completed"
                if hasattr(callback, 'trajectory'):
                    exp.trajectory_data = callback.trajectory.copy()
                db.commit()

            # Broadcast completion
            asyncio.run_coroutine_threadsafe(
                broadcast_func(
                    experiment_id,
                    {
                        "type": "progress",
                        "data": {
                            "experiment_id": experiment_id,
                            "step": total_steps,
                            "total_steps": total_steps,
                            "metrics": {"completed": True},
                            "status": "Completed",
                            "timestamp": time.time(),
                        },
                    },
                ),
                loop
            )

    def _handle_training_error(self, db, experiment_id, error):
        """Handle training errors."""
        from .. import crud, models as db_models
        
        print(f"\n!!! Training error for experiment {experiment_id} !!!")
        print(f"Error type: {type(error).__name__}")
        print(f"Error message: {error}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        
        # Update experiment status
        experiment = crud.get_entity(db, db_models.Experiment, experiment_id)
        if experiment:
            experiment.status = "Cancelled"
            db.commit()
