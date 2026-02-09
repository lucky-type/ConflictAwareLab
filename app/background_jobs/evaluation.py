"""Evaluation-related functionality for background experiments."""

import asyncio
import time
import threading
from typing import Dict, Any, List
import numpy as np
from sqlalchemy.orm import Session

from stable_baselines3 import PPO, SAC


class EvaluationMixin:
    """Mixin class for evaluation-related experiment functionality."""
    
    def _calculate_evaluation_metrics(
        self,
        episode_rewards,
        episode_lengths,
        episode_successes,
        episode_crashes,
        episode_timeouts,
        episode_costs,
        episode_near_misses,
        episode_danger_time,
        lagrange_state,
        safety_config,
        residual_magnitudes=None,
        base_magnitudes=None,
        cosine_similarities=None,
        intervention_flags=None,
        episode_shield_interventions=None,
        # CARS K metrics
        effective_k_values=None,
        k_conf_values=None,
        k_risk_values=None,
        window_size=None,
    ):
        """Calculate comprehensive metrics matching training callback format."""

        metrics = {}
        
        # Convert Pydantic model to dict if needed
        if safety_config is not None and hasattr(safety_config, 'model_dump'):
            safety_config = safety_config.model_dump()
        elif safety_config is not None and hasattr(safety_config, 'dict'):
            safety_config = safety_config.dict()
        
        # Use window or all data
        if window_size and len(episode_rewards) > window_size:
            recent_rewards = episode_rewards[-window_size:]
            recent_lengths = episode_lengths[-window_size:]
            recent_successes = episode_successes[-window_size:]
            recent_crashes = episode_crashes[-window_size:]
            recent_timeouts = episode_timeouts[-window_size:]
            recent_costs = episode_costs[-window_size:]
            recent_near_misses = episode_near_misses[-window_size:]
            recent_danger_time = episode_danger_time[-window_size:]
            recent_shield_interventions = episode_shield_interventions[-window_size:] if episode_shield_interventions else []
        else:
            recent_rewards = episode_rewards if episode_rewards else []
            recent_lengths = episode_lengths if episode_lengths else []
            recent_successes = episode_successes if episode_successes else []
            recent_crashes = episode_crashes if episode_crashes else []
            recent_timeouts = episode_timeouts if episode_timeouts else []
            recent_costs = episode_costs if episode_costs else []
            recent_near_misses = episode_near_misses if episode_near_misses else []
            recent_danger_time = episode_danger_time if episode_danger_time else []
            recent_shield_interventions = episode_shield_interventions if episode_shield_interventions else []
        
        # A) Performance Metrics
        if len(recent_rewards) > 0:
            metrics["reward_raw_mean"] = float(np.mean(recent_rewards))
            metrics["reward_raw_std"] = float(np.std(recent_rewards))
            metrics["mean_ep_length"] = float(np.mean(recent_lengths))
            metrics["ep_length_std"] = float(np.std(recent_lengths))
            
            # Calculate shaped rewards: r' = r - λ*cost (if safety enabled)
            # In evaluation, rewards are raw, so shaped = raw - lambda*cost
            if lagrange_state and len(recent_costs) > 0 and len(recent_rewards) == len(recent_costs) and lagrange_state.lam > 0:
                # Reconstruct shaped rewards from raw rewards and costs
                # shaped_reward = raw_reward - lambda * cumulative_cost
                shaped_rewards = []
                for raw_reward, cost in zip(recent_rewards, recent_costs):
                    shaped = raw_reward - (lagrange_state.lam * cost)
                    shaped_rewards.append(shaped)
                if shaped_rewards:
                    metrics["reward_shaped_mean"] = float(np.mean(shaped_rewards))
                else:
                    metrics["reward_shaped_mean"] = metrics["reward_raw_mean"]
            else:
                # No shaping, shaped = raw
                metrics["reward_shaped_mean"] = metrics["reward_raw_mean"]
            
            # Round to 0.5% precision (nearest 0.5) like training callback
            metrics["success_rate"] = round(float(np.mean(recent_successes)) * 100.0 * 2) / 2
            metrics["crash_rate"] = round(float(np.mean(recent_crashes)) * 100.0 * 2) / 2
            metrics["timeout_rate"] = round(float(np.mean(recent_timeouts)) * 100.0 * 2) / 2
            
            # NEW: Total episode counts for UI display
            # (currentEpisode / successCount (green) / failedCount (red) / timeoutCount (yellow))
            metrics["total_episodes"] = len(recent_rewards)
            metrics["total_successes"] = int(sum(recent_successes))
            metrics["total_failures"] = int(sum(recent_crashes))
            metrics["total_timeouts"] = int(sum(recent_timeouts))
            
            # Legacy compatibility
            metrics["mean_reward"] = metrics["reward_raw_mean"]
        else:
            metrics.update({
                "reward_raw_mean": 0.0,
                "reward_raw_std": 0.0,
                "reward_shaped_mean": 0.0,
                "success_rate": 0.0,
                "crash_rate": 0.0,
                "timeout_rate": 0.0,
                "mean_ep_length": 0.0,
                "ep_length_std": 0.0,
                "mean_reward": 0.0,
                "total_episodes": 0,
                "total_successes": 0,
                "total_failures": 0,
            })
        
        # B) Safety/Risk Metrics
        if len(recent_costs) > 0:
            metrics["cost_mean"] = float(np.mean(recent_costs))
            metrics["cost_std"] = float(np.std(recent_costs))
            metrics["near_miss_mean"] = float(np.mean(recent_near_misses))
            metrics["danger_time_mean"] = float(np.mean(recent_danger_time))
            
            # Calculate violation rate (same logic as training callback)
            # Violation = (cumulative_cost / episode_length) > epsilon
            if safety_config and safety_config.get('enabled', False):
                epsilon = safety_config.get('risk_budget', 0.3)
                metrics["epsilon"] = float(epsilon)
                
                if len(recent_costs) == len(recent_lengths):
                    violations = [
                        (cost / max(1, length)) > epsilon
                        for cost, length in zip(recent_costs, recent_lengths)
                    ]
                    violation_rate_value = float(np.mean(violations)) * 100.0
                    metrics["violation_rate"] = violation_rate_value
                else:
                    metrics["violation_rate"] = 0.0
            else:
                metrics["epsilon"] = 0.0
                metrics["violation_rate"] = 0.0
        else:
            metrics.update({
                "cost_mean": 0.0,
                "cost_std": 0.0,
                "near_miss_mean": 0.0,
                "danger_time_mean": 0.0,
                "violation_rate": 0.0,
                "epsilon": safety_config.get('risk_budget', 0.3) if safety_config and safety_config.get('enabled', False) else 0.0,
            })
        
        metrics["shield_intervention_rate"] = 0.0
        metrics["shield_interventions_per_episode"] = 0.0
        
        # D) Lagrangian Dynamics
        if lagrange_state:
            metrics["lambda"] = float(lagrange_state.lam)
        else:
            metrics["lambda"] = 0.0
        
        # Warmup metrics (not applicable for evaluation, but set for compatibility)
        metrics["warmup_target_lambda"] = None
        metrics["warmup_progress"] = None
        
        # E) Residual Learning Metrics (Group 3 - Enhanced)
        if residual_magnitudes and len(residual_magnitudes) > 0:
            mean_res_mag = float(np.mean(residual_magnitudes))
            mean_base_mag = float(np.mean(base_magnitudes)) if base_magnitudes and len(base_magnitudes) > 0 else 0.0
            
            metrics["residual_correction_magnitude"] = mean_res_mag
            metrics["residual_base_magnitude"] = mean_base_mag
            
            # Calculate intervention ratio (legacy)
            if mean_base_mag > 1e-6:
                metrics["residual_intervention_ratio"] = float(mean_res_mag / mean_base_mag)
            else:
                metrics["residual_intervention_ratio"] = 0.0
            
            # NEW: Residual Contribution Ratio = ||a_res|| / (||a_base|| + ||a_res|| + eps)
            # Calculate per-step then average for accuracy
            if base_magnitudes and len(base_magnitudes) == len(residual_magnitudes):
                contribution_ratios = []
                for res_mag, base_mag in zip(residual_magnitudes, base_magnitudes):
                    ratio = res_mag / (base_mag + res_mag + 1e-8)
                    contribution_ratios.append(ratio)
                metrics["residual_contribution_mean"] = float(np.mean(contribution_ratios))
            else:
                # Fallback: use aggregated magnitudes
                metrics["residual_contribution_mean"] = mean_res_mag / (mean_base_mag + mean_res_mag + 1e-8)
        else:
            metrics["residual_correction_magnitude"] = 0.0
            metrics["residual_base_magnitude"] = 0.0
            metrics["residual_intervention_ratio"] = 0.0
            metrics["residual_contribution_mean"] = 0.0
        
        # NEW: Policy Conflict (Cosine Similarity) metrics
        if cosine_similarities and len(cosine_similarities) > 0:
            metrics["conflict_mean"] = float(np.mean(cosine_similarities))
            metrics["conflict_std"] = float(np.std(cosine_similarities))
        else:
            metrics["conflict_mean"] = 0.0
            metrics["conflict_std"] = 0.0
        
        # NEW: Intervention Rate (% of steps where residual > threshold)
        if intervention_flags and len(intervention_flags) > 0:
            metrics["intervention_rate"] = float(np.mean(intervention_flags)) * 100.0
        else:
            metrics["intervention_rate"] = 0.0
        
        # G) CARS K Metrics
        if effective_k_values and len(effective_k_values) > 0:
            metrics["effective_k_mean"] = float(np.mean(effective_k_values))
        else:
            metrics["effective_k_mean"] = 0.0
        
        if k_conf_values and len(k_conf_values) > 0:
            metrics["k_conf_mean"] = float(np.mean(k_conf_values))
        else:
            metrics["k_conf_mean"] = 1.0
        
        if k_risk_values and len(k_risk_values) > 0:
            metrics["k_risk_mean"] = float(np.mean(k_risk_values))
        else:
            metrics["k_risk_mean"] = 1.0
        
        metrics["k_shield_mean"] = 1.0
        
        # F) Algorithm-specific metrics (not available in evaluation, but set for compatibility)
        metrics["policy_loss"] = None
        metrics["value_loss"] = None
        metrics["actor_loss"] = None
        metrics["critic_loss"] = None
        metrics["entropy"] = None
        metrics["learning_rate"] = None
        
        return metrics
    
    async def start_evaluation(
        self,
        experiment: "models.Experiment",
        db: Session,
        broadcast_func,
    ):
        """Start an evaluation experiment in the background."""
        experiment_id = experiment.id

        if experiment_id in self.running_tasks:
            raise RuntimeError(f"Experiment {experiment_id} is already running")

        # Check if we have capacity for a new job
        if not self.can_start_new_job():
            raise RuntimeError(f"Maximum concurrent jobs ({self.max_concurrent_jobs}) reached. Please wait for a job to complete.")

        # Create cancellation event for this experiment
        cancel_event = threading.Event()
        self.cancel_events[experiment_id] = cancel_event

        task = asyncio.create_task(
            self._run_evaluation_wrapper(experiment, db, broadcast_func, cancel_event)
        )
        self.running_tasks[experiment_id] = task
        self.paused_experiments[experiment_id] = False

        from .. import crud

        crud.update_experiment_status(db, experiment_id, "In Progress")

    async def _run_evaluation_wrapper(
        self,
        experiment: "models.Experiment",
        db: Session,
        broadcast_func,
        cancel_event: threading.Event,
    ):
        """Wrapper that runs evaluation in executor thread."""
        experiment_id = experiment.id
        
        try:
            # Run evaluation in a thread pool (similar to training)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._run_evaluation_sync,
                experiment.id,
                broadcast_func,
                loop,
                cancel_event
            )
        except Exception as e:
            print(f"\n!!! Evaluation error for experiment {experiment_id} !!!")
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

    def _run_evaluation_sync(
        self,
        experiment_id: int,
        broadcast_func,
        loop,
        cancel_event: threading.Event,
    ):
        """Synchronous evaluation runner executed in thread pool."""
        from .. import crud, models as db_models
        from ..database import SessionLocal
        from ..drone_env import DroneSwarmEnv
        from ..lagrangian_safety import LagrangeState, AppendLambdaWrapper
        from ..wrappers import ResidualActionWrapper
        import pybullet as p

        print(f"\n=== Starting evaluation for experiment {experiment_id} ===")

        # Create new DB session for this thread
        db = SessionLocal()
        
        try:
            # Load experiment and related entities
            experiment = crud.get_entity(db, db_models.Experiment, experiment_id)
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")

            if not experiment.evaluation_episodes:
                raise ValueError("Evaluation requires evaluation_episodes to be set")

            environment = crud.get_entity(db, db_models.Environment, experiment.env_id)
            agent = crud.get_entity(db, db_models.Agent, experiment.agent_id)
            reward_function = crud.get_entity(db, db_models.RewardFunction, experiment.reward_id)
            
            # Determine which model to load (base model or residual model)
            base_snapshot_id = None
            residual_snapshot_id = None
            
            if experiment.training_mode == "residual" and experiment.residual_base_model_id:
                base_snapshot_id = experiment.residual_base_model_id
                residual_snapshot_id = experiment.model_snapshot_id
                print(f"[Evaluation] Residual mode: base={base_snapshot_id}, residual={residual_snapshot_id}")
            elif experiment.model_snapshot_id:
                base_snapshot_id = experiment.model_snapshot_id
                print(f"[Evaluation] Standard mode: model={base_snapshot_id}")
            else:
                raise ValueError("Evaluation requires a model snapshot")

            # Load base model snapshot
            base_snapshot = crud.get_entity(db, db_models.ModelSnapshot, base_snapshot_id)
            if not base_snapshot:
                raise ValueError(f"Base model snapshot {base_snapshot_id} not found")

            print(f"[Evaluation] Base model: {base_snapshot.file_path}")
            print(f"[Evaluation] Environment: {environment.name}")
            print(f"[Evaluation] Agent: {agent.name}")
            print(f"[Evaluation] Episodes to run: {experiment.evaluation_episodes}")
            print(f"[Evaluation] FPS delay: {experiment.fps_delay or 0}ms")

            # Get the agent configuration from the snapshot metadata
            # This agent was used during training, so we recreate env with its exact config
            snapshot_agent_id = base_snapshot.metrics_at_save.get('agent_id') if base_snapshot.metrics_at_save else None
            training_agent = agent  # Default to current agent
            
            if snapshot_agent_id:
                print(f"[Evaluation] Loading agent config from snapshot (agent_id={snapshot_agent_id})")
                training_agent = crud.get_entity(db, db_models.Agent, snapshot_agent_id)
                
                if training_agent:
                    print(f"[Evaluation] Found training agent: {training_agent.name}")
                    print(f"[Evaluation] Training config: kinematic={getattr(training_agent, 'kinematic_type', 'holonomic')}, lidar_rays={training_agent.lidar_rays}")
                else:
                    print(f"[Evaluation] Warning: Agent {snapshot_agent_id} not found, using current agent")
                    training_agent = agent
            else:
                print(f"[Evaluation] Warning: No agent_id in snapshot metadata, using current agent config")

            # Create environment with training agent configuration
            obstacles = environment.obstacles if environment.obstacles else []
            num_swarm_drones = len(obstacles)
            
            reward_params = {
                'w_progress': reward_function.w_progress,
                'w_time': reward_function.w_time,
                'w_jerk': reward_function.w_jerk,
                'success_reward': reward_function.success_reward,
                'crash_reward': reward_function.crash_reward
            }
            
            env = DroneSwarmEnv(
                corridor_length=environment.length,
                corridor_width=environment.width,
                corridor_height=environment.height,
                buffer_start=environment.buffer_start,
                buffer_end=environment.buffer_end,
                agent_diameter=training_agent.agent_diameter,
                agent_max_speed=training_agent.agent_max_speed,
                agent_max_acceleration=getattr(training_agent, 'agent_max_acceleration', 5.0),
                num_neighbors=5,
                num_swarm_drones=num_swarm_drones,
                swarm_obstacles=obstacles,
                reward_params=reward_params,
                lidar_range=training_agent.lidar_range,
                lidar_rays=training_agent.lidar_rays,
                target_diameter=environment.target_diameter,
                max_ep_length=experiment.max_ep_length,
                kinematic_type=getattr(training_agent, 'kinematic_type', 'holonomic'),
                # Predicted mode for reproducible scenarios
                predicted_seed=int(environment.created_at.timestamp()) if getattr(environment, 'is_predicted', False) and environment.created_at else None
            )
            
            if getattr(environment, 'is_predicted', False) and environment.created_at:
                print(f"[Evaluation] Predicted mode ENABLED: seed={int(environment.created_at.timestamp())}")

            print(f"[Evaluation] Created DroneSwarmEnv with training agent config")
            print(f"[Evaluation] Observation space: {env.observation_space}")
            print(f"[Evaluation] Action space: {env.action_space}")
            print(f"[Evaluation] Kinematic type: {getattr(training_agent, 'kinematic_type', 'holonomic')}")
            print(f"[Evaluation] Lidar: range={training_agent.lidar_range}m, rays={training_agent.lidar_rays}")

            # Configure safety constraint if enabled
            # Match wrapper order from training so Shield sees the combined action.
            # DroneSwarmEnv → SafetyShieldWrapper → ResidualActionWrapper → AppendLambdaWrapper
            # This ensures Shield sees the combined action (base + K * residual), not raw residual.
            
            # Load lambda value from snapshot metadata (prioritize residual model lambda if in residual mode)
            stored_lambda = 0.0
            
            # For residual training mode, prioritize residual model's lambda over base model's lambda
            if experiment.training_mode == "residual" and residual_snapshot_id:
                residual_snapshot = crud.get_entity(db, db_models.ModelSnapshot, residual_snapshot_id)
                if residual_snapshot and residual_snapshot.metrics_at_save and 'lambda' in residual_snapshot.metrics_at_save:
                    stored_lambda = float(residual_snapshot.metrics_at_save['lambda'])
                    print(f"[Evaluation] Using λ={stored_lambda:.4f} from residual model snapshot")
                elif base_snapshot.metrics_at_save and 'lambda' in base_snapshot.metrics_at_save:
                    stored_lambda = float(base_snapshot.metrics_at_save['lambda'])
                    print(f"[Evaluation] Using λ={stored_lambda:.4f} from base model snapshot (residual model λ not found)")
                else:
                    print(f"[Evaluation] No λ found in either residual or base model snapshots, using λ=0.0")
            else:
                # Standard training mode or no residual snapshot - use base model lambda
                if base_snapshot.metrics_at_save and 'lambda' in base_snapshot.metrics_at_save:
                    stored_lambda = float(base_snapshot.metrics_at_save['lambda'])
                    print(f"[Evaluation] Using λ={stored_lambda:.4f} from base model snapshot")
                else:
                    print(f"[Evaluation] No λ in base model snapshot metadata, using λ=0.0")
            
            lagrange_state = LagrangeState(lam=stored_lambda)
            
            # Check Shield configuration FIRST (before applying wrappers)
            safety_config = experiment.safety_constraint if hasattr(experiment, 'safety_constraint') else None
            
            # Convert Pydantic model to dict if needed for consistent access
            if safety_config is not None and hasattr(safety_config, 'model_dump'):
                safety_config_dict = safety_config.model_dump()
            elif safety_config is not None and hasattr(safety_config, 'dict'):
                safety_config_dict = safety_config.dict()
            elif isinstance(safety_config, dict):
                safety_config_dict = safety_config
            else:
                safety_config_dict = {}
            
            shield_enabled = False

            print(f"\n[Evaluation] === Safety Shield Configuration Check ===")
            print(f"[Evaluation] shield_enabled result: {shield_enabled}")
            print(f"[Evaluation] ==========================================\n")

            shield_wrapper = None
            print(f"[Evaluation] ❌ SafetyShield DISABLED")
            
            # STEP 2: Apply ResidualActionWrapper (if in residual mode) - AFTER Shield
            if experiment.training_mode == "residual" and residual_snapshot_id:
                residual_snapshot = crud.get_entity(db, db_models.ModelSnapshot, residual_snapshot_id)
                
                if not residual_snapshot:
                    raise ValueError(f"Residual snapshot {residual_snapshot_id} not found")
                
                # Try to get connector_id from snapshot metadata first, fallback to experiment
                residual_connector_id = None
                if residual_snapshot.metrics_at_save and 'residual_connector_id' in residual_snapshot.metrics_at_save:
                    residual_connector_id = residual_snapshot.metrics_at_save['residual_connector_id']
                    print(f"[Evaluation] Using residual_connector_id={residual_connector_id} from snapshot metadata")
                elif experiment.residual_connector_id:
                    residual_connector_id = experiment.residual_connector_id
                    print(f"[Evaluation] Using residual_connector_id={residual_connector_id} from experiment (snapshot metadata not available)")
                else:
                    snapshot_keys = list(residual_snapshot.metrics_at_save.keys()) if residual_snapshot.metrics_at_save else []
                    raise ValueError(
                        f"Residual evaluation requires residual_connector_id. "
                        f"Snapshot {residual_snapshot_id} metadata keys: {snapshot_keys}. "
                        f"Experiment residual_connector_id: {experiment.residual_connector_id}. "
                        f"Please run the backfill migration script or manually select a residual connector."
                    )
                
                residual_connector = crud.get_entity(db, db_models.ResidualConnector, residual_connector_id)

                if not residual_connector:
                    raise ValueError(
                        f"Residual connector {residual_connector_id} not found. "
                        f"It may have been deleted. Please select a different connector or recreate it."
                    )
                
                print(f"[Evaluation] Using residual wrapper: base={base_snapshot.file_path}, residual={residual_snapshot.file_path}")
                
                # Get base model algorithm from snapshot metadata
                base_model_agent_id = base_snapshot.metrics_at_save.get('agent_id') if base_snapshot.metrics_at_save else None
                if base_model_agent_id:
                    base_model_agent = crud.get_entity(db, db_models.Agent, base_model_agent_id)
                    base_algorithm = base_model_agent.type if base_model_agent else "PPO"
                else:
                    base_algorithm = "PPO"  # Default
                
                print(f"[Evaluation] Base model algorithm: {base_algorithm}")
                
                # Get base model's lambda from base model snapshot (for frozen base model)
                base_model_lambda = 0.0
                if base_snapshot.metrics_at_save and 'lambda' in base_snapshot.metrics_at_save:
                    base_model_lambda = float(base_snapshot.metrics_at_save['lambda'])
                print(f"[Evaluation] Base model λ (frozen): {base_model_lambda:.4f}")
                
                # Get K-factor and adaptive_k from residual connector
                k_factor = getattr(residual_connector, 'k_factor', 0.15)
                adaptive_k = getattr(residual_connector, 'adaptive_k', False)
                enable_k_conf = getattr(residual_connector, 'enable_k_conf', True)
                enable_k_risk = getattr(residual_connector, 'enable_k_risk', True)
                print(f"[Evaluation] K-factor: {k_factor}, Adaptive K: {adaptive_k}")
                if adaptive_k:
                    print(f"[Evaluation] CARS Ablation: K_conf={enable_k_conf}, K_risk={enable_k_risk}")

                env = ResidualActionWrapper(
                    env=env,
                    base_model_path=base_snapshot.file_path,
                    base_algorithm=base_algorithm,
                    lagrange_state=lagrange_state,  # For CARS K_risk
                    base_lambda_frozen=base_model_lambda,
                    k_factor=k_factor,
                    adaptive_k=adaptive_k,
                    # CARS: Ablation control
                    enable_k_conf=enable_k_conf,
                    enable_k_risk=enable_k_risk,
                )
                if adaptive_k:
                    print(f"[Evaluation] CARS enabled")
                print(f"[Evaluation] Environment wrapped with ResidualActionWrapper")
            
            # STEP 3: Apply AppendLambdaWrapper LAST (before loading model)
            env = AppendLambdaWrapper(env, lagrange_state)
            print(f"[Evaluation] Wrapped environment with AppendLambdaWrapper (λ={stored_lambda:.4f})")

            
            print(f"[Evaluation] Final observation space: {env.observation_space}")

            # Load the appropriate model
            if experiment.training_mode == "residual" and residual_snapshot_id:
                # Load residual policy model
                try:
                    model = PPO.load(residual_snapshot.file_path, env=env)
                    print(f"[Evaluation] Loaded residual PPO model from {residual_snapshot.file_path}")
                except Exception:
                    try:
                        model = SAC.load(residual_snapshot.file_path, env=env)
                        print(f"[Evaluation] Loaded residual SAC model from {residual_snapshot.file_path}")
                    except Exception as e:
                        raise ValueError(f"Failed to load residual model: {e}")
            else:
                # Load standard model
                try:
                    model = PPO.load(base_snapshot.file_path, env=env)
                    print(f"[Evaluation] Loaded PPO model")
                except Exception:
                    try:
                        model = SAC.load(base_snapshot.file_path, env=env)
                        print(f"[Evaluation] Loaded SAC model")
                    except Exception as e:
                        raise ValueError(f"Failed to load model: {e}")

            # Initialize metrics tracking (similar to ExperimentCallback)
            episode_rewards = []
            episode_lengths = []
            episode_successes = []
            episode_crashes = []
            episode_timeouts = []
            episode_costs = []
            episode_near_misses = []
            episode_danger_time = []
            residual_magnitudes = []
            base_magnitudes = []
            
            # NEW: Per-step data collection for advanced Residual metrics (Group 3)
            all_cosine_sims = []        # Cosine similarity between base and residual actions
            all_intervention_flags = [] # Boolean flags for significant residual interventions
            all_effective_k = []        # Effective K values
            
            # CARS K components for charting
            all_k_conf = []
            all_k_risk = []
            
            # Trajectory storage for replay
            trajectory = []
            
            # Run evaluation episodes
            total_episodes = experiment.evaluation_episodes
            fps_delay_seconds = (experiment.fps_delay or 0) / 1000.0
            
            for episode in range(total_episodes):
                if cancel_event.is_set():
                    print(f"[Evaluation] Cancelled at episode {episode}")
                    break
                
                obs, info = env.reset()
                done = False
                truncated = False
                step = 0
                episode_reward = 0.0
                episode_cost = 0.0
                episode_near_miss_count = 0
                episode_danger_count = 0
                episode_frames = []
                
                print(f"[Evaluation] Episode {episode + 1}/{total_episodes}")
                
                # Get the base environment for position data (unwrap wrappers)
                base_env = env
                while hasattr(base_env, 'env'):
                    base_env = base_env.env
                
                while not (done or truncated) and not cancel_event.is_set():
                    # Use deterministic predictions for reproducible evaluation
                    action, _ = model.predict(obs, deterministic=True)
                    
                    obs, reward, done, truncated, info = env.step(action)
                    
                    episode_reward += reward
                    step += 1
                    
                    # Track safety metrics if available
                    if 'cost' in info:
                        episode_cost += info['cost']
                    
                    if 'near_miss' in info and info['near_miss']:
                        episode_near_miss_count += 1
                    if 'in_danger_zone' in info and info['in_danger_zone']:
                        episode_danger_count += 1
                    
                    # NEW: Extract residual info and calculate advanced metrics (Group 3)
                    # The ResidualActionWrapper adds 'residual_info' to the step info dict
                    if 'residual_info' in info:
                        res_info = info['residual_info']
                        
                        # Track magnitudes (also captured from env.predict, but ensure step-based)
                        if 'residual_magnitude' in res_info:
                            residual_magnitudes.append(res_info['residual_magnitude'])
                        if 'base_magnitude' in res_info:
                            base_magnitudes.append(res_info['base_magnitude'])
                        
                        # Extract action vectors for cosine similarity
                        if 'base_action' in res_info and 'residual_action' in res_info:
                            base_action = np.array(res_info['base_action'])
                            residual_action = np.array(res_info['residual_action'])
                            
                            # Compute cosine similarity: S = (a_base · a_res) / (||a_base|| ||a_res|| + eps)
                            base_norm = np.linalg.norm(base_action)
                            res_norm = np.linalg.norm(residual_action)
                            
                            if base_norm > 1e-8 and res_norm > 1e-8:
                                cos_sim = float(np.dot(base_action, residual_action) / (base_norm * res_norm))
                            else:
                                cos_sim = 0.0  # Default when vectors are near-zero
                            all_cosine_sims.append(cos_sim)
                            
                            # Check for significant intervention (residual magnitude > threshold)
                            intervention_threshold = 0.05
                            is_intervention = res_norm > intervention_threshold
                            all_intervention_flags.append(is_intervention)
                        
                        # Collect CARS K metrics for charting
                        if 'effective_k' in res_info:
                            all_effective_k.append(res_info['effective_k'])
                        if 'K_conf' in res_info:
                            all_k_conf.append(res_info['K_conf'])
                        if 'K_risk' in res_info:
                            all_k_risk.append(res_info['K_risk'])
                    
                    # Get agent position and velocity from base environment (same method as training callback)
                    agent_pos = [0, 0, 0]
                    agent_vel = [0, 0, 0]
                    agent_orient = [0, 0, 0]
                    goal_pos = [0, 0, 0]
                    obstacles_data = []
                    lidar_readings = []
                    lidar_hit_info = []
                    
                    try:
                        # Use drone_id (same as training callback), not agent_id
                        if hasattr(base_env, 'drone_id') and base_env.drone_id is not None:
                            pos, orient_quat = p.getBasePositionAndOrientation(base_env.drone_id, physicsClientId=base_env.physics_client)
                            vel, _ = p.getBaseVelocity(base_env.drone_id, physicsClientId=base_env.physics_client)
                            orient = p.getEulerFromQuaternion(orient_quat)
                            agent_pos = [float(pos[0]), float(pos[1]), float(pos[2])]
                            agent_vel = [float(vel[0]), float(vel[1]), float(vel[2])]
                            agent_orient = [float(orient[0]), float(orient[1]), float(orient[2])]
                        
                        # Get goal position from environment goal_position attribute if available
                        if hasattr(base_env, 'goal_position') and base_env.goal_position is not None:
                            goal_pos = [float(base_env.goal_position[0]), float(base_env.goal_position[1]), float(base_env.goal_position[2])]
                        elif hasattr(base_env, 'goal_id') and base_env.goal_id is not None:
                            goal_p, _ = p.getBasePositionAndOrientation(base_env.goal_id, physicsClientId=base_env.physics_client)
                            goal_pos = [float(goal_p[0]), float(goal_p[1]), float(goal_p[2])]
                    except Exception as e:
                        if step % 100 == 0:  # Log occasionally
                            print(f"[Evaluation] Could not get position from PyBullet at step {step}: {e}")
                    
                    # Get swarm drone positions (outside try-catch to ensure it runs)
                    try:
                        if hasattr(base_env, 'swarm_drones'):
                            for drone in base_env.swarm_drones:
                                d_pos, _ = p.getBasePositionAndOrientation(drone.id, physicsClientId=base_env.physics_client)
                                d_vel, _ = p.getBaseVelocity(drone.id, physicsClientId=base_env.physics_client)
                                obstacles_data.append({
                                    "x": float(d_pos[0]),
                                    "y": float(d_pos[1]),
                                    "z": float(d_pos[2]),
                                    "diameter": float(drone.diameter),
                                    "speed": float(drone.max_speed),
                                    "strategy": drone.strategy,
                                    "chaos": float(drone.chaos),
                                    "vx": float(d_vel[0]),
                                    "vy": float(d_vel[1]),
                                    "vz": float(d_vel[2])
                                })
                    except Exception as e:
                        if step % 100 == 0:
                            print(f"[Evaluation] Could not get swarm drone positions: {e}")
                        
                        # Get lidar readings from info
                        if 'lidar_readings' in info:
                            lidar_readings = [float(x) for x in info['lidar_readings']]
                        if 'lidar_hit_info' in info:
                            for hit in info['lidar_hit_info']:
                                distance, hit_type, obj_id, direction_vec = hit
                                lidar_hit_info.append({
                                    "distance": float(distance),
                                    "type": hit_type,
                                    "direction": [float(direction_vec[0]), float(direction_vec[1]), float(direction_vec[2])]
                                })
                    except Exception as e:
                        if step == 1:
                            print(f"[Evaluation] Warning: Could not get environment data: {e}")
                    
                    # Create full frame data for trajectory replay and live visualization
                    frame_data = {
                        'experiment_id': experiment_id,
                        'frame': step,
                        'episode': episode + 1,
                        'timestamp': time.time(),
                        'agent_position': {'x': agent_pos[0], 'y': agent_pos[1], 'z': agent_pos[2]},
                        'agent_velocity': {'x': agent_vel[0], 'y': agent_vel[1], 'z': agent_vel[2]},
                        'agent_orientation': {'roll': agent_orient[0], 'pitch': agent_orient[1], 'yaw': agent_orient[2]},
                        'target_position': {'x': goal_pos[0], 'y': goal_pos[1], 'z': goal_pos[2]},
                        'obstacles': obstacles_data,
                        'lidar_readings': lidar_readings,
                        'lidar_hit_info': lidar_hit_info,
                        'reward': float(reward),
                        'done': bool(done),
                        'success': bool(info.get('success', False)),
                        'crashed': bool(info.get('crashed', False)),  # Fixed: was 'crash', should be 'crashed'
                        'crash_type': str(info.get('crash_type', 'none')),
                    }
                    episode_frames.append(frame_data)
                    trajectory.append(frame_data)
                    
                    # Apply FPS delay for visualization
                    if fps_delay_seconds > 0:
                        time.sleep(fps_delay_seconds)
                    
                    # Broadcast frame data for live visualization (every step for smooth animation)
                    if fps_delay_seconds > 0 or step % 10 == 0:
                        loop.call_soon_threadsafe(
                            lambda fd=frame_data: asyncio.create_task(broadcast_func(experiment_id, {
                                "type": "simulation_frame",
                                "data": fd
                            }))
                        )
                    
                    # Broadcast progress and metrics periodically
                    if step % 50 == 0:
                        # Calculate comprehensive metrics using helper function
                        # Only use completed episodes (don't include current incomplete episode)
                        try:
                            metrics = self._calculate_evaluation_metrics(
                                episode_rewards,
                                episode_lengths,
                                episode_successes,
                                episode_crashes,
                                episode_timeouts,
                                episode_costs,
                                episode_near_misses,
                                episode_danger_time,
                                lagrange_state,
                                experiment.safety_constraint if hasattr(experiment, 'safety_constraint') else None,
                                residual_magnitudes if residual_magnitudes else None,
                                base_magnitudes if base_magnitudes else None,
                                all_cosine_sims if all_cosine_sims else None,
                                all_intervention_flags if all_intervention_flags else None,
                                None,  # episode_shield_intervention_counts not used
                                effective_k_values=all_effective_k if all_effective_k else None,
                                k_conf_values=all_k_conf if all_k_conf else None,
                                k_risk_values=all_k_risk if all_k_risk else None,
                                window_size=50  # Use recent 50 episodes for rolling metrics
                            )
                        except Exception as e:
                            print(f"[Evaluation] Error calculating metrics at step {step}: {e}")
                            import traceback
                            traceback.print_exc()
                            # Use minimal metrics on error
                            metrics = {
                                'reward_raw_mean': float(episode_reward) if 'episode_reward' in locals() else 0.0,
                                'mean_reward': float(episode_reward) if 'episode_reward' in locals() else 0.0,
                                'lambda': float(lagrange_state.lam) if lagrange_state else 0.0,
                            }
                        
                        progress_data = {
                            'experiment_id': experiment_id,
                            'step': 0,  # Not used for evaluation but required by frontend
                            'total_steps': 0,  # Not used for evaluation but required by frontend
                            'episode': episode + 1,
                            'total_episodes': total_episodes,
                            'metrics': metrics,
                            'status': 'In Progress',
                            'timestamp': time.time()
                        }
                        
                        # Send progress update via WebSocket using same pattern as training callback
                        try:
                            future = asyncio.run_coroutine_threadsafe(
                                broadcast_func(experiment_id, {
                                    "type": "progress",
                                    "data": progress_data
                                }),
                                loop
                            )
                            # Don't block waiting for result
                        except Exception as e:
                            if step % 500 == 0:  # Log occasionally
                                print(f"[Evaluation] WebSocket broadcast error at step {step}: {e}")
                # Determine episode outcome
                success = done and 'success' in info and info['success']
                crash = done and 'crashed' in info and info['crashed']  # Environment uses 'crashed' not 'crash'
                timeout = truncated and not done
                
                # Store episode results (after outcome determination)
                episode_rewards.append(episode_reward)
                episode_lengths.append(step)
                episode_successes.append(success)
                episode_crashes.append(crash)
                episode_timeouts.append(timeout)
                episode_costs.append(episode_cost)
                episode_near_misses.append(episode_near_miss_count)
                episode_danger_time.append(episode_danger_count)
                
                print(f"[Evaluation] Episode {episode + 1} complete: reward={episode_reward:.2f}, steps={step}, success={success}, crash={crash}, timeout={timeout}")
                
                # Update Lagrangian multiplier if safety constraint is enabled
                if experiment.safety_constraint and experiment.safety_constraint.get('enabled', False):
                    lagrange_state.update(episode_cost)
                
                # Calculate comprehensive metrics for this episode completion
                try:
                    metrics = self._calculate_evaluation_metrics(
                        episode_rewards,
                        episode_lengths,
                        episode_successes,
                        episode_crashes,
                        episode_timeouts,
                        episode_costs,
                        episode_near_misses,
                        episode_danger_time,
                        lagrange_state,
                        experiment.safety_constraint if hasattr(experiment, 'safety_constraint') else None,
                        residual_magnitudes if residual_magnitudes else None,
                        base_magnitudes if base_magnitudes else None,
                        all_cosine_sims if all_cosine_sims else None,
                        all_intervention_flags if all_intervention_flags else None,
                        None,  # episode_shield_intervention_counts not used
                        effective_k_values=all_effective_k if all_effective_k else None,
                        k_conf_values=all_k_conf if all_k_conf else None,
                        k_risk_values=all_k_risk if all_k_risk else None,
                        window_size=None  # Use all episodes for cumulative metrics
                    )
                except Exception as e:
                    print(f"[Evaluation] Error calculating metrics at episode {episode + 1}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Use minimal metrics on error
                    metrics = {
                        'reward_raw_mean': float(episode_reward),
                        'mean_reward': float(episode_reward),
                        'mean_ep_length': int(step),
                        'success_rate': round(float(np.mean(episode_successes)) * 100.0 * 2) / 2 if episode_successes else 0.0,
                        'crash_rate': round(float(np.mean(episode_crashes)) * 100.0 * 2) / 2 if episode_crashes else 0.0,
                        'timeout_rate': round(float(np.mean(episode_timeouts)) * 100.0 * 2) / 2 if episode_timeouts else 0.0,
                        'cost_mean': float(episode_cost),
                        'near_miss_mean': float(episode_near_miss_count),
                        'danger_time_mean': float(episode_danger_count),
                        'lambda': float(lagrange_state.lam) if lagrange_state else 0.0,
                    }
                
                # Save metrics to database for historical viewing
                # Use episode number as the "step" for evaluation mode
                metrics_to_save = metrics.copy()
                crud.create_metric(db, experiment_id, episode + 1, metrics_to_save)
                
                # Broadcast episode completion with proper structure
                episode_progress_data = {
                    'experiment_id': experiment_id,
                    'step': 0,  # Not used for evaluation
                    'total_steps': 0,  # Not used for evaluation  
                    'episode': episode + 1,
                    'total_episodes': total_episodes,
                    'metrics': metrics,
                    'status': 'In Progress',
                    'timestamp': time.time()
                }
                
                # Send episode completion via WebSocket using same pattern as training callback
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        broadcast_func(experiment_id, {
                            "type": "progress", 
                            "data": episode_progress_data
                        }),
                        loop
                    )
                    # Don't block waiting for result
                except Exception as e:
                    print(f"[Evaluation] Episode completion WebSocket broadcast error: {e}")
            
            # Evaluation completed - save trajectory and final metrics
            if trajectory:
                crud.update_experiment_trajectory(db, experiment_id, trajectory)
                print(f"[Evaluation] Saved trajectory with {len(trajectory)} frames")
            
            # Update experiment status
            final_status = "Cancelled" if cancel_event.is_set() else "Completed"
            crud.update_experiment_status(db, experiment_id, final_status)
            
            # Calculate final comprehensive metrics
            try:
                final_metrics = self._calculate_evaluation_metrics(
                    episode_rewards,
                    episode_lengths,
                    episode_successes,
                    episode_crashes,
                    episode_timeouts,
                    episode_costs,
                    episode_near_misses,
                    episode_danger_time,
                    lagrange_state,
                    experiment.safety_constraint if hasattr(experiment, 'safety_constraint') else None,
                    residual_magnitudes if residual_magnitudes else None,
                    base_magnitudes if base_magnitudes else None,
                    all_cosine_sims if all_cosine_sims else None,
                    all_intervention_flags if all_intervention_flags else None,
                    None,  # episode_shield_intervention_counts not used
                    effective_k_values=all_effective_k if all_effective_k else None,
                    k_conf_values=all_k_conf if all_k_conf else None,
                    k_risk_values=all_k_risk if all_k_risk else None,
                    window_size=None  # Use all episodes for final metrics
                )
                
                # Add legacy field names for compatibility
                final_metrics['avg_reward'] = final_metrics.get('reward_raw_mean', 0.0)
                final_metrics['avg_cost'] = final_metrics.get('cost_mean', 0.0)
                final_metrics['avg_episode_length'] = final_metrics.get('mean_ep_length', 0.0)
            except Exception as e:
                print(f"[Evaluation] Error calculating final metrics: {e}")
                import traceback
                traceback.print_exc()
                # Use minimal metrics on error
                final_metrics = {
                    'reward_raw_mean': float(np.mean(episode_rewards)) if episode_rewards else 0.0,
                    'avg_reward': float(np.mean(episode_rewards)) if episode_rewards else 0.0,
                    'success_rate': round(float(np.mean(episode_successes)) * 100.0 * 2) / 2 if episode_successes else 0.0,
                    'crash_rate': round(float(np.mean(episode_crashes)) * 100.0 * 2) / 2 if episode_crashes else 0.0,
                    'timeout_rate': round(float(np.mean(episode_timeouts)) * 100.0 * 2) / 2 if episode_timeouts else 0.0,
                    'avg_cost': float(np.mean(episode_costs)) if episode_costs else 0.0,
                    'avg_episode_length': float(np.mean(episode_lengths)) if episode_lengths else 0.0,
                    'lambda': float(lagrange_state.lam) if lagrange_state else 0.0,
                }
            
            # Broadcast final completion with proper structure
            final_progress_data = {
                'experiment_id': experiment_id,
                'step': 0,  # Not used for evaluation
                'total_steps': 0,  # Not used for evaluation
                'episode': len(episode_rewards),
                'total_episodes': total_episodes,
                'metrics': final_metrics,
                'status': final_status,
                'timestamp': time.time()
            }
            
            # Send final completion via WebSocket using same pattern as training callback
            try:
                future = asyncio.run_coroutine_threadsafe(
                    broadcast_func(experiment_id, {
                        "type": "progress",
                        "data": final_progress_data
                    }),
                    loop
                )
                # Don't block waiting for result
            except Exception as e:
                print(f"[Evaluation] Final WebSocket broadcast error: {e}")
            
            print(f"[Evaluation] Completed {len(episode_rewards)} episodes")
            print(f"[Evaluation] Final status: {final_status}")
            print(f"[Evaluation] Final avg reward: {final_progress_data['metrics']['avg_reward']:.2f}")
            print(f"[Evaluation] Final success rate: {final_progress_data['metrics']['success_rate']:.2f}")
            
        except Exception as e:
            print(f"[Evaluation] Error: {e}")
            import traceback
            traceback.print_exc()
            crud.update_experiment_status(db, experiment_id, "Cancelled")
            raise
        finally:
            db.close()
            # Close environment to release PyBullet connection
            try:
                if 'env' in locals():
                    env.close()
                    print(f"[Evaluation] Closed PyBullet environment")
            except Exception as e:
                print(f"[Evaluation] Error closing environment: {e}")
