"""
Curriculum Orchestrator - manages multi-stage training pipelines.

This module implements a state machine that:
1. Monitors running curriculums
2. Starts experiments for each step
3. Links models between sequential steps
4. Checks transition conditions (success rate thresholds)
5. Advances to next step when criteria are met
"""
"""
Curriculum Orchestrator - manages multi-stage training pipelines.

This module implements a state machine that:
1. Monitors running curriculums
2. Starts experiments for each step
3. Links models between sequential steps
4. Checks transition conditions (success rate thresholds)
5. Advances to next step when criteria are met
"""
# from __future__ import annotations

# import logging
# from typing import TYPE_CHECKING, Optional

# import numpy as np
# from sqlalchemy.orm import Session

# from . import models
# from .database import SessionLocal

# if TYPE_CHECKING:
#     from .background_jobs import ExperimentRunner

# logger = logging.getLogger(__name__)


# class CurriculumOrchestrator:
#     """Manages sequential training pipelines with automatic model linking."""
    
#     def __init__(self, experiment_runner: Optional['ExperimentRunner'] = None, main_loop=None):
#         """
#         Initialize the orchestrator.
        
#         Args:
#             experiment_runner: Reference to the experiment runner for starting/stopping experiments
#             main_loop: The main application event loop for scheduling async tasks
#         """
#         self.experiment_runner = experiment_runner
#         self.main_loop = main_loop
#         logger.info("CurriculumOrchestrator initialized")
    
#     def set_experiment_runner(self, runner: 'ExperimentRunner'):
#         """Set the experiment runner reference (for dependency injection)."""
#         self.experiment_runner = runner
    
#     def tick(self):
#         """
#         Main state machine loop - called periodically by background thread.
#         Processes all running curriculums and manages their execution.
#         """
#         db = SessionLocal()
#         try:
#             # Fetch all running curriculums
#             running_curriculums = db.query(models.Curriculum).filter(
#                 models.Curriculum.status == models.CurriculumStatus.RUNNING
#             ).all()
            
#             for curriculum in running_curriculums:
#                 try:
#                     self._process_curriculum(db, curriculum)
#                 except Exception as e:
#                     logger.error(f"Error processing curriculum {curriculum.id} ({curriculum.name}): {e}")
#                     import traceback
#                     logger.error(traceback.format_exc())
#                     curriculum.status = models.CurriculumStatus.FAILED
#                     db.add(curriculum)
#                     db.commit()
#         except Exception as e:
#             logger.error(f"Error in orchestrator tick: {e}")
#         finally:
#             db.close()
    
#     def _process_curriculum(self, db: Session, curriculum: models.Curriculum):
#         """Process a single curriculum - check active step and manage transitions."""
#         # Get the current active step
#         active_step = db.query(models.CurriculumStep).filter(
#             models.CurriculumStep.curriculum_id == curriculum.id,
#             models.CurriculumStep.order == curriculum.current_step_order
#         ).first()
        
#         if not active_step:
#             # No more steps - curriculum is complete
#             logger.info(f"Curriculum {curriculum.id} ({curriculum.name}) has no more steps. Marking as completed.")
#             curriculum.status = models.CurriculumStatus.COMPLETED
#             db.add(curriculum)
#             db.commit()
#             return
        
#         # Refresh to get the latest experiment relationship
#         db.refresh(active_step)
#         logger.debug(f"Processing curriculum {curriculum.id}, step {active_step.order}, has experiment: {active_step.experiment is not None}")
        
#         # Check if step has an experiment running
#         if not active_step.experiment:
#             # Step hasn't started yet - create and start experiment
#             self._start_step(db, curriculum, active_step)
#         else:
#             # Check if experiment failed or was cancelled
#             if active_step.experiment.status == models.ExperimentStatus.CANCELLED:
#                 logger.error(f"Experiment {active_step.experiment.id} was cancelled for curriculum {curriculum.id}")
#                 curriculum.status = models.CurriculumStatus.FAILED
#                 db.add(curriculum)
#                 db.commit()
#                 return
            
#             # Step is running - check if it should transition to next step
#             self._check_transition(db, curriculum, active_step)
    
#     def _start_step(self, db: Session, curriculum: models.Curriculum, step: models.CurriculumStep):
#         """
#         Create and start an experiment for a curriculum step.
#         Links to previous step's best model if step > 1.
#         """
#         logger.info(f"Starting step {step.order} for curriculum {curriculum.id} ({curriculum.name})")
        
#         # Build experiment configuration
#         config = step.config.copy()
#         experiment_name = f"{curriculum.name}_Step{step.order}"
        
#         # Validate required configuration
#         if not config.get('env_id'):
#             raise ValueError(f"Step {step.order}: Missing environment ID")
#         if not config.get('reward_id'):
#             raise ValueError(f"Step {step.order}: Missing reward function ID")
        
#         # Determine experiment type based on step type
#         if step.step_type == "training":
#             experiment_type = models.ExperimentType.TRAINING
#             training_mode = models.TrainingMode.STANDARD
#             if not config.get('agent_id'):
#                 raise ValueError(f"Step {step.order}: Training requires an agent ID")
#         elif step.step_type == "fine_tuning":
#             experiment_type = models.ExperimentType.FINE_TUNING
#             training_mode = models.TrainingMode.STANDARD
#             if not config.get('agent_id'):
#                 raise ValueError(f"Step {step.order}: Fine-tuning requires an agent ID")
#         elif step.step_type == "residual":
#             experiment_type = models.ExperimentType.TRAINING
#             training_mode = models.TrainingMode.RESIDUAL
#             if not config.get('residual_connector_id'):
#                 raise ValueError(f"Step {step.order}: Residual learning requires a connector ID")
#         else:
#             raise ValueError(f"Unknown step type: {step.step_type}")
        
#         # Verify that all required entities exist in database
#         env = db.query(models.Environment).get(config.get("env_id"))
#         if not env:
#             raise ValueError(f"Step {step.order}: Environment with ID {config.get('env_id')} not found")
        
#         reward = db.query(models.RewardFunction).get(config.get("reward_id"))
#         if not reward:
#             raise ValueError(f"Step {step.order}: Reward function with ID {config.get('reward_id')} not found")
        
#         if config.get('agent_id'):
#             agent = db.query(models.Agent).get(config.get("agent_id"))
#             if not agent:
#                 raise ValueError(f"Step {step.order}: Agent with ID {config.get('agent_id')} not found")
        
#         if config.get('residual_connector_id'):
#             connector = db.query(models.ResidualConnector).get(config.get("residual_connector_id"))
#             if not connector:
#                 raise ValueError(f"Step {step.order}: Residual connector with ID {config.get('residual_connector_id')} not found")
        
#         # Link to previous step's best model if not the first step
#         base_model_snapshot_id = None
#         residual_base_model_id = None
        
#         if step.order > 1:
#             # Find the previous step
#             prev_step = db.query(models.CurriculumStep).filter(
#                 models.CurriculumStep.curriculum_id == curriculum.id,
#                 models.CurriculumStep.order == step.order - 1
#             ).first()
            
#             if prev_step and prev_step.experiment:
#                 # Select the best snapshot from the previous step
#                 best_snapshot = self._select_best_snapshot(db, prev_step.experiment.id)
                
#                 if best_snapshot:
#                     logger.info(f"Linking step {step.order} to snapshot {best_snapshot.id} from step {prev_step.order}")
                    
#                     # Both fine-tuning and residual use base_model_snapshot_id
#                     if step.step_type == "fine_tuning" or step.step_type == "residual":
#                         base_model_snapshot_id = best_snapshot.id
#                         if step.step_type == "residual":
#                             residual_base_model_id = best_snapshot.id
                    
#                     logger.info(f"Set base_model_snapshot_id={base_model_snapshot_id} for {step.step_type}")
#                 else:
#                     logger.warning(f"No snapshot found for previous step {prev_step.order}")
        
#         # Create the experiment
#         experiment = models.Experiment(
#             name=experiment_name,
#             type=experiment_type,
#             env_id=config.get("env_id"),
#             agent_id=config.get("agent_id"),
#             reward_id=config.get("reward_id"),
#             total_steps=config.get("total_steps", 100000),
#             snapshot_freq=config.get("snapshot_freq", 10000),
#             max_ep_length=config.get("max_ep_length", 2000),
#             training_mode=training_mode,
#             base_model_snapshot_id=base_model_snapshot_id,
#             residual_base_model_id=residual_base_model_id,
#             residual_connector_id=config.get("residual_connector_id"),
#             safety_constraint=step.safety_config,
#             curriculum_step_id=step.id,
#             status=models.ExperimentStatus.IN_PROGRESS
#         )
        
#         db.add(experiment)
#         db.commit()
#         db.refresh(experiment)
        
#         logger.info(f"Created experiment {experiment.id} for curriculum step {step.id}")
        
#         # Lock related entities
#         env = db.query(models.Environment).get(experiment.env_id)
#         if env:
#             env.is_locked = True
#             db.add(env)
        
#         if experiment.agent_id:
#             agent = db.query(models.Agent).get(experiment.agent_id)
#             if agent:
#                 agent.is_locked = True
#                 db.add(agent)
        
#         if experiment.residual_connector_id:
#             connector = db.query(models.ResidualConnector).get(experiment.residual_connector_id)
#             if connector:
#                 connector.is_locked = True
#                 db.add(connector)
        
#         reward = db.query(models.RewardFunction).get(experiment.reward_id)
#         if reward:
#             reward.is_locked = True
#             db.add(reward)
        
#         db.commit()
        
#         # Start the experiment using the experiment runner
#         if not self.experiment_runner:
#             logger.error("Experiment runner not available - cannot start experiment")
#             raise RuntimeError("Experiment runner not available")
        
#         logger.info(f"Attempting to start experiment {experiment.id} for curriculum step {step.id}")
        
#         import asyncio
#         from .websocket import connection_manager
        
#         # Use the main event loop that was passed during initialization
#         if not self.main_loop:
#             logger.error("No main event loop available")
#             raise RuntimeError("No event loop available for curriculum orchestrator")
        
#         logger.info(f"Using main event loop: {type(self.main_loop)}")
        
#         # Schedule the experiment start on the main event loop
#         try:
#             future = asyncio.run_coroutine_threadsafe(
#                 self.experiment_runner.start_training(experiment, db, connection_manager.broadcast),
#                 self.main_loop
#             )
#             # Wait briefly to ensure it starts (but don't block the orchestrator)
#             try:
#                 future.result(timeout=2.0)
#                 logger.info(f"Successfully started experiment {experiment.id} for curriculum step {step.id}")
#             except TimeoutError:
#                 # This is actually OK - the experiment is running in the background
#                 logger.info(f"Experiment {experiment.id} start initiated (running in background)")
#         except Exception as e:
#             logger.error(f"Failed to start experiment {experiment.id}: {e}")
#             import traceback
#             logger.error(traceback.format_exc())
#             # Mark experiment as cancelled
#             experiment.status = models.ExperimentStatus.CANCELLED
#             db.add(experiment)
#             db.commit()
#             raise
    
#     def _check_transition(self, db: Session, curriculum: models.Curriculum, step: models.CurriculumStep):
#         """
#         Check if the current step meets transition criteria.
#         If yes, stop the experiment and advance to next step.
#         """
#         if not step.experiment:
#             return
        
#         experiment = step.experiment
#         if not experiment:
#             return
        
#         # If experiment already completed naturally, advance to next step
#         if experiment.status == models.ExperimentStatus.COMPLETED:
#             logger.info(f"Step {step.order} experiment completed naturally")
            
#             # Mark step as completed
#             step.is_completed = True
#             db.add(step)
            
#             # Check if this is the last step
#             total_steps = db.query(models.CurriculumStep).filter(
#                 models.CurriculumStep.curriculum_id == curriculum.id
#             ).count()
            
#             if step.order >= total_steps:
#                 # This was the last step - mark curriculum as completed
#                 curriculum.status = models.CurriculumStatus.COMPLETED
#                 logger.info(f"Curriculum {curriculum.id} completed all {total_steps} steps")
#             else:
#                 # Advance to next step
#                 curriculum.current_step_order += 1
#                 logger.info(f"Curriculum {curriculum.id} advanced to step {curriculum.current_step_order}")
            
#             curriculum.updated_at = models.dt.datetime.utcnow()
#             db.add(curriculum)
#             db.commit()
#             return
        
#         # Only check transition criteria if experiment is still in progress
#         if experiment.status != models.ExperimentStatus.IN_PROGRESS:
#             return
        
#         # Ensure minimum 10k steps before checking transition criteria
#         # This prevents premature transitions when success rate is 100% immediately
#         MIN_STEPS_BEFORE_TRANSITION = 10000
#         current_step = experiment.current_step or 0
        
#         if current_step < MIN_STEPS_BEFORE_TRANSITION:
#             logger.debug(
#                 f"Curriculum {curriculum.id} Step {step.order}: Waiting for min steps "
#                 f"({current_step}/{MIN_STEPS_BEFORE_TRANSITION}) before checking transition"
#             )
#             return
        
#         # Calculate moving average success rate
#         success_rate = self._calculate_moving_average_success(db, experiment.id, window=step.min_episodes)
        
#         logger.debug(
#             f"Curriculum {curriculum.id} Step {step.order}: Current success rate: {success_rate:.2%}, "
#             f"Target: {step.target_success_rate:.2%}, Min episodes: {step.min_episodes}, Steps: {current_step}"
#         )
        
#         if success_rate >= step.target_success_rate:
#             logger.info(
#                 f"Step {step.order} reached target success rate: {success_rate:.2%} >= {step.target_success_rate:.2%}"
#             )
            
#             # Stop the experiment
#             if self.experiment_runner:
#                 # Use the main loop to await the async cancel_experiment call
#                 if self.main_loop:
#                     import asyncio
#                     asyncio.run_coroutine_threadsafe(
#                         self.experiment_runner.cancel_experiment(experiment.id), 
#                         self.main_loop
#                     )
#                 else:
#                     # Fallback: set the cancel event directly
#                     if experiment.id in self.experiment_runner.cancel_events:
#                         self.experiment_runner.cancel_events[experiment.id].set()
            
#             # Mark experiment as completed
#             experiment.status = models.ExperimentStatus.COMPLETED
#             db.add(experiment)
            
#             # Mark step as completed
#             step.is_completed = True
#             db.add(step)
            
#             # Check if this is the last step
#             total_steps = db.query(models.CurriculumStep).filter(
#                 models.CurriculumStep.curriculum_id == curriculum.id
#             ).count()
            
#             if step.order >= total_steps:
#                 # This was the last step - mark curriculum as completed
#                 curriculum.status = models.CurriculumStatus.COMPLETED
#                 logger.info(f"Curriculum {curriculum.id} completed all {total_steps} steps")
#             else:
#                 # Advance to next step
#                 curriculum.current_step_order += 1
#                 logger.info(f"Curriculum {curriculum.id} advanced to step {curriculum.current_step_order}")
            
#             curriculum.updated_at = models.dt.datetime.utcnow()
#             db.add(curriculum)
#             db.commit()
    
#     def _select_best_snapshot(self, db: Session, experiment_id: int) -> Optional[models.ModelSnapshot]:
#         """
#         Select the best snapshot from an experiment based on metrics.
        
#         Selection priority:
#         1. Primary: success_rate (higher is better)
#         2. Secondary: mean_reward / reward (for tie-breaking)
        
#         This ensures residual/fine-tuning starts from the most stable base,
#         not necessarily the highest reward (which may be unstable).
#         """
#         snapshots = db.query(models.ModelSnapshot).filter(
#             models.ModelSnapshot.experiment_id == experiment_id
#         ).all()
        
#         if not snapshots:
#             return None
        
#         def snapshot_score(s):
#             """Returns (success_rate, reward) tuple for sorting.
            
#             success_rate: Normalized to 0-1 (stored as 0-100 percentage)
#             reward: Used as secondary criterion for tie-breaking
#             """
#             metrics = s.metrics_at_save or {}
#             # success_rate stored as 0-100 percentage, normalize to 0-1
#             success_rate = metrics.get("success_rate", 0.0) / 100.0
#             reward = metrics.get("mean_reward", metrics.get("reward", -float("inf")))
#             return (success_rate, reward)
        
#         # Select snapshot with highest success_rate, using reward as tie-breaker
#         best_snapshot = max(snapshots, key=snapshot_score)
        
#         # Log selection for debugging
#         metrics = best_snapshot.metrics_at_save or {}
#         logger.info(
#             f"Selected best snapshot {best_snapshot.id} from experiment {experiment_id}: "
#             f"success_rate={metrics.get('success_rate', 0):.1f}%, "
#             f"reward={metrics.get('mean_reward', metrics.get('reward', 'N/A'))}"
#         )
        
#         return best_snapshot
    
#     def _calculate_moving_average_success(self, db: Session, experiment_id: int, window: int = 50) -> float:
#         """
#         Calculate success rate over the last N episodes.
        
#         Args:
#             db: Database session
#             experiment_id: ID of experiment to analyze
#             window: Number of recent episodes to consider
        
#         Returns:
#             Success rate as a float (0.0 to 1.0)
#         """
#         # CRITICAL FIX: Query enough metrics to cover the desired episode window
#         # Metrics are saved every ~100 steps (save_interval), but episodes can vary in length
#         # To ensure we capture at least 'window' episodes worth of data:
#         # - Average episode length is ~500-2000 steps (varies by difficulty)
#         # - Conservative estimate: fetch 10x window to ensure coverage
#         # - This prevents premature curriculum advancement from small sample sizes
#         query_limit = max(100, window * 25)
        
#         metrics = db.query(models.ExperimentMetric).filter(
#             models.ExperimentMetric.experiment_id == experiment_id
#         ).order_by(models.ExperimentMetric.step.desc()).limit(query_limit).all()
        
#         if not metrics:
#             logger.debug(f"No metrics found for experiment {experiment_id}")
#             return 0.0
        
#         # CRITICAL FIX: Collect ALL success_rate values and calculate the actual mean
#         success_rates = []
#         for metric in metrics:
#             values = metric.values or {}
#             if "success_rate" in values:
#                 # success_rate is stored as percentage (0-100), convert to 0-1
#                 rate = values["success_rate"] / 100.0
#                 success_rates.append(rate)
        
#         if not success_rates:
#             logger.debug(f"No success_rate metric found for experiment {experiment_id}")
#             return 0.0
        
#         # Calculate arithmetic mean
#         mean_success_rate = float(np.mean(success_rates))
#         logger.info(f"Experiment {experiment_id} moving avg success_rate: {mean_success_rate:.3f} "
#                     f"(from {len(success_rates)} metrics, window={window}, queried={query_limit})")
#         return mean_success_rate


# # Global orchestrator instance (will be initialized by background_jobs.py)
# curriculum_orchestrator = None
