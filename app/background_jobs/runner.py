"""Main experiment runner that manages background training, simulation, and evaluation."""
from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict

from .training import TrainingMixin
from .simulation import SimulationMixin
from .evaluation import EvaluationMixin

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from .. import models


class ExperimentRunner(TrainingMixin, SimulationMixin, EvaluationMixin):
    """Manages background training, simulation, and evaluation experiments."""

    def __init__(self):
        self.running_tasks: Dict[int, asyncio.Task] = {}
        self.paused_experiments: Dict[int, bool] = {}
        self.cancel_events: Dict[int, threading.Event] = {}  # For signaling cancellation to threads
        self.models_cache: Dict[int, Any] = {}
        self.envs_cache: Dict[int, Any] = {}
        # Thread pool for running blocking training operations
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.max_concurrent_jobs = 8

    def can_start_new_job(self) -> bool:
        """Check if there's capacity to start a new job."""
        return len(self.running_tasks) < self.max_concurrent_jobs

    def get_job_capacity_info(self) -> dict:
        """Get information about job capacity."""
        # Only count tasks that are actually running (not paused)
        active_count = sum(1 for exp_id in self.running_tasks.keys() 
                          if exp_id not in self.paused_experiments or not self.paused_experiments[exp_id])
        return {
            "running_jobs": active_count,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "available_slots": self.max_concurrent_jobs - active_count,
            "can_start_new": self.can_start_new_job()
        }

    async def cancel_experiment(self, experiment_id: int):
        """Cancel a running experiment."""
        if experiment_id in self.cancel_events:
            self.cancel_events[experiment_id].set()
            print(f"[ExperimentRunner] Cancellation signal sent to experiment {experiment_id}")

    async def pause_experiment(self, experiment_id: int):
        """Pause a running experiment."""
        if experiment_id in self.running_tasks:
            self.paused_experiments[experiment_id] = True
            print(f"[ExperimentRunner] Experiment {experiment_id} paused")

    async def resume_experiment(self, experiment_id: int):
        """Resume a paused experiment."""
        if experiment_id in self.running_tasks:
            self.paused_experiments[experiment_id] = False
            print(f"[ExperimentRunner] Experiment {experiment_id} resumed")

    def is_experiment_running(self, experiment_id: int) -> bool:
        """Check if an experiment is currently running."""
        return experiment_id in self.running_tasks

    def is_experiment_paused(self, experiment_id: int) -> bool:
        """Check if an experiment is currently paused."""
        return self.paused_experiments.get(experiment_id, False)

    def cleanup_experiment(self, experiment_id: int):
        """Clean up resources for a completed/cancelled experiment."""
        if experiment_id in self.running_tasks:
            del self.running_tasks[experiment_id]
        if experiment_id in self.paused_experiments:
            del self.paused_experiments[experiment_id]
        if experiment_id in self.cancel_events:
            del self.cancel_events[experiment_id]
        if experiment_id in self.models_cache:
            del self.models_cache[experiment_id]
        if experiment_id in self.envs_cache:
            try:
                self.envs_cache[experiment_id].close()
            except:
                pass
            del self.envs_cache[experiment_id]

    def clear_env_cache(self, experiment_id: int):
        """Clear cached environment for an experiment to force regeneration."""
        if experiment_id in self.envs_cache:
            try:
                self.envs_cache[experiment_id].close()
            except:
                pass
            del self.envs_cache[experiment_id]
            print(f"[ExperimentRunner] Cleared environment cache for experiment {experiment_id}")