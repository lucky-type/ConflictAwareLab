"""Training management using Stable-Baselines3 PPO and PyBullet environments."""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import gymnasium as gym
import pybullet_envs  # noqa: F401 - registers PyBullet environments
from stable_baselines3 import PPO

from . import models

StatusCallback = Callable[[dict], None]


class TrainingManager:
    """Manage background PPO training jobs per experiment."""

    def __init__(self):
        self._threads: Dict[int, threading.Thread] = {}
        self._listeners: dict[int, List[StatusCallback]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, experiment_id: int, listener: StatusCallback) -> None:
        with self._lock:
            self._listeners[experiment_id].append(listener)

    def unsubscribe(self, experiment_id: int, listener: StatusCallback) -> None:
        with self._lock:
            if experiment_id in self._listeners and listener in self._listeners[experiment_id]:
                self._listeners[experiment_id].remove(listener)

    def _notify(self, experiment_id: int, payload: dict) -> None:
        listeners = list(self._listeners.get(experiment_id, []))
        for listener in listeners:
            listener(payload)

    def start_training(self, experiment: models.Experiment) -> None:
        with self._lock:
            if experiment.id in self._threads and self._threads[experiment.id].is_alive():
                raise RuntimeError("Training already running for this experiment")

            thread = threading.Thread(
                target=self._train_experiment, args=(experiment,), daemon=True
            )
            self._threads[experiment.id] = thread
            thread.start()

    def _train_experiment(self, experiment: models.Experiment) -> None:
        experiment_id = experiment.id
        total_timesteps = experiment.total_timesteps
        learning_rate = experiment.learning_rate

        self._notify(
            experiment_id,
            {
                "status": "starting",
                "progress": 0.0,
                "message": f"Initializing PPO with learning_rate={learning_rate}",
            },
        )

        env = gym.make("CartPoleBulletEnv-v1")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=learning_rate,
        )

        # Train in short chunks to provide incremental updates.
        chunk = max(1_000, total_timesteps // 10)
        steps_remaining = total_timesteps
        try:
            while steps_remaining > 0:
                current_chunk = min(chunk, steps_remaining)
                model.learn(total_timesteps=current_chunk)
                steps_remaining -= current_chunk
                progress = 1 - (steps_remaining / total_timesteps)
                self._notify(
                    experiment_id,
                    {
                        "status": "running",
                        "progress": progress,
                        "message": f"Completed {progress*100:.1f}%",
                    },
                )
                time.sleep(0.1)

            # Save the trained model for re-use.
            model.save(f"trained_model_{experiment_id}.zip")
            self._notify(
                experiment_id,
                {
                    "status": "finished",
                    "progress": 1.0,
                    "message": "Training completed",
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._notify(
                experiment_id,
                {
                    "status": "error",
                    "progress": 0.0,
                    "message": str(exc),
                },
            )
        finally:
            env.close()


training_manager = TrainingManager()
