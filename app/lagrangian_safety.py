"""Lagrangian Safety Constraint Implementation for Risk-aware RL.

This module provides a unified implementation that works with all SB3 algorithms
(PPO, A2C, SAC, TD3, DDPG) by:
1. Using a VecEnv wrapper to append λ to observations
2. Using a VecEnv wrapper to shape rewards: r' = r - λ * cost
3. Using a Callback to update λ based on episode cost statistics
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


@dataclass
class LagrangeState:
    """Shared state for Lagrangian multiplier λ between wrapper and callback."""
    lam: float = 0.0  # Current Lagrangian multiplier


class AppendLambdaWrapper(gym.ObservationWrapper):
    """Wrapper that appends current λ value to observation.
    
    This ensures the observation space is consistent across all algorithms,
    whether using Lagrangian safety or not. For non-Lagrangian algorithms,
    λ will always be 0.0.
    """
    
    def __init__(self, env: gym.Env, state: LagrangeState):
        """
        Args:
            env: Base environment
            state: Shared LagrangeState object (may be updated by callback or stay at 0.0)
        """
        super().__init__(env)
        self.state = state
        
        # Extend observation space by 1 dimension for λ
        old_obs_space = env.observation_space
        if isinstance(old_obs_space, gym.spaces.Box):
            old_shape = old_obs_space.shape
            new_shape = (old_shape[0] + 1,)
            self.observation_space = gym.spaces.Box(
                low=np.append(old_obs_space.low, 0.0),  # λ >= 0
                high=np.append(old_obs_space.high, np.inf),  # λ unbounded
                shape=new_shape,
                dtype=old_obs_space.dtype
            )
        else:
            raise NotImplementedError("AppendLambdaWrapper only supports Box observation spaces")
    
    def observation(self, obs):
        """Append current λ to observation."""
        return np.append(obs, self.state.lam).astype(np.float32)
    

class LagrangianRewardVecWrapper(VecEnvWrapper):
    """VecEnv wrapper that shapes rewards using Lagrangian method: r' = r - λ * cost.
    
    Reads info["cost"] from each environment step and adjusts the reward.
    Works with any number of parallel environments.
    """
    
    def __init__(self, venv: VecEnv, state: LagrangeState, cost_key: str = "cost"):
        """
        Args:
            venv: Base vectorized environment
            state: Shared LagrangeState object (updated by callback)
            cost_key: Key in info dict containing the cost signal
        """
        super().__init__(venv)
        self.state = state
        self.cost_key = cost_key
        
    def step_wait(self):
        """Override step_wait to shape rewards with current λ."""
        obs, rewards, dones, infos = self.venv.step_wait()
        
        # Shape rewards: r' = r - λ * cost
        shaped_rewards = np.zeros_like(rewards)
        for i in range(len(rewards)):
            cost = infos[i].get(self.cost_key, 0.0)
            shaped_rewards[i] = rewards[i] - self.state.lam * cost
            infos[i]["lagrange_lambda"] = self.state.lam
            
        return obs, shaped_rewards, dones, infos
    
    def reset(self):
        """Reset environment."""
        return self.venv.reset()


class LagrangianCallback(BaseCallback):
    """Callback that updates λ based on CUMULATIVE episode cost statistics.
    
    Uses cumulative episode cost (not per-step average) when updating λ.
    
    Updates λ every N completed episodes using:
    λ ← max(0, λ + α * (C_bar - ε))
    
    where:
    - C_bar is the mean CUMULATIVE cost over the last N episodes
    - ε is the per-episode risk budget (e.g., 0.05 means max 5% episodes can have cost)
    - α is the learning rate for λ
    
    A crash = 1.0 cumulative cost, regardless of episode length.
    This ensures Lambda increases when crashes occur, not decreases.
    """
    
    def __init__(
        self,
        state: LagrangeState,
        epsilon: float,
        alpha: float,
        update_every_n_episodes: int,
        target_lambda: Optional[float] = None,
        warmup_episodes: int = 0,
        warmup_schedule: str = "exponential",
        verbose: int = 1,
    ):
        """
        Args:
            state: Shared LagrangeState object (also used by wrapper)
            epsilon: Per-episode risk budget ε (maximum acceptable cumulative cost, e.g., 0.05)
            alpha: Learning rate α for updating λ
            update_every_n_episodes: Update λ every N completed episodes
            target_lambda: Target λ for warmup schedule. If None, no warmup cap is applied.
            warmup_episodes: Number of episodes for λ warmup. If 0, warmup is disabled.
            warmup_schedule: Warmup schedule type: 'linear' or 'exponential'
            verbose: Verbosity level (0=silent, 1=info, 2=debug)
        """
        super().__init__(verbose)
        self.state = state
        self.epsilon = epsilon
        self.alpha = alpha
        self.update_every_n = update_every_n_episodes
        
        # Warmup parameters
        self.target_lambda = target_lambda
        self.warmup_episodes = warmup_episodes
        self.warmup_schedule = warmup_schedule
        
        # Track episode costs per parallel environment
        self.episode_costs = None  # Will be initialized in _on_training_start
        self.episode_lengths = None  # Track episode lengths for debugging
        self.completed_episode_costs = []  # List of CUMULATIVE costs per episode
        self.episode_count = 0
        
        # SHIELDING COMMENTED OUT
        # Shield intervention tracking
        # self.episode_shield_interventions = None  # Per-env episode intervention count
        # self.total_shield_interventions = 0
        # self.total_shield_steps = 0
        
    def _on_training_start(self) -> None:
        """Initialize episode cost tracking for all parallel environments."""
        n_envs = self.training_env.num_envs
        
        # Only initialize if not already set (prevents reset on chunked training)
        if self.episode_costs is None:
            self.episode_costs = [0.0] * n_envs
            self.episode_lengths = [0] * n_envs
            # SHIELDING COMMENTED OUT
            # self.episode_shield_interventions = [0] * n_envs
            
            if self.verbose >= 1:
                print(f"\n[LagrangianCallback] Initialized for {n_envs} parallel environments")
                print(f"  Risk budget ε: {self.epsilon:.3f}")
                print(f"  λ learning rate α: {self.alpha:.3f}")
                print(f"  Update frequency: every {self.update_every_n} episodes")
                print(f"  Initial λ: {self.state.lam:.3f}")
                
                # Add warmup info
                if self.target_lambda is not None and self.warmup_episodes > 0:
                    print(f"  🔥 WARMUP ENABLED:")
                    print(f"     Target λ: {self.target_lambda:.1f}")
                    print(f"     Warmup episodes: {self.warmup_episodes}")
                    print(f"     Schedule: {self.warmup_schedule}")
                    
                    # Show warmup curve preview
                    milestones = [0, 10, 25, 50, 75, 100]
                    print(f"     Warmup preview:")
                    for pct in milestones:
                        ep = int(self.warmup_episodes * pct / 100)
                        if ep <= self.warmup_episodes:
                            cap = self._calculate_warmup_cap(ep)
                            print(f"       Episode {ep:3d} ({pct:3d}%): λ_cap = {cap:5.2f}")
                print()
    
    def _calculate_warmup_cap(self, episode_num: int) -> float:
        """
        Calculate λ cap for given episode during warmup.
        
        Returns:
            Maximum allowed λ at this episode (inf if warmup disabled/finished)
        """
        # No warmup configured
        if self.target_lambda is None or self.warmup_episodes == 0:
            return float('inf')
        
        # Warmup finished
        if episode_num >= self.warmup_episodes:
            return float('inf')
        
        # Calculate progress (0.0 to 1.0)
        progress = episode_num / self.warmup_episodes
        
        if self.warmup_schedule == "linear":
            # Linear: λ_cap grows linearly from 0 to target_λ
            # Formula: λ_cap = target_λ * progress
            lambda_cap = self.target_lambda * progress
        
        elif self.warmup_schedule == "exponential":
            # Exponential: fast growth early, slower near target
            # FIX: Normalize so it reaches exactly 100% at progress=1.0
            # Raw formula: f(x) = 1 - exp(-k * x), where k controls curve steepness
            # Normalized: f_norm(x) = (1 - exp(-k*x)) / (1 - exp(-k))
            # At x=1: f_norm(1) = 1.0 (exactly 100%)
            k = 3.0  # Steepness factor (higher = faster early growth)
            normalizer = 1.0 - np.exp(-k)  # ≈ 0.9502
            exp_factor = (1.0 - np.exp(-k * progress)) / normalizer
            lambda_cap = self.target_lambda * exp_factor
        
        else:
            raise ValueError(
                f"Unknown warmup_schedule: '{self.warmup_schedule}'. "
                f"Must be 'linear' or 'exponential'"
            )
        
        return lambda_cap
        
    def _on_step(self) -> bool:
        """Called after each environment step."""
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        
        for i, info in enumerate(infos):
            # 1. Get raw cost
            cost = info.get("cost", 0.0)
            
            # 2. Fix for "silent" violations: 
            # If ignore_walls=True, wall crashes have cost=0.0. 
            # We ONLY accumulate 'cost' which comes from drone_env.py logic.
            # (drone_env.py already handles the ignore_walls logic correctly now)
            violation_cost = cost
            
            self.episode_costs[i] += violation_cost
            self.episode_lengths[i] += 1
            
            # SHIELDING COMMENTED OUT
            # 3. Track shield interventions
            # if info.get("shield_intervention", False):
            #     self.episode_shield_interventions[i] += 1
            # self.total_shield_steps += 1
            
            if dones[i]:
                # 4. Use cumulative episode cost so a crash is 1.0 regardless of length.
                cumulative_cost = self.episode_costs[i]
                
                self.completed_episode_costs.append(cumulative_cost)
                self.episode_count += 1
                
                # SHIELDING COMMENTED OUT
                # Track shield interventions for this episode
                # self.total_shield_interventions += self.episode_shield_interventions[i]
                
                if self.verbose >= 2:
                    print(f"[SAFETY] Ep {self.episode_count}: Cost={cumulative_cost:.3f} Eps={self.epsilon}")
                
                self.episode_costs[i] = 0.0
                self.episode_lengths[i] = 0
                # SHIELDING COMMENTED OUT
                # self.episode_shield_interventions[i] = 0
                
                if self.episode_count % self.update_every_n == 0:
                    self._update_lambda()
        
        return True
    
    def _update_lambda(self) -> None:
        """Update λ based on recent CUMULATIVE episode costs."""
        if len(self.completed_episode_costs) == 0:
            return
        
        # Compute mean cumulative cost over last N episodes
        recent_costs = self.completed_episode_costs[-self.update_every_n:]
        mean_cost = np.mean(recent_costs)
        
        old_lambda = self.state.lam
        warmup_active = False
        lambda_scheduled = None
        
        # Check if warmup is active
        if self.target_lambda is not None and self.warmup_episodes > 0:
            if self.episode_count < self.warmup_episodes:
                # WARMUP MODE: Directly set λ according to warmup schedule
                warmup_active = True
                lambda_scheduled = self._calculate_warmup_cap(self.episode_count)
                new_lambda = lambda_scheduled
                delta = new_lambda - old_lambda  # For logging
            else:
                # Warmup finished, use normal adaptive updates
                delta = self.alpha * (mean_cost - self.epsilon)
                new_lambda = max(0.0, old_lambda + delta)
        else:
            # No warmup configured, use normal adaptive updates
            delta = self.alpha * (mean_cost - self.epsilon)
            new_lambda = max(0.0, old_lambda + delta)
        
        self.state.lam = new_lambda
        
        # TensorBoard logging
        self.logger.record("safety/lambda", self.state.lam)
        self.logger.record("safety/cost_mean_cumulative", mean_cost)
        self.logger.record("safety/cost_limit_epsilon", self.epsilon)
        self.logger.record("safety/lambda_delta", delta)
        
        # Log warmup metrics
        if warmup_active:
            self.logger.record("safety/warmup_lambda_scheduled", lambda_scheduled)
            self.logger.record("safety/warmup_progress", 
                             self.episode_count / self.warmup_episodes)
        
        # SHIELDING COMMENTED OUT
        # Log shield metrics
        # if self.total_shield_steps > 0:
        #     shield_rate = (self.total_shield_interventions / self.total_shield_steps) * 100
        #     self.logger.record("safety/shield_interventions_total", self.total_shield_interventions)
        #     self.logger.record("safety/shield_intervention_rate", shield_rate)
        
        # Console logging
        warmup_str = (f" [WARMUP {self.episode_count}/{self.warmup_episodes} eps, "
                      f"λ={lambda_scheduled:.2f}]" if warmup_active else "")
        
        print(f"[LAGRANGE]{warmup_str} C̄={mean_cost:.3f} ε={self.epsilon:.3f} "
              f"λ={old_lambda:.4f}→{self.state.lam:.4f} Δ={delta:.4f}")
        
        if self.verbose >= 1:
            print(f"\n{'='*60}")
            if warmup_active:
                print(f"[LagrangianCallback] λ UPDATE (WARMUP MODE - Episode {self.episode_count}/{self.warmup_episodes})")
                print(f"  Warmup progress: {100*self.episode_count/self.warmup_episodes:.1f}%")
                print(f"  λ_scheduled: {lambda_scheduled:.3f}")
                print(f"  λ set directly from warmup schedule (not from cost)")
            else:
                print(f"[LagrangianCallback] λ UPDATE (ADAPTIVE MODE - after {self.episode_count} episodes)")
                print(f"  C̄ (mean cumulative cost): {mean_cost:.3f}")
                print(f"  ε (risk budget): {self.epsilon:.3f}")
                print(f"  C̄ - ε: {mean_cost - self.epsilon:.3f}")
                print(f"  α (learning rate): {self.alpha:.4f}")
                print(f"  Δλ = α × (C̄ - ε): {delta:.4f}")
            print(f"{'='*60}")
            print(f"  λ_before: {old_lambda:.4f}")
            print(f"  λ_after: {self.state.lam:.4f}")
            if not warmup_active:
                print(f"  Recent costs: {[f'{c:.3f}' for c in recent_costs[-5:]]}")
            print(f"{'='*60}\n")
