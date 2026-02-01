"""Custom Gym wrappers for advanced training modes."""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG
from typing import Any, Dict


class ResidualActionWrapper(gym.Wrapper):
    """
    Wrapper for Residual Learning mode with CARS (Conflict-Aware Residual Scaling).
    
    Loads a frozen base model and allows training a residual policy that learns
    delta actions (corrections) on top of the base model's actions.
    
    Final action = clip(base_action + K * residual_action, -1, 1)
    
    CARS computes K as a product of modular components:
        K = clip(K_base × K_conf × K_risk × K_shield × K_progress, K_min, K_max)
    
    Each component is optional - if not available, defaults to 1.0:
        - K_conf: Conflict-aware scaling based on cosine similarity
        - K_risk: Risk-aware scaling based on Lagrange λ
        - K_shield: Safety-aware scaling based on Shield intervention rate
        - K_progress: Progress-aware boost when base policy is stuck (placeholder)
    """
    
    def __init__(
        self,
        env: gym.Env,
        base_model_path: str,
        base_algorithm: str = "PPO",
        lagrange_state = None,
        base_lambda_frozen: float = None,
        k_factor: float = 0.15,
        k_min: float = 0.01,  # Absolute minimum K
        k_max: float = 1.0,   # Absolute maximum K
        # CARS: Conflict component (K_conf)
        adaptive_k: bool = False,  # Enable CARS
        k_min_ratio: float = 0.7,  # Minimum K as ratio of k_factor when conflicting (was 0.5 - too aggressive)
        conflict_tau: float = -0.3,  # Threshold for cos_sim (below this = conflict). Negative = only strong opposition triggers reduction
        conflict_gamma: float = 2.0,  # Sigmoid steepness (was 5.0 - too sharp). Lower = smoother transition
        k_conf_ema_alpha: float = 0.05,  # EMA alpha (was 0.1). Slower smoothing to avoid oscillation
        # CARS: Risk component (K_risk)
        risk_alpha: float = 0.05,  # Sensitivity: tanh(alpha * lambda)
        risk_boost_max: float = 0.5,  # Max boost for risk (e.g. 0.5 = +50% -> 1.5x)
        # SHIELDING COMMENTED OUT - CARS: Shield component (K_shield)
        # shield_wrapper = None,  # Reference to SafetyShieldWrapper (optional)
        # shield_beta: float = 0.5,  # Sensitivity: K_shield = 1 - beta * intervention_rate. Was 1.0 - too aggressive
        # CARS: Progress component (K_progress) - placeholder
        progress_boost: float = 1.2,  # Boost multiplier when stuck
        # CARS: Ablation control (for ablation studies)
        enable_k_conf: bool = True,  # Enable conflict-aware component
        enable_k_risk: bool = True,  # Enable risk-aware component
        # SHIELDING COMMENTED OUT
        # enable_k_shield: bool = True,  # Enable shield-aware component
    ):
        """
        Initialize the Residual Action Wrapper with CARS.
        
        Args:
            env: The base Gym environment
            base_model_path: Path to the saved base model (.zip file)
            base_algorithm: Algorithm type ("PPO", "A2C", "SAC", "TD3", "DDPG")
            lagrange_state: LagrangeState object for residual policy safety constraint
            base_lambda_frozen: Frozen lambda value for base model
            k_factor: Base residual action scaling factor (K_base)
            k_min: Absolute minimum K value
            k_max: Absolute maximum K value
            adaptive_k: If True, enable CARS (dynamic K based on context)
            k_min_ratio: Minimum K ratio when fully conflicting
            conflict_tau: Cosine similarity threshold (below = conflict)
            conflict_gamma: Sigmoid steepness for K_conf transition
            risk_alpha: λ sensitivity for K_risk
            risk_boost_max: Max boost factor for K_risk (additive)
            # SHIELDING COMMENTED OUT
            # shield_wrapper: Optional SafetyShieldWrapper reference for K_shield
            # shield_beta: Intervention rate sensitivity for K_shield
            progress_boost: K multiplier when base policy stuck (placeholder)
            enable_k_conf: Enable conflict-aware component (for ablation studies)
            enable_k_risk: Enable risk-aware component (for ablation studies)
            # SHIELDING COMMENTED OUT
            # enable_k_shield: Enable shield-aware component (for ablation studies)
        """
        super().__init__(env)
        
        self.base_model_path = base_model_path
        self.base_algorithm = base_algorithm
        self.lagrange_state = lagrange_state
        self.base_lambda_frozen = base_lambda_frozen if base_lambda_frozen is not None else 0.0
        
        # CARS parameters
        self.k_factor = k_factor
        self.k_min = k_min
        self.k_max = k_max
        self.adaptive_k = adaptive_k
        self.k_min_ratio = k_min_ratio
        self.conflict_tau = conflict_tau
        self.conflict_gamma = conflict_gamma
        self.risk_alpha = risk_alpha
        self.risk_boost_max = risk_boost_max
        # SHIELDING COMMENTED OUT
        # self.shield_wrapper = shield_wrapper
        # self.shield_beta = shield_beta
        self.progress_boost = progress_boost

        # CARS Ablation control
        self.enable_k_conf = enable_k_conf
        self.enable_k_risk = enable_k_risk
        # SHIELDING COMMENTED OUT
        # self.enable_k_shield = enable_k_shield

        # EMA initialization
        self.k_conf_ema_alpha = k_conf_ema_alpha
        self._ema_k_conf = 1.0  # Initial state
        
        # Tracking for CARS statistics
        self._effective_k_sum = 0.0
        self._effective_k_count = 0
        self._last_effective_k = k_factor
        self._last_cos_sim = 0.0
        self._last_K_conf = 1.0
        self._last_K_risk = 1.0
        self._last_K_shield = 1.0
        self._last_K_progress = 1.0
        
        print(f"[ResidualWrapper] === CARS Configuration ===")
        print(f"[ResidualWrapper] K_base (k_factor): {self.k_factor}")
        print(f"[ResidualWrapper] CARS enabled: {self.adaptive_k}")
        if self.adaptive_k:
            print(f"[ResidualWrapper]   K_conf: {'ENABLED' if enable_k_conf else 'DISABLED'} | tau={self.conflict_tau}, gamma={self.conflict_gamma}, min_ratio={self.k_min_ratio}")
            print(f"[ResidualWrapper]   K_risk: {'ENABLED' if enable_k_risk else 'DISABLED'} | alpha={self.risk_alpha}, λ_source={'LagrangeState' if lagrange_state else 'None'}")
            # SHIELDING COMMENTED OUT
            # print(f"[ResidualWrapper]   K_shield: {'ENABLED' if enable_k_shield else 'DISABLED'} | beta={self.shield_beta}, shield={'Connected' if shield_wrapper else 'None'}")
            print(f"[ResidualWrapper]   K bounds: [{self.k_min}, {self.k_max}]")
        print(f"[ResidualWrapper] Base model λ (frozen): {self.base_lambda_frozen:.4f}")
        if lagrange_state:
            print(f"[ResidualWrapper] Residual policy λ (adaptive): {lagrange_state.lam:.4f}")
        
        # Load the base model (frozen - will be used for inference only)
        algorithm_map = {
            "PPO": PPO,
            "A2C": A2C,
            "SAC": SAC,
            "TD3": TD3,
            "DDPG": DDPG,
        }
        
        if base_algorithm not in algorithm_map:
            raise ValueError(f"Unsupported base algorithm: {base_algorithm}. "
                           f"Supported: {list(algorithm_map.keys())}")
        
        AlgorithmClass = algorithm_map[base_algorithm]
        
        print(f"[ResidualWrapper] Loading base model from {base_model_path}")
        print(f"[ResidualWrapper] Algorithm: {base_algorithm}")
        
        # Load the base model without setting env (we'll use it for prediction only)
        # This keeps it frozen and independent
        self.base_model = AlgorithmClass.load(base_model_path)
        
        # Check if base model expects lambda in observation
        self.base_expects_lambda = False
        if hasattr(self.base_model.observation_space, 'shape'):
            base_dim = self.base_model.observation_space.shape[0]
            env_dim = env.observation_space.shape[0]
            if base_dim == env_dim + 1:
                self.base_expects_lambda = True
                print(f"[ResidualWrapper] Base model expects Lambda appended (Dim: {base_dim} vs {env_dim})")
                print(f"[ResidualWrapper] Will append lambda to observation for base model prediction")
        
        print(f"[ResidualWrapper] Base model loaded successfully")
        print(f"[ResidualWrapper] Base model will provide actions deterministically")
        print(f"[ResidualWrapper] Residual policy will learn corrections (delta actions)")
        
        # Store base action for analysis/logging
        self.last_base_action = None
        self.last_residual_action = None
        self.last_final_action = None
    
    def step(self, residual_action: np.ndarray):
        """
        Execute a step with residual action.
        
        Args:
            residual_action: The action from the residual policy (delta action)
        
        Returns:
            observation, reward, done, truncated, info
        """
        # Get the current observation (stored from last reset/step)
        obs = self._last_obs

        # Residual policy obs dim = base_env_dim + 1 (lambda from AppendLambdaWrapper), so strip it.
        base_env_obs_dim = self.env.observation_space.shape[0]  # Inner env dim

        if obs.shape[0] > base_env_obs_dim:
            # obs has lambda appended - strip it
            obs_without_lambda = obs[:base_env_obs_dim]
        else:
            obs_without_lambda = obs

        # Now prepare for base model
        # Use frozen lambda for base model so its behavior matches training.
        if self.base_expects_lambda:
            # Append the frozen lambda expected by the base policy.
            base_input_obs = np.append(obs_without_lambda, self.base_lambda_frozen).astype(np.float32)
        else:
            base_input_obs = obs_without_lambda

        # Get base action from frozen base model (deterministic)
        base_action, _ = self.base_model.predict(base_input_obs, deterministic=True)
        
        # Ensure both are numpy arrays
        base_action = np.array(base_action, dtype=np.float32)
        residual_action = np.array(residual_action, dtype=np.float32)
        
        # Calculate cosine similarity for adaptive K and logging
        base_mag = float(np.linalg.norm(base_action))
        res_mag = float(np.linalg.norm(residual_action))
        
        if base_mag > 1e-6 and res_mag > 1e-6:
            cos_sim = float(np.dot(base_action, residual_action) / (base_mag * res_mag))
        else:
            cos_sim = 0.0
        
        self._last_cos_sim = cos_sim
        
        # =============================================
        # CARS: Conflict-Aware Residual Scaling
        # K = clip(K_base × K_conf × K_risk × K_shield × K_progress, K_min, K_max)
        # =============================================
        
        if self.adaptive_k:
            # --- K_conf: Conflict-aware scaling (INVERTED Logic) ---
            # OLD Logic: Conflict -> Reduce K (Bad for safety)
            # NEW Logic: Conflict -> Increase K (Trust Residual/Safety more)
            #
            # Range: Scales from k_factor (at no conflict) up to k_max (at high conflict)
            K_conf = 1.0  # Default multiplier
            if self.enable_k_conf:
                # Sigmoid input: gamma * (tau - cos_sim)
                # If cos_sim < tau (Conflict): input is positive -> sigmoid -> 1.0
                # If cos_sim > tau (Alignment): input is negative -> sigmoid -> 0.0
                sigmoid_input = self.conflict_gamma * (self.conflict_tau - cos_sim)
                sigmoid_val = 1.0 / (1.0 + np.exp(-sigmoid_input))
                
                # Target K value driven by conflict
                target_k_conf = self.k_factor + (self.k_max - self.k_factor) * sigmoid_val
                
                # Convert to multiplier (so it integrates with the product formula below)
                if self.k_factor > 1e-6:
                    K_conf = target_k_conf / self.k_factor
                else:
                    K_conf = 1.0

            # --- K_risk: Risk-aware scaling (INVERTED Logic) ---
            # OLD Logic: High Lambda -> Lower K (Conservative)
            # NEW Logic: High Lambda -> Higher K (Emergency Override)
            #
            # When historical constraints are violated (high lambda), the base agent 
            # is proven unsafe. We boost the residual agent to take control.
            # K_risk = 1.0 + boost_max * tanh(alpha * lambda)
            K_risk = 1.0  # Default: no boost
            if self.enable_k_risk and self.lagrange_state is not None:
                current_lambda = getattr(self.lagrange_state, 'lam', 0.0)
                if current_lambda > 0:
                    # Tanh scales lambda to [0, 1], multiplied by boost_max
                    # e.g., if boost_max=0.5, K_risk can reach 1.5
                    boost = self.risk_boost_max * np.tanh(self.risk_alpha * current_lambda)
                    K_risk = 1.0 + boost

            # SHIELDING COMMENTED OUT - K_shield: Safety-aware scaling from Shield interventions
            # Higher intervention rate = residual is "destabilizing" = lower K
            K_shield = 1.0  # Default: no scaling (shielding disabled)
            # if self.enable_k_shield and self.shield_wrapper is not None:
            #     shield_total_steps = getattr(self.shield_wrapper, 'total_steps', 0)
            #     shield_interventions = getattr(self.shield_wrapper, 'intervention_count', 0)
            #     if shield_total_steps > 0:
            #         intervention_rate = shield_interventions / shield_total_steps
            #         K_shield = max(0.1, 1.0 - self.shield_beta * intervention_rate)
            
            # --- K_progress: Progress-aware boost (currently fixed at 1.0) ---
            K_progress = 1.0
            
            # --- EMA Smoothing for K_conf ---
            # Smooth out the jittery conflict signal to stabilize learning
            # K_smooth = alpha * K_new + (1-alpha) * K_old
            self._ema_k_conf = self.k_conf_ema_alpha * K_conf + (1.0 - self.k_conf_ema_alpha) * self._ema_k_conf
            K_conf = self._ema_k_conf
            
            # --- Final K computation ---
            effective_k = self.k_factor * K_conf * K_risk * K_shield * K_progress
            effective_k = np.clip(effective_k, self.k_min, self.k_max)
        else:
            # CARS disabled: use static k_factor
            K_conf = 1.0
            K_risk = 1.0
            K_shield = 1.0  # SHIELDING COMMENTED OUT - always 1.0
            K_progress = 1.0
            effective_k = self.k_factor
            # Reset EMA state when disabled
            self._ema_k_conf = 1.0
        
        # Store CARS component values for logging
        self._last_K_conf = K_conf
        self._last_K_risk = K_risk
        self._last_K_shield = K_shield
        self._last_K_progress = K_progress
        self._last_effective_k = effective_k
        self._effective_k_sum += effective_k
        self._effective_k_count += 1
        
        # Combine: final_action = base_action + (effective_K * residual_action)
        combined_action = base_action + (effective_k * residual_action)
        
        # Clip to valid action space bounds
        if isinstance(self.action_space, gym.spaces.Box):
            final_action = np.clip(
                combined_action,
                self.action_space.low,
                self.action_space.high
            )
        else:
            final_action = combined_action
        
        # Store for analysis/logging
        self.last_base_action = base_action
        self.last_residual_action = residual_action
        self.last_final_action = final_action
        self.last_effective_k = effective_k
        
        # Increment step counter for diagnostic logging
        if not hasattr(self, '_step_count'):
            self._step_count = 0
            self._total_residual_magnitude = 0.0
            self._total_base_magnitude = 0.0
            self._conflict_count = 0  # Times residual opposes base
        
        self._step_count += 1
        res_mag = float(np.linalg.norm(residual_action))
        base_mag = float(np.linalg.norm(base_action))
        self._total_residual_magnitude += res_mag
        self._total_base_magnitude += base_mag
        
        # Check if residual opposes base action (cosine similarity < 0)
        if base_mag > 1e-6 and res_mag > 1e-6:
            cos_sim = float(np.dot(base_action, residual_action) / (base_mag * res_mag))
            if cos_sim < -0.3:  # Significantly opposing
                self._conflict_count += 1
        
        # Log diagnostics every 100 steps (extended with CARS components)
        if self._step_count % 100 == 0:
            avg_res = self._total_residual_magnitude / self._step_count
            avg_base = self._total_base_magnitude / self._step_count
            conflict_rate = (self._conflict_count / self._step_count) * 100
            avg_effective_k = self._effective_k_sum / max(1, self._effective_k_count)
            
            if self.adaptive_k:
                # SHIELDING COMMENTED OUT - removed shield from logging
                print(f"[CARS] Step {self._step_count}: "
                      f"K={avg_effective_k:.3f} (conf={K_conf:.2f} risk={K_risk:.2f}), "
                      f"conflict_rate={conflict_rate:.1f}%")
            else:
                print(f"[ResidualDiag] Step {self._step_count}: "
                      f"avg_res_mag={avg_res:.3f}, avg_base_mag={avg_base:.3f}, "
                      f"K*res={self.k_factor * avg_res:.3f}, conflict_rate={conflict_rate:.1f}%")
        
        # Execute the combined action in the environment
        obs, reward, done, truncated, info = self.env.step(final_action)
        
        # Store observation for next step
        self._last_obs = obs
        
        # Add residual-specific info for analysis (extended with CARS components)
        avg_effective_k = self._effective_k_sum / max(1, self._effective_k_count)
        info['residual_info'] = {
            'base_action': base_action.tolist(),
            'residual_action': residual_action.tolist(),
            'final_action': final_action.tolist(),
            'residual_magnitude': res_mag,
            'base_magnitude': base_mag,
            'cos_sim': cos_sim,
            'effective_k': effective_k,
            'avg_effective_k': avg_effective_k,
            'adaptive_k_enabled': self.adaptive_k,
            # CARS components
            'K_conf': K_conf,
            'K_risk': K_risk,
            # SHIELDING COMMENTED OUT
            # 'K_shield': K_shield,
            'K_progress': K_progress,
            'cars_enabled': self.adaptive_k,
        }
        
        return obs, reward, done, truncated, info
    
    def reset(self, **kwargs):
        """Reset the environment and store initial observation."""
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        
        # Reset action tracking
        self.last_base_action = None
        self.last_residual_action = None
        self.last_final_action = None
        
        return obs, info


def initialize_residual_policy_weights(model: Any, initial_log_std: float = -2.0):
    """
    Zero-initialize the residual policy's action network weights.
    
    This ensures the residual policy starts by outputting near-zero actions,
    making the combined policy initially equivalent to the base model.
    
    Args:
        model: The SB3 model (PPO, A2C, etc.)
        initial_log_std: Initial log std for action distribution (default: -2.0 for low exploration)
    """
    import torch
    
    print("[ResidualInit] Initializing residual policy with zero weights...")
    
    zeroed_layers = []
    
    try:
        # For on-policy algorithms (PPO, A2C) with MlpPolicy
        if hasattr(model.policy, 'action_net'):
            # Zero out the final action layer
            with torch.no_grad():
                if model.policy.action_net.weight is not None:
                    model.policy.action_net.weight.fill_(0.0)
                    print(f"[ResidualInit] ✅ Zeroed action_net weights: {model.policy.action_net.weight.shape}")
                    zeroed_layers.append("action_net.weight")
                
                if model.policy.action_net.bias is not None:
                    model.policy.action_net.bias.fill_(0.0)
                    print(f"[ResidualInit] ✅ Zeroed action_net bias: {model.policy.action_net.bias.shape}")
                    zeroed_layers.append("action_net.bias")
            
            # Reduce log_std for low initial exploration
            if hasattr(model.policy, 'log_std'):
                with torch.no_grad():
                    model.policy.log_std.fill_(initial_log_std)
                    print(f"[ResidualInit] ✅ Set log_std to {initial_log_std} (low exploration)")
                    zeroed_layers.append("log_std")
            
            # Run a test prediction to confirm zero-initialized residual outputs.
            test_obs = np.zeros(model.observation_space.shape, dtype=np.float32)
            test_action, _ = model.predict(test_obs, deterministic=True)
            mag = np.linalg.norm(test_action)
            print(f"[ResidualInit] Verification: Magnitude = {mag:.6f}")
            if mag > 0.01:
                print("⚠️ WARNING: Residual model is NOT zeroed! Actions may be unstable initially.")
            else:
                print("✅ Verification passed: Residual outputs near-zero actions")
        
        # For off-policy algorithms (SAC, TD3, DDPG) with actor
        elif hasattr(model.policy, 'actor'):
            actor = model.policy.actor
            
            # Zero out the final layer of the actor network
            if hasattr(actor, 'mu'):  # TD3/DDPG style
                with torch.no_grad():
                    if actor.mu.weight is not None:
                        actor.mu.weight.fill_(0.0)
                        print(f"[ResidualInit] ✅ Zeroed actor.mu weights: {actor.mu.weight.shape}")
                        zeroed_layers.append("actor.mu.weight")
                    
                    if actor.mu.bias is not None:
                        actor.mu.bias.fill_(0.0)
                        print(f"[ResidualInit] ✅ Zeroed actor.mu bias: {actor.mu.bias.shape}")
                        zeroed_layers.append("actor.mu.bias")
            
            elif hasattr(actor, 'latent_pi'):  # SAC style
                # For SAC, the final layer is after latent_pi
                # Find the output layer
                layers = list(actor.children())
                if len(layers) > 0:
                    final_layer = layers[-1]
                    if hasattr(final_layer, 'weight'):
                        with torch.no_grad():
                            final_layer.weight.fill_(0.0)
                            print(f"[ResidualInit] ✅ Zeroed final actor layer weights: {final_layer.weight.shape}")
                            zeroed_layers.append("actor.final_layer.weight")
                        
                        if hasattr(final_layer, 'bias') and final_layer.bias is not None:
                            final_layer.bias.fill_(0.0)
                            print(f"[ResidualInit] ✅ Zeroed final actor layer bias: {final_layer.bias.shape}")
                            zeroed_layers.append("actor.final_layer.bias")
            
            # Fallback: dynamically find and zero the last linear layer.
            if not zeroed_layers:
                print("[ResidualInit] ⚠️  Standard attributes (mu, latent_pi) not found, attempting dynamic search...")
                
                # Try to find the last Linear layer in the actor
                last_linear = None
                for module in reversed(list(actor.modules())):
                    if isinstance(module, torch.nn.Linear):
                        last_linear = module
                        break
                
                if last_linear is not None:
                    with torch.no_grad():
                        if last_linear.weight is not None:
                            last_linear.weight.fill_(0.0)
                            print(f"[ResidualInit] ✅ Dynamically zeroed last Linear layer weights: {last_linear.weight.shape}")
                            zeroed_layers.append("actor.last_linear.weight (dynamic)")
                        
                        if last_linear.bias is not None:
                            last_linear.bias.fill_(0.0)
                            print(f"[ResidualInit] ✅ Dynamically zeroed last Linear layer bias: {last_linear.bias.shape}")
                            zeroed_layers.append("actor.last_linear.bias (dynamic)")
                else:
                    print("[ResidualInit] ⚠️  Could not find any Linear layer in actor")
        
        # Summary
        if zeroed_layers:
            print(f"[ResidualInit] ✅ Residual policy initialized successfully")
            print(f"[ResidualInit] Zeroed layers: {', '.join(zeroed_layers)}")
            print(f"[ResidualInit] Initial actions will be ~0, making combined policy ≈ base model")
        else:
            print(f"[ResidualInit] ⚠️  Warning: No layers were zeroed (unknown policy structure)")
            print(f"[ResidualInit] Proceeding with random initialization (may cause initial instability)")
        
    except Exception as e:
        print(f"[ResidualInit] ⚠️  Error during initialization: {e}")
        print(f"[ResidualInit] Zeroed layers before error: {zeroed_layers}")
        print(f"[ResidualInit] Proceeding with partial/random initialization (may cause initial instability)")
