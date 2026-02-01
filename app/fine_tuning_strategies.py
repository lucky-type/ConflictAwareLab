"""Fine-Tuning Strategy Implementation for Stable-Baselines3 (PPO, A2C, SAC, TD3, DDPG)

NOTE: Fine-tuning strategies (freezing layers) have been removed. 
All operations now default to full fine-tuning (training all parameters).
"""


def set_trainable(module, trainable: bool) -> None:
    """Set all parameters in a module to trainable or frozen."""
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = trainable


def rebuild_ppo_optimizer(model) -> None:
    """Rebuild optimizer to include only trainable parameters."""
    import torch as th
    
    # Get learning rate from schedule
    lr = model.lr_schedule(1.0) if hasattr(model, "lr_schedule") else 3e-4
    
    # Get only trainable parameters
    trainable_params = [p for p in model.policy.parameters() if p.requires_grad]
    if not trainable_params:
        # Should not happen in full fine-tune
        return
    
    # Get Adam epsilon if available
    adam_eps = getattr(model.policy, "adam_eps", 1e-5)
    
    # Create new optimizer with only trainable params
    model.policy.optimizer = th.optim.Adam(trainable_params, lr=lr, eps=adam_eps)
    
    print(f"[Fine-Tuning] Rebuilt optimizer with {len(trainable_params)} trainable parameter groups")


def rebuild_a2c_optimizer(model) -> None:
    """Rebuild A2C optimizer (uses RMSprop) to include only trainable parameters."""
    import torch as th
    
    # Get learning rate from schedule
    lr = model.lr_schedule(1.0) if hasattr(model, "lr_schedule") else 7e-4
    
    # Get only trainable parameters
    trainable_params = [p for p in model.policy.parameters() if p.requires_grad]
    if not trainable_params:
        return
    
    # Get RMSprop epsilon if available
    adam_eps = getattr(model.policy, "adam_eps", 1e-5)
    
    # Create new optimizer with only trainable params (A2C uses RMSprop by default)
    model.policy.optimizer = th.optim.RMSprop(trainable_params, lr=lr, eps=adam_eps)
    
    print(f"[Fine-Tuning A2C] Rebuilt RMSprop optimizer with {len(trainable_params)} trainable parameter groups")


def apply_ppo_fine_tuning_strategy(model, strategy: str) -> None:
    """
    Apply fine-tuning strategy to PPO model.
    Always uses full fine-tuning (all parameters trainable).
    """
    print(f"[Fine-Tuning PPO] Strategy requested: {strategy}. Defaulting to full fine-tune.")
    
    # Ensure all parameters are trainable
    set_trainable(model.policy, True)
    
    # Rebuild optimizer
    rebuild_ppo_optimizer(model)


def apply_a2c_fine_tuning_strategy(model, strategy: str) -> None:
    """
    Apply fine-tuning strategy to A2C model.
    Always uses full fine-tuning.
    """
    print(f"[Fine-Tuning A2C] Strategy requested: {strategy}. Defaulting to full fine-tune.")
    
    set_trainable(model.policy, True)
    rebuild_a2c_optimizer(model)


def apply_sac_fine_tuning_strategy(model, strategy: str) -> None:
    """
    Apply fine-tuning strategy to SAC model.
    Always uses full fine-tuning.
    """
    print(f"[Fine-Tuning SAC] Strategy requested: {strategy}. Defaulting to full fine-tune.")
    
    policy = model.policy
    set_trainable(policy, True)
    
    # Handle entropy coef if regularized
    log_ent_coef = getattr(model, "log_ent_coef", None)
    if log_ent_coef is not None:
        try:
            log_ent_coef.requires_grad_(True)
        except:
            pass


def apply_td3_fine_tuning_strategy(model, strategy: str) -> None:
    """
    Apply fine-tuning strategy to TD3 model.
    Always uses full fine-tuning.
    """
    print(f"[Fine-Tuning TD3] Strategy requested: {strategy}. Defaulting to full fine-tune.")
    set_trainable(model.policy, True)


def apply_ddpg_fine_tuning_strategy(model, strategy: str) -> None:
    """
    Apply fine-tuning strategy to DDPG model.
    Always uses full fine-tuning.
    """
    print(f"[Fine-Tuning DDPG] Strategy requested: {strategy}. Defaulting to full fine-tune.")
    set_trainable(model.policy, True)
    
    # Calculate parameter counts for logging
    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"  Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")


# ==================== Main Dispatcher ====================

def apply_fine_tuning_strategy(model, strategy: str) -> None:
    """
    Apply fine-tuning strategy to a model (PPO, SAC, TD3, DDPG, A2C).
    Automatically detects algorithm type and routes to appropriate handler.
    
    Args:
        model: Stable-Baselines3 model (PPO, SAC, TD3, DDPG, A2C)
        strategy: One of:
            - full_finetune (R0): Train all parameters
            - freeze_perception (R1): Freeze features_extractor only
            - freeze_value (R2): Freeze value/critic branch
            - freeze_policy (R3): Freeze policy/actor branch
            - freeze_perception_policy (R4): Freeze perception + policy
    """
    # Detect algorithm type from model class name
    algo_name = model.__class__.__name__.lower()
    
    if "ppo" in algo_name:
        # PPO uses Adam optimizer
        apply_ppo_fine_tuning_strategy(model, strategy)
    elif "a2c" in algo_name:
        # A2C uses RMSprop optimizer (similar architecture to PPO)
        apply_a2c_fine_tuning_strategy(model, strategy)
    elif "sac" in algo_name:
        apply_sac_fine_tuning_strategy(model, strategy)
    elif "td3" in algo_name:
        # TD3 has deterministic actor + twin critics (no entropy tuning)
        apply_td3_fine_tuning_strategy(model, strategy)
    elif "ddpg" in algo_name:
        # DDPG similar to TD3 but with single critic (legacy algorithm)
        apply_ddpg_fine_tuning_strategy(model, strategy)
    else:
        # Fallback: try PPO handler for unknown algorithms
        print(f"[Fine-Tuning] Unknown algorithm '{algo_name}', attempting PPO strategy handler")
        apply_ppo_fine_tuning_strategy(model, strategy)
