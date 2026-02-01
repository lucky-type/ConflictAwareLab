"""Pydantic schemas used by the FastAPI endpoints."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .models import ExperimentStatus, ExperimentType


# ==================== Environments ====================
class ObstacleConfig(BaseModel):
    type: str = Field(..., description="Obstacle type")
    diameter: float = Field(..., gt=0)
    speed: float = Field(..., ge=0)
    acceleration: float = Field(1.5, gt=0, description="Maximum acceleration factor (multiplied by speed)")
    strategy: str = Field(..., description="Movement strategy: linear, circular, zigzag")
    chaos: float = Field(..., ge=0, le=1, description="Chaos factor 0-1")


class EnvironmentBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    length: float = Field(..., gt=0, description="Corridor length in meters")
    width: float = Field(..., gt=0, description="Corridor width in meters")
    height: float = Field(..., gt=0, description="Corridor height in meters")
    buffer_start: float = Field(0.1, ge=0, le=1, description="Start zone as percentage")
    buffer_end: float = Field(0.9, ge=0, le=1, description="End zone as percentage")
    target_diameter: float = Field(..., gt=0)
    obstacles: List[ObstacleConfig] = Field(default_factory=list)
    is_predicted: bool = Field(False, description="Enable deterministic seeding based on created_at + episode")


class EnvironmentCreate(EnvironmentBase):
    pass


class EnvironmentUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    length: Optional[float] = Field(None, gt=0)
    width: Optional[float] = Field(None, gt=0)
    height: Optional[float] = Field(None, gt=0)
    buffer_start: Optional[float] = Field(None, ge=0, le=1)
    buffer_end: Optional[float] = Field(None, ge=0, le=1)
    target_diameter: Optional[float] = Field(None, gt=0)
    obstacles: Optional[List[ObstacleConfig]] = None
    is_predicted: Optional[bool] = None


class Environment(EnvironmentBase):
    id: int
    is_locked: bool
    created_at: dt.datetime

    class Config:
        from_attributes = True


# ==================== Agents ====================
class AgentBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    type: str = Field(..., description="Algorithm type: PPO, SAC, DQN, etc.")
    agent_diameter: float = Field(..., gt=0, description="Agent diameter in meters")
    agent_max_speed: float = Field(..., gt=0, description="Maximum speed in m/s")
    agent_max_acceleration: float = Field(5.0, gt=0, description="Maximum acceleration in m/s² (multiplied by max_speed)")
    lidar_range: float = Field(..., gt=0, description="Lidar range in meters")
    lidar_rays: int = Field(36, gt=0, description="Number of lidar rays for sensing")
    kinematic_type: str = Field("holonomic", description="Kinematic model: holonomic or semi-holonomic")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameters as JSON")


class AgentCreate(AgentBase):
    pass


class AgentUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    type: Optional[str] = None
    agent_diameter: Optional[float] = Field(None, gt=0)
    agent_max_speed: Optional[float] = Field(None, gt=0)
    agent_max_acceleration: Optional[float] = Field(None, gt=0)
    lidar_range: Optional[float] = Field(None, gt=0)
    lidar_rays: Optional[int] = Field(None, gt=0)
    kinematic_type: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Agent(AgentBase):
    id: int
    is_locked: bool
    created_at: dt.datetime

    class Config:
        from_attributes = True


# ==================== Residual Connectors ====================
class ResidualConnectorBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    parent_agent_id: int = Field(..., description="Parent agent ID to inherit physical specs from")
    algorithm: str = Field(..., description="Algorithm for residual learner: PPO, A2C, etc.")
    net_arch: List[int] = Field(default_factory=lambda: [64, 64], description="Neural network architecture")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Residual algorithm hyperparameters")
    k_factor: float = Field(0.15, gt=0, le=1.0, description="K-factor for residual correction scaling (typically 0.05-0.5)")
    adaptive_k: bool = Field(False, description="Enable adaptive K-factor that reduces K when residual opposes base action")
    enable_k_conf: bool = Field(True, description="Enable conflict-aware scaling component (K_conf)")
    enable_k_risk: bool = Field(True, description="Enable risk-aware scaling component (K_risk)")
    # SHIELDING COMMENTED OUT
    # enable_k_shield: bool = Field(True, description="Enable shield-aware scaling component (K_shield)")


class ResidualConnectorCreate(ResidualConnectorBase):
    pass


class ResidualConnectorUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    algorithm: Optional[str] = None
    net_arch: Optional[List[int]] = None
    parameters: Optional[Dict[str, Any]] = None
    k_factor: Optional[float] = Field(None, gt=0, le=1.0, description="K-factor for residual correction scaling")
    adaptive_k: Optional[bool] = Field(None, description="Enable adaptive K-factor")
    enable_k_conf: Optional[bool] = Field(None, description="Enable conflict-aware scaling component (K_conf)")
    enable_k_risk: Optional[bool] = Field(None, description="Enable risk-aware scaling component (K_risk)")
    # SHIELDING COMMENTED OUT
    # enable_k_shield: Optional[bool] = Field(None, description="Enable shield-aware scaling component (K_shield)")


class ResidualConnector(ResidualConnectorBase):
    id: int
    is_locked: bool
    created_at: dt.datetime
    parent_agent: Optional['Agent'] = None

    class Config:
        from_attributes = True


# ==================== Reward Functions ====================
class RewardFunctionBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    w_progress: float = Field(..., gt=0, description="Weight for progress component (must be positive)")
    w_time: float = Field(..., ge=0, description="Weight for time penalty (0 to disable)")
    w_jerk: float = Field(..., ge=0, description="Weight for jerk penalty (0 to disable)")
    success_reward: float = Field(..., gt=0, description="Bonus for reaching goal")
    crash_reward: float = Field(..., lt=0, description="Penalty for collision (negative)")


class RewardFunctionCreate(RewardFunctionBase):
    pass


class RewardFunctionUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    w_progress: Optional[float] = Field(None, gt=0)
    w_time: Optional[float] = Field(None, ge=0)
    w_jerk: Optional[float] = Field(None, ge=0)
    success_reward: Optional[float] = Field(None, gt=0)
    crash_reward: Optional[float] = Field(None, lt=0)


class RewardFunction(RewardFunctionBase):
    id: int
    is_locked: bool
    created_at: dt.datetime

    class Config:
        from_attributes = True


# ==================== Experiments ====================
class SafetyConstraintConfig(BaseModel):
    """Safety constraint configuration for Lagrangian / Risk-aware RL."""
    enabled: bool = Field(False, description="Enable safety constraint")
    risk_budget: float = Field(0.5, gt=0, description="Risk budget ε (max cumulative cost per episode, e.g., 0.5 ensures crashes with cost>=1.0 violate constraint)")
    initial_lambda: float = Field(0.0, ge=0, description="Initial Lagrangian multiplier λ")
    lambda_learning_rate: float = Field(0.02, gt=0, description="Learning rate α for λ updates")
    update_frequency: int = Field(50, gt=0, description="Update λ every N episodes")
    
    # Lambda warmup schedule parameters
    target_lambda: Optional[float] = Field(
        None, 
        ge=0, 
        description="Target λ for warmup schedule. If None, no warmup cap is applied."
    )
    warmup_episodes: int = Field(
        0, 
        ge=0, 
        description="Number of episodes for λ warmup. If 0, warmup is disabled."
    )
    warmup_schedule: str = Field(
        "exponential", 
        description="Warmup schedule type: 'linear' or 'exponential'"
    )
    
    cost_signal_preset: str = Field("balanced", description="Preset: balanced, strict_safety, near_miss_focused, collision_only")
    collision_weight: float = Field(1.0, ge=0, description="Weight for collision cost (cumulative formulation: 1.0 means crash = 1.0 cost, violating risk_budget=0.5)")
    near_miss_weight: float = Field(0.1, ge=0, description="Weight for near-miss cost (per-step, 0.1 = 0.001 cost per step when in near-miss zone)")
    danger_zone_weight: float = Field(0.05, ge=0, description="Weight for danger zone cost (per-step, 0.05 = 0.0005 cost per step when in danger zone)")
    near_miss_threshold: float = Field(1.5, gt=0, description="Distance threshold for near-miss (× agent diameter)")
    ignore_walls: bool = Field(True, description="If True, walls are completely ignored (no crashes, no proximity costs) - recommended for corridor environments")
    wall_near_miss_weight: float = Field(0.005, ge=0, description="Weight for wall near-miss cost (per-step, active only when ignore_walls=False)")
    wall_danger_zone_weight: float = Field(0.01, ge=0, description="Weight for wall danger zone cost (per-step, additive, active only when ignore_walls=False)")
    wall_near_miss_threshold: float = Field(1.0, gt=0, description="Distance threshold for wall near-miss in meters")
    

    # SHIELDING COMMENTED OUT - Safety Shield (runtime action filtering)
    # shield_enabled: bool = Field(False, description="Enable runtime Safety Shield that blocks dangerous actions")
    # shield_threshold: float = Field(0.3, gt=0, le=1.0, description="Min LIDAR distance (normalized 0-1) to trigger shield for DRONES. Lower = more aggressive")
    # shield_wall_threshold: float = Field(0.1, gt=0, le=1.0, description="Min LIDAR distance (normalized 0-1) to trigger shield for WALLS. Lower = more aggressive")
    # shield_fallback_strength: float = Field(0.8, gt=0, le=1.0, description="Strength of retreat action when shield activates (0-1)")



class ExperimentBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    type: ExperimentType
    env_id: int
    agent_id: int
    reward_id: int
    total_steps: int = Field(..., gt=0)
    snapshot_freq: int = Field(..., gt=0, description="Save model every N steps")
    max_ep_length: int = Field(2000, gt=0, description="Max steps per episode before truncation")
    model_snapshot_id: Optional[int] = Field(None, description="Model snapshot for simulations")
    base_model_snapshot_id: Optional[int] = Field(None, description="Base model snapshot for fine-tuning")
    fine_tuning_strategy: Optional[str] = Field("full_finetune", description="Fine-tuning strategy: full_finetune")
    training_mode: Optional[str] = Field("standard", description="Training mode: standard or residual")
    residual_base_model_id: Optional[int] = Field(None, description="Base model snapshot for residual learning (frozen)")
    residual_connector_id: Optional[int] = Field(None, description="Residual connector defining residual policy architecture and algorithm")
    safety_constraint: Optional[SafetyConstraintConfig] = Field(None, description="Safety constraint configuration")
    evaluation_episodes: Optional[int] = Field(None, gt=0, description="Number of episodes for evaluation mode")
    fps_delay: Optional[float] = Field(None, ge=0, description="Delay in milliseconds between steps for visualization")
    seed: Optional[int] = Field(None, ge=0, description="Random seed for reproducibility. If None, a random seed will be used.")


class ExperimentCreate(ExperimentBase):
    pass


class ExperimentUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    status: Optional[ExperimentStatus] = None
    total_steps: Optional[int] = Field(None, gt=0)
    snapshot_freq: Optional[int] = Field(None, gt=0)
    evaluation_episodes: Optional[int] = Field(None, gt=0)
    fps_delay: Optional[float] = Field(None, ge=0)
    seed: Optional[int] = Field(None, ge=0)


class ExperimentSummary(ExperimentBase):
    """
    Lightweight experiment schema for list views.
    Excludes heavy fields like trajectory_data.
    """
    id: int
    status: ExperimentStatus
    current_step: int = 0
    created_at: dt.datetime
    environment: Optional['Environment'] = None
    agent: Optional['Agent'] = None
    reward_function: Optional['RewardFunction'] = None
    base_model_snapshot: Optional['ModelSnapshot'] = None
    residual_base_model: Optional['ModelSnapshot'] = None
    residual_connector: Optional['ResidualConnector'] = None

    class Config:
        from_attributes = True


class Experiment(ExperimentBase):
    id: int
    status: ExperimentStatus
    current_step: int = 0
    trajectory_data: Optional[List[Dict[str, Any]]] = None
    created_at: dt.datetime
    environment: Optional['Environment'] = None  # Include environment with full details
    agent: Optional['Agent'] = None  # Include agent details
    reward_function: Optional['RewardFunction'] = None  # Include reward function details
    base_model_snapshot: Optional['ModelSnapshot'] = None  # Include base model for fine-tuning
    residual_base_model: Optional['ModelSnapshot'] = None  # Include base model for residual learning

    class Config:
        from_attributes = True


# ==================== Model Snapshots ====================
class ModelSnapshotBase(BaseModel):
    iteration: int = Field(..., ge=0)
    file_path: str
    metrics_at_save: Dict[str, Any] = Field(default_factory=dict)


class ModelSnapshotCreate(ModelSnapshotBase):
    experiment_id: int


class ModelSnapshot(ModelSnapshotBase):
    id: int
    experiment_id: int
    created_at: dt.datetime

    class Config:
        from_attributes = True


# ==================== Experiment Metrics ====================
class ExperimentMetricBase(BaseModel):
    step: int = Field(..., ge=0)
    values: Dict[str, Any] = Field(default_factory=dict)
    logs: Optional[str] = None


class ExperimentMetricCreate(ExperimentMetricBase):
    experiment_id: int


class ExperimentMetric(ExperimentMetricBase):
    id: int
    experiment_id: int
    created_at: dt.datetime

    class Config:
        from_attributes = True


# ==================== Training & Telemetry ====================
class TrainingRequest(BaseModel):
    experiment_id: int
    resume: bool = False


class TrainingStatus(BaseModel):
    experiment_id: int
    status: str
    progress: float
    message: Optional[str] = None


class TelemetryMessage(BaseModel):
    timestamp: float
    position: tuple[float, float, float]
    orientation: tuple[float, float, float]
    linear_velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]


# ==================== Simulation ====================
class SimulationFrame(BaseModel):
    experiment_id: int
    frame: int
    timestamp: float
    agent_position: Dict[str, float]
    agent_velocity: Optional[Dict[str, float]] = None
    agent_orientation: Optional[Dict[str, float]] = None
    target_position: Dict[str, float]
    obstacles: List[Dict[str, Any]] = Field(default_factory=list)
    lidar_readings: List[float] = Field(default_factory=list)
    reward: float = 0.0
    done: bool = False
    success: bool = False
    crashed: bool = False


# # ==================== Curriculum Learning ====================
# class CurriculumStepConfigBase(BaseModel):
#     """Configuration for a single curriculum step."""
#     env_id: int
#     agent_id: Optional[int] = None
#     residual_connector_id: Optional[int] = None
#     reward_id: int
#     total_steps: int = Field(100000, gt=0)
#     snapshot_freq: int = Field(10000, gt=0)
#     max_ep_length: int = Field(2000, gt=0)


# class CurriculumStepBase(BaseModel):
#     """Base schema for curriculum step."""
#     order: int = Field(..., ge=1, description="Step order in pipeline (1-indexed)")
#     step_type: str = Field(..., description="Step type: training, fine_tuning, or residual")
#     config: CurriculumStepConfigBase
#     safety_config: Optional[SafetyConstraintConfig] = None
#     target_success_rate: float = Field(0.95, ge=0, le=1, description="Success rate threshold to advance (0-1)")
#     min_episodes: int = Field(50, gt=0, description="Minimum episodes before checking transition")


# class CurriculumStepCreate(CurriculumStepBase):
#     """Schema for creating a curriculum step."""
#     pass


# class CurriculumStepRead(CurriculumStepBase):
#     """Schema for reading a curriculum step."""
#     id: int
#     curriculum_id: int
#     is_completed: bool

#     class Config:
#         from_attributes = True


# class CurriculumBase(BaseModel):
#     """Base schema for curriculum."""
#     name: str = Field(..., min_length=1, max_length=100)
#     description: Optional[str] = None


# class CurriculumCreate(CurriculumBase):
#     """Schema for creating a curriculum."""
#     steps: List[CurriculumStepCreate] = Field(..., min_items=1, description="List of sequential training steps")


# class CurriculumRead(CurriculumBase):
#     """Schema for reading a curriculum."""
#     id: int
#     status: str
#     current_step_order: int
#     created_at: dt.datetime
#     updated_at: Optional[dt.datetime] = None
#     steps: List[CurriculumStepRead] = []

#     class Config:
#         from_attributes = True


# Resolve forward references for nested models
Experiment.model_rebuild()
ResidualConnector.model_rebuild() 
