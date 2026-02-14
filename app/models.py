"""SQLAlchemy models for the ConflictAwareLab domain."""
from __future__ import annotations

import datetime as dt
import enum

from sqlalchemy import Boolean, Column, DateTime, Enum, Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import relationship

from .database import Base


class ExperimentType(str, enum.Enum):
    TRAINING = "Training"
    SIMULATION = "Simulation"
    FINE_TUNING = "Fine-Tuning"
    EVALUATION = "Evaluation"


class TrainingMode(str, enum.Enum):
    STANDARD = "standard"
    RESIDUAL = "residual"


class EvaluationPolicyMode(str, enum.Enum):
    MODEL = "model"
    HEURISTIC = "heuristic"


class HeuristicAlgorithm(str, enum.Enum):
    POTENTIAL_FIELD = "potential_field"
    VFH_LITE = "vfh_lite"


class ExperimentStatus(str, enum.Enum):
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"
    PAUSED = "Paused"
    CANCELLED = "Cancelled"


# class CurriculumStatus(str, enum.Enum):
#     PENDING = "Pending"
#     RUNNING = "Running"
#     PAUSED = "Paused"
#     COMPLETED = "Completed"
#     FAILED = "Failed"


class Environment(Base):
    __tablename__ = "environments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    
    # Dimensions
    length = Column(Float, nullable=False)
    width = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
    
    # Boundaries (as percentages 0.0-1.0)
    buffer_start = Column(Float, nullable=False, default=0.1)  # 10%
    buffer_end = Column(Float, nullable=False, default=0.9)    # 90%
    
    # Target configuration
    target_diameter = Column(Float, nullable=False)
    
    # Obstacles stored as JSON array (using JSON for SQLite compatibility)
    obstacles = Column(JSON, nullable=False, default=list)
    
    # Locking mechanism
    is_locked = Column(Boolean, default=False, nullable=False)
    
    # Predicted mode: when enabled, random generation is seeded with created_at + episode number
    # This ensures reproducible scenarios across different experiments on the same environment
    is_predicted = Column(Boolean, default=False, nullable=False)
    
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    # Relationships
    experiments = relationship("Experiment", back_populates="environment")


class Agent(Base):
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    type = Column(String(50), nullable=False)  # "PPO", "SAC", etc.
    
    # Agent physical configuration
    agent_diameter = Column(Float, nullable=False)
    agent_max_speed = Column(Float, nullable=False)
    agent_max_acceleration = Column(Float, nullable=False, default=5.0)  # Acceleration factor
    lidar_range = Column(Float, nullable=False)
    lidar_rays = Column(Integer, nullable=False, default=36)  # Number of lidar rays
    kinematic_type = Column(String(50), nullable=False, default="holonomic")  # "holonomic" or "semi-holonomic"
    
    # Hyperparameters stored as JSON
    parameters = Column(JSON, nullable=False, default=dict)
    
    # Locking mechanism
    is_locked = Column(Boolean, default=False, nullable=False)
    
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    # Relationships
    experiments = relationship("Experiment", back_populates="agent")
    residual_connectors = relationship("ResidualConnector", back_populates="parent_agent")


class ResidualConnector(Base):
    """
    A ResidualConnector defines a 'Safety Corrector' layer on top of a frozen base agent.
    It inherits physical specs from the parent agent but trains its own residual policy.
    """
    __tablename__ = "residual_connectors"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    
    # Link to parent agent (inherit physical specs)
    parent_agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    
    # Residual algorithm configuration
    algorithm = Column(String(50), nullable=False)  # "PPO", "A2C", etc.
    net_arch = Column(JSON, nullable=False, default=list)  # Neural network architecture e.g., [64, 64]
    parameters = Column(JSON, nullable=False, default=dict)  # Residual algorithm hyperparameters
    
    # K-factor for residual correction scaling (typically 0.05-0.5)
    k_factor = Column(Float, nullable=False, default=0.15)  # Scaling factor for residual corrections

    # Adaptive K-factor: dynamically reduce K when residual opposes base action
    adaptive_k = Column(Boolean, nullable=False, default=False)

    # CARS Ablation: Individual component toggles (for ablation studies)
    enable_k_conf = Column(Boolean, nullable=False, default=True)    # Enable conflict-aware scaling
    enable_k_risk = Column(Boolean, nullable=False, default=True)    # Enable risk-aware scaling (Lagrange λ)

    # Locking mechanism
    is_locked = Column(Boolean, default=False, nullable=False)
    
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    # Relationships
    parent_agent = relationship("Agent", back_populates="residual_connectors")


class RewardFunction(Base):
    __tablename__ = "reward_functions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    
    # Weight components
    w_progress = Column(Float, nullable=False)
    w_time = Column(Float, nullable=False)
    w_jerk = Column(Float, nullable=False)
    
    # Rewards/penalties
    success_reward = Column(Float, nullable=False)
    crash_reward = Column(Float, nullable=False)  # Should be negative
    
    # Locking mechanism
    is_locked = Column(Boolean, default=False, nullable=False)
    
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    # Relationships
    experiments = relationship("Experiment", back_populates="reward_function")


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    type = Column(Enum(ExperimentType), nullable=False)
    
    # Foreign keys
    env_id = Column(Integer, ForeignKey("environments.id"), nullable=False)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    reward_id = Column(Integer, ForeignKey("reward_functions.id"), nullable=False)
    model_snapshot_id = Column(Integer, ForeignKey("model_snapshots.id"), nullable=True)  # For simulations
    base_model_snapshot_id = Column(Integer, ForeignKey("model_snapshots.id"), nullable=True)  # For fine-tuning: the starting model
    fine_tuning_strategy = Column(String(50), nullable=True, default="full_finetune")  # Fine-tuning strategy
    
    # Residual Learning mode (new)
    training_mode = Column(String(50), nullable=True, default="standard")  # "standard" or "residual"
    residual_base_model_id = Column(Integer, ForeignKey("model_snapshots.id"), nullable=True)  # For residual: frozen base model
    residual_connector_id = Column(Integer, ForeignKey("residual_connectors.id"), nullable=True)  # For residual: connector configuration
    
    # Status and progress
    status = Column(Enum(ExperimentStatus), default=ExperimentStatus.IN_PROGRESS, nullable=False)
    total_steps = Column(Integer, nullable=False)
    current_step = Column(Integer, default=0, nullable=False)  # Actual steps completed
    snapshot_freq = Column(Integer, nullable=False)  # Save model every N steps
    max_ep_length = Column(Integer, nullable=False, default=2000)  # Max steps per episode before truncation
    
    # Evaluation mode specific fields
    evaluation_episodes = Column(Integer, nullable=True)  # Number of episodes for evaluation mode
    fps_delay = Column(Float, nullable=True)  # Delay in milliseconds between steps for visualization
    run_without_simulation = Column(
        Boolean, nullable=False, default=False
    )  # Fast mode: skip simulation frames and trajectory recording
    evaluation_policy_mode = Column(
        String(20), nullable=False, default=EvaluationPolicyMode.MODEL.value
    )  # "model" or "heuristic"
    heuristic_algorithm = Column(
        String(50), nullable=True
    )  # "potential_field" or "vfh_lite" when evaluation_policy_mode="heuristic"
    
    # Simulation-specific trajectory storage
    trajectory_data = Column(JSON, nullable=True, default=None)  # Store full flight path for replay
    
    # Safety constraint configuration (Lagrangian RL)
    safety_constraint = Column(JSON, nullable=True, default=None)  # Safety constraint config

    # Random seed for reproducibility
    seed = Column(Integer, nullable=True, default=None)  # If None, random seed will be used

    # Curriculum Learning (link to curriculum step if part of a pipeline)
    # curriculum_step_id = Column(Integer, ForeignKey("curriculum_steps.id"), nullable=True)
    
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    # Relationships
    environment = relationship("Environment", back_populates="experiments")
    agent = relationship("Agent", back_populates="experiments")
    reward_function = relationship("RewardFunction", back_populates="experiments")
    models = relationship(
        "ModelSnapshot", 
        back_populates="experiment", 
        cascade="all, delete-orphan",
        foreign_keys="ModelSnapshot.experiment_id"
    )
    base_model_snapshot = relationship(
        "ModelSnapshot",
        foreign_keys=[base_model_snapshot_id],
        uselist=False
    )
    residual_base_model = relationship(
        "ModelSnapshot",
        foreign_keys=[residual_base_model_id],
        uselist=False
    )
    residual_connector = relationship(
        "ResidualConnector",
        foreign_keys=[residual_connector_id],
        uselist=False
    )
    # curriculum_step = relationship(
    #     "CurriculumStep",
    #     foreign_keys=[curriculum_step_id],
    #     back_populates="experiment"
    # )
    metrics = relationship("ExperimentMetric", back_populates="experiment", cascade="all, delete-orphan")


class ModelSnapshot(Base):
    __tablename__ = "model_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    
    iteration = Column(Integer, nullable=False)  # Step number
    file_path = Column(String(500), nullable=False)  # Path to saved model
    
    # Metrics at time of save (using JSON for SQLite compatibility)
    metrics_at_save = Column(JSON, nullable=True, default=dict)
    
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    # Relationships
    experiment = relationship(
        "Experiment", 
        back_populates="models",
        foreign_keys=[experiment_id]
    )


class ExperimentMetric(Base):
    __tablename__ = "experiment_metrics"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False, index=True)
    
    step = Column(Integer, nullable=False)
    
    # Store all metrics as JSON for flexibility
    values = Column(JSON, nullable=False, default=dict)
    
    # Optional log message
    logs = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    # Relationships
    experiment = relationship("Experiment", back_populates="metrics")


# class Curriculum(Base):
#     """
#     Represents a multi-stage training pipeline.
#     Each curriculum contains multiple steps that execute sequentially.
#     """
#     __tablename__ = "curriculums"

#     id = Column(Integer, primary_key=True, index=True)
#     name = Column(String(100), unique=True, nullable=False, index=True)
#     description = Column(Text, nullable=True)
    
#     # Pipeline status
#     status = Column(Enum(CurriculumStatus), default=CurriculumStatus.PENDING, nullable=False)
    
#     # Current step being executed (1-indexed)
#     current_step_order = Column(Integer, default=1, nullable=False)
    
#     created_at = Column(DateTime, default=dt.datetime.utcnow)
#     updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)

#     # Relationships
#     steps = relationship("CurriculumStep", back_populates="curriculum", cascade="all, delete-orphan", order_by="CurriculumStep.order")


# class CurriculumStep(Base):
#     """
#     Represents a single stage in a curriculum pipeline.
#     Each step defines training configuration, stop conditions, and links to experiments.
#     """
#     __tablename__ = "curriculum_steps"

#     id = Column(Integer, primary_key=True, index=True)
#     curriculum_id = Column(Integer, ForeignKey("curriculums.id"), nullable=False)
    
#     # Step ordering (1, 2, 3...)
#     order = Column(Integer, nullable=False)
    
#     # Type of training step
#     step_type = Column(String(50), nullable=False)  # "training", "fine_tuning", "residual"
    
#     # Training configuration stored as JSON
#     config = Column(JSON, nullable=False, default=dict)
#     # Example: {"env_id": 1, "agent_id": 2, "reward_id": 3, "total_steps": 100000, ...}
    
#     # Safety constraint configuration (Lagrangian RL)
#     safety_config = Column(JSON, nullable=True, default=None)
#     # Example: {"enabled": true, "epsilon": 0.1, "lambda_init": 0.5}
    
#     # Stop condition: advance to next step when this success rate is reached
#     target_success_rate = Column(Float, nullable=False, default=0.95)
    
#     # Minimum number of episodes to evaluate before allowing transition
#     min_episodes = Column(Integer, nullable=False, default=50)
    
#     # Completion flag
#     is_completed = Column(Boolean, default=False, nullable=False)
    
#     # Relationships
#     curriculum = relationship("Curriculum", back_populates="steps")
#     experiment = relationship(
#         "Experiment",
#         back_populates="curriculum_step",
#         uselist=False,
#         viewonly=True
#     )
