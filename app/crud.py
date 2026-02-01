"""CRUD helpers for SQLAlchemy models."""
from __future__ import annotations

from typing import Optional, Type, TypeVar

from fastapi import HTTPException
from sqlalchemy.orm import Session

from . import models

ModelType = TypeVar("ModelType", bound=models.Base)


def list_entities(db: Session, model: Type[ModelType]):
    """List all entities of a given model."""
    return db.query(model).all()


def get_entity(db: Session, model: Type[ModelType], entity_id: int) -> Optional[ModelType]:
    """Get a single entity by ID."""
    return db.query(model).filter(model.id == entity_id).first()


def create_entity(db: Session, model: Type[ModelType], payload):
    """Create a new entity."""
    data = payload.dict() if hasattr(payload, 'dict') else payload
    
    # Check for duplicate name if model has 'name' field
    if hasattr(model, 'name') and 'name' in data:
        existing = db.query(model).filter(model.name == data['name']).first()
        if existing:
            raise HTTPException(
                status_code=400, 
                detail=f"{model.__name__} with name '{data['name']}' already exists"
            )
    
    db_obj = model(**data)
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def update_entity(db: Session, db_obj, payload):
    """Update an existing entity."""
    data = payload.dict(exclude_unset=True)
    for key, value in data.items():
        setattr(db_obj, key, value)
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def delete_entity(db: Session, db_obj):
    """Delete an entity."""
    db.delete(db_obj)
    db.commit()
    return db_obj


# ==================== Business Logic Functions ====================

def check_entity_locked(entity) -> None:
    """
    Check if entity is locked and raise exception if it is.
    Used for Environments, Agents, and RewardFunctions.
    """
    if hasattr(entity, 'is_locked') and entity.is_locked:
        raise HTTPException(
            status_code=400,
            detail=f"This preset is being used in an active experiment and cannot be modified or deleted."
        )


def check_experiments_using_entity(db: Session, entity_type: str, entity_id: int) -> None:
    """
    Check if any experiments are using this entity.
    Raises HTTPException if experiments exist.
    
    Args:
        entity_type: 'environment', 'agent', or 'reward_function'
        entity_id: ID of the entity to check
    """
    query = db.query(models.Experiment)
    
    if entity_type == 'environment':
        count = query.filter(models.Experiment.env_id == entity_id).count()
    elif entity_type == 'agent':
        count = query.filter(models.Experiment.agent_id == entity_id).count()
    elif entity_type == 'reward_function':
        count = query.filter(models.Experiment.reward_id == entity_id).count()
    else:
        raise ValueError(f"Unknown entity type: {entity_type}")
    
    if count > 0:
        raise HTTPException(
            status_code=400,
            detail=f"This preset is being used in {count} experiment(s). Deletion is not allowed."
        )


def lock_entity(db: Session, entity) -> None:
    """Lock an entity when it's used in an experiment."""
    if hasattr(entity, 'is_locked'):
        entity.is_locked = True
        db.add(entity)
        db.commit()


def create_experiment_and_lock_entities(db: Session, payload: dict) -> models.Experiment:
    """
    Create an experiment and lock all related entities.
    This ensures the presets cannot be modified while experiment is active.
    """
    # Create the experiment
    experiment = models.Experiment(**payload)
    db.add(experiment)
    db.flush()  # Get the ID without committing
    
    # Lock all related entities
    env = get_entity(db, models.Environment, payload['env_id'])
    agent = get_entity(db, models.Agent, payload['agent_id'])
    reward = get_entity(db, models.RewardFunction, payload['reward_id'])
    
    if env:
        lock_entity(db, env)
    if agent:
        lock_entity(db, agent)
    if reward:
        lock_entity(db, reward)
    
    db.commit()
    db.refresh(experiment)
    return experiment


# ==================== Specialized CRUD for Metrics ====================

def create_metric(db: Session, experiment_id: int, step: int, values: dict, logs: str = None) -> models.ExperimentMetric:
    """Create a new metric entry for an experiment."""
    metric = models.ExperimentMetric(
        experiment_id=experiment_id,
        step=step,
        values=values,
        logs=logs
    )
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return metric


def get_metrics_for_experiment(db: Session, experiment_id: int, limit: int = None) -> list[models.ExperimentMetric]:
    """Get all metrics for an experiment, optionally limited."""
    query = db.query(models.ExperimentMetric).filter(
        models.ExperimentMetric.experiment_id == experiment_id
    ).order_by(models.ExperimentMetric.step)
    
    if limit:
        query = query.limit(limit)
    
    return query.all()


def get_bulk_latest_metrics(db: Session, experiment_ids: list[int]) -> dict[int, models.ExperimentMetric]:
    """Get the latest metric for each experiment in a single query.
    
    This is optimized for bulk fetching to avoid N+1 query problems.
    Returns a dict mapping experiment_id -> latest ExperimentMetric.
    """
    if not experiment_ids:
        return {}
    
    from sqlalchemy import func
    
    # Subquery to get max step per experiment
    subq = db.query(
        models.ExperimentMetric.experiment_id,
        func.max(models.ExperimentMetric.step).label('max_step')
    ).filter(
        models.ExperimentMetric.experiment_id.in_(experiment_ids)
    ).group_by(models.ExperimentMetric.experiment_id).subquery()
    
    # Join to get full metric rows for the latest step of each experiment
    metrics = db.query(models.ExperimentMetric).join(
        subq,
        (models.ExperimentMetric.experiment_id == subq.c.experiment_id) &
        (models.ExperimentMetric.step == subq.c.max_step)
    ).all()
    
    return {m.experiment_id: m for m in metrics}


def create_model_snapshot(
    db: Session, 
    experiment_id: int, 
    iteration: int, 
    file_path: str, 
    metrics: dict = None
) -> models.ModelSnapshot:
    """Create a model snapshot for an experiment."""
    snapshot = models.ModelSnapshot(
        experiment_id=experiment_id,
        iteration=iteration,
        file_path=file_path,
        metrics_at_save=metrics or {}
    )
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)
    return snapshot


def get_snapshots_for_experiment(db: Session, experiment_id: int) -> list[models.ModelSnapshot]:
    """Get all model snapshots for an experiment."""
    return db.query(models.ModelSnapshot).filter(
        models.ModelSnapshot.experiment_id == experiment_id
    ).order_by(models.ModelSnapshot.iteration).all()


def create_experiment_snapshot(
    db: Session,
    experiment_id: int,
    iteration: int,
    file_path: str,
    metrics: dict = None,
) -> models.ModelSnapshot:
    """Create an experiment snapshot (alias for backward compatibility)."""
    return create_model_snapshot(db, experiment_id, iteration, file_path, metrics)


def update_experiment_status(db: Session, experiment_id: int, status: str):
    """Update experiment status."""
    experiment = get_entity(db, models.Experiment, experiment_id)
    if experiment:
        experiment.status = status
        db.add(experiment)
        db.commit()
        db.refresh(experiment)
    return experiment


def update_experiment_trajectory(db: Session, experiment_id: int, trajectory: list):
    """Update experiment trajectory data."""
    experiment = get_entity(db, models.Experiment, experiment_id)
    if experiment:
        # Store trajectory data as JSON string
        import json
        import numpy as np
        
        # Custom JSON encoder to handle numpy types and other non-serializable objects
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            elif hasattr(obj, '__dict__'):
                return str(obj)  # Convert complex objects to string
            return obj
            
        # Convert trajectory to serializable format
        serializable_trajectory = []
        for frame in trajectory:
            serializable_frame = {}
            for key, value in frame.items():
                serializable_frame[key] = convert_to_serializable(value)
            serializable_trajectory.append(serializable_frame)
        
        # SQLAlchemy JSON type handles serialization automatically
        # Assigning it directly ensures it's stored/retrieved as a list/dict, not a string
        experiment.trajectory_data = serializable_trajectory
        db.add(experiment)
        db.commit()
        db.refresh(experiment)
        print(f"[CRUD] Updated experiment {experiment_id} trajectory with {len(trajectory)} frames")
    return experiment


def create_experiment_metric(
    db: Session,
    experiment_id: int,
    step: int,
    metric_name: str,
    metric_value: float,
) -> models.ExperimentMetric:
    """Create an experiment metric entry."""
    metric = models.ExperimentMetric(
        experiment_id=experiment_id,
        step=step,
        metric_name=metric_name,
        metric_value=metric_value,
    )
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return metric
