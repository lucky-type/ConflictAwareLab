"""Pydantic schemas used by the FastAPI endpoints."""
from __future__ import annotations

import datetime as dt
from typing import Optional

from pydantic import BaseModel, Field


class EnvironmentBase(BaseModel):
    name: str = Field(..., example="Urban Canyon")
    description: Optional[str] = Field(None, example="Dense city environment with wind gusts")


class EnvironmentCreate(EnvironmentBase):
    pass


class EnvironmentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class Environment(EnvironmentBase):
    id: int
    created_at: dt.datetime

    class Config:
        from_attributes = True


class AlgorithmBase(BaseModel):
    name: str = Field(..., example="PPO")
    version: Optional[str] = Field(None, example="1.8.0")
    details: Optional[str] = Field(None, example="Default PPO hyperparameters")


class AlgorithmCreate(AlgorithmBase):
    pass


class AlgorithmUpdate(BaseModel):
    name: Optional[str] = None
    version: Optional[str] = None
    details: Optional[str] = None


class Algorithm(AlgorithmBase):
    id: int
    created_at: dt.datetime

    class Config:
        from_attributes = True


class RewardFunctionBase(BaseModel):
    name: str
    expression: str


class RewardFunctionCreate(RewardFunctionBase):
    pass


class RewardFunctionUpdate(BaseModel):
    name: Optional[str] = None
    expression: Optional[str] = None


class RewardFunction(RewardFunctionBase):
    id: int
    created_at: dt.datetime

    class Config:
        from_attributes = True


class ModelBase(BaseModel):
    name: str
    architecture: Optional[str] = None


class ModelCreate(ModelBase):
    pass


class ModelUpdate(BaseModel):
    name: Optional[str] = None
    architecture: Optional[str] = None


class Model(ModelBase):
    id: int
    created_at: dt.datetime

    class Config:
        from_attributes = True


class ExperimentBase(BaseModel):
    name: str
    description: Optional[str] = None
    environment_id: int
    algorithm_id: int
    reward_function_id: int
    model_id: int
    total_timesteps: int = Field(10_000, gt=0)
    learning_rate: float = Field(0.0003, gt=0)


class ExperimentCreate(ExperimentBase):
    pass


class ExperimentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    environment_id: Optional[int] = None
    algorithm_id: Optional[int] = None
    reward_function_id: Optional[int] = None
    model_id: Optional[int] = None
    total_timesteps: Optional[int] = Field(None, gt=0)
    learning_rate: Optional[float] = Field(None, gt=0)


class Experiment(ExperimentBase):
    id: int
    created_at: dt.datetime

    class Config:
        from_attributes = True


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

