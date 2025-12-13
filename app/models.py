"""SQLAlchemy models for the DroneSim domain."""
from __future__ import annotations

import datetime as dt

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from .database import Base


class Environment(Base):
    __tablename__ = "environments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    experiments = relationship("Experiment", back_populates="environment")


class Algorithm(Base):
    __tablename__ = "algorithms"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    version = Column(String(50), nullable=True)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    experiments = relationship("Experiment", back_populates="algorithm")


class RewardFunction(Base):
    __tablename__ = "reward_functions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    expression = Column(Text, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    experiments = relationship("Experiment", back_populates="reward_function")


class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    architecture = Column(Text, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    experiments = relationship("Experiment", back_populates="model")


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    environment_id = Column(Integer, ForeignKey("environments.id"), nullable=False)
    algorithm_id = Column(Integer, ForeignKey("algorithms.id"), nullable=False)
    reward_function_id = Column(Integer, ForeignKey("reward_functions.id"), nullable=False)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    total_timesteps = Column(Integer, default=10_000)
    learning_rate = Column(Float, default=0.0003)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    environment = relationship("Environment", back_populates="experiments")
    algorithm = relationship("Algorithm", back_populates="experiments")
    reward_function = relationship("RewardFunction", back_populates="experiments")
    model = relationship("Model", back_populates="experiments")
