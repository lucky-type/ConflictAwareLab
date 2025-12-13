"""CRUD helpers for SQLAlchemy models."""
from __future__ import annotations

from typing import Optional, Type, TypeVar

from sqlalchemy.orm import Session

from . import models

ModelType = TypeVar("ModelType", bound=models.Base)


def list_entities(db: Session, model: Type[ModelType]):
    return db.query(model).all()


def get_entity(db: Session, model: Type[ModelType], entity_id: int) -> Optional[ModelType]:
    return db.query(model).filter(model.id == entity_id).first()


def create_entity(db: Session, model: Type[ModelType], payload):
    db_obj = model(**payload.dict())
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def update_entity(db: Session, db_obj, payload):
    data = payload.dict(exclude_unset=True)
    for key, value in data.items():
        setattr(db_obj, key, value)
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def delete_entity(db: Session, db_obj):
    db.delete(db_obj)
    db.commit()
    return db_obj
