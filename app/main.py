from __future__ import annotations

import asyncio
import time
from typing import List

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from . import crud, models, schemas
from .database import Base, engine, get_db
from .training import training_manager

app = FastAPI(title="DroneSim", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)


# ------------------------------ CRUD helpers ------------------------------ #


def _get_or_404(db: Session, model, entity_id: int):
    entity = crud.get_entity(db, model, entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Item not found")
    return entity


# Environments
@app.post("/environments", response_model=schemas.Environment)
def create_environment(
    payload: schemas.EnvironmentCreate, db: Session = Depends(get_db)
):
    return crud.create_entity(db, models.Environment, payload)


@app.get("/environments", response_model=List[schemas.Environment])
def list_environments(db: Session = Depends(get_db)):
    return crud.list_entities(db, models.Environment)


@app.get("/environments/{environment_id}", response_model=schemas.Environment)
def get_environment(environment_id: int, db: Session = Depends(get_db)):
    return _get_or_404(db, models.Environment, environment_id)


@app.put("/environments/{environment_id}", response_model=schemas.Environment)
def update_environment(
    environment_id: int,
    payload: schemas.EnvironmentUpdate,
    db: Session = Depends(get_db),
):
    env = _get_or_404(db, models.Environment, environment_id)
    return crud.update_entity(db, env, payload)


@app.delete("/environments/{environment_id}")
def delete_environment(environment_id: int, db: Session = Depends(get_db)):
    env = _get_or_404(db, models.Environment, environment_id)
    crud.delete_entity(db, env)
    return {"status": "deleted"}


# Algorithms
@app.post("/algorithms", response_model=schemas.Algorithm)
def create_algorithm(payload: schemas.AlgorithmCreate, db: Session = Depends(get_db)):
    return crud.create_entity(db, models.Algorithm, payload)


@app.get("/algorithms", response_model=List[schemas.Algorithm])
def list_algorithms(db: Session = Depends(get_db)):
    return crud.list_entities(db, models.Algorithm)


@app.get("/algorithms/{algorithm_id}", response_model=schemas.Algorithm)
def get_algorithm(algorithm_id: int, db: Session = Depends(get_db)):
    return _get_or_404(db, models.Algorithm, algorithm_id)


@app.put("/algorithms/{algorithm_id}", response_model=schemas.Algorithm)
def update_algorithm(
    algorithm_id: int, payload: schemas.AlgorithmUpdate, db: Session = Depends(get_db)
):
    algo = _get_or_404(db, models.Algorithm, algorithm_id)
    return crud.update_entity(db, algo, payload)


@app.delete("/algorithms/{algorithm_id}")
def delete_algorithm(algorithm_id: int, db: Session = Depends(get_db)):
    algo = _get_or_404(db, models.Algorithm, algorithm_id)
    crud.delete_entity(db, algo)
    return {"status": "deleted"}


# Reward functions
@app.post("/reward-functions", response_model=schemas.RewardFunction)
def create_reward_function(
    payload: schemas.RewardFunctionCreate, db: Session = Depends(get_db)
):
    return crud.create_entity(db, models.RewardFunction, payload)


@app.get("/reward-functions", response_model=List[schemas.RewardFunction])
def list_reward_functions(db: Session = Depends(get_db)):
    return crud.list_entities(db, models.RewardFunction)


@app.get(
    "/reward-functions/{reward_function_id}", response_model=schemas.RewardFunction
)
def get_reward_function(reward_function_id: int, db: Session = Depends(get_db)):
    return _get_or_404(db, models.RewardFunction, reward_function_id)


@app.put(
    "/reward-functions/{reward_function_id}", response_model=schemas.RewardFunction
)
def update_reward_function(
    reward_function_id: int,
    payload: schemas.RewardFunctionUpdate,
    db: Session = Depends(get_db),
):
    reward_fn = _get_or_404(db, models.RewardFunction, reward_function_id)
    return crud.update_entity(db, reward_fn, payload)


@app.delete("/reward-functions/{reward_function_id}")
def delete_reward_function(reward_function_id: int, db: Session = Depends(get_db)):
    reward_fn = _get_or_404(db, models.RewardFunction, reward_function_id)
    crud.delete_entity(db, reward_fn)
    return {"status": "deleted"}


# Models
@app.post("/models", response_model=schemas.Model)
def create_model(payload: schemas.ModelCreate, db: Session = Depends(get_db)):
    return crud.create_entity(db, models.Model, payload)


@app.get("/models", response_model=List[schemas.Model])
def list_models(db: Session = Depends(get_db)):
    return crud.list_entities(db, models.Model)


@app.get("/models/{model_id}", response_model=schemas.Model)
def get_model(model_id: int, db: Session = Depends(get_db)):
    return _get_or_404(db, models.Model, model_id)


@app.put("/models/{model_id}", response_model=schemas.Model)
def update_model(model_id: int, payload: schemas.ModelUpdate, db: Session = Depends(get_db)):
    model = _get_or_404(db, models.Model, model_id)
    return crud.update_entity(db, model, payload)


@app.delete("/models/{model_id}")
def delete_model(model_id: int, db: Session = Depends(get_db)):
    model = _get_or_404(db, models.Model, model_id)
    crud.delete_entity(db, model)
    return {"status": "deleted"}


# Experiments
@app.post("/experiments", response_model=schemas.Experiment)
def create_experiment(
    payload: schemas.ExperimentCreate, db: Session = Depends(get_db)
):
    return crud.create_entity(db, models.Experiment, payload)


@app.get("/experiments", response_model=List[schemas.Experiment])
def list_experiments(db: Session = Depends(get_db)):
    return crud.list_entities(db, models.Experiment)


@app.get("/experiments/{experiment_id}", response_model=schemas.Experiment)
def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    return _get_or_404(db, models.Experiment, experiment_id)


@app.put("/experiments/{experiment_id}", response_model=schemas.Experiment)
def update_experiment(
    experiment_id: int, payload: schemas.ExperimentUpdate, db: Session = Depends(get_db)
):
    experiment = _get_or_404(db, models.Experiment, experiment_id)
    return crud.update_entity(db, experiment, payload)


@app.delete("/experiments/{experiment_id}")
def delete_experiment(experiment_id: int, db: Session = Depends(get_db)):
    experiment = _get_or_404(db, models.Experiment, experiment_id)
    crud.delete_entity(db, experiment)
    return {"status": "deleted"}


# ------------------------------ Training ------------------------------ #


@app.post("/train", response_model=schemas.TrainingStatus)
def start_training(
    request: schemas.TrainingRequest, db: Session = Depends(get_db)
):
    experiment = _get_or_404(db, models.Experiment, request.experiment_id)
    try:
        training_manager.start_training(experiment)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return schemas.TrainingStatus(
        experiment_id=experiment.id,
        status="starting",
        progress=0.0,
        message="Training scheduled",
    )


@app.websocket("/ws/training/{experiment_id}")
async def training_ws(websocket: WebSocket, experiment_id: int):
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue()

    def listener(payload: dict) -> None:
        queue.put_nowait(payload)

    training_manager.subscribe(experiment_id, listener)
    try:
        while True:
            payload = await queue.get()
            await websocket.send_json({"experiment_id": experiment_id, **payload})
    except WebSocketDisconnect:
        training_manager.unsubscribe(experiment_id, listener)
    finally:
        training_manager.unsubscribe(experiment_id, listener)


# ------------------------------ Telemetry ------------------------------ #


async def telemetry_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            now = time.time()
            payload = schemas.TelemetryMessage(
                timestamp=now,
                position=(0.0, 0.0, 1.0),
                orientation=(0.0, 0.0, 0.0),
                linear_velocity=(0.0, 0.0, 0.0),
                angular_velocity=(0.0, 0.0, 0.0),
            )
            await websocket.send_json(payload.dict())
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return


@app.websocket("/ws/telemetry")
async def telemetry_ws(websocket: WebSocket):
    await telemetry_stream(websocket)


@app.get("/")
def root():
    return {"message": "DroneSim API running"}
