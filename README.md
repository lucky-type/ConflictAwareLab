# DroneSim

This project provides a FastAPI backend for managing reinforcement learning experiments and streaming telemetry/training updates for a drone simulator.

## Features
- CRUD APIs for environments, algorithms, reward functions, models, and experiments (SQLAlchemy + Pydantic)
- WebSocket streams for simulation telemetry and training status
- Background PPO training loop using Stable-Baselines3 and PyBullet environments
- Simple startup script and dependency list

## Getting started
1. Create a virtual environment (optional):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the API:
   ```bash
   ./start.sh
   ```
4. Open the interactive docs at http://localhost:8000/docs

## API highlights
- CRUD endpoints under `/environments`, `/algorithms`, `/reward-functions`, `/models`, and `/experiments`
- Start training: `POST /train` with `{ "experiment_id": <id> }`
- Training WebSocket: `/ws/training/{experiment_id}` (JSON status updates)
- Telemetry WebSocket: `/ws/telemetry` (JSON pose/velocity payloads)

The service uses SQLite by default (`dronesim.db`). Adjust `SQLALCHEMY_DATABASE_URL` in `app/database.py` for other databases.
