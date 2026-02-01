# ConflictAwareLab

ConflictAwareLab is a comprehensive platform for the development, training, and evaluation of **safe reinforcement learning** policies for drone navigation. It specifically focuses on a novel method: **Conflict-Aware Residual Scaling (CARS)** with adaptive K-factors in Lagrangian residual learning.

The system combines a **FastAPI** backend with a **React** dashboard to provide a seamless workflow for configuring environments, managing agents, and running large-scale simulation experiments with **PyBullet** and **Stable-Baselines3**.

---

## 🚀 Research Goal: CARS Methodology

The core research objective of this platform is to validate the effectiveness of **Conflict-Aware Residual Scaling (CARS)**.

### The Problem
Standard residual learning (Base Policy + Residual Policy) often struggles with balancing performance and safety. A fixed weighting factor ($k$) for the residual policy can lead to either unsafe behaviors (if $k$ is too high) or inability to correct base model mistakes (if $k$ is too low).

### The Solution: Adaptive K
CARS dynamically adjusts the residual contribution ($k_{effective}$) based on three real-time signals:

$$k_{effective} = k_{factor} \times K_{conf} \times K_{risk} \times K_{shield}$$

1.  **$K_{conf}$ (Conflict-Aware):** Reduces $k$ when the residual action vectors conflict with the base policy (measured via cosine similarity).
2.  **$K_{risk}$ (Risk-Aware):** Reduces $k$ when the Lagrangian multiplier ($\lambda$) is high, indicating increased violation of safety constraints.
3.  **$K_{shield}$ (Shield-Aware):** Reduces $k$ when the LIDAR-based safety shield frequently intervenes.

---

## ✨ Features

- **Experiment Management:** Full CRUD for Environments, Agents, Reward Functions, and Residual Connectors.
- **Simulation Engine:**
    - Physics-based drone simulation using **PyBullet**.
    - Holonomic kinematic models.
    - Support for swarm obstacles with configurable behaviors (linear, zigzag, circular, random).
- **Visualization:**
    - Real-time **WebSocket** streaming of simulation frames.
    - Interactive **Three.js** 3D viewer in the browser.
    - Live charts for reward, cost, and safety metrics.
- **Safety Pipelines:**
    - **Lagrangian Relaxation** for constrained updates.
    - **Safety Shield** (LIDAR-based) for emergency braking and overriding.
    - **CARS** ablation study support (granular control over $K$ components).
- **Analysis:**
    - Automated metrics collection (Success Rate, Violation Rate, Crash Rate).
    - Excel export for statistical analysis.

---

## 🛠 Tech Stack

- **Frontend:** React, TypeScript, Vite, Tailwind CSS, Three.js, Recharts.
- **Backend:** FastAPI, SQLAlchemy, Pydantic, WebSockets.
- **Machine Learning:** Gymnasium, Stable-Baselines3, PyTorch.
- **Simulation:** PyBullet.
- **Database:** SQLite (with support for WAL mode).

---

## 🧪 Standard Experimental Setup

The platform is pre-configured with standard benchmarks for evaluating safe navigation:

### Environments
- **Easy Corridor:** 100x20x20m, 3 dynamic obstacles.
- **Medium Corridor:** 100x20x20m, 5 dynamic obstacles (Standard Benchmark).
- **Hard Corridor:** 100x15x15m, 7 dynamic obstacles.

### Agents
- **Base:** PPO Agent (standard policy).
- **Residual Connectors:**
    - **Static:** Fixed $k \in \{0.10, 0.15, 0.20\}$.
    - **CARS:** Adaptive $k$ with enabled conflict, risk, and shield scaling.

---

## 📦 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 18+

### 1. Backend Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize Database
python init_db.py

# Run Server
./start.sh
# OR
uvicorn app.main:app --reload
```

### 2. Frontend Setup
```bash
# In a new terminal
npm install
npm run dev
```

### 3. Access
Open [http://localhost:5173](http://localhost:5173) to view the dashboard.

---

## 📄 License
MIT License.
