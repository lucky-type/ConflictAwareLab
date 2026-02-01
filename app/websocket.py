"""WebSocket connection manager for real-time experiment updates."""
from __future__ import annotations

import json
from typing import Dict, List

from fastapi import WebSocket


class ConnectionManager:
    """Manages WebSocket connections for experiments."""

    def __init__(self):
        # Map experiment_id -> list of active WebSocket connections
        self.active_connections: Dict[int, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, experiment_id: int):
        """Accept a new WebSocket connection for an experiment."""
        await websocket.accept()
        if experiment_id not in self.active_connections:
            self.active_connections[experiment_id] = []
        self.active_connections[experiment_id].append(websocket)
        print(f"WebSocket connected for experiment {experiment_id}")

    def disconnect(self, websocket: WebSocket, experiment_id: int):
        """Remove a WebSocket connection."""
        if experiment_id in self.active_connections:
            if websocket in self.active_connections[experiment_id]:
                self.active_connections[experiment_id].remove(websocket)
            # Clean up empty lists
            if not self.active_connections[experiment_id]:
                del self.active_connections[experiment_id]
        print(f"WebSocket disconnected for experiment {experiment_id}")

    async def broadcast(self, experiment_id: int, message: dict):
        """Broadcast a message to all connected clients for an experiment."""
        message_type = message.get('type', 'unknown')
        message_data = message.get('data', {})
        episode_info = f"episode {message_data.get('episode', '?')}/{message_data.get('total_episodes', '?')}" if 'episode' in message_data else "no episode info"
        print(f"[WebSocket] Broadcasting to experiment {experiment_id}: {message_type} ({episode_info})")
        
        if experiment_id not in self.active_connections:
            print(f"[WebSocket] No connections for experiment {experiment_id}")
            return

        # Send to all connected clients
        disconnected = []
        for websocket in self.active_connections[experiment_id]:
            try:
                await websocket.send_json(message)
                print(f"[WebSocket] Sent {message_type} message to client (exp {experiment_id})")
            except Exception as e:
                print(f"Error sending to websocket: {e}")
                disconnected.append(websocket)

        # Clean up disconnected clients
        for websocket in disconnected:
            self.disconnect(websocket, experiment_id)


# Global connection manager instance
connection_manager = ConnectionManager()
