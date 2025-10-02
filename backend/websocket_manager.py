import asyncio
import json
from datetime import datetime
from typing import Dict, Set

from fastapi import WebSocket


class WebSocketManager:
    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict) -> None:
        payload = json.dumps({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **message,
        })
        async with self._lock:
            dead = []
            for connection in list(self.active_connections):
                try:
                    await connection.send_text(payload)
                except Exception:
                    dead.append(connection)
            for d in dead:
                self.active_connections.discard(d)


ws_manager = WebSocketManager()




