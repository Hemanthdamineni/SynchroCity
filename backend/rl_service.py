import asyncio
from typing import Optional

from .websocket_manager import ws_manager


class RLService:
    def __init__(self) -> None:
        self.training_task: Optional[asyncio.Task] = None
        self.control_task: Optional[asyncio.Task] = None
        self.controlling: bool = False

    async def start_training(self) -> None:
        if self.training_task and not self.training_task.done():
            await ws_manager.broadcast({"type": "rl", "message": "Training already running"})
            return
        await ws_manager.broadcast({"type": "rl", "message": "Starting RL training"})
        self.training_task = asyncio.create_task(self._fake_train_loop())

    async def _fake_train_loop(self) -> None:
        for i in range(5):
            await asyncio.sleep(1)
            await ws_manager.broadcast({"type": "rl", "message": "Training progress", "details": {"step_block": i + 1, "total": 5}})
        await ws_manager.broadcast({"type": "rl", "message": "Training complete"})

    async def start_control(self) -> None:
        if self.controlling:
            return
        self.controlling = True
        await ws_manager.broadcast({"type": "rl", "message": "Starting control loop"})

    async def stop_control(self) -> None:
        if not self.controlling:
            return
        self.controlling = False
        await ws_manager.broadcast({"type": "rl", "message": "Stopped control loop"})


rl_service = RLService()




