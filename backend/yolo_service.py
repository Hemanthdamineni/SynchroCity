import asyncio
import os
from typing import Optional

from .websocket_manager import ws_manager


class YOLOService:
    def __init__(self) -> None:
        self.training_task: Optional[asyncio.Task] = None
        self.inference_task: Optional[asyncio.Task] = None
        self.running_inference: bool = False

    async def start_training(self, script_path: str, data_yaml: str, weights_out: str) -> None:
        if self.training_task and not self.training_task.done():
            await ws_manager.broadcast({"type": "yolo", "message": "Training already running"})
            return
        await ws_manager.broadcast({"type": "yolo", "message": "Starting YOLO training"})
        self.training_task = asyncio.create_task(self._fake_train_loop())

    async def _fake_train_loop(self) -> None:
        # Placeholder progress loop to integrate UI; real train script is provided in python/yolo
        for i in range(5):
            await asyncio.sleep(1)
            await ws_manager.broadcast({"type": "yolo", "message": "Training progress", "details": {"epoch": i + 1, "total": 5}})
        await ws_manager.broadcast({"type": "yolo", "message": "Training complete"})

    async def start_inference(self) -> None:
        if self.running_inference:
            return
        self.running_inference = True
        await ws_manager.broadcast({"type": "yolo", "message": "Starting inference"})

    async def stop_inference(self) -> None:
        if not self.running_inference:
            return
        self.running_inference = False
        await ws_manager.broadcast({"type": "yolo", "message": "Stopped inference"})


yolo_service = YOLOService()




