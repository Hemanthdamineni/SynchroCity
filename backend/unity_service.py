import asyncio
import os
import time
from typing import Optional, Tuple

import zmq
import zmq.asyncio
import numpy as np

from .websocket_manager import ws_manager


class UnityFrameSubscriber:
    def __init__(self, port: int, fps: int = 15, fallback: bool = False) -> None:
        self.port = port
        self.fps = fps
        self.ctx = zmq.asyncio.Context.instance()
        self.socket: Optional[zmq.asyncio.Socket] = None
        self.running = False
        self.latest_frame: Optional[np.ndarray] = None
        self.fallback = fallback
        self._last_broadcast = 0.0

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        await ws_manager.broadcast({"type": "unity", "message": f"Starting Unity subscriber on {self.port}"})
        try:
            self.socket = self.ctx.socket(zmq.SUB)
            self.socket.setsockopt(zmq.CONFLATE, 1)
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self.socket.connect(f"tcp://127.0.0.1:{self.port}")
        except Exception as e:
            await ws_manager.broadcast({"type": "error", "message": "ZeroMQ connect failed", "details": {"port": self.port, "error": str(e)}})
            if not self.fallback:
                self.fallback = True

        asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self.running = False
        if self.socket is not None:
            self.socket.close(0)
            self.socket = None

    async def _run_loop(self) -> None:
        period = 1.0 / max(1, self.fps)
        while self.running:
            start = time.time()
            try:
                if not self.fallback and self.socket is not None:
                    msg = await asyncio.wait_for(self.socket.recv(), timeout=2.0)
                    frame = self._deserialize_frame(msg)
                else:
                    frame = self._synthetic_frame()
                self.latest_frame = frame
                now = time.time()
                if now - self._last_broadcast > 1.0:
                    self._last_broadcast = now
                    await ws_manager.broadcast({
                        "type": "unity",
                        "message": "Frame received",
                        "details": {"shape": list(frame.shape), "fallback": self.fallback},
                    })
            except Exception as e:
                await ws_manager.broadcast({"type": "error", "message": "Unity frame loop error", "details": {"error": str(e)}})
                self.fallback = True
            elapsed = time.time() - start
            await asyncio.sleep(max(0.0, period - elapsed))

    def _deserialize_frame(self, msg: bytes) -> np.ndarray:
        try:
            # Expect header: width,height,channels; then raw bytes
            header, raw = msg.split(b"|", 1)
            w_str, h_str, c_str = header.decode("utf-8").split(",")
            w, h, c = int(w_str), int(h_str), int(c_str)
            arr = np.frombuffer(raw, dtype=np.uint8)
            return arr.reshape((h, w, c))
        except Exception:
            # Fallback to numpy compressed bytes if provided
            try:
                arr = np.loadb(np.frombuffer(msg, dtype=np.uint8))  # type: ignore[attr-defined]
            except Exception:
                # As last resort, create synthetic
                arr = self._synthetic_frame()
            return arr

    def _synthetic_frame(self) -> np.ndarray:
        w, h = 320, 240
        t = int(time.time() * 10) % 255
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 0] = (t + np.arange(w, dtype=np.uint8)) % 255
        img[:, :, 1] = (t + np.arange(h, dtype=np.uint8)[:, None]) % 255
        img[:, :, 2] = t
        return img


unity_subscriber: Optional[UnityFrameSubscriber] = None




