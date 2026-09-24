#!/usr/bin/env python3
"""Run CUDA tongue inference in a sidecar process.

torch.load holds the receiver GIL long enough that the headset TCP client
stops draining, the client-lease dies, and preview freezes on the first frame.
A spawned process loads CUDA on its own GIL so the camera socket stays live.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path
from queue import Empty, Full
from typing import Any

import numpy as np


def _process_main(
    command_queue: multiprocessing.Queue[object],
    result_queue: multiprocessing.Queue[object],
    init: dict[str, Any],
) -> None:
    from tongue_model_preview import LiveTongueModelPreview

    try:
        preview = LiveTongueModelPreview(
            init["checkpoint_path"],
            device_name=init["device_name"],
            direction_checkpoint_path=init["direction_checkpoint_path"],
            smoothing=init["smoothing"],
            visibility_mode=init["visibility_mode"],
        )
        result_queue.put(
            {
                "type": "ready",
                "device": str(preview.device),
                "checkpoint_path": str(preview.checkpoint_path),
                "direction_checkpoint_path": (
                    str(preview.direction_checkpoint_path)
                    if preview.direction_checkpoint_path is not None
                    else None
                ),
                "target_names": list(preview.target_names),
            }
        )
    except Exception as error:
        result_queue.put(
            {"type": "error", "message": f"{type(error).__name__}: {error}"}
        )
        return

    while True:
        cmd = command_queue.get()
        if cmd is None:
            return
        while True:
            try:
                newer = command_queue.get_nowait()
            except Empty:
                break
            if newer is None:
                return
            cmd = newer
        try:
            prediction = preview.predict(
                cmd["strip"], cmd["factory_sample"], cmd["factory_names"]
            )
            image = preview.render(
                prediction, output_enabled=bool(cmd.get("output_enabled"))
            )
            result_queue.put(
                {
                    "type": "frame",
                    "prediction": prediction,
                    "image": image,
                }
            )
        except Exception as error:
            result_queue.put(
                {"type": "error", "message": f"{type(error).__name__}: {error}"}
            )
            return


class TongueInferenceProcess:
    """Latest-frame-first sidecar that must not import torch in the parent."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device_name: str = "auto",
        direction_checkpoint_path: str | Path | None = None,
        smoothing: float = 0.35,
        visibility_mode: str = "weighted",
    ) -> None:
        ctx = multiprocessing.get_context("spawn")
        self._commands: multiprocessing.Queue[object] = ctx.Queue(maxsize=1)
        self._results: multiprocessing.Queue[object] = ctx.Queue(maxsize=2)
        self._process = ctx.Process(
            target=_process_main,
            args=(
                self._commands,
                self._results,
                {
                    "checkpoint_path": str(Path(checkpoint_path).resolve()),
                    "device_name": device_name,
                    "direction_checkpoint_path": (
                        str(Path(direction_checkpoint_path).resolve())
                        if direction_checkpoint_path
                        else None
                    ),
                    "smoothing": float(smoothing),
                    "visibility_mode": visibility_mode,
                },
            ),
            name="tongue-inference",
            daemon=False,
        )
        self._process.start()
        self.ready = False
        self.device = device_name
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.direction_checkpoint_path = (
            Path(direction_checkpoint_path).resolve()
            if direction_checkpoint_path
            else None
        )
        self.target_names: list[str] = []
        self._latest: tuple[object, np.ndarray] | None = None

    def pump(self) -> str | None:
        status = None
        while True:
            try:
                item = self._results.get_nowait()
            except Empty:
                break
            kind = item.get("type")
            if kind == "ready":
                self.ready = True
                self.device = str(item["device"])
                self.checkpoint_path = Path(item["checkpoint_path"])
                direction = item.get("direction_checkpoint_path")
                self.direction_checkpoint_path = (
                    Path(direction) if direction else None
                )
                self.target_names = list(item.get("target_names") or [])
                status = (
                    f"Loaded opt-in stereo tongue model on {self.device}: "
                    f"{self.checkpoint_path}"
                )
            elif kind == "frame":
                self._latest = (item["prediction"], item["image"])
            elif kind == "error":
                raise RuntimeError(
                    f"Tongue inference process failed: {item.get('message')}"
                )
        if not self.ready and self._process.exitcode not in (None, 0):
            raise RuntimeError(
                f"Tongue inference process exited with code {self._process.exitcode}"
            )
        return status

    def submit(
        self,
        strip: np.ndarray,
        factory_sample: dict[str, object] | None,
        factory_names: list[str],
        *,
        output_enabled: bool,
    ) -> None:
        if not self.ready:
            return
        payload = {
            "strip": np.ascontiguousarray(strip),
            "factory_sample": factory_sample,
            "factory_names": list(factory_names),
            "output_enabled": bool(output_enabled),
        }
        try:
            self._commands.put_nowait(payload)
        except Full:
            try:
                self._commands.get_nowait()
            except Empty:
                pass
            try:
                self._commands.put_nowait(payload)
            except Full:
                pass

    def latest(self) -> tuple[object | None, np.ndarray | None]:
        if self._latest is None:
            return None, None
        return self._latest

    def close(self) -> None:
        try:
            self._commands.put_nowait(None)
        except Exception:
            pass
        if self._process.is_alive():
            self._process.join(timeout=3.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
