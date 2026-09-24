#!/usr/bin/env python3
"""Torch-free VRCFT tongue UDP override and shared prediction types."""

from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass

import numpy as np


TONGUE_PACKET = struct.Struct("<4sBBH12f")
TONGUE_MAGIC = b"QPTO"
TONGUE_VERSION = 1


@dataclass(frozen=True)
class TonguePrediction:
    values: np.ndarray
    native_tongue_out: float
    fused_visibility: float
    visible: bool
    inference_ms: float
    pipeline_ms: float = 0.0
    dropped_frames: int = 0


def vrcft_tongue_values(
    prediction: TonguePrediction, target_names: list[str]
) -> np.ndarray:
    """Map ten model heads to VRCFT's twelve detailed tongue expressions."""
    values = {
        name: float(prediction.values[index])
        for index, name in enumerate(target_names)
    }
    if not prediction.visible:
        return np.zeros(12, dtype=np.float32)
    horizontal = float(np.clip(values.get("horizontal", 0.0), -1.0, 1.0))
    vertical = float(np.clip(values.get("vertical", 0.0), -1.0, 1.0))
    twist = float(np.clip(values.get("twist", 0.0), -1.0, 1.0))
    tongue_out = max(
        float(np.clip(prediction.fused_visibility, 0.0, 1.0)),
        float(np.clip(values.get("extension", 0.0), 0.0, 1.0)),
    )
    return np.asarray(
        [
            tongue_out,
            max(vertical, 0.0),
            max(-vertical, 0.0),
            max(-horizontal, 0.0),
            max(horizontal, 0.0),
            float(np.clip(values.get("roll", 0.0), 0.0, 1.0)),
            float(np.clip(values.get("bend_down", 0.0), 0.0, 1.0)),
            float(np.clip(values.get("curl_up", 0.0), 0.0, 1.0)),
            float(np.clip(values.get("squish", 0.0), 0.0, 1.0)),
            float(np.clip(values.get("flat", 0.0), 0.0, 1.0)),
            max(-twist, 0.0),
            max(twist, 0.0),
        ],
        dtype=np.float32,
    )


def encode_tongue_packet(values: np.ndarray, enabled: bool) -> bytes:
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (12,):
        raise ValueError("A VRCFT tongue packet needs exactly twelve values")
    return TONGUE_PACKET.pack(
        TONGUE_MAGIC, TONGUE_VERSION, int(enabled), 0,
        *[float(np.clip(value, 0.0, 1.0)) for value in values],
    )


class TongueBroadcaster:
    """Opt-in UDP override; disabled or stale packets restore stock tracking."""

    def __init__(self, port: int = 27276, *, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self._address = ("127.0.0.1", int(port))
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._last_values: np.ndarray | None = None
        self._last_sent = 0.0
        self._minimum_interval = 1.0 / 24.0
        self._keepalive_interval = 0.20

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        if not self.enabled:
            self._send(np.zeros(12, dtype=np.float32), enabled=False)
        return self.enabled

    def send_prediction(
        self, prediction: TonguePrediction, target_names: list[str]
    ) -> None:
        if not self.enabled:
            return
        values = vrcft_tongue_values(prediction, target_names)
        now = time.perf_counter()
        elapsed = now - self._last_sent
        if elapsed < self._minimum_interval:
            return
        changed = (
            self._last_values is None
            or float(np.max(np.abs(values - self._last_values))) >= 0.015
        )
        if not changed and elapsed < self._keepalive_interval:
            return
        self._send(values, enabled=True)
        self._last_values = values.copy()
        self._last_sent = now

    def _send(self, values: np.ndarray, *, enabled: bool) -> None:
        self._socket.sendto(encode_tongue_packet(values, enabled), self._address)

    def close(self) -> None:
        try:
            self._send(np.zeros(12, dtype=np.float32), enabled=False)
        finally:
            self._socket.close()
