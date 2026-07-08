"""Edge export / compression for the Nordic nRF54LM20B — NOT STARTED.

Plan (per the two-stage strategy): once the off-device model establishes the accuracy
ceiling, compress it to fit the SoC's memory/power budget and run inference locally in
real time (no cloud). This module is a placeholder that documents the intended path and
fails loudly if called, so nobody assumes an edge build exists yet.

Intended steps:
  1. quantization-aware training / post-training int8 quantization
  2. structured pruning of the CNN + LSTM
  3. reduce the acoustic path to a compact feature set (mic model is bandwidth-heavy
     and the full-audio model will NOT fit the SoC)
  4. export to a runtime deployable on the Nordic toolchain (e.g. TFLite Micro / CMSIS-NN)
  5. on-device latency/power validation against the 15-20 min session budget
"""
from __future__ import annotations

from ..models import MilkEazeNet


def export_for_nordic(model: MilkEazeNet, out_path: str) -> None:  # pragma: no cover
    raise NotImplementedError(
        "Edge export for nRF54LM20B is not implemented yet. This is stage 2 of the "
        "plan (compress the off-device model). See module docstring for the roadmap."
    )
