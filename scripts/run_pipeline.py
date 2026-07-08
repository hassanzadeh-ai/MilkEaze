"""Run the end-to-end signal pipeline on one real session directory and print a summary.

    python scripts/run_pipeline.py --session path/to/session_dir

Runs ingestion -> calibration -> resampling -> windowing -> features. If a trained
checkpoint is given, also runs inference and prints the session summary.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from milkeaze.config import ModelConfig
from milkeaze.features.assemble import FEATURE_DIM
from milkeaze.inference.predict import predict_session
from milkeaze.models import MilkEazeNet, ModelDims
from milkeaze.pipeline import process_session
from milkeaze.utils.logging import get_logger

log = get_logger("run_pipeline")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    proc = process_session(args.session)
    log.info("windows=%d frames=%s hand=%s has_volume=%s has_labels=%s",
             len(proc.windows), proc.frames.shape, proc.hand.shape,
             proc.volume is not None, proc.labels is not None)

    if args.checkpoint:
        cfg = ModelConfig.load()
        dims = ModelDims.from_config(cfg.model, FEATURE_DIM)
        model = MilkEazeNet(dims).to(args.device)
        state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(state["model_state"])
        result = predict_session(model, proc.frames, proc.hand, device=args.device)
        log.info("SESSION SUMMARY: %s", result.summary())


if __name__ == "__main__":
    main()
