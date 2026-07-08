"""Pretrain the MilkEaze backbone on synthetic sessions.

Runnable end to end with no real data:

    python scripts/pretrain_synthetic.py --sessions 64 --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from milkeaze.config import ModelConfig
from milkeaze.features.assemble import FEATURE_DIM
from milkeaze.synthetic.generator import generate_dataset
from milkeaze.training.train import pretrain_synthetic
from milkeaze.utils.logging import get_logger

log = get_logger("pretrain")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=64)
    ap.add_argument("--windows", type=int, default=120)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--ckpt-dir", type=str, default="checkpoints")
    args = ap.parse_args()

    cfg = ModelConfig.load()
    log.info("generating %d synthetic sessions (feature_dim=%d)", args.sessions, FEATURE_DIM)
    sessions = generate_dataset(n_sessions=args.sessions, n_windows=args.windows,
                                in_channels=int(cfg.model["cnn"]["in_channels"]))

    pretrain_synthetic(
        sessions=sessions, model_cfg=cfg.model, training_cfg=cfg.training,
        hand_feature_dim=FEATURE_DIM, device=args.device, ckpt_dir=args.ckpt_dir,
    )


if __name__ == "__main__":
    main()
