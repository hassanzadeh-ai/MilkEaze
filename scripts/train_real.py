"""Fine-tune a synthetic-pretrained model on real session directories.

    python scripts/train_real.py --data-root path/to/sessions --checkpoint checkpoints/synthetic_pretrained.pt

Loads each session through the pipeline, builds SessionSamples, and fine-tunes. The
classifier only trains for sessions that ship an events.csv; otherwise only the volume
head updates (current data limitation — see milkeaze/data/labels.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from milkeaze.config import ModelConfig
from milkeaze.data.rig_session import discover_stems
from milkeaze.features.assemble import FEATURE_DIM
from milkeaze.models import MilkEazeNet, ModelDims
from milkeaze.pipeline import process_session
from milkeaze.training.dataset import SessionSample
from milkeaze.training.train import finetune_real
from milkeaze.utils.logging import get_logger

log = get_logger("train_real")


def _iter_sessions(root: Path) -> list[tuple[Path, str | None]]:
    """Yield ``(directory, stem)`` pairs for every session under ``root``.

    A rig-layout directory holds several captures side by side rather than one per
    subdirectory, so it is enumerated by stem instead of by folder.
    """
    stems = discover_stems(root)
    if stems:
        return [(root, stem) for stem in stems]
    return [(d, None) for d in sorted(root.iterdir()) if d.is_dir()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    cfg = ModelConfig.load()
    dims = ModelDims.from_config(cfg.model, FEATURE_DIM)
    model = MilkEazeNet(dims).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(state["model_state"])

    samples: list[SessionSample] = []
    for session_dir, stem in _iter_sessions(Path(args.data_root)):
        name = stem or session_dir.name
        try:
            proc = process_session(session_dir, stem=stem)
        except Exception as exc:  # fail loud per-session, keep going
            log.error("skipping %s: %s", name, exc)
            continue
        samples.append(SessionSample(frames=proc.frames, hand=proc.hand,
                                     labels=proc.labels, volume=proc.volume))

    if not samples:
        log.error("no usable sessions found under %s", args.data_root)
        return

    finetune_real(model, samples, cfg.training, device=args.device)


if __name__ == "__main__":
    main()
