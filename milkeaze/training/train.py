"""Training loops.

``pretrain_synthetic`` is fully functional and runnable: it trains the whole model
(both heads) on synthetic sessions, which have both class labels and per-window volume.

``finetune_real`` is the path we take once real sessions arrive. It's implemented for
the volume head; the classifier portion is gated on per-event labels being present
(see the guard) and is otherwise skipped — this is the current real-data limitation.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..models import MilkEazeNet, ModelDims
from ..synthetic.generator import SyntheticSession
from ..utils.logging import get_logger
from ..utils.seed import set_seed
from .dataset import SessionDataset, SessionSample, collate_sessions
from .losses import combined_loss

log = get_logger(__name__)


def _synth_to_sample(s: SyntheticSession) -> SessionSample:
    return SessionSample(frames=s.frames, hand=s.hand, labels=s.labels, volume=s.volume)


def build_model(model_cfg: dict, hand_feature_dim: int, device: str = "cpu") -> MilkEazeNet:
    dims = ModelDims.from_config(model_cfg, hand_feature_dim)
    return MilkEazeNet(dims).to(device)


def _run_epoch(model, loader, optimizer, cfg, device, train_classifier, train=True):
    model.train(train)
    w = cfg["loss_weights"]
    agg = {"loss": 0.0, "cls": 0.0, "vol": 0.0}
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch["frames"], batch["hand"], batch["lengths"])
        loss, parts = combined_loss(
            outputs, batch,
            w_cls=float(w["classification"]), w_vol=float(w["volume"]),
            train_classifier=train_classifier,
        )
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        for k in agg:
            agg[k] += parts[k]
        n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}


def pretrain_synthetic(sessions: list[SyntheticSession], model_cfg: dict, training_cfg: dict,
                       hand_feature_dim: int, device: str = "cpu",
                       ckpt_dir: str | Path = "checkpoints") -> MilkEazeNet:
    set_seed(int(training_cfg.get("seed", 1337)))
    model = build_model(model_cfg, hand_feature_dim, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    ds = SessionDataset([_synth_to_sample(s) for s in sessions])
    loader = DataLoader(ds, batch_size=int(training_cfg["batch_size"]),
                        shuffle=True, collate_fn=collate_sessions)

    epochs = int(training_cfg["epochs_synthetic"])
    for ep in range(1, epochs + 1):
        stats = _run_epoch(model, loader, optimizer, training_cfg, device,
                           train_classifier=True, train=True)
        if ep % max(epochs // 10, 1) == 0 or ep == 1:
            log.info("[synthetic] epoch %d/%d loss=%.4f cls=%.4f vol=%.4f",
                     ep, epochs, stats["loss"], stats["cls"], stats["vol"])

    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / "synthetic_pretrained.pt"
    torch.save({"model_state": model.state_dict(), "dims": model.dims.__dict__}, path)
    log.info("saved synthetic-pretrained checkpoint to %s", path)
    return model


def finetune_real(model: MilkEazeNet, samples: list[SessionSample], training_cfg: dict,
                  device: str = "cpu") -> MilkEazeNet:
    """Fine-tune on real sessions.

    Volume head trains against the scale. The classifier only trains if per-event
    labels are present in the batch; otherwise it's skipped (current limitation).
    """
    any_labels = any(s.labels is not None for s in samples)
    require = bool(training_cfg.get("require_event_labels_for_classifier", True))
    train_classifier = any_labels or not require
    if not train_classifier:
        log.warning(
            "no per-event labels found in real sessions -> training VOLUME head only. "
            "Provide events.csv (t_ms,type) to unblock the classifier."
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    ds = SessionDataset(samples)
    loader = DataLoader(ds, batch_size=int(training_cfg["batch_size"]),
                        shuffle=True, collate_fn=collate_sessions)

    epochs = int(training_cfg["epochs_finetune"])
    for ep in range(1, epochs + 1):
        stats = _run_epoch(model, loader, optimizer, training_cfg, device,
                           train_classifier=train_classifier, train=True)
        log.info("[finetune] epoch %d/%d loss=%.4f cls=%.4f vol=%.4f",
                 ep, epochs, stats["loss"], stats["cls"], stats["vol"])
    return model
