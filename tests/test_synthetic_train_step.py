import torch

from milkeaze.config import ModelConfig
from milkeaze.features.assemble import FEATURE_DIM
from milkeaze.synthetic.generator import generate_dataset
from milkeaze.training.dataset import SessionDataset, SessionSample, collate_sessions
from milkeaze.training.losses import combined_loss
from milkeaze.training.train import build_model
from torch.utils.data import DataLoader


def test_one_training_step_decreases_or_runs():
    cfg = ModelConfig.load()
    sessions = generate_dataset(n_sessions=4, n_windows=20,
                                in_channels=int(cfg.model["cnn"]["in_channels"]))
    samples = [SessionSample(s.frames, s.hand, s.labels, s.volume) for s in sessions]
    loader = DataLoader(SessionDataset(samples), batch_size=2, collate_fn=collate_sessions)

    model = build_model(cfg.model, FEATURE_DIM)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch = next(iter(loader))
    out = model(batch["frames"], batch["hand"], batch["lengths"])
    loss, parts = combined_loss(out, batch)
    opt.zero_grad()
    loss.backward()
    opt.step()

    assert parts["loss"] >= 0
    assert parts["vol"] >= 0
