import torch

from milkeaze.config import ModelConfig
from milkeaze.features.assemble import FEATURE_DIM
from milkeaze.models import MilkEazeNet, ModelDims


def test_forward_shapes():
    cfg = ModelConfig.load()
    dims = ModelDims.from_config(cfg.model, FEATURE_DIM)
    model = MilkEazeNet(dims)

    b, seq, C, T = 2, 5, dims.cnn_in_channels, 400
    frames = torch.randn(b, seq, C, T)
    hand = torch.randn(b, seq, FEATURE_DIM)

    out = model(frames, hand)
    assert out["logits"].shape == (b, seq, dims.num_classes)
    assert out["volume"].shape == (b, seq)
    assert torch.all(out["volume"] >= 0)  # Softplus -> non-negative volume
