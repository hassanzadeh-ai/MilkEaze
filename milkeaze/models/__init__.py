"""Model backbone: per-window CNN encoder -> fuse with hand features -> LSTM over
windows -> classification + volume-regression heads."""

from .milkeaze_net import MilkEazeNet, ModelDims

__all__ = ["MilkEazeNet", "ModelDims"]
