"""Hand-engineered per-window features (strain, acoustic, IMU) + assembly.

Two representations are built per window and fused downstream:
  (a) raw frames -> CNN embedding  (see models/cnn_encoder.py)
  (b) this hand-feature vector     (assemble.window_features)
"""

from .assemble import window_features, FEATURE_DIM

__all__ = ["window_features", "FEATURE_DIM"]
