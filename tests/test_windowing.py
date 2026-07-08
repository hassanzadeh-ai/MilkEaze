import numpy as np

from milkeaze.config import PipelineConfig
from milkeaze.data.windowing import make_windows


def test_window_count_and_overlap():
    pipeline = PipelineConfig.load()
    grid_hz = pipeline.target_grid_hz
    n = int(10 * grid_hz)  # 10 seconds
    t = np.arange(n) * (1000.0 / grid_hz)
    frames = np.random.randn(n, 16).astype(np.float32)

    windows = make_windows(t, frames, pipeline)

    win_len = int(pipeline.window_s * grid_hz)
    hop = int(pipeline.hop_s * grid_hz)
    expected = (n - win_len) // hop + 1
    assert len(windows) == expected
    assert windows[0].frames.shape == (win_len, 16)
    # 50% overlap -> second window starts one hop in
    assert windows[1].start_sample == hop
