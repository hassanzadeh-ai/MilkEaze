import numpy as np

from milkeaze.data.resampling import make_grid, resample_linear


def test_resample_linear_recovers_line():
    # a straight line resampled onto a finer grid stays the same line
    t_src = np.linspace(0, 1000, 20)      # ms
    x = (2.0 * t_src + 5.0)[:, None]
    grid = make_grid(0, 1000, grid_hz=200.0)
    out = resample_linear(t_src, x, grid)
    assert np.allclose(out[:, 0], 2.0 * grid + 5.0, atol=1e-6)


def test_grid_spacing():
    grid = make_grid(0, 1000, grid_hz=200.0)
    dt = np.diff(grid)
    assert np.allclose(dt, 5.0)  # 200 Hz -> 5 ms
