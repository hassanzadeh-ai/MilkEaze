import numpy as np

from milkeaze.config import PipelineConfig
from milkeaze.data.quality import check_timestamps


def test_clean_timestamps_pass():
    pipeline = PipelineConfig.load()
    # 19 Hz strain -> ~52 ms spacing, well under the 250 ms dropout threshold
    t = np.arange(200) * (1000.0 / 19.0)
    rep = check_timestamps(t, pipeline, "strain")
    assert rep.ok
    assert rep.reasons == []


def test_dropout_is_flagged():
    pipeline = PipelineConfig.load()
    t = np.arange(200) * (1000.0 / 19.0)
    t[100:] += 500.0  # inject a 500 ms gap (> max_dropout_ms)
    rep = check_timestamps(t, pipeline, "strain")
    assert not rep.ok  # reject_on_fail is true in config
    assert any("dropouts" in r for r in rep.reasons)
