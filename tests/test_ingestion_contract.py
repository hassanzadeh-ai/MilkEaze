import json

import numpy as np
import pytest

from milkeaze.config import SensorConfig
from milkeaze.data.contract import ContractError, validate_session
from milkeaze.data.ingestion import (
    MAX_BACKSTEP_MS, RawSession, SessionLayout, detect_layout, load_session, repair_monotonic,
)
from milkeaze.synthetic.write_session import write_session


def _raw(sensors, n=500, rate_hz=50.0):  # 10 s, comfortably past the overlap floor
    t = np.arange(n) * (1000.0 / rate_hz)
    return RawSession(
        session_id="unit",
        strain_t_ms=t, imu_t_ms=t, mic_t_ms=t, scale_t_ms=t,
        strain=np.zeros((n, len(sensors.strain_channel_names())), np.float32),
        imu=np.zeros((n, len(sensors.imu_channel_names())), np.float32),
        mic=np.zeros((n, len(sensors.mic_channel_names())), np.float32),
        scale_g=np.linspace(0, 50, n).astype(np.float32),
        has_scale=True,
        measured_rates_hz={"strain": rate_hz, "imu": rate_hz, "mic": rate_hz},
    )


def test_valid_session_passes():
    sensors = SensorConfig.load()
    validate_session(_raw(sensors), sensors)


def test_wrong_channel_count_is_a_contract_error():
    sensors = SensorConfig.load()
    raw = _raw(sensors)
    raw.strain = raw.strain[:, :4]
    with pytest.raises(ContractError, match="expected 8 channels"):
        validate_session(raw, sensors)


def test_non_overlapping_streams_are_a_contract_error():
    sensors = SensorConfig.load()
    raw = _raw(sensors)
    raw.mic_t_ms = raw.mic_t_ms + 1e6  # mic recorded a different 4 seconds entirely
    with pytest.raises(ContractError, match="overlap"):
        validate_session(raw, sensors)


def test_nan_samples_are_a_contract_error():
    sensors = SensorConfig.load()
    raw = _raw(sensors)
    raw.imu[3, 2] = np.nan
    with pytest.raises(ContractError, match="non-finite"):
        validate_session(raw, sensors)


def test_empty_stream_is_a_contract_error():
    sensors = SensorConfig.load()
    raw = _raw(sensors)
    raw.strain = np.zeros((0, 8), np.float32)
    raw.strain_t_ms = np.zeros(0)
    with pytest.raises(ContractError, match="empty"):
        validate_session(raw, sensors)


def test_repair_clamps_transport_jitter():
    t = np.arange(10, dtype=float) * 4.0
    t[5] = t[4] - 2.0  # block arrives stamped before the previous one ended
    repaired, stats = repair_monotonic(t, "mic")
    assert stats["n_backsteps"] == 1
    assert stats["worst_backstep_ms"] == pytest.approx(2.0)
    assert np.all(np.diff(repaired) >= 0)
    assert repaired[5] == pytest.approx(t[4])  # clamped forward, order preserved


def test_repair_refuses_a_large_reordering():
    t = np.arange(10, dtype=float) * 4.0
    t[5] -= MAX_BACKSTEP_MS * 3
    with pytest.raises(ValueError, match="not trustworthy"):
        repair_monotonic(t, "mic")


def test_monotonic_stream_is_returned_unchanged():
    t = np.arange(10, dtype=float) * 4.0
    repaired, stats = repair_monotonic(t, "strain")
    assert stats["n_backsteps"] == 0
    assert np.array_equal(repaired, t)


def test_flat_layout_round_trips_through_the_loader(tmp_path):
    sensors = SensorConfig.load()
    session_dir = write_session(tmp_path / "sess", duration_s=20.0, seed=0, sensors=sensors)

    assert detect_layout(session_dir) is SessionLayout.FLAT
    raw = load_session(session_dir, sensors)
    validate_session(raw, sensors)

    assert raw.strain.shape[1] == len(sensors.strain_channel_names())
    assert raw.mic.shape[1] == len(sensors.mic_channel_names())
    assert raw.has_scale
    assert raw.measured_rates_hz["imu"] == pytest.approx(sensors.sample_rates["imu"], rel=0.05)
    assert json.loads((session_dir / "meta.json").read_text())["session_id"] == raw.session_id


def test_unrecognised_directory_raises(tmp_path):
    (tmp_path / "nothing.txt").write_text("x")
    with pytest.raises(FileNotFoundError, match="no recognised session layout"):
        detect_layout(tmp_path)
