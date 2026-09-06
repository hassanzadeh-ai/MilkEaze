"""Tests for the dual-board capture reader, built on a fabricated capture.

The clock model is the part of ingestion most able to fail silently: an inverted skew
still yields a smooth, plausible timebase that is simply wrong by a few milliseconds
per minute. These tests fix the convention (``host = (1 + skew) * device + offset``)
against a capture whose true skew is known by construction.
"""
import json

import numpy as np
import pandas as pd
import pytest

from milkeaze.config import SensorConfig
from milkeaze.data.rig_session import (
    POOLED, SENSOR_BOARD, STRICT, discover_stems, fill_dropped_strain, load_pressure,
    load_rig_session, load_temperature, open_capture, pooled_skew_ppm,
)
from milkeaze.data.schema import validate_sidecars

STRAIN_COLS = ["ch0_Radial_N", "ch1_Radial_E", "ch2_Radial_S", "ch3_Radial_W",
               "ch4_Arc_Inner_E", "ch5_Arc_Outer_E", "ch6_Arc_Inner_W", "ch7_Arc_Outer_W"]

P_OUT_MIN, P_OUT_MAX, P_MIN, P_MAX = 1677722.0, 15099494.0, -15.0, 15.0


def _psi_to_raw(psi):
    return (psi - P_MIN) * (P_OUT_MAX - P_OUT_MIN) / (P_MAX - P_MIN) + P_OUT_MIN


def _conversion():
    return {
        "imu": {"accel_ms2_per_count": 0.00059820565, "gyro_dps_per_count": 0.00875},
        "rig": {
            "pressure": {"part": "ABP2", "out_min_counts": P_OUT_MIN, "out_max_counts": P_OUT_MAX,
                         "p_min_psi": P_MIN, "p_max_psi": P_MAX, "units": "psi"},
            "tmp117": {"c_per_count": 0.0078125},
        },
        "strain": {"stretch_rref_ohm": 1500.0, "bend_rref_ohm": 4700.0},
    }


def write_capture(root, stem, duration_s=60.0, sensor_skew_ppm=25.0, rig_skew_ppm=-35.0,
                  sensor_confidence="resolved", sensor_se_ppm=2.0, cpm=42.0,
                  host_t0=1000.0, device_t0_s=500.0):
    """Fabricate a dual-board capture whose true clock relationship is known.

    Device time advances such that ``host = (1 + skew*1e-6) * device + offset``.
    """
    root.mkdir(parents=True, exist_ok=True)

    def device_us(host_s, skew_ppm, dev0):
        # invert the documented relation to place a host instant on the device clock,
        # then wrap it the way a 32-bit microsecond counter does on the hardware
        exact = ((host_s - host_t0) / (1.0 + skew_ppm * 1e-6) + dev0) * 1e6
        return exact % (2 ** 32)

    for board, skew in ((SENSOR_BOARD, sensor_skew_ppm), ("rig", rig_skew_ppm)):
        host = host_t0 + np.arange(0.0, duration_s, 1.0)
        pd.DataFrame({"host_mono_s": host,
                      "device_ts_us": device_us(host, skew, device_t0_s)}
                     ).to_csv(root / f"{stem}_{board}_sync.csv", index=False)

    # sensor board streams
    n_strain = int(duration_s * 80)
    host = host_t0 + np.linspace(0, duration_s, n_strain, endpoint=False)
    strain = pd.DataFrame({"scan_t_us": device_us(host, sensor_skew_ppm, device_t0_s)})
    for i, col in enumerate(STRAIN_COLS):
        strain[col] = np.linspace(0, 1000, n_strain) + i
    strain.to_csv(root / f"{stem}_sensor_strain.csv", index=False)

    n_imu = int(duration_s * 416)
    host = host_t0 + np.linspace(0, duration_s, n_imu, endpoint=False)
    imu = pd.DataFrame({"time_us": device_us(host, sensor_skew_ppm, device_t0_s)})
    for col, val in zip(["ax_ms2", "ay_ms2", "az_ms2"], [0.1, 0.2, 9.7]):
        imu[col] = val + np.linspace(0, 0.01, n_imu)
    for col in ["gx_dps", "gy_dps", "gz_dps"]:
        imu[col] = np.linspace(0, 0.5, n_imu)
    imu.to_csv(root / f"{stem}_sensor_imu.csv", index=False)

    n_audio = int(duration_s * 2000)
    host = host_t0 + np.linspace(0, duration_s, n_audio, endpoint=False)
    pd.DataFrame({
        "time_us": device_us(host, sensor_skew_ppm, device_t0_s),
        "block_idx": np.arange(n_audio) // 32,
        "left": (300 * np.sin(np.linspace(0, 200, n_audio))).astype(int),
        "right": np.full(n_audio, 9999),  # the faulty mic: constant garbage
    }).to_csv(root / f"{stem}_sensor_audio.csv", index=False)

    n_scale = int(duration_s * 10)
    pd.DataFrame({
        "host_mono_s": host_t0 + np.linspace(0, duration_s, n_scale, endpoint=False),
        "grams": np.linspace(0, 103.0, n_scale),
    }).to_csv(root / f"{stem}_sensor_scale.csv", index=False)

    # rig board streams: a clean pump cycle at `cpm`
    n_press = int(duration_s * 180)
    host = host_t0 + np.linspace(0, duration_s, n_press, endpoint=False)
    psi = -1.0 * np.cos(2 * np.pi * (cpm / 60.0) * (host - host_t0)) - 0.5
    pd.DataFrame({"time_us": device_us(host, rig_skew_ppm, device_t0_s),
                  "p_raw": _psi_to_raw(psi), "t_raw": 6300000, "status": 64}
                 ).to_csv(root / f"{stem}_rig_pressure.csv", index=False)

    n_temp = int(duration_s)
    host = host_t0 + np.arange(n_temp, dtype=float)
    pd.DataFrame({"time_us": device_us(host, rig_skew_ppm, device_t0_s),
                  "tmp117_raw": 2936}).to_csv(root / f"{stem}_rig_temp.csv", index=False)

    def alignment(skew, confidence, se):
        return {"skew_ppm": skew, "skew_se_ppm": se,
                "skew_confidence": confidence,
                "skew_applied": skew if confidence == "resolved" else 0.0,
                "jitter_ms": 0.3, "duration_s": duration_s, "n_sync_points": int(duration_s)}

    (root / f"{stem}_session.json").write_text(json.dumps({
        "session": stem, "elapsed_s": duration_s, "interrupted": False,
        "boards": {"sensor": {"stem": f"{stem}_sensor"}, "rig": {"stem": f"{stem}_rig"}},
        "alignment": {
            "sensor": alignment(sensor_skew_ppm, sensor_confidence, sensor_se_ppm),
            "rig": alignment(rig_skew_ppm, "resolved", 0.3),
        },
    }), encoding="utf-8")

    (root / f"{stem}_sensor.json").write_text(json.dumps({
        "session": f"{stem}_sensor",
        "device": {"strain_ch_mask": 0,
                   "active": {"imu": True, "bend": True, "stretch": True,
                              "mic_l": True, "mic_r": True}},
        "conversion": _conversion(),
        "run": {"imu_mic_mounting": "ring", "orientation": "up-2", "vacuum_level": 3,
                "cycle_rate_cpm": cpm, "weight_before_g": None, "weight_after_g": None},
        "files": {"strain": f"{stem}_sensor_strain.csv"},
    }), encoding="utf-8")

    (root / f"{stem}_rig.json").write_text(json.dumps({
        "session": f"{stem}_rig", "conversion": _conversion(),
        "run": {"orientation": "up-1", "cycle_rate_cpm": cpm},
        "files": {"pressure": f"{stem}_rig_pressure.csv"},
    }), encoding="utf-8")
    return stem


def test_clock_maps_device_time_onto_host_time(tmp_path):
    write_capture(tmp_path, "cap", sensor_skew_ppm=25.0, rig_skew_ppm=-35.0)
    capture = open_capture(tmp_path, "cap")

    for board, expected in ((SENSOR_BOARD, 25.0), ("rig", -35.0)):
        clock = capture.clocks[board]
        assert clock.skew_ppm == pytest.approx(expected)
        # the fabricated sync points must land back on host time to within a microsecond
        sync = pd.read_csv(tmp_path / f"cap_{board}_sync.csv")
        recovered = clock.to_host_s(sync["device_ts_us"].to_numpy())
        assert np.allclose(recovered, sync["host_mono_s"].to_numpy(), atol=1e-6)


def test_boards_share_one_timebase(tmp_path):
    """Two clocks with opposite skew must still land on the same session timeline."""
    write_capture(tmp_path, "cap", duration_s=60.0)
    raw = load_rig_session(tmp_path, SensorConfig.load(), stem="cap")
    t_ms, _ = load_pressure(open_capture(tmp_path, "cap"))
    # the two streams run at different rates, so they can only agree to within one
    # sample of the coarser one; a mishandled skew would drift by far more than that
    strain_period_ms = float(np.median(np.diff(raw.strain_t_ms)))
    assert raw.strain_t_ms[0] == pytest.approx(t_ms[0], abs=strain_period_ms)
    assert raw.strain_t_ms[-1] == pytest.approx(t_ms[-1], abs=strain_period_ms)


def test_pooled_skew_ignores_unresolved_captures(tmp_path):
    write_capture(tmp_path, "good1", sensor_skew_ppm=24.0, sensor_se_ppm=2.0)
    write_capture(tmp_path, "good2", sensor_skew_ppm=20.0, sensor_se_ppm=2.0)
    write_capture(tmp_path, "bad", sensor_skew_ppm=-99.0,
                  sensor_confidence="unresolved", sensor_se_ppm=40.0)

    pooled = pooled_skew_ppm(tmp_path, SENSOR_BOARD)
    assert pooled == pytest.approx(22.0, abs=0.5)  # -99 must not drag it
    assert set(discover_stems(tmp_path)) == {"good1", "good2", "bad"}


def test_unresolved_capture_falls_back_to_the_pooled_skew(tmp_path):
    write_capture(tmp_path, "good", sensor_skew_ppm=24.0, sensor_se_ppm=2.0)
    write_capture(tmp_path, "bad", sensor_skew_ppm=-99.0,
                  sensor_confidence="unresolved", sensor_se_ppm=40.0)

    clock = open_capture(tmp_path, "bad", skew_fallback=POOLED).clocks[SENSOR_BOARD]
    assert clock.skew_source == "pooled"
    assert clock.skew_ppm == pytest.approx(24.0, abs=0.5)

    kept = open_capture(tmp_path, "bad", skew_fallback="session").clocks[SENSOR_BOARD]
    assert kept.skew_source == "session"
    assert kept.skew_ppm == pytest.approx(-99.0)


def test_strict_mode_refuses_an_unresolved_clock(tmp_path):
    write_capture(tmp_path, "bad", sensor_confidence="unresolved")
    with pytest.raises(ValueError, match="unresolved"):
        open_capture(tmp_path, "bad", skew_fallback=STRICT)


def test_ambiguous_directory_requires_a_stem(tmp_path):
    write_capture(tmp_path, "cap1")
    write_capture(tmp_path, "cap2")
    with pytest.raises(ValueError, match="pass stem="):
        open_capture(tmp_path)


def test_faulty_mic_is_zero_filled_but_keeps_the_contract_width(tmp_path):
    write_capture(tmp_path, "cap")
    sensors = SensorConfig.load()
    raw = load_rig_session(tmp_path, sensors, stem="cap")

    contract = sensors.mic_channel_names()
    assert raw.mic.shape[1] == len(contract)
    for i, name in enumerate(contract):
        if name in sensors.mic_active_channels():
            assert np.any(raw.mic[:, i] != 0)
        else:
            assert np.all(raw.mic[:, i] == 0)  # the 9999 garbage never reaches the model


def test_imu_is_marked_already_physical(tmp_path):
    write_capture(tmp_path, "cap")
    raw = load_rig_session(tmp_path, SensorConfig.load(), stem="cap")
    assert raw.unit_of("imu") == "physical"
    assert raw.unit_of("strain") == "counts"


def test_pressure_converts_with_suction_negative(tmp_path):
    write_capture(tmp_path, "cap")
    _, psi = load_pressure(open_capture(tmp_path, "cap"))
    assert psi.min() < -1.0
    assert psi.mean() == pytest.approx(-0.5, abs=0.2)


#: places a 60 s run so that the counter rolls over 30 s in
_WRAP_S = 2 ** 32 / 1e6
_STRADDLES_THE_BOUNDARY = _WRAP_S - 30.0


def test_a_counter_rollover_mid_run_does_not_break_the_clock(tmp_path):
    """A run straddling the 71.6-minute boundary must still align.

    Without unwrapping, the fit does not degrade gracefully: it returns a slope near
    -0.045, i.e. device time running backwards at 45x, and every cross-board window in
    the capture lands somewhere arbitrary. This happened for real on the rig board of
    sweep_v2_c46_r1 in the 20260816 batch.
    """
    write_capture(tmp_path, "cap", duration_s=60.0, device_t0_s=_STRADDLES_THE_BOUNDARY)
    capture = open_capture(tmp_path, "cap")

    for board in ("sensor", "rig"):
        clock = capture.clocks[board]
        assert clock.wraps == 1, f"{board}: rollover not detected"
        assert clock.residual_ms < 1.0, f"{board}: residual {clock.residual_ms:.1f} ms"
        assert clock.slope == pytest.approx(1.0, abs=1e-4)

    t_ms, _ = load_pressure(capture)
    assert np.all(np.diff(t_ms) > 0)
    assert t_ms.max() - t_ms.min() == pytest.approx(60_000.0, rel=0.01)


def test_streams_are_placed_in_the_same_counter_epoch_as_the_sync_file(tmp_path):
    """A stream whose samples all sit past the rollover must not be read an epoch early."""
    write_capture(tmp_path, "cap", duration_s=60.0, device_t0_s=_STRADDLES_THE_BOUNDARY)
    capture = open_capture(tmp_path, "cap")

    # the temperature stream is sparse enough that its own samples may show no rollover,
    # so its epoch has to come from the clock rather than from the stream itself
    t_ms, degc = load_temperature(capture)
    assert degc.size > 0
    assert t_ms.min() > -1_000.0
    assert t_ms.max() < 120_000.0, f"temperature landed at {t_ms.max():.0f} ms, an epoch out"

    strain_t, _ = load_pressure(capture)
    assert abs(float(np.median(t_ms)) - float(np.median(strain_t))) < 5_000.0


def test_schema_report_flags_the_known_gaps(tmp_path):
    write_capture(tmp_path, "cap")
    report = validate_sidecars(open_capture(tmp_path, "cap"))
    flagged = {issue.field for issue in report.issues}
    assert "schema" in flagged
    assert "fill" in flagged
    assert "run.outlet" in flagged
    assert "run.orientation" in flagged          # boards disagree by construction
    assert "device.strain_ch_mask" in flagged
    assert report.ok                              # all warnings, nothing unusable


def test_schema_report_errors_on_a_missing_conversion(tmp_path):
    write_capture(tmp_path, "cap")
    rig_path = tmp_path / "cap_rig.json"
    meta = json.loads(rig_path.read_text())
    del meta["conversion"]["rig"]["pressure"]
    rig_path.write_text(json.dumps(meta), encoding="utf-8")

    report = validate_sidecars(open_capture(tmp_path, "cap"))
    assert not report.ok
    assert any("pressure" in issue.field for issue in report.errors)


def test_dropped_strain_samples_are_interpolated_not_left_as_zeros():
    """A zero is a dropped sample in this format; a zero-phase filter cannot see a NaN."""
    ramp = np.arange(1.0, 21.0)
    block = np.column_stack([ramp.copy(), ramp.copy()])
    block[5, 0] = 0.0
    block[10:13, 1] = 0.0

    filled = fill_dropped_strain(block)
    assert not (filled == 0.0).any()
    assert np.isfinite(filled).all()
    assert filled[:, 0] == pytest.approx(ramp)
    assert filled[:, 1] == pytest.approx(ramp)


def test_a_strain_channel_with_no_real_samples_stays_flat():
    """Flat is what the in-band power check downstream needs, not an invented cycle."""
    block = np.column_stack([np.arange(1.0, 11.0), np.zeros(10)])
    filled = fill_dropped_strain(block)
    assert (filled[:, 1] == 0.0).all()


def test_fill_dropped_strain_rejects_a_single_channel_vector():
    with pytest.raises(ValueError, match="n_time, n_channels"):
        fill_dropped_strain(np.arange(10.0))
