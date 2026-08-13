# MilkEaze - ML Pipeline

Machine-learning brain for the MilkEaze breastfeeding monitor: turns a multi-sensor
recording of a feed (strain + microphone + IMU) into per-event **suck / swallow**
classification and an estimate of **milk volume (mL)**, validated against a gram scale.

```
ingestion → calibration → signal-quality → resampling → windowing
→ feature extraction (hand features + CNN embedding) → CNN+LSTM (classification + volume)
→ session aggregation (suck count, swallow count, cumulative volume)
```

## Status

| Area | State |
|------|-------|
| Data contract + ingestion (flat dev layout **and** production dual-board captures) | working |
| Calibration (ADC→physical units) + EMA drift compensation | working |
| Cross-board clock alignment (per-board skew, pooled fallback) | working |
| Multi-rate resampling to a unified window grid | working |
| Signal-quality / fail-loud checks | working |
| Windowing (2 s / 50% overlap) | working |
| Feature extraction (strain, acoustic, IMU) | working |
| Model backbone (CNN encoder + fusion + LSTM + dual heads) | implemented |
| Synthetic generator + synthetic pretraining | working (runnable) |
| **Volume-regression head** training | trains against scale grams |
| **Per-event suck labels** from rig pressure | working — derived, 42.0 cpm at CV 0.02 on all 5 captures |
| **Classifier** training | suck unblocked on rig data; **swallow still has no ground truth** |
| Sidecar schema validation | working (graded report, pending freeze) |
| Real-data fine-tuning | partial (single condition axis, ~1,200 windows) |
| Edge export / Nordic nRF54LM20B compression | not started (placeholder) |

## Labels: half the blocker is retired

The scale gives clean continuous **volume** ground truth, so the volume head trains.
The classifier needed something marking *where* each event occurs.

On the pump rig, the vacuum-line pressure cycle **is** the suck event, so
`milkeaze/data/pressure_events.py` derives an `events.csv` (`t_ms, type=suck`) directly
from the rig board, together with a pinned parameter file recording exactly how it was
produced. `t_ms` marks **peak suction** by convention.

What is still missing is **swallow**: the pump has no swallow mechanism, so no amount
of rig data produces those labels. That half of the classifier stays blocked until live
testing supplies a swallow ground truth.

## Quick start

```bash
pip install -r requirements.txt

python scripts/pretrain_synthetic.py                          # synthetic data only
python scripts/validate_dataset.py --root new_dataset/20260718   # grade the sidecars
python scripts/derive_events.py  --root new_dataset/20260718 --dry-run   # tune the detector
python scripts/derive_events.py  --root new_dataset/20260718            # write events.csv
python scripts/run_pipeline.py   --session new_dataset/20260718 --stem test_vac1_20260718_130431
```

## Session layouts

`milkeaze.data.load_session` detects which of two layouts a directory holds:

- **flat** — `strain.csv` / `imu.csv` / `mic.csv` / `scale.csv` + `meta.json`, one shared
  `t_ms` column. Written by `milkeaze.synthetic.write_session` for tests.
- **rig** — the production capture: a sensor board and a rig board, each on its own
  device clock, several captures per directory. Select one with `stem=`.

Sensor schema is in `configs/sensors.yaml` (16 model channels: 8 strain + 2 mic + 6 IMU;
scale and rig pressure are ground truth, not inputs). The mic contract keeps **two**
slots even though the hardware is currently mono, so the trained model shape survives
the second microphone being repaired.

## Data

- `new_dataset/20260718` — 5 sessions, vacuum L1–5, with the pressure/temperature rig
  board. Runs end to end; ~1,200 windows on a single condition axis, so it validates the
  pipeline but cannot fine-tune the model. See `reports/vac_sweep_20260718_report.md`.
- `RuthV0_bottle` — 7 bottle sessions (pre-ring-mount).
