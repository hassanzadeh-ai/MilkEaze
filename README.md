# MilkEaze - ML Pipeline

Machine-learning brain for the MilkEaze breastfeeding monitor: turns a multi-sensor
recording of a feed (strain + microphone + IMU) into per-event **suck / swallow**
classification and an estimate of **milk volume (mL)**, validated against a gram scale.

```
ingestion → calibration → signal-quality → resampling → windowing
→ feature extraction (hand features + CNN embedding) → CNN+LSTM (classification + volume)
→ session aggregation (suck count, swallow count, cumulative volume)
```

## Status (handover snapshot)

| Area | State |
|------|-------|
| Data contract + ingestion (CSV strain/mic/IMU + JSON sidecar) | working |
| Calibration (ADC→physical units) + EMA drift compensation | working |
| Multi-rate resampling to a unified window grid | working |
| Signal-quality / fail-loud checks | working |
| Windowing (2 s / 50% overlap) | working |
| Feature extraction (strain, acoustic, IMU) | working |
| Model backbone (CNN encoder + fusion + LSTM + dual heads) | implemented |
| Synthetic generator + synthetic pretraining | working (runnable) |
| **Volume-regression head** training | trains against scale grams |
| **Classifier** training | **blocked - no per-event labels yet** (see below) |
| Real-data fine-tuning | partial (waiting on labeled sessions) |
| Edge export / Nordic nRF54LM20B compression | not started (placeholder) |

## The #1 open blocker: per-event labels

The scale gives clean continuous **volume** ground truth, so the volume head trains.
But nothing marks *where* each suck/swallow occurs, so the classifier has no targets.
`milkeaze/data/labels.py` defines the intended `events.csv` (`t_ms, type`) contract;
until real event labels arrive, classifier training runs only on synthetic data.

## Quick start

```bash
pip install -r requirements.txt
python scripts/pretrain_synthetic.py --config configs/model.yaml   # runs on synthetic data
python scripts/run_pipeline.py --session path/to/session_dir       # end-to-end on one session
```

## Data

- `scale-vac-ramp` - 5 sessions, vacuum L1–5 (proves pipeline end-to-end; not enough to train).
- `RuthV0_bottle` - 7 bottle sessions.

Sensor schema is defined in `configs/sensors.yaml` (16 model channels: 8 strain + 2 mic + 6 IMU;
scale is ground truth, not an input).
