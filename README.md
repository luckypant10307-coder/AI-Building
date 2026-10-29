# Smart Property AI — prototype

Finds property problems before anyone can see them — a concealed pipe leak, roof
seepage, condensation, a running fixture — says which room, predicts when the
damage becomes visible, and tells the owner what to do about it.

This repository is the **sensor spine**: stages 1 and 2 of the model plan. It
runs end to end today on simulated readings, with no GPU and no model downloads.

## What is here, and what is not

| # | Component | Status |
|---|---|---|
| 1 | Sensor anomaly detector — Isolation Forest, one per room | built |
| 2 | Fault rules and localiser — building physics, no ML | built |
| 3 | Damage forecaster — days to visible damp or mould risk | built (trend; Chronos-2 optional) |
| 9 | Repair advisor — plain language, English and Hindi | built (templates; LLM optional) |
| — | Alert fusion + technician confirmation | built |
| — | Simulator, ingest API, dashboard | built |
| 4 | Learned fault classifier (XGBoost) | dataset export only |
| 5 | Crack detection and segmentation | **not built** — needs a GPU fine-tune |
| 6 | Crack width and growth (OpenCV + ArUco) | **not built** |
| 7 | Damp and mould classifier (ViT-B/16 on BD3) | **not built** |
| 8 | Acoustic leak classifier (PANNs) | **not built** — needs own recordings |

Nothing in this repository has seen a real flat. Every number below comes from
the simulator.

## Quick start

```bash
pip install -r requirements.txt

python scripts/run_demo.py --fault hidden_pipe_leak --room bathroom --days 18
python scripts/run_demo.py --fault roof_seepage --room living --language hi
python scripts/benchmark.py --seeds 25
python app.py                      # dashboard at http://127.0.0.1:5000
```

A healthy flat is the default: `python scripts/run_demo.py` should raise nothing.

## Results

From `python scripts/benchmark.py --seeds 25`, on **simulated data**:

| Measure | Result | Scale |
|---|---|---|
| False alarms, fault-free flats | **0 rooms flagged** | 25 flats × 60 days = 6,000 room-days |
| Fault kind *and* room both correct | **98 / 100** | 25 seeds × 4 fault kinds |
| Days-to-visible-damp, mean absolute error | **1.15 days** | 35 evaluable cases |
| Predictions within 2 days of the truth | **100%** | — |

Per fault kind: concealed pipe leak 25/25, condensation 25/25, running fixture
25/25, roof seepage 23/25. Seepage is the hard one — it only wets the wall while
it rains, so a dry spell at the end of a run can leave too little recent signal.

In 9 of 44 forecast cases the system declined to predict a crossing within seven
days. Those are counted and reported rather than hidden; declining to guess is
the correct behaviour when the trend is flat.

## How it decides

Readings (15-minute) → per-room detector → physics rules → forecast → fusion → advice.

Every fault kind has a **precondition**: the symptom without which that fault is
not what is happening. A wall that is not wetter than its own baseline cannot be
a leak or seepage, however much else matches. This is what keeps a healthy flat
silent, and it is why the false-alarm count is zero rather than merely low.

What separates the four faults:

| Evidence | Leak | Seepage | Condensation | Running fixture |
|---|---|---|---|---|
| Wall wetter than baseline | yes | yes | no | no |
| Rise follows the rain (2-day window) | no (≤0.33) | yes (≥0.50) | — | — |
| Water drawn at night | yes | no | no | yes |
| Pipe noise above normal | yes | no | no | yes |
| Surface humidity at mould risk | no | no | yes | no |

The rain-response figure is a correlation between each day's *rise* in moisture
and the rain of that day and the day before. Measured over 12 seeds per fault,
seepage scores 0.50 and above and a leak 0.33 and below, so the 0.45 cut
separates them cleanly. Correlating moisture *levels* instead of rises does not
work: any slow climb then looks rain-driven.

Alert level comes from the forecast, not the detector — damage in 2 days
outranks damage in a month. Urgent alerts, and uncertain action-level ones, are
marked **needs technician confirmation**; each verdict a technician returns is a
real label, which is how this earns real training data.

## The forecaster

Default is a least-squares trend with a band from its own residuals. No
dependencies, always available.

If `chronos-forecasting>=2.0` is installed, Chronos-2 is used instead, zero-shot.
Series are resampled to hourly first. At 15-minute spacing a 7-day horizon is 672
steps, past Chronos-Bolt's 64-step maximum; hourly keeps the horizon inside a
single pass and costs nothing for a figure reported in days. Chronos-2 itself
allows up to 1,024 steps and takes covariates, so rainfall could enter the
forecast directly rather than only through the seepage physics.

Swap back ends with `forecast_room(df, backend="chronos")`.

## Simulator

One flat, four rooms, driven by a diurnal outdoor temperature cycle,
monsoon-style rain events and a daily occupancy pattern. Surface humidity is
computed properly — vapour pressure set by the room air, saturation pressure by
the wall — so a cold wall sits at a higher RH than the room it is in, which is
why condensation appears on a thermal bridge first.

```python
from smart_property.simulator import simulate
from smart_property.pipeline import analyse_flat

df = simulate(seed=3, days=18, fault_kind="hidden_pipe_leak", room="bathroom")
for result in analyse_flat(df):
    print(result["room"], result["alert"]["level"])
```

## Ingest API

```bash
curl -X POST http://127.0.0.1:5000/api/ingest \
  -H 'Content-Type: application/json' \
  -d '{"readings":[{"ts":"2026-06-01T00:00:00","room":"bathroom",
       "wall_moisture":8.1,"wall_surface_rh":58.0,"air_temp":27.2,
       "air_rh":56.0,"wall_temp":26.6,"flow_lpm":0.0,"pipe_db":28.1}]}'
```

`rain_mm_h` is optional and defaults to 0, but the seepage rule needs it to
tell rain-driven moisture from a leak.

## Honest limitations

- **Simulated only.** The detector, the rules and the forecaster have been
  tested against a simulator written by the same project. Good numbers here mean
  the pipeline is coherent, not that it works on a real wall.
- **The simulator is the ground truth.** It encodes the same physics the rules
  look for, so the rules are not being tested against nature.
- **No vision.** Components 5, 6 and 7 are not built, so no figure here comes
  from real public data. Crack segmentation on Crack-seg is the one result that
  would.
- **Readings are in memory.** `app.py` keeps readings in a process-local dict.

## Layout

```
smart_property/
  config.py      rooms, sensors, damage thresholds
  physics.py     vapour pressure, surface RH, dew point
  simulator.py   building-physics simulator with injectable faults
  detector.py    Isolation Forest per room (component 1)
  rules.py       physics evidence and localiser (component 2)
  forecast.py    days to damage, trend or Chronos-2 (component 3)
  fusion.py      one alert level per room
  advisor.py     plain-language advice, en/hi (component 9)
  pipeline.py    readings in, alerts out
scripts/
  run_demo.py        one flat, printed
  benchmark.py       the three measurements above
  export_dataset.py  labelled windows for component 4
app.py           Flask ingest API and dashboard
```
