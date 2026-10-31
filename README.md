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
| 7 | Damp and mould classifier (ResNet18 on BD3) | pipeline built, **needs the full dataset** |
| 8 | Acoustic leak classifier (PANNs) | **not built** — needs own recordings |

The sensor side of this repository has never seen a real flat: every sensor
number below comes from the simulator. Component 7 is the exception — it trains
on photographs of real buildings — but it is not yet trained on enough of them
to report a figure. See [The one component with real data](#the-one-component-with-real-data).

## Quick start

```bash
pip install -r requirements-base.txt

python scripts/run_demo.py --fault hidden_pipe_leak --room bathroom --days 18
python scripts/run_demo.py --fault roof_seepage --room living --language hi
python scripts/benchmark.py --seeds 25
python scripts/train_defect_classifier.py --data data/bd3_raw   # real photos
```

### The dashboard

```bash
python server.py                   # http://127.0.0.1:5000
```

`server.py` serves the dashboard from `web/artifact/` and the API it calls:
`api/config`, `api/state`, `api/history`, `api/inspections` and the
`api/simulate/*` controls. A clock thread advances the flat about one
simulated hour per second while a slower thread re-runs the detectors, so the
readings on screen stay live even while a verdict is being recomputed.

Deploying it:

```bash
docker build -t smartprop .                          # sensor spine, 679 MB
docker build -t smartprop --build-arg WITH_VISION=1 .  # adds components 5 and 7
```

`render.yaml` deploys the first of those as a Render web service.

The page shows which of the ten components are actually live on this machine,
one card per room, and takes a wall photograph per room to bring components 5
and 7 into the decision. Optional extras switch themselves on when present:
`GEMINI_API_KEY` for LLM-written advice, `models/bd3_grouped.pt` for the damp
classifier, `models/yolov8n-crack-seg-best.pt` for crack segmentation,
`models/sequence_cnn.pt` for the sequence model. Nothing is required -- the
sensor side runs on numpy, pandas and scikit-learn alone, and the UI says so
rather than implying the whole architecture is running.

A healthy flat is the default: `python scripts/run_demo.py` should raise nothing.

## Results

From `python scripts/benchmark.py --seeds 25`, on **simulated data**:

| Measure | Result | Scale |
|---|---|---|
| False alarms, fault-free flats | **0 rooms flagged** | 25 flats × 60 days = 6,000 room-days |
| Fault kind *and* room both correct | **98 / 100** | 25 seeds × 4 fault kinds |
| Days-to-visible-damp within 2 days | **26 / 35 (74%)** | every case, declines included |

Detection per fault kind: concealed pipe leak 25/25, condensation 25/25, running
fixture 25/25, roof seepage 23/25. Seepage is the hard one — it only wets the
wall while it rains, so a dry spell at the end of a run leaves too little recent
signal.

### The forecast number deserves a closer look

| Fault | Cases | Declined | Within 2d | Bias | Spread |
|---|---|---|---|---|---|
| Concealed pipe leak | 25 | 0 | 25/25 | **+1.14 d** | **0.04 d** |
| Roof seepage | 10 | 9 | 1/10 | −1.32 d | — |
| Overall | 35 | 9 | 26/35 | +1.05 d | 0.48 d |

Two things in that table matter more than the headline.

**The leak forecast is biased, not noisy.** A spread of 0.04 days — about an
hour — across 25 different random seeds is not accuracy. It is a near-constant
offset: the forecaster says "visible damp in N days" and is late by 1.14 days,
every time. The cause is structural. The predictor is a straight line
(`np.polyfit(..., 1)`), the simulator's leak accelerates
(`0.75·elapsed + 0.012·elapsed²`), and a line cannot see the acceleration, so it
always arrives at the threshold too late. Reporting this as a 1.15-day *mean
absolute error* would disguise a fixed, correctable offset as random scatter.

This also means the leak result is **circular**: the forecaster looks good
because the simulator generates almost exactly the shape the forecaster assumes.

**On the non-linear fault it mostly gives up.** Roof seepage is episodic and
rain-driven, so a line fits it badly: the forecaster declined to predict in 9 of
10 cases. Those declines are counted as failures here. An earlier version of
this README reported "100% within 2 days" by computing the rate over only the
cases where the model was willing to answer — which quietly excluded exactly the
cases it could not handle. The honest figure is 74%.

Declining to guess is reasonable behaviour on its own; counting only the
non-declines is not.

## The one component with real data

Every figure above is simulated, and a model scored on a simulator is being
scored against assumptions somebody wrote down. Component 7 exists to break
that: it trains on [BD3](https://github.com/Praveenkottari/BD3-Dataset), 3,965
photographs taken by hand at more than 50 real buildings aged 10 to 60 years.

```bash
python scripts/train_defect_classifier.py --data data/bd3_raw --epochs 12
```

**Status: the pipeline runs on real photographs, and has not been shown to
work.** BD3's full set is distributed through Kaggle and needs an account
token, which this machine does not have. What runs today is the 37 sample
images the BD3 repository ships on GitHub — enough to prove the code path
(ImageNet weights, fine-tune, held-out evaluation, saved checkpoint,
inference), far too few to mean anything. With a 7-image test set the script
prints its numbers and then refuses to let them stand:

```
SMOKE TEST ONLY -- 7 test images is far too few for these numbers to mean anything.
```

To get the real figure: put a Kaggle API token at `~/.kaggle/kaggle.json`, then

```bash
pip install kaggle
kaggle datasets download -d praveenkottari/bd3-dataset-for-building-defect-detection -p data/ --unzip
python scripts/train_defect_classifier.py --data data/<extracted-folder> --epochs 12
```

BD3's own paper reports ResNet18 at **0.8301 F1** on its original split, which
is the number to be judged against.

### Why there are two split modes

```bash
python scripts/train_defect_classifier.py --split grouped   # default
python scripts/train_defect_classifier.py --split random    # what the paper does
```

BD3 filenames are sequential within each class (`cls00_001`, `cls00_002`, …),
and consecutive numbers tend to be the same wall photographed from slightly
different angles. A stratified *random* split therefore puts near-duplicate
photographs of one wall into train and test at once, and the score goes up
without the model getting better. `grouped` holds out a contiguous block of the
highest indices per class instead, which approximates splitting by wall.

Report both. The gap between them is the size of the leak, not a result. This
is the image-domain version of the rule in `export_dataset.py` — split by seed,
never by row.

### What the photograph is for

The classifier collapses to one binary question the rest of the system actually
asks: *algae or stain, versus everything else.* Those are the classes that mean
water. The sensors predict that a wall is wetting days ahead; the camera says
whether a mark has appeared on it yet.

```python
from smart_property.vision import DefectClassifier, confirms_damp

clf = DefectClassifier.load("models/defect_resnet18.pt")
prediction = clf.predict("wall.jpg")
confirms_damp(prediction)      # does the photo corroborate the sensor alert?
```

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

**What is actually predicting, today, is a straight line.** `backend="auto"`
falls through to the trend path unless `chronos-forecasting` is installed, and
it is not installed by default — so every forecast number above came from
`np.polyfit(x, y, 1)` over the last 168 hourly points, extended 168 hours. The
band is `1.28 × residual σ × √(1 + h/24)`, which is an assumption about how
uncertainty grows, not something measured.

Worth being blunt about what that means for the word "AI" in this project: the
only model in the repository that gets `.fit()` called on it is the per-room
Isolation Forest, and that is a *detector*, not a predictor. The rules are
hand-written thresholds and the advisor is string templates. Nothing here has
learned to forecast anything.

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
- **The forecast test is rigged in the forecaster's favour.** A linear predictor
  is scored on a simulator that generates near-linear ramps. The 0.04-day spread
  on the leak case is the evidence: that is the simulator's regularity showing
  through, not a property of real walls. A fair test needs a fault that
  saturates, or dries between rain events — which is why seepage, the one
  non-linear fault here, is also the one it fails.
- **No vision.** Components 5, 6 and 7 are not built, so no figure here comes
  from real public data. Crack segmentation on Crack-seg is the one result that
  would.
- **State is in memory.** `server.py` holds the running flat in one process, which is why it must run with a single worker.

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
  vision.py      BD3 defect classifier, real photos (component 7)
scripts/
  run_demo.py                 one flat, printed
  benchmark.py                the three measurements above
  export_dataset.py           labelled windows for component 4
  train_defect_classifier.py  fine-tune ResNet18 on BD3
server.py        API and dashboard host
web/artifact/    the dashboard itself
```
