# Smart Property AI — Architecture

Finds property problems while they are still hidden — a concealed pipe leak,
roof seepage, condensation, a running fixture — says which room and which wall,
predicts when the damage becomes visible, and tells the owner what to do.

A note on honesty before anything else, because it governs how to read this
document. **The sensor layer is simulated.** No physical sensor has ever fed
this system; readings come from a building-physics simulator. The vision models
are the exception — they are trained and measured on photographs of real
buildings. Every section below marks which of the two it is, and so does the
dashboard, which reports how many components are actually live rather than
implying the whole architecture is running.

---

## 1. Sensors: what is measured and why

Eight channels per room, sampled every **15 minutes**. Each one exists to
separate two faults that would otherwise look identical.

| Channel | Unit | What it is for | Hardware (planned) |
|---|---|---|---|
| `wall_moisture` | % volumetric | **The damage variable.** Water inside the plaster, long before the surface changes. Visible damp at 20%. | Capacitive probe embedded in the plaster, read on an ADC |
| `wall_surface_rh` | % RH | **Mould risk.** Humidity in the film of air touching the wall. Mould becomes possible above 80%. | SHT41 mounted flush to the wall face |
| `wall_temp` | °C | **Evaporative cooling.** A wet wall runs cooler than a dry one as water evaporates — the tell that moisture is active, not historic. | Thermistor or the SHT41's own sensor |
| `air_temp` | °C | Room baseline, and the dew-point calculation. | SHT41 / BME280 in free air |
| `air_rh` | % | With `air_temp`, gives vapour pressure, which with `wall_temp` gives surface RH. | Same sensor |
| `flow_lpm` | L/min | **The leak signature.** Water drawn at 03:00, when nobody is awake, means something is running that should not be. | Hall-effect flow sensor on the branch, or a pulse-output meter |
| `pipe_db` | dB | **Structure-borne noise.** A pressurised leak makes the pipe hiss continuously. | Piezo contact disc clamped to the pipe |
| `rain_mm_h` | mm/h | **The discriminator.** Moisture that answers the rain is seepage; moisture that ignores it is a leak. | Tipping-bucket gauge, or a weather API by postcode |

### Why these eight and not fewer

Any single channel is ambiguous. Moisture rising could be a leak, seepage or
condensation. The separation only appears across channels:

| Evidence | Pipe leak | Roof seepage | Condensation | Running fixture |
|---|---|---|---|---|
| Wall wetter than its own baseline | yes | yes | no | no |
| Rise follows the rain | **no** | **yes** | — | — |
| Water drawn at night | **yes** | no | no | **yes** |
| Pipe noise above normal | yes | no | no | yes |
| Surface humidity at mould risk | no | no | **yes** | no |

`rain_mm_h` is what makes the leak/seepage split possible, and `flow_lpm` at
night is what makes the leak/condensation split possible. Drop either and two
faults collapse into one.

### Placement

Five rooms, ten walls, one moisture probe per room on a named wall. Each wall
carries a **printed ArUco marker** of known size. The marker is what turns a
photograph into a measurement: it fixes the scale, so a crack is reported in
millimetres rather than pixels, and it fixes the viewing angle, so two photos
from different visits can be compared. Acoustic sensing is only in the kitchen
and bathroom, which is where the pressurised pipework runs.

---

## 2. Raspberry Pi: the edge layer

**Status: designed, not built.** No hardware exists. This section describes the
intended deployment so the software boundaries make sense; treat it as a plan.

```
  per room                      per flat                      cloud
┌──────────────┐          ┌──────────────────┐        ┌──────────────┐
│ ESP32 node   │  Wi-Fi   │ Raspberry Pi 4/5 │  HTTPS │ dashboard /  │
│ SHT41        ├─ MQTT ──►│ 4 GB             ├───────►│ owner's      │
│ moisture ADC │          │                  │        │ phone        │
│ piezo ADC    │          │ • Mosquitto      │        └──────────────┘
└──────────────┘          │ • SQLite store   │
┌──────────────┐          │ • detectors      │
│ flow sensor  ├─ pulse ─►│ • forecaster     │
└──────────────┘          │ • fusion         │
┌──────────────┐          │ • Flask API      │
│ phone camera ├─ upload ►│                  │
└──────────────┘          └──────────────────┘
```

**Why a Pi rather than sending everything to a server.** Three reasons, in
order of weight:

1. **Bandwidth.** Eight channels × five rooms × every 15 minutes is 3,840
   readings a day. Trivial to store locally, wasteful to stream.
2. **It keeps working offline.** Indian flats lose connectivity. The detectors
   are local, so a leak is still found and logged when the internet is down.
3. **Photographs stay in the building.** Wall photos are uploaded to the Pi and
   classified there. Nothing identifiable leaves the property unless the owner
   shares a finding.

**What fits.** The sensor spine — Isolation Forest, physics rules, forecaster,
fusion, advisor — measured at **275 MiB** of RAM in its container, comfortable
on a 4 GB Pi. Adding the vision models takes it to **438 MiB**, still
comfortable, though ResNet18 and YOLO on a Pi CPU take seconds per photograph
rather than milliseconds. That is acceptable: a wall is photographed on a
visit, not continuously.

**What the Pi does not do.** No training. Models are trained elsewhere and
copied to the device as weight files.

---

## 3. What the system does with the data

Ten components. Numbering matches the model plan and the dashboard.

```
  INPUTS              DETECT                      DECIDE          ACT
┌──────────┐   ┌───────────────────────┐    ┌────────────┐  ┌──────────────┐
│ room     ├──►│ 1 anomaly detector    ├───►│            │  │ 3 forecaster │
│ sensors  │   │   Isolation Forest    │    │            ├─►│   days to    │
└──────────┘   ├───────────────────────┤    │            │  │   damage     │
               │ 2 fault rules         ├───►│   alert    │  └──────┬───────┘
┌──────────┐   │   building physics    │    │   fusion   │         ▼
│ meter +  ├──►├───────────────────────┤    │            │  ┌──────────────┐
│ pipe mic │   │ 4 sequence model      ├───►│ combines   │  │ 9 advisor    │
└──────────┘   │   1D-CNN, 48 h        │    │ every      ├─►│   plain      │
               ├───────────────────────┤    │ score;     │  │   language   │
┌──────────┐   │ 5 crack segmentation  ├───►│ urgent     │  └──────┬───────┘
│ wall     ├──►│   YOLO on Crack-seg   │    │ needs a    │         ▼
│ photos   │   ├───────────────────────┤    │ technician │  ┌──────────────┐
│ + ArUco  │   │ 6 crack width (mm)    ├───►│ check      │  │ dashboard    │
└──────────┘   │   OpenCV + marker     │    │            │  │ app + API    │
               ├───────────────────────┤    │            │  └──────────────┘
               │ 7 damp classifier     ├───►│            │
               │   ResNet18 on BD3     │    └────────────┘
               └───────────────────────┘
               │ 8 acoustic classifier │  not built
               └───────────────────────┘
```

### The decision chain, in order

**Learn normal (component 1).** Each room's first week is its baseline. An
Isolation Forest is fitted per room on that week alone and then frozen — no
pretrained model knows what normal looks like inside one particular flat, and a
baseline that slides forward would eventually learn a running leak as normal.

**Name the fault (component 2).** Hand-written physics, not a model, so every
alert arrives with reasons a home owner can check: *water flowing at night*,
*pipe noise above normal*, *the rise does not follow the rain*. Each fault has
a **precondition** — the symptom without which that fault is not what is
happening. A wall no wetter than its own baseline cannot be a leak, however
much else matches. That is what keeps a healthy flat silent.

**Second opinion (component 4).** A 1D-CNN reads the *shape* of the last 48
hours across all eight channels, z-scored against the room's own baseline. It
exists because the rules are deliberately conservative: they wait for the
plaster to be measurably wetter, which takes days, while flow and pipe noise
step up within hours.

**Predict (component 3).** Forecasts moisture and surface humidity 7 days out
and reads off the days until 20% moisture or 80% surface RH.

**Confirm by camera (components 5, 6, 7).** A photograph of the wall. Crack
segmentation finds cracks; the ArUco marker converts the mask to millimetres;
the classifier says whether damp or algae is visible yet.

**Fuse.** One alert level per room. The camera may **corroborate but never
cancel** — a clean photo does not mean a dry wall, because the sensors are
reading inside the plaster and the entire claim of the product is that they see
it days before anything surfaces.

**Explain (component 9).** Cause, urgency, action and who to call, in English
or Hindi. The language model only rewrites evidence the detectors produced; it
never decides whether a problem exists, and that is enforced in code, not only
in the prompt.

### What this buys, measured

| | Rules alone | With the sequence model |
|---|---|---|
| Correct 6 h after onset | 0.250 | **0.844** |
| Correct 24 h after onset | 0.281 | **0.875** |
| Correct 96 h after onset | 0.875 | **1.000** |

Roughly two days of extra warning, on seeds held out of training.

---

## 4. Tech stack

| Layer | Technology | Why this one |
|---|---|---|
| Language | Python 3.12 | The whole scientific and vision stack lives here |
| Numerics | NumPy, pandas | Time-series handling; 15-minute sampling, rolling windows |
| Classical ML | scikit-learn | Isolation Forest, one per room, fitted on a week of data |
| Deep learning | PyTorch (CPU) | ResNet18 and the 1D-CNN; CPU wheels only, no GPU needed to run |
| Segmentation | Ultralytics YOLO | Crack masks; pretrained Crack-seg weights used as-is |
| Image measurement | OpenCV (headless) | ArUco detection, perspective rectification, distance transform |
| Skeletonisation | scikit-image | Crack centreline, for width from the distance transform |
| Forecasting | Least-squares trend, optional Chronos-2 | Trend is the default — it was measured as the better of the two here |
| Language | Gemini (`gemini-3.5-flash-lite`) | Rewrites advice; templates are the fallback, so a failed call is invisible |
| API | Flask + gunicorn | One worker, many threads: the running flat is process state |
| Front end | Vanilla JS, SVG | Floor plan, wall elevations and charts drawn as SVG; no build step |
| Storage | SQLite | Readings, frozen baselines, alert lifecycle, technician verdicts |
| Packaging | Docker | Two builds from one file: slim 679 MB, vision 2.97 GB |
| Hosting | Render | Free instance runs the slim build at 275 MiB |

### Two decisions worth defending

**Gunicorn runs a single worker.** The flat is a live object with a clock
thread. A second worker would be a second flat with its own clock, and
consecutive requests would land on different simulations. Concurrency comes
from threads sharing one engine.

**The forecaster defaults to a straight line, not the foundation model.**
Measured on 17 held-out runs: the trend line predicted a threshold crossing in
12 of them, Chronos-2 in none. Chronos-2 extrapolates the ramp and then damps
it — a sound prior for a general time series, the wrong one for water
accumulating in a wall. It remains available with `backend="chronos"`.

---

## 5. Where the data comes from

### Sensor data — simulated

`smart_property/simulator.py`. A building-physics model of one flat: diurnal
outdoor temperature, monsoon-style rain events, a daily occupancy pattern, and
four injectable faults. Surface humidity is computed properly — vapour pressure
set by the room air, saturation pressure by the wall — so a cold wall sits at a
higher RH than the room, which is why condensation appears on a thermal bridge
first.

It exists because real labelled fault data does not. A real building gives
readings but never says *that was a pipe leak behind the bathroom west wall,
starting Tuesday*. Without injected faults there is nothing to score against,
and nothing to train component 4 on.

**Its limits, stated plainly.** The rules were written against the same physics
the simulator encodes, so a high score partly measures self-consistency. Two
places where that showed: the sequence model scores exactly 1.000 on faults
older than 96 hours, having learned the generative rules outright; and the
trend forecaster's error is a constant +1.14 days with a spread of 0.04 across
25 seeds — a fixed structural offset, not noise.

### Image data — real

| Dataset | Source | Size | Licence | Used for |
|---|---|---|---|---|
| **BD3** | [Kaggle](https://www.kaggle.com/datasets/praveenkottari/bd3-dataset-for-building-defect-detection) · [code](https://github.com/Praveenkottari/BD3-Dataset) | 3,965 photos, 7 classes | see source | Training component 7 |
| **house-wall-damage** | [Roboflow Universe](https://universe.roboflow.com/salehs/house-wall-damage) | 3,449 photos, 5 classes | CC BY 4.0 | Cross-dataset test |
| **Crack-seg** | [Ultralytics](https://docs.ultralytics.com/datasets/segment/crack-seg) | 4,029 photos with masks | — | Component 5 weights |
| **Water-testbed leak** | [Mendeley](https://data.mendeley.com/datasets/tbrnp6vrnj) | 280 recordings, 30 s | CC BY 4.0 | Component 8, not built |

BD3 was collected by hand at more than 50 real buildings aged 10 to 60 years,
photographed about a metre from the wall.

### The results that are not circular

| Measure | Result | On |
|---|---|---|
| 7-class accuracy, by-wall split | **0.875** | 793 held-out BD3 photos |
| Damp question, F1 | **0.94** | algae or stain vs rest |
| **Cross-dataset accuracy** | **0.926** | 283 photos, different photographer |
| In-domain, same 4 classes | 0.933 | the fair comparison |
| Crack width error | **+0.08 mm** | lines of known width |
| Crack recall @ 0.40 conf | 1.00 at 10% false positives | BD3, cross-dataset for these weights |

The cross-dataset row is the one that matters. A model tested only on held-out
photos from the same collection has shown it generalises across pictures of the
same buildings taken by the same person with the same phone. Tested on
house-wall-damage — different photographer, different buildings — accuracy
falls from 0.933 to 0.926, a gap indistinguishable from noise on 283 images.

Collapsing BD3's `major_crack` and `minor_crack` into one class is what makes
those two numbers comparable; the 7-class figure is 0.875 and is the honest
headline.

---

## 6. What is not built

| | Why |
|---|---|
| **Component 8**, acoustic leak classifier | The simulator emits a dB level, not waveforms. PANNs needs audio that does not exist here. |
| **Physical sensors** | Section 2 is a design. Nothing has been wired. |
| **Photo upload in the dashboard** | Components 5 and 7 are loaded and wired into the pipeline, but the inspection form posts metadata rather than an image, so the camera cross-check currently runs on typed input. |
| **Component 10**, failure risk score | A cold-start problem. It needs repair history that only accumulates after deployment; the technician-verdict log that feeds it is already being collected. |

---

## 7. Repository map

```
smart_property/
  config.py        rooms, sensors, damage thresholds
  flat_model.py    5 rooms, 10 walls, ArUco markers, probe positions
  physics.py       vapour pressure, surface RH, dew point
  simulator.py     building-physics simulator with injectable faults
  detector.py      Isolation Forest per room            component 1
  rules.py         physics evidence and localiser       component 2
  forecast.py      days to damage, trend or Chronos-2   component 3
  sequence.py      1D-CNN over a 48 h window            component 4
  segmentation.py  YOLO crack masks                     component 5
  crack.py         width and growth in mm, ArUco        component 6
  vision.py        BD3 damp classifier                  component 7
  advisor.py       plain-language advice, en/hi         component 9
  llm.py           Gemini adapter with a hard guardrail
  fusion.py        one alert level per room
  pipeline.py      readings and photos in, alerts out
  registry.py      which components are available here
  store.py         SQLite: readings, baselines, alerts, verdicts
scripts/
  benchmark.py                false alarms, localisation, forecast error
  compare_layers.py           rules against the sequence model
  train_sequence_model.py     component 4
  train_defect_classifier.py  component 7
  cross_dataset_eval.py       the non-circular number
  validate_crack_width.py     component 6 against known widths
  run_demo.py                 one flat, printed
server.py          API and dashboard host
web/artifact/      the dashboard
Dockerfile         slim and vision builds
render.yaml        deployment blueprint
```
