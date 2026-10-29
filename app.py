"""Flask backend and dashboard for Smart Property AI.

    python app.py                 then open http://127.0.0.1:5000

Endpoints
    GET  /                 the dashboard
    GET  /api/alerts       analyse whatever readings are currently loaded
    POST /api/ingest       append readings  {"readings": [ {...}, ... ]}
    POST /api/demo         load a simulated flat  {"fault": ..., "room": ...}
"""

import warnings

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

warnings.filterwarnings("ignore")

from smart_property.config import FAULT_KINDS, ROOMS, SENSORS
from smart_property.pipeline import analyse_flat
from smart_property.simulator import simulate

app = Flask(__name__)

# A prototype keeps its readings in memory. A real deployment would write them
# to a time-series store and analyse a rolling window.
STORE = {"readings": None}

REQUIRED = {"ts", "room", *SENSORS}


def _load(df):
    STORE["readings"] = df


@app.post("/api/ingest")
def ingest():
    payload = request.get_json(silent=True) or {}
    rows = payload.get("readings")
    if not rows:
        return jsonify({"error": "send {'readings': [...]}"}), 400

    frame = pd.DataFrame(rows)
    missing = REQUIRED - set(frame.columns)
    if missing:
        return jsonify({"error": f"missing columns: {sorted(missing)}"}), 400
    frame["ts"] = pd.to_datetime(frame["ts"])
    if "rain_mm_h" not in frame:
        frame["rain_mm_h"] = 0.0

    current = STORE["readings"]
    STORE["readings"] = frame if current is None else pd.concat(
        [current, frame], ignore_index=True)
    return jsonify({"ok": True, "rows": len(STORE["readings"])})


@app.post("/api/demo")
def demo():
    payload = request.get_json(silent=True) or {}
    fault = payload.get("fault") or None
    if fault and fault not in FAULT_KINDS:
        return jsonify({"error": f"fault must be one of {FAULT_KINDS}"}), 400
    _load(simulate(
        seed=int(payload.get("seed", 3)),
        days=int(payload.get("days", 18)),
        fault_kind=fault,
        room=payload.get("room", "bathroom") if fault else None,
        start_day=float(payload.get("start_day", 9.0)),
    ))
    return jsonify({"ok": True, "fault": fault, "room": payload.get("room")})


@app.get("/api/alerts")
def alerts():
    df = STORE["readings"]
    if df is None:
        return jsonify({"error": "no readings loaded"}), 404
    language = request.args.get("language", "en")
    results = analyse_flat(df, language=language)
    return jsonify({
        "rooms": results,
        "raised": sum(r["alert"]["level"] != "none" for r in results),
        "simulated": True,
    })


@app.get("/")
def dashboard():
    return render_template_string(PAGE, faults=FAULT_KINDS, rooms=ROOMS)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smart Property AI</title>
<style>
  :root { --bg:#fbfbfa; --card:#fff; --ink:#1a1a18; --quiet:#6b6b66;
          --line:#e4e2dd; --none:#9b9b95; --watch:#9a7b20; --action:#b4581a;
          --urgent:#b3261e; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
    --bg:#1a1a18; --card:#232320; --ink:#f2f1ee; --quiet:#a8a69f; --line:#36352f;
    --watch:#e0bb57; --action:#f0915a; --urgent:#f08a82; } }
  * { box-sizing:border-box }
  body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5
         ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
  .wrap { max-width:900px; margin:0 auto; padding:32px 16px 64px; }
  h1 { font-size:22px; margin:0 0 4px }
  .sub { color:var(--quiet); font-size:13px; margin-bottom:24px }
  .bar { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:24px }
  select,button { font:inherit; padding:8px 12px; border-radius:8px;
    border:1px solid var(--line); background:var(--card); color:var(--ink); }
  button { cursor:pointer; font-weight:600 }
  .card { background:var(--card); border:1px solid var(--line);
          border-radius:12px; padding:16px; margin-bottom:12px }
  .head { display:flex; justify-content:space-between; align-items:baseline; gap:12px }
  .room { font-weight:600; text-transform:capitalize }
  .lvl { font-size:12px; font-weight:700; letter-spacing:.04em; text-transform:uppercase }
  .none{color:var(--none)} .watch{color:var(--watch)}
  .action{color:var(--action)} .urgent{color:var(--urgent)}
  .meta { color:var(--quiet); font-size:13px; margin-top:6px }
  ul { margin:8px 0 0; padding-left:20px; font-size:13px }
  .advice { margin-top:12px; padding-top:12px; border-top:1px solid var(--line);
            font-size:14px }
  .advice b { font-weight:600 }
  .flag { display:inline-block; margin-top:8px; font-size:12px; font-weight:600;
          color:var(--urgent) }
</style></head><body><div class="wrap">
<h1>Smart Property AI</h1>
<div class="sub">Sensor readings are simulated. Every figure on this page comes
from the simulator, not from a real flat.</div>
<div class="bar">
  <select id="fault"><option value="">no fault (healthy flat)</option>
  {% for f in faults %}<option value="{{f}}">{{f}}</option>{% endfor %}</select>
  <select id="room">{% for r in rooms %}<option value="{{r}}">{{r}}</option>{% endfor %}</select>
  <select id="lang"><option value="en">English</option><option value="hi">हिन्दी</option></select>
  <button onclick="run()">Run</button>
</div>
<div id="out"></div>
</div><script>
async function run() {
  const fault = document.getElementById('fault').value;
  const room = document.getElementById('room').value;
  const lang = document.getElementById('lang').value;
  document.getElementById('out').innerHTML = '<div class="card">Simulating…</div>';
  await fetch('/api/demo', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({fault, room})});
  const res = await fetch('/api/alerts?language=' + lang);
  render(await res.json());
}
function render(data) {
  if (data.error) { document.getElementById('out').textContent = data.error; return; }
  document.getElementById('out').innerHTML = data.rooms.map(r => {
    const a = r.alert, f = r.forecast;
    let body = '';
    if (a.level === 'none') {
      body = '<div class="meta">Nothing outside this room\\'s normal week.</div>';
    } else {
      const damp = f.visible_damp.days, mould = f.mould_risk.days;
      body = '<div class="meta">' + a.kind.replace(/_/g,' ') +
        ' &middot; confidence ' + Math.round(a.confidence*100) + '%' +
        ' &middot; forecaster: ' + f.backend + '</div>' +
        '<div class="meta">' +
        (damp === null ? 'No visible damp within 7 days.'
                       : 'Visible damp in about ' + damp.toFixed(1) + ' days.') +
        ' ' + (mould === null ? '' : 'Mould risk in about ' + mould.toFixed(1) + ' days.') +
        '</div><ul>' + a.evidence_for.map(e => '<li>' + e + '</li>').join('') + '</ul>';
      if (r.advice) {
        body += '<div class="advice"><b>Cause.</b> ' + r.advice.cause +
          '<br><b>Do.</b> ' + r.advice.action +
          '<br><b>Call.</b> ' + r.advice.who_to_call + '</div>';
      }
      if (a.needs_technician_confirmation) {
        body += '<div class="flag">Needs technician confirmation</div>';
      }
    }
    return '<div class="card"><div class="head"><span class="room">' + r.room +
      '</span><span class="lvl ' + a.level + '">' + a.level + '</span></div>' +
      body + '</div>';
  }).join('');
}
run();
</script></body></html>"""


if __name__ == "__main__":
    app.run(debug=False, port=5000)
