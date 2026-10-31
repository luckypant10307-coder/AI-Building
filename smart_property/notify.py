"""Notifications: tell the owner when something changes, and only then.

The detectors re-run every few seconds, and a leak stays a leak for days. A
notifier that sent on every analysis would ping the owner hundreds of times for
one problem and be muted by lunchtime. So this keeps a little state per room and
sends only on a transition:

    new        a room that was quiet now has a problem
    escalated  same problem, higher level (watch -> alert)
    changed    the diagnosis itself changed (seepage -> leak)
    resolved   quiet for CLEAR_CHECKS consecutive analyses

An unchanged problem at an unchanged level sends nothing.

Sending happens on its own thread so a slow or unreachable Telegram never holds
up detection. Every attempt is logged, sent or not, so it is always possible to
show exactly what the owner was told and when.

Configuration, all from the environment (a local .env works):

    TELEGRAM_BOT_TOKEN   from @BotFather
    TELEGRAM_CHAT_ID     who receives it; scripts/telegram_setup.py finds it
    NOTIFY_LANGUAGE      en (default) or hi
    NOTIFY_QUIET_HOURS   e.g. 22-7; watch-level messages wait, alerts do not

With no token set, the dry-run channel records messages instead of sending
them, so the rest of the system behaves identically either way.
"""

import html
import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path

LEVEL_RANK = {"watch": 1, "alert": 2}
CLEAR_CHECKS = 3                      # quiet analyses before "resolved"
LOG_PATH = Path("data/notifications.jsonl")

ICON = {"new": "🚨", "escalated": "⬆️", "changed": "🔄", "resolved": "✅",
        "test": "🔔"}
HEADLINE = {
    "en": {"new": "Problem found", "escalated": "Getting worse",
           "changed": "Diagnosis changed", "resolved": "All clear",
           "test": "Test message"},
    "hi": {"new": "समस्या मिली", "escalated": "स्थिति बिगड़ रही है",
           "changed": "निदान बदला", "resolved": "सब ठीक है",
           "test": "परीक्षण संदेश"},
}
LABELS = {
    "en": {"room": "Room", "confidence": "Confidence", "why": "Why",
           "cause": "Cause", "do": "Do this", "call": "Call",
           "forecast": "Forecast", "cleared": "No problem detected for the "
           "last few checks. The alert has been closed."},
    "hi": {"room": "कमरा", "confidence": "विश्वास", "why": "कारण",
           "cause": "वजह", "do": "क्या करें", "call": "किसे बुलाएँ",
           "forecast": "अनुमान", "cleared": "पिछली कुछ जाँचों में कोई समस्या "
           "नहीं मिली। अलर्ट बंद कर दिया गया है।"},
}


# Hindi for the parts of a finding that are not advice text. Headings alone
# translated over an English body reads as a half-finished message, so the
# room, the fault and every evidence line the rule engine can produce are here.
ROOM_HI = {"living": "बैठक", "master": "मुख्य शयनकक्ष", "bedroom2": "शयनकक्ष 2",
           "kitchen": "रसोई", "bathroom": "बाथरूम"}
FAULT_HI = {"pipe_leak": "दीवार में छिपा पाइप लीक",
            "roof_seepage": "छत / टेरेस से रिसाव",
            "condensation": "नमी जमना और फफूंद का खतरा",
            "fixture_leak": "चलता हुआ फ्लश / टपकता नल"}
LEVEL_HI = {"watch": "निगरानी", "alert": "समस्या"}
EVIDENCE_HI = {
    "moisture climbing": "नमी बढ़ रही है",
    "moisture climbing steadily": "नमी लगातार बढ़ रही है",
    "no extra pipe noise": "पाइप में कोई अतिरिक्त आवाज़ नहीं",
    "no extra water being drawn": "अतिरिक्त पानी नहीं खिंच रहा",
    "pipe noise above normal": "पाइप की आवाज़ सामान्य से ज़्यादा",
    "plaster not yet wet through": "प्लास्टर अभी पूरी तरह गीला नहीं",
    "plaster still dry": "प्लास्टर अभी सूखा है",
    "the rise does not follow the rain": "नमी बारिश के साथ नहीं बढ़ रही",
    "the rise follows the rain": "नमी बारिश के साथ बढ़ रही है",
    "wall cooled by evaporation": "भाप बनने से दीवार ठंडी है",
    "wall near the dew point": "दीवार ओस बिंदु के पास है",
    "wall surface not humid": "दीवार की सतह नम नहीं है",
    "water flowing all night": "पूरी रात पानी बह रहा है",
    "water flowing at night": "रात में पानी बह रहा है",
    "photo confirms damp on the surface": "फ़ोटो में सतह पर सीलन दिख रही है",
}


def _hi(table, key, fallback):
    return table.get(key, fallback)


# ------------------------------------------------------------------ channels

class TelegramChannel:
    name = "telegram"

    def __init__(self, token, chat_id, timeout=15.0):
        self.url = f"https://api.telegram.org/bot{token}/sendMessage"
        self.chat_id = chat_id
        self.timeout = timeout

    def send(self, text):
        body = json.dumps({"chat_id": self.chat_id, "text": text,
                           "parse_mode": "HTML",
                           "disable_web_page_preview": True}).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=body, method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as r:
                reply = json.load(r)
            return bool(reply.get("ok")), None
        except urllib.error.HTTPError as exc:
            # 400 with "chat not found" is the usual setup mistake: the owner
            # has to message the bot once before it is allowed to reply
            return False, f"HTTP {exc.code}: {exc.read().decode('utf-8')[:200]}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return False, f"{type(exc).__name__}: {exc}"


class DryRunChannel:
    """Records instead of sending. Used when no bot is configured."""
    name = "dry-run"

    def send(self, text):
        return True, None


def channel_from_env():
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if token and chat:
        return TelegramChannel(token, chat)
    return DryRunChannel()


def _quiet_hours():
    spec = os.environ.get("NOTIFY_QUIET_HOURS", "").strip()
    if not spec:
        return None
    try:
        start, end = (int(x) for x in spec.split("-"))
        return start % 24, end % 24
    except ValueError:
        return None


def _in_quiet_hours(hours, now):
    if not hours:
        return False
    start, end = hours
    h = now.hour
    return start <= h < end if start < end else (h >= start or h < end)


# ------------------------------------------------------------------ messages

def _e(text):
    return html.escape(str(text), quote=False)


def compose(event, room_name, finding=None, language="en", advice=None,
            room=None):
    """The text of one message, Telegram HTML."""
    lang = language if language in HEADLINE else "en"
    L, H = LABELS[lang], HEADLINE[lang]
    hindi = lang == "hi"
    shown_room = _hi(ROOM_HI, room, room_name) if hindi else room_name
    lines = [f"{ICON[event]} <b>{_e(H[event])}</b> · {_e(shown_room)}"]

    if event == "resolved":
        lines.append(_e(L["cleared"]))
        return "\n".join(lines)
    if not finding:
        return "\n".join(lines)

    title = (_hi(FAULT_HI, finding["type"], finding["title"]) if hindi
             else finding["title"])
    level = (_hi(LEVEL_HI, finding["level"], finding["level"]) if hindi
             else finding["level"])
    lines.append(f"<b>{_e(title)}</b> ({_e(level)}, {L['confidence']} "
                 f"{round(finding['confidence'] * 100)}%)")
    if finding.get("forecast") and not hindi:
        # forecasts are generated sentences; shown in English only
        lines.append(f"{_e(L['forecast'])}: {_e(finding['forecast'])}")
    evidence = finding.get("evidence") or []
    if evidence:
        lines.append(f"\n<b>{_e(L['why'])}</b>")
        lines.extend(f"• {_e(_hi(EVIDENCE_HI, e, e) if hindi else e)}"
                     for e in evidence[:4])
    advice = advice or finding.get("advice") or {}
    if advice.get("cause"):
        lines.append(f"\n<b>{_e(L['cause'])}:</b> {_e(advice['cause'])}")
    if advice.get("action"):
        lines.append(f"<b>{_e(L['do'])}:</b> {_e(advice['action'])}")
    who = advice.get("who") or advice.get("who_to_call")
    if who:
        lines.append(f"<b>{_e(L['call'])}:</b> {_e(who)}")
    return "\n".join(lines)


# ------------------------------------------------------------------ notifier

class Notifier:
    """Watches per-room findings and sends on transitions only."""

    def __init__(self, channel=None, language=None, log_path=LOG_PATH):
        self.channel = channel or channel_from_env()
        self.language = language or os.environ.get("NOTIFY_LANGUAGE", "en")
        self.quiet = _quiet_hours()
        self.log_path = Path(log_path) if log_path else None
        self.state = {}               # room -> {kind, level, clear}
        self.history = deque(maxlen=200)
        self.held = []                # watch messages waiting out quiet hours
        self.outbox = queue.Queue()
        self.lock = threading.Lock()
        threading.Thread(target=self._drain, daemon=True,
                         name="notifier").start()

    # -- decide ----------------------------------------------------------

    def observe(self, room, room_name, finding, advice_for=None):
        """Feed one room's current finding (or None). Returns the event sent."""
        with self.lock:
            prev = self.state.get(room)
            event = None
            if finding:
                kind, level = finding["type"], finding["level"]
                if prev is None:
                    event = "new"
                elif prev["kind"] != kind:
                    event = "changed"
                elif LEVEL_RANK.get(level, 0) > LEVEL_RANK.get(prev["level"], 0):
                    event = "escalated"
                self.state[room] = {"kind": kind, "level": level, "clear": 0}
            elif prev is not None:
                prev["clear"] += 1
                if prev["clear"] >= CLEAR_CHECKS:
                    event = "resolved"
                    del self.state[room]
            if event is None:
                return None

        advice = None
        if finding and advice_for and self.language != "en":
            try:
                advice = advice_for(self.language)
            except Exception:
                advice = None
        text = compose(event, room_name, finding, self.language, advice, room)
        urgent = bool(finding) and finding["level"] == "alert"
        self._queue(event, room, text, urgent)
        return event

    def test(self, text=None):
        message = text or compose("test", "Smart Property AI",
                                  language=self.language) + \
            "\nNotifications are connected."
        self._queue("test", None, message, urgent=True)

    # -- send ------------------------------------------------------------

    def _queue(self, event, room, text, urgent):
        if not urgent and _in_quiet_hours(self.quiet, datetime.now()):
            self.held.append((event, room, text))
            self._log(event, room, text, sent=False, error="quiet hours, held")
            return
        self.outbox.put((event, room, text))

    def release_held(self):
        if _in_quiet_hours(self.quiet, datetime.now()):
            return 0
        held, self.held = self.held, []
        for item in held:
            self.outbox.put(item)
        return len(held)

    def _drain(self):
        while True:
            event, room, text = self.outbox.get()
            ok, error = self.channel.send(text)
            self._log(event, room, text, sent=ok, error=error)
            time.sleep(1.1)           # Telegram allows about 1 message/s per chat

    def _log(self, event, room, text, sent, error=None):
        entry = {"at": datetime.now().isoformat(timespec="seconds"),
                 "event": event, "room": room, "channel": self.channel.name,
                 "sent": sent, "error": error, "text": text}
        self.history.appendleft(entry)
        if self.log_path:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except OSError:
                pass

    def status(self):
        return {"channel": self.channel.name, "language": self.language,
                "quiet_hours": self.quiet, "open": dict(self.state),
                "held": len(self.held), "recent": list(self.history)[:20]}
