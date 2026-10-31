"""Gemini adapter for the repair advisor (component 9).

The plan's guardrail is the whole design here:

    The LLM only rewrites evidence. It never decides whether a problem exists.

That is enforced in code, not only in the prompt. The model is handed a
conclusion the detectors have already reached and asked to phrase it; whatever
comes back is filtered down to the four wording fields before it is used. If it
tries to change the fault, the urgency level, or the days-to-damage, those
fields are discarded. If the call fails, times out, or returns something
unparseable, the template text stands and the owner sees that instead.

The API key is read from the environment and never written to a file:

    export GEMINI_API_KEY=...            # bash
    $env:GEMINI_API_KEY = "..."          # PowerShell
"""

import json
import os
import sys
import urllib.error
import urllib.request

ENDPOINT = ("https://generativelanguage.googleapis.com/v1beta/"
            "models/{model}:generateContent")
# Listing a model does not mean this account may call it: gemini-2.5-flash is
# listed but answers 404 "no longer available to new users", and gemini-3.8-flash
# answered 503 "high demand". A lite model is the right size for rewriting four
# short fields anyway -- measured at 0.7s against 11.7s-then-fail.
DEFAULT_MODEL = "gemini-3.5-flash-lite"
# Tried in order when the chosen model is overloaded or withdrawn.
FALLBACK_MODELS = ("gemini-flash-lite-latest", "gemini-flash-latest")
RETRY_STATUS = {429, 500, 503}

# The only fields the model is allowed to influence. Everything else about the
# alert -- the fault, the level, the numbers -- is the detectors' to decide.
WRITABLE = ("cause", "urgency", "action", "who_to_call")

SYSTEM = """You rewrite building-maintenance findings for a flat owner in India.

The finding is already decided by sensors and is not yours to question. Never
say a problem might not exist, never change what the fault is, never change how
urgent it is, and never invent evidence, causes, measurements or costs that are
not given to you.

Rewrite only the wording, so that someone with no building knowledge
understands what is happening and what to do today. Keep each field to one or
two short sentences. Be concrete and calm. No preamble, no markdown.

Reply with a JSON object containing exactly these keys: cause, urgency, action,
who_to_call. Nothing else."""


class GeminiAdvisor:
    """Callable that fits the `llm` hook in advisor.advise()."""

    def __init__(self, api_key=None, model=None, timeout=30.0):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "no Gemini API key: set GEMINI_API_KEY in the environment")
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        self.timeout = timeout
        # Falling back to templates silently once hid a wrong model name for a
        # whole debugging session. Record why, and say it the first time.
        self.last_error = None
        self._warned = False

    # ------------------------------------------------------------------ call

    def __call__(self, advice, language="en"):
        prompt = self._prompt(advice, language)
        body = {
            "systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        }
        raw = self._post(body)
        return self._accept(raw) if raw else None

    # --------------------------------------------------------------- helpers

    def _prompt(self, advice, language):
        evidence = "\n".join(f"- {line}" for line in advice.get("evidence", [])) \
            or "- (none recorded)"
        tongue = ("Reply in Hindi (Devanagari script)." if language == "hi"
                  else "Reply in English.")
        return (
            f"{tongue}\n\n"
            f"Room: {advice['room']}\n"
            f"Fault the sensors identified: {advice['kind']}\n"
            f"Urgency level already decided: {advice['level']}\n"
            f"Timing already decided: {advice['urgency']}\n\n"
            f"Evidence the detectors recorded:\n{evidence}\n\n"
            f"Current wording to improve:\n"
            f"  cause: {advice['cause']}\n"
            f"  action: {advice['action']}\n"
            f"  who_to_call: {advice['who_to_call']}\n"
        )

    def _post(self, body):
        """Try the chosen model, then the fallbacks if it is overloaded."""
        for model in (self.model, *FALLBACK_MODELS):
            text, retry = self._post_once(body, model)
            if text is not None:
                if model != self.model:
                    self._note(f"fell back to {model}")
                return text
            if not retry:
                break                        # a real error, not transient
        return None

    def _post_once(self, body, model):
        url = ENDPOINT.format(model=model)
        request = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "x-goog-api-key": self.api_key},
            method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            self._note(f"{model} HTTP {exc.code}: {exc.read().decode('utf-8')[:160]}")
            return None, exc.code in RETRY_STATUS
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            self._note(f"{model} {type(exc).__name__}: {exc}")
            return None, True                # network wobble: worth another model
        try:
            return payload["candidates"][0]["content"]["parts"][0]["text"], False
        except (KeyError, IndexError, TypeError):
            self._note(f"{model} unexpected response: {str(payload)[:160]}")
            return None, False

    def _note(self, message):
        self.last_error = message
        if not self._warned:
            self._warned = True
            print(f"[advisor] Gemini unavailable, using template text. {message}",
                  file=sys.stderr)

    @staticmethod
    def _accept(raw):
        """Keep only the wording fields. Everything else the model says is dropped."""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        kept = {k: str(parsed[k]).strip() for k in WRITABLE
                if isinstance(parsed.get(k), str) and parsed[k].strip()}
        return kept or None


def build_advisor(language="en"):
    """A Gemini advisor if a key is configured, otherwise None (templates)."""
    try:
        gemini = GeminiAdvisor()
    except RuntimeError:
        return None
    return lambda advice: gemini(advice, language)
