"""Repair advisor (component 9).

Turns the evidence into plain-language cause, urgency, action and who to call,
in English or Hindi.

The template text below is the fallback and the default. An LLM may be passed
in to rewrite it, but it only ever rewrites evidence that the detectors have
already produced -- it never decides whether a problem exists, and if the call
fails the template text is what the owner sees.
"""

TEMPLATES = {
    "hidden_pipe_leak": {
        "en": {
            "cause": "A concealed pipe inside the wall is leaking. The plaster is "
                     "taking up water even when no tap is open.",
            "action": "Call a plumber and have the pipework inside this wall "
                      "pressure-tested. Shut the supply to this branch if you can.",
            "who": "Plumber",
        },
        "hi": {
            "cause": "दीवार के अंदर लगे पाइप से पानी रिस रहा है। बिना नल खोले भी "
                     "प्लास्टर में नमी बढ़ रही है।",
            "action": "प्लंबर को बुलाएँ और इस दीवार की पाइपलाइन की जाँच कराएँ। "
                      "हो सके तो इस लाइन का पानी बंद कर दें।",
            "who": "प्लंबर",
        },
    },
    "roof_seepage": {
        "en": {
            "cause": "Rain is getting in through the roof or the outside wall. The "
                     "moisture rises after rain and dries back between showers.",
            "action": "Have the roof and the outside face of this wall waterproofed "
                      "before the next heavy spell.",
            "who": "Waterproofing contractor",
        },
        "hi": {
            "cause": "छत या बाहरी दीवार से बारिश का पानी अंदर आ रहा है। बारिश के बाद "
                     "नमी बढ़ती है और फिर कुछ कम हो जाती है।",
            "action": "अगली तेज़ बारिश से पहले छत और बाहरी दीवार की वॉटरप्रूफिंग कराएँ।",
            "who": "वॉटरप्रूफिंग ठेकेदार",
        },
    },
    "condensation": {
        "en": {
            "cause": "This wall is colder than the room, so moisture in the air is "
                     "settling on it. This is condensation, not a leak.",
            "action": "Air the room daily and run an exhaust fan while cooking or "
                      "bathing. If it keeps up, the wall needs insulating.",
            "who": "No one yet, you can try this first",
        },
        "hi": {
            "cause": "यह दीवार कमरे से ठंडी है, इसलिए हवा की नमी उस पर जम रही है। "
                     "यह रिसाव नहीं, सीलन है।",
            "action": "कमरे में रोज़ हवा आने-जाने दें और नहाते या खाना बनाते समय "
                      "एग्ज़ॉस्ट फैन चलाएँ। फिर भी ठीक न हो तो दीवार पर इन्सुलेशन कराएँ।",
            "who": "अभी किसी को नहीं, पहले यह आज़माएँ",
        },
    },
    "running_fixture": {
        "en": {
            "cause": "A tap or flush is running continuously. Water is being drawn "
                     "through the night, but the wall is still dry.",
            "action": "Check the taps and the flush tanks in this room. This is "
                      "wasting water, not damaging the wall.",
            "who": "No one yet, you can try this first",
        },
        "hi": {
            "cause": "कोई नल या फ्लश लगातार चल रहा है। रात भर पानी बह रहा है, "
                     "लेकिन दीवार अभी सूखी है।",
            "action": "इस कमरे के नल और फ्लश टंकी जाँचें। इससे पानी बर्बाद हो रहा है, "
                      "दीवार को नुकसान नहीं।",
            "who": "अभी किसी को नहीं, पहले यह आज़माएँ",
        },
    },
}

URGENCY = {
    "en": {
        "already": "Already showing: this wall is past the damage threshold now.",
        "urgent": "Urgent: damage becomes visible in about {days} days.",
        "action": "Act this week: damage becomes visible in about {days} days.",
        "watch": "Keep watching: no visible damage expected within a week.",
    },
    "hi": {
        "already": "अभी से दिख रहा है: यह दीवार नुकसान की सीमा पार कर चुकी है।",
        "urgent": "तुरंत ध्यान दें: लगभग {days} दिन में नुकसान दिखने लगेगा।",
        "action": "इस हफ़्ते कुछ करें: लगभग {days} दिन में नुकसान दिखने लगेगा।",
        "watch": "नज़र रखें: एक हफ़्ते में दिखने वाला नुकसान नहीं है।",
    },
}

CONFIRM = {
    "en": "A technician should confirm this before any repair is paid for.",
    "hi": "मरम्मत का ख़र्च करने से पहले किसी तकनीशियन से पुष्टि करा लें।",
}


def advise(alert, room, language="en", llm=None):
    """Build the owner-facing advice for one fused alert."""
    if alert["level"] == "none" or alert["kind"] not in TEMPLATES:
        return None

    text = TEMPLATES[alert["kind"]][language]
    days = alert.get("days_to_damage")
    # past the threshold already: "in about 0 days" would be nonsense
    key = "already" if (days is not None and days < 0.5) else alert["level"]
    urgency = URGENCY[language][key]
    if "{days}" in urgency:
        urgency = urgency.format(days=int(round(days)) if days is not None else "?")

    advice = {
        "room": room,
        "kind": alert["kind"],
        "level": alert["level"],
        "cause": text["cause"],
        "urgency": urgency,
        "action": text["action"],
        "who_to_call": text["who"],
        "evidence": alert.get("evidence_for", []),
        "source": "template",
    }
    if alert["needs_technician_confirmation"]:
        advice["confirmation"] = CONFIRM[language]

    if llm is not None:
        try:
            rewritten = llm(advice)
            if rewritten:
                advice.update(rewritten)
                advice["source"] = "llm"
        except Exception:
            pass        # the template text stands if the model is unavailable
    return advice
