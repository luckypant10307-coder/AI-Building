"""Run one flat through the whole pipeline and print what the owner would see.

Examples:
    python scripts/run_demo.py --fault hidden_pipe_leak --room bathroom
    python scripts/run_demo.py --fault roof_seepage --room living --language hi
    python scripts/run_demo.py                      # a healthy flat
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from smart_property.config import FAULT_KINDS, ROOMS      # noqa: E402
from smart_property.pipeline import analyse_flat          # noqa: E402
from smart_property.simulator import simulate             # noqa: E402

BAR = "-" * 72


def describe(result, language):
    alert = result["alert"]
    print(f"\n{result['room'].upper()}   [{alert['level'].upper()}]")
    if alert["level"] == "none":
        print("   nothing outside this room's normal week")
        return

    print(f"   fault      : {alert['kind']}")
    print(f"   confidence : {alert['confidence']:.0%}"
          f"   (anomalous hours in the last day: {alert['anomaly_fraction']:.0%})")

    damp = result["forecast"]["visible_damp"]
    mould = result["forecast"]["mould_risk"]
    print(f"   forecaster : {result['forecast']['backend']}")
    for name, item in (("visible damp", damp), ("mould risk", mould)):
        if item["days"] is None:
            print(f"   {name:<12}: not within 7 days "
                  f"(now {item['now']:.1f}, threshold {item['threshold']:.0f})")
        else:
            soonest = ("" if item["days_low"] is None
                       else f", earliest {item['days_low']:.1f}")
            print(f"   {name:<12}: ~{item['days']:.1f} days"
                  f"{soonest}  (now {item['now']:.1f})")

    print("   evidence   :")
    for line in alert["evidence_for"]:
        print(f"      + {line}")
    for line in alert["evidence_against"]:
        print(f"      - {line} (not seen)")

    advice = result["advice"]
    if advice:
        print("   advice     :")
        print(f"      cause  : {advice['cause']}")
        print(f"      urgency: {advice['urgency']}")
        print(f"      action : {advice['action']}")
        print(f"      call   : {advice['who_to_call']}")
        if "confirmation" in advice:
            print(f"      confirm: {advice['confirmation']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fault", choices=FAULT_KINDS, default=None)
    parser.add_argument("--room", choices=ROOMS, default="bathroom")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--days", type=int, default=25)
    parser.add_argument("--start-day", type=float, default=10.0)
    parser.add_argument("--language", choices=["en", "hi"], default="en")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--llm", choices=["none", "gemini"], default="none",
                        help="gemini rewrites the advice; needs GEMINI_API_KEY")
    args = parser.parse_args()

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    df = simulate(seed=args.seed, days=args.days, fault_kind=args.fault,
                  room=args.room if args.fault else None, start_day=args.start_day)
    llm = None
    if args.llm == "gemini":
        from smart_property.llm import build_advisor
        llm = build_advisor(args.language)
        if llm is None:
            print("GEMINI_API_KEY is not set; using template advice.\n")
    results = analyse_flat(df, language=args.language, llm=llm)

    if args.json:
        print(json.dumps(results, indent=2, default=str))
        return

    truth = f"{args.fault} in {args.room}" if args.fault else "no fault"
    print(BAR)
    print(f"Smart Property AI  |  {args.days} simulated days  |  ground truth: {truth}")
    print(BAR)
    for result in results:
        describe(result, args.language)
    print(f"\n{BAR}")
    raised = [r for r in results if r["alert"]["level"] != "none"]
    print(f"{len(raised)} room(s) raised of {len(results)}. "
          "All readings are simulated.")


if __name__ == "__main__":
    main()
