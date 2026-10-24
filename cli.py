"""MedReel CLI
  python cli.py make  <video> [--style modern_educational] [--brand brand.json] [--preview] [--approve-fillers]
  python cli.py edit  <project_dir> "Make the subtitles more dynamic"
  python cli.py render <project_dir> [--preview]
"""
import argparse, json, os, sys
from engine import pipeline, commands, planner

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "projects")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("video"); m.add_argument("--style", default="modern_educational")
    m.add_argument("--brand"); m.add_argument("--preview", action="store_true"); m.add_argument("--approve-fillers", action="store_true"); m.add_argument("--id")
    e = sub.add_parser("edit"); e.add_argument("project"); e.add_argument("command")
    r = sub.add_parser("render"); r.add_argument("project"); r.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    if a.cmd == "make":
        brand = json.load(open(a.brand)) if a.brand else None
        pid, d, src = pipeline.ingest(ROOT, a.video, a.id)
        plan = pipeline.analyze_and_plan(d, src, a.style, brand)
        if a.approve_fillers:
            commands.execute(os.path.join(d, "plan.json"), "approve all filler cuts")
        res = pipeline.render_project(d, preview=a.preview)
        print(json.dumps(res, indent=1)); print("project:", d)
    elif a.cmd == "edit":
        print(json.dumps(commands.execute(os.path.join(a.project, "plan.json"), a.command), indent=1))
    else:
        print(json.dumps(pipeline.render_project(a.project, preview=a.preview), indent=1))


if __name__ == "__main__":
    main()
