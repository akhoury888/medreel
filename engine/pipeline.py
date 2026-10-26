"""Orchestration: project folder layout + analyze -> plan -> render. Used by the CLI and the web app.

projects/<id>/
  source<ext>          original upload (read-only; never modified)
  transcript.json      ASR words
  analysis.json        analysis pass
  plan.json            editable EditPlan (+ plan.json.history.json for undo)
  renders/preview.mp4, renders/final.mp4 (+ .srt)
"""
import json, os, shutil, stat, time
from . import asr, analyze as A, planner as P, render as R


def project_dir(root, pid): return os.path.join(root, pid)


def ingest(root, src_path, pid=None):
    pid = pid or time.strftime("%Y%m%d-%H%M%S")
    d = project_dir(root, pid); os.makedirs(os.path.join(d, "renders"), exist_ok=True)
    dst = os.path.join(d, "source" + os.path.splitext(src_path)[1].lower())
    if os.path.abspath(src_path) != os.path.abspath(dst): shutil.copy2(src_path, dst)
    os.chmod(dst, stat.S_IRUSR | stat.S_IRGRP)  # source is read-only
    return pid, d, dst


def analyze_and_plan(d, src, style="modern_educational", branding=None, log=print, progress=None, language="auto", title=None, ai_budget=None):
    def p(x, msg):
        log(msg)
        if progress: progress(x, msg)
    p(0.05, "Transcribing speech… (Arabic takes a few minutes on a laptop)" if language == "ar" else "Transcribing speech…")
    ctx = {k: (branding or {}).get(k) for k in ("name", "specialty", "clinic")}; ctx["title"] = title
    backend, words, lang = asr.transcribe(src, language, context=ctx)
    json.dump({"backend": backend, "language": lang, "words": words}, open(os.path.join(d, "transcript.json"), "w"))
    p(0.45, f"Transcribed {len(words)} words ({backend}, {lang}). Analysing pauses, retakes, medical content, framing and audio…")
    an = A.analyze(src, words, lang)
    json.dump(an, open(os.path.join(d, "analysis.json"), "w"))
    p(0.75, "Building the editing plan…")
    plan = P.build_plan(src, backend, words, an, style=style, branding=branding, language=lang)
    if title and len(title) <= 60:  # the doctor's own title beats a sentence picked from the transcript
        plan["title"].update({"text": title, "enabled": True})
    _stock(plan, d, p, ai_budget)
    plan["mode"] = "auto_edit"
    P.save(plan, os.path.join(d, "plan.json"), history=False)
    p(0.8, "Plan ready.")
    return plan


def _stock(plan, d, p, ai_budget=None):
    """ai_budget (from the account's AI credits): {"credits", "image", "animation", "max_images", "max_animations"};
    None = no limit (local/CLI use). What was actually made is recorded in plan["ai_usage"] for billing."""
    from . import broll, imagegen
    plan["ai_usage"] = {"image": 0, "animation": 0}
    if imagegen.available() and (ai_budget is None or ai_budget["credits"] >= ai_budget["image"]):
        p(0.76, "Creating medical illustrations…")
        try:
            b = ai_budget
            cap = None if b is None else max(0, min(b["max_images"], b["credits"] // b["image"]))
            n = imagegen.auto(plan, d, cap=cap) if cap != 0 else 0
            plan["ai_usage"]["image"] = n
            left = None if b is None else b["credits"] - n * b["image"]
            k_cap = None if b is None else max(0, min(b["max_animations"], left // b["animation"]))
            if n and k_cap != 0:
                p(0.78, f"Added {n} illustration(s). Animating the key ones (1–3 min)…")
                env_n = int(os.environ.get("MEDREEL_AI_ANIMATE", "2"))
                plan["ai_usage"]["animation"] = imagegen.auto_animate(plan, d, n=env_n if k_cap is None else min(env_n, k_cap))
        except Exception:
            pass  # illustrations are a bonus: never fail a video because of them
    if not broll.available(): return
    p(0.78, "Adding stock footage…")
    try:
        n = broll.auto(plan, d)
        if n: p(0.8, f"Added {n} stock clip(s).")
    except Exception:
        pass  # footage is a bonus: never fail a video because of it


MODES = {"auto_edit": "Upload a recording — AI edits everything",
         "own_script": "Upload a recording + your script — exact captions, AI graphics",
         "teleprompter": "Prompt → AI script → record with the teleprompter → AI edits",
         "ai_voice": "Prompt or script → video with AI narration (no recording)"}


def create(d, mode, src=None, script=None, style="modern_educational", branding=None, language="auto", log=print, progress=None, title=None, ai_budget=None):
    """Single entry point for all creation modes. Returns the plan (also saved to plan.json)."""
    from . import script_align, synth
    def p(x, msg):
        log(msg)
        if progress: progress(x, msg)
    if mode == "auto_edit" or (mode in ("own_script", "teleprompter") and not (script or "").strip()):
        return analyze_and_plan(d, src, style, branding, log, progress, language, title, ai_budget)
    lang = language if language in ("en", "fr", "ar") else script_align.detect_script_language(script)
    if mode == "ai_voice":
        if not synth.available(lang): raise RuntimeError(f"No AI voice installed for '{lang}'. Run tools/setup_models.sh.")
        p(0.05, "Generating the AI narration…")
        src = os.path.join(d, "source.mp4")
        words, dur = synth.make_source(script, lang, src)
        os.chmod(src, stat.S_IRUSR | stat.S_IRGRP)
        json.dump({"backend": "tts", "language": lang, "words": words}, open(os.path.join(d, "transcript.json"), "w"))
        p(0.4, "Matching medical animations to the script…")
        an = A.analyze(src, words, lang, presenter=False)
        json.dump(an, open(os.path.join(d, "analysis.json"), "w"))
        brand = dict(branding or {})
        brand["disclaimer"] = ((brand.get("disclaimer") or P.DEFAULT_BRAND["disclaimer"]) if lang == "en" else
                               (brand.get("disclaimer") or (P.FR_DISCLAIMER if lang == "fr" else P.AR_DISCLAIMER))) + " " + synth.DISCLOSURE[lang]
        plan = P.build_plan(src, "tts", words, an, style=style, branding=brand, language=lang, presenter=False)
    else:  # own_script / teleprompter: the script is the caption text; ASR only times it
        p(0.05, "Timing your script against the recording…")
        backend, asr_words, lang2 = asr.transcribe(src, lang, quality="fast")
        audio = asr.load_audio(src)
        retakes = A._merge_retakes(A.detect_retakes(asr_words))  # false starts the script doesn't contain
        words, coverage = script_align.align(asr_words, script, audio)
        json.dump({"backend": backend + "+script", "language": lang, "words": words, "asr_words": asr_words, "coverage": coverage},
                  open(os.path.join(d, "transcript.json"), "w"))
        p(0.45, f"Script aligned ({coverage*100:.0f}% of words anchored). Analysing the recording…")
        an = A.analyze(src, words, lang, extra_retakes=retakes)
        json.dump(an, open(os.path.join(d, "analysis.json"), "w"))
        plan = P.build_plan(src, backend + "+script", words, an, style=style, branding=branding, language=lang)
        plan["script_alignment"] = {"coverage": round(coverage, 3)}
        if coverage < 0.35:
            plan["analysis"]["qa"]["issues"].append({"area": "script", "level": "warn",
                "msg": "The recording doesn't follow the script closely — check caption timing, or switch to automatic captions."})
    if title and len(title) <= 60: plan["title"].update({"text": title, "enabled": True})
    _stock(plan, d, p, ai_budget)
    plan["mode"] = mode
    P.save(plan, os.path.join(d, "plan.json"), history=False)
    p(0.8, "Plan ready.")
    return plan


def render_project(d, preview=False, log=print, progress=None):
    plan = json.load(open(os.path.join(d, "plan.json")))
    out = os.path.join(d, "renders", "preview.mp4" if preview else "final.mp4")
    tmp = out + ".part.mp4"
    res = R.render(plan, tmp, os.path.join(d, "work"), preview=preview, progress=progress, log=log)
    os.replace(tmp, out)
    import glob
    for srt in glob.glob(os.path.splitext(tmp)[0] + "*.srt"):  # .srt and .<lang>.srt
        os.replace(srt, os.path.splitext(out)[0] + srt[len(os.path.splitext(tmp)[0]):])
    res["path"] = out
    return res


def delete_project(d):
    """Hard data deletion: source, derived media, transcripts, plans, caches."""
    for root, dirs, files in os.walk(d):
        for f in files: os.chmod(os.path.join(root, f), stat.S_IWUSR | stat.S_IRUSR)
    shutil.rmtree(d)
