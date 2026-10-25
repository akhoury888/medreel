"""Subtitle translation (e.g. Lebanese Arabic -> English / French) via Claude.

Translations live in plan.captions.translations {sentence_index: text}; they are editable by
the doctor and are never invented when no API key is configured."""
import json, os, urllib.request

NAMES = {"en": "English", "fr": "French", "ar": "Arabic (Modern Standard, easy to read)"}


class TranslationUnavailable(RuntimeError):
    pass


def translate_sentences(sentences, source_lang, target_lang):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise TranslationUnavailable("Subtitle translation needs ANTHROPIC_API_KEY on the server. "
                                     "You can also type the translations by hand in the Transcript tab.")
    src = {"ar": "Lebanese Arabic (dialect, may mix French and English medical words)"}.get(source_lang, NAMES.get(source_lang, source_lang))
    prompt = (f"Translate these spoken {src} sentences from a cardiologist's educational video into short, natural "
              f"{NAMES.get(target_lang, target_lang)} subtitles. Keep medical meaning exact, keep numbers, do not add or soften claims. "
              f"Return ONLY a JSON array of strings, same length and order.\n\n" + json.dumps(sentences, ensure_ascii=False))
    body = json.dumps({"model": os.environ.get("MEDREEL_LLM", "claude-sonnet-5-5"), "max_tokens": 4000,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", body,
                                 {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    txt = json.loads(urllib.request.urlopen(req, timeout=120).read())["content"][0]["text"]
    out = json.loads(txt[txt.index("["): txt.rindex("]") + 1])
    if len(out) != len(sentences): raise RuntimeError("translation length mismatch")
    return [str(x) for x in out]


def fill_plan_translations(plan, target):
    sents = [s["text"] for s in plan["analysis"]["sentences"]]
    tr = translate_sentences(sents, plan.get("language", "en"), target)
    plan["captions"]["translations"] = {str(i): t for i, t in enumerate(tr)}
    plan["captions"]["target"] = target
    return len(tr)


def clean_transcript(phrases, lang="ar", alts=None, context=None):
    """Fix speech-recognition errors in dialect phrases without rewording. `alts` = the same phrases
    from a second recognizer (each model makes different mistakes); `context` = speaker name,
    specialty, clinic and video title so names and places come out right. Same number of phrases back."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key: raise TranslationUnavailable("no key")
    ctx = context or {}
    about = "; ".join(f"{k}: {v}" for k, v in (("speaker", ctx.get("name")), ("specialty", ctx.get("specialty")),
                                                  ("clinic / hospital", ctx.get("clinic")), ("video title", ctx.get("title"))) if v)
    if alts and len(alts) == len(phrases):
        items = [{"a": x, "b": y} for x, y in zip(phrases, alts)]
        shape = ("Each item has two recognitions of the same spoken phrase: 'a' (more accurate model) and 'b' (another model). "
                 "Where they disagree, choose or combine what was most plausibly said.")
    else:
        items, shape = phrases, "Each item is one recognized spoken phrase."
    dialect = {"ar": "Lebanese Arabic (with French/English medical words)", "fr": "French", "en": "English"}.get(lang, lang)
    prompt = (f"These are speech-recognition phrases from a physician speaking {dialect} in a short educational social-media video"
              + (f" ({about})" if about else "") + ". " + shape +
              " Correct recognition errors only: misheard medical terms and test names (e.g. coroscanner, calcium score, stent, "
              "cholesterol), the speaker's name, hospital and place names, words wrongly split or merged, dropped letters. "
              "Write foreign medical words the way the speaker said them (Latin script is fine for English/French terms). "
              "Keep the dialect exactly as spoken — never convert to Modern Standard Arabic, never add, remove or summarise content, "
              "no heavy punctuation. Return ONLY a JSON array of strings: one corrected phrase per item, same length and order.\n\n"
              + json.dumps(items, ensure_ascii=False))
    body = json.dumps({"model": os.environ.get("MEDREEL_LLM", "claude-sonnet-5-5"), "max_tokens": 4000,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", body,
                                 {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    txt = json.loads(urllib.request.urlopen(req, timeout=120).read())["content"][0]["text"]
    out = json.loads(txt[txt.index("["): txt.rindex("]") + 1])
    return [str(x) for x in out] if len(out) == len(phrases) else phrases
