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


def clean_transcript(phrases, lang="ar"):
    """Fix obvious ASR errors in Lebanese-dialect phrases (medical terms, split words) without rewording.
    Same number of phrases back; dialect kept as spoken."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key: raise TranslationUnavailable("no key")
    prompt = ("These are speech-recognition phrases from a Lebanese cardiologist speaking Lebanese Arabic "
              "(may include French/English medical words). Correct only clear recognition errors (misheard medical terms, "
              "words wrongly split or merged, dropped letters). Keep the Lebanese dialect wording exactly as spoken — do not "
              "convert to Modern Standard Arabic, do not add or remove content, do not punctuate heavily. Return ONLY a JSON array "
              "of strings, same length and order.\n\n" + json.dumps(phrases, ensure_ascii=False))
    body = json.dumps({"model": os.environ.get("MEDREEL_LLM", "claude-sonnet-5-5"), "max_tokens": 4000,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", body,
                                 {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    txt = json.loads(urllib.request.urlopen(req, timeout=120).read())["content"][0]["text"]
    out = json.loads(txt[txt.index("["): txt.rindex("]") + 1])
    return [str(x) for x in out] if len(out) == len(phrases) else phrases
