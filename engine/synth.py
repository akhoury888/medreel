"""Mode "AI-voice video": a script becomes a video with no presenter — synthetic narration
(Piper voices, clearly disclosed), medical animations and kinetic captions.

The voices are generic, not clones of any real person. Every output carries an
"AI narration" note on the end card, and the doctor's own voice/face are never imitated."""
import os, re, subprocess, wave
import numpy as np
from . import knowledge as K

MODELS = os.environ.get("MEDREEL_MODELS", os.path.join(os.path.dirname(__file__), "..", "..", "models"))
# (model dir, model file, speaker id, dataset licence). Only voices whose training data allows commercial use.
VOICES = {"en": ("vits-piper-en_US-libritts_r-medium", "en_US-libritts_r-medium.onnx", int(os.environ.get("MEDREEL_VOICE_EN_SID", "48")), "LibriTTS-R, CC BY 4.0"),
          "ar": ("vits-piper-ar_JO-kareem-medium", "ar_JO-kareem-medium.onnx", 0, "UNVERIFIED — confirm before commercial launch"),
          "fr": ("vits-piper-fr_FR-siwis-medium", "fr_FR-siwis-medium.onnx", 0, "SIWIS, CC BY 4.0")}
DISCLOSURE = {"en": "Narration: AI voice.", "fr": "Narration : voix de synthèse.", "ar": "التعليق الصوتي: صوت مُولَّد بالذكاء الاصطناعي."}
_tts = {}


def available(lang):
    d, m = VOICES.get(lang, VOICES["en"])[:2]; return os.path.exists(os.path.join(MODELS, d, m))


def _engine(lang):
    import sherpa_onnx
    if lang not in _tts:
        d, m = VOICES[lang][:2]; d = os.path.join(MODELS, d)
        cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=f"{d}/{m}", tokens=f"{d}/tokens.txt", data_dir=f"{d}/espeak-ng-data"),
            num_threads=max(1, os.cpu_count() or 1)))
        _tts[lang] = sherpa_onnx.OfflineTts(cfg)
    return _tts[lang]


def split_sentences(script):
    parts = re.split(r"(?<=[.!?؟…])\s+|\n+", script.strip())
    return [p.strip() for p in parts if p.strip()]


def _speakable(text, lang):
    """What the voice reads (captions keep the original): '130/80' -> words."""
    if lang == "fr": text = re.sub(r"\b(1[0-9])\s*/\s*([5-9])\b", r"\1 \2", text)
    sep = {"en": " over ", "fr": " sur ", "ar": " على "}[lang]
    return re.sub(r"\b(\d{2,3})\s*/\s*(\d{2,3})\b", lambda m: m.group(1) + sep + m.group(2), text)


def make_source(script, lang, out_mp4, speed=1.0, gap=0.35):
    """-> (words with exact sentence timing, duration). Writes a plain 1080x1920 source with the narration."""
    eng = _engine(lang)
    sents = split_sentences(script)
    sr, chunks, words, t = None, [], [], 0.4
    for sent in sents:
        g = eng.generate(_speakable(sent, lang), sid=VOICES[lang][2], speed=speed); sr = g.sample_rate
        a = np.array(g.samples, np.float32)
        if not chunks: chunks.append(np.zeros(int(sr * 0.4), np.float32))
        # trim TTS leading/trailing silence so word timing is tight
        env = np.abs(a) > 0.01; idx = np.where(env)[0]
        if len(idx): a = a[max(0, idx[0] - int(sr * .03)): idx[-1] + int(sr * .05)]
        d = len(a) / sr
        toks = sent.split(); wts = [len(K.core(x)) + 1 for x in toks]; W = float(sum(wts)); acc = 0.0
        for tok, wt in zip(toks, wts):
            s0 = t + d * acc / W; acc += wt; s1 = t + d * acc / W
            words.append({"text": tok, "start": round(s0, 3), "end": round(s1 - 0.02, 3), "conf": 1.0, "script": True, "tts": True})
        chunks += [a, np.zeros(int(sr * gap), np.float32)]; t += d + gap
    y = np.concatenate(chunks + [np.zeros(int(sr * 0.5), np.float32)])
    y = y / (np.abs(y).max() + 1e-9) * 0.6
    wav = out_mp4 + ".wav"
    with wave.open(wav, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((y * 32767).astype(np.int16).tobytes())
    dur = len(y) / sr
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c=0x0B1730:s=1080x1920:r=30:d={dur:.3f}", "-i", wav,
                    "-c:v", "libx264", "-preset", "ultrafast", "-tune", "stillimage", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out_mp4], check=True)
    os.remove(wav)
    return words, dur
