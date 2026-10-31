"""Make a talking AI presenter clip for MedReel's own ads (a FICTIONAL doctor — never a real person).

  1. FLUX.2 (fal.ai) creates a photo of a fictional doctor (kept and reused, so every language has the same face)
  2. ElevenLabs reads the script aloud (multilingual voice)
  3. Kling AI Avatar v2 (fal.ai) makes the photo speak the audio, lips in sync

Run inside the MedReel container (it already has FAL_KEY and ELEVENLABS_API_KEY):
  docker compose -f docker-compose.local.yml exec medreel python tools/make_presenter.py en
  docker compose -f docker-compose.local.yml cp medreel:/data/presenter/presenter_en.mp4 .
Options: en | fr | ar | lb (Lebanese Arabic)   --new-face (make a different doctor)   --male (a male doctor)   --pro (sharper lips, ~2x price)
         --audio=/data/presenter/my_voice.m4a   use a real voice recording instead of the AI voice (best for Lebanese)
         --text="..."                           your own script for the AI voice
Cost: about $1 per 18-second clip (standard) + a few cents for the photo.
The presenter is AI-generated: label it as such in ads, and never present it as a real doctor or patient testimonial."""
import base64, json, os, sys, time, urllib.request, urllib.error
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine.imagegen import _fal_queue, FAL_MODEL

OUT = os.environ.get("PRESENTER_DIR", "/data/presenter")
SCRIPTS = {
    "en": "Did you know that high cholesterol usually has no symptoms? Over the years, it slowly builds plaque inside your arteries and narrows them. "
          "That's why a simple blood test matters. Check your cholesterol, and talk to your doctor about your heart.",
    "fr": "Saviez-vous que le cholestérol élevé ne donne généralement aucun symptôme ? Avec les années, il forme des plaques dans vos artères et les rétrécit. "
          "Une simple prise de sang peut tout changer. Faites contrôler votre cholestérol et parlez-en à votre médecin.",
    "lb": "بتعرفوا إنو الكوليسترول العالي عادةً ما بيعطي أي عوارض؟ مع السنين، بيتجمّع جوّا الشرايين وبيضيّقها. "
          "لهيك فحص دم بسيط بيفرق كتير. افحصوا الكوليسترول، واحكوا مع حكيمكن عن صحة قلبكن.",
    "ar": "هل تعلم أن ارتفاع الكوليسترول عادةً لا يسبب أي أعراض؟ مع السنين، يتراكم داخل الشرايين ويضيّقها. "
          "لهذا فحص دم بسيط مهم جداً. افحص الكوليسترول، واسأل طبيبك عن صحة قلبك.",
}
FACE_PROMPT = ("Photorealistic vertical portrait of a fictional friendly female doctor in her late thirties, white coat over a navy top, "
               "stethoscope around the neck, looking straight at the camera with a warm natural expression, mouth closed, head and shoulders "
               "centred in the upper half of the frame, modern bright clinic softly blurred behind, soft window light, sharp focus on the face, "
               "natural skin texture, shot on a 50mm lens, no text, no logo")
VOICE = os.environ.get("PRESENTER_VOICE", "21m00Tcm4TlvDq8ikWAM")  # an ElevenLabs premade voice (change to any voice id you like)


def say(m): print(f"[presenter] {m}", flush=True)


def face(new=False, male=False):
    p = os.path.join(OUT, "face_male.jpg" if male else "face.jpg")
    if os.path.exists(p) and not new: say("using the existing face (add --new-face for another one)"); return p
    say("creating the doctor's photo (FLUX.2)…")
    hdr = {"Authorization": f"Key {os.environ['FAL_KEY']}", "Content-Type": "application/json"}
    prompt = FACE_PROMPT.replace("female doctor in her late thirties", "male doctor in his early forties, short dark hair, neat beard") if male else FACE_PROMPT
    body = {"prompt": prompt, "image_size": {"width": 720, "height": 1280}, "output_format": "jpeg", "enable_safety_checker": True}
    res = json.loads(urllib.request.urlopen(urllib.request.Request(f"https://fal.run/{FAL_MODEL}", json.dumps(body).encode(), hdr), timeout=180).read())
    url = res["images"][0]["url"]
    data = base64.b64decode(url.split(",", 1)[1]) if url.startswith("data:") else urllib.request.urlopen(url, timeout=60).read()
    open(p, "wb").write(data); say(f"photo saved: {p}")
    return p


def voice(lang, text=None, male=False):
    p = os.path.join(OUT, f"voice_{lang}.mp3")
    say("recording the voice (ElevenLabs)…")
    vid = os.environ.get("PRESENTER_VOICE") or ("pNInz6obpgDQGcFmaJgB" if male else VOICE)
    body = {"text": text or SCRIPTS[lang], "model_id": os.environ.get("PRESENTER_TTS_MODEL", "eleven_multilingual_v2"), "voice_settings": {"stability": 0.45, "similarity_boost": 0.8, "style": 0.2}}
    req = urllib.request.Request(f"https://api.elevenlabs.io/v1/text-to-speech/{vid}?output_format=mp3_44100_128", json.dumps(body).encode(),
                                 {"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "Content-Type": "application/json"})
    try:
        open(p, "wb").write(urllib.request.urlopen(req, timeout=120).read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:300]
        if "permission" in detail.lower() or e.code in (401, 403):
            sys.exit("ElevenLabs refused text-to-speech. In elevenlabs.io → Developers → API keys, edit your key and turn ON "
                     "'Text to Speech' (your key may only allow Speech to Text). Then run this again.\n" + detail)
        sys.exit(f"ElevenLabs error {e.code}: {detail}")
    say(f"voice saved: {p}")
    return p


def talk(img, audio, lang, pro=False):
    mime = {".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".wav": "audio/wav", ".ogg": "audio/ogg", ".aac": "audio/aac"}.get(os.path.splitext(audio)[1].lower(), "audio/mpeg")
    model = "fal-ai/kling-video/ai-avatar/v2/" + ("pro" if pro else "standard")
    say(f"making the photo speak ({model}) — usually 2-6 minutes…")
    uri = lambda path, mime: f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()
    res = _fal_queue(model, {"image_url": uri(img, "image/jpeg"), "audio_url": uri(audio, mime),
                             "prompt": "A friendly doctor explaining calmly to the camera, natural small head movements and blinks, professional, steady camera"},
                     timeout=1500, on_status=lambda m: say(m))
    url = (res.get("video") or {}).get("url")
    if not url: sys.exit("no video returned: " + json.dumps(res)[:300])
    out = os.path.join(OUT, f"presenter_{lang}.mp4")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "MedReel/1.0"}), timeout=300) as r, open(out, "wb") as f:
        f.write(r.read())
    say(f"DONE: {out}")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    lang = next((a for a in args if a in SCRIPTS), "en")
    for k in ("FAL_KEY", "ELEVENLABS_API_KEY"):
        if not os.environ.get(k): sys.exit(f"{k} is missing in your .env file.")
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    male = "--male" in args
    own = next((a.split("=", 1)[1] for a in args if a.startswith("--audio=")), None)
    text = next((a.split("=", 1)[1] for a in args if a.startswith("--text=")), None)
    if own and not os.path.exists(own): sys.exit(f"Audio file not found: {own}")
    if own:
        import subprocess  # a real voice recording: tidy it (mono, loudness) so the lips follow it well
        clean = os.path.join(OUT, f"voice_{lang}_own.mp3")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", own, "-ac", "1", "-ar", "44100", "-af", "highpass=f=80,loudnorm=I=-16:TP=-1.5",
                        "-b:a", "160k", clean], check=True)
        say(f"using your recording: {own}")
    talk(face("--new-face" in args, male), clean if own else voice(lang, text, male), lang, "--pro" in args)
    say(f"finished in {int(time.time() - t0)} s")
