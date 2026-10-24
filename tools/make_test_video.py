"""Builds a ~60 s TEST FIXTURE that mimics a raw phone recording of a cardiologist:
long pauses, an "um", a false start / retake and one overstated claim.
The voice is synthetic (Piper TTS via sherpa-onnx) and the presenter is an
illustrated stand-in — it exists only to exercise the pipeline. Real use = a real recording."""
import os, sys, subprocess, wave
import numpy as np, cv2

MODELS = os.environ.get("MEDREEL_MODELS", os.path.join(os.path.dirname(__file__), "..", "..", "models"))
SCRIPT = [
    ("Hi, I'm a cardiologist, and this is something I explain to my patients every single week.", 0.45),
    ("Today I want to explain what really happens inside your heart arteries.", 1.9),
    ("When cholesterol accumulates inside the coronary arteries, it gradually narrows the vessel.", 0.5),
    ("Umm.", 0.45),
    ("This process is called atherosclerosis, and it can", 0.9),
    ("This process is called atherosclerosis, and it can develop silently for many years.", 0.6),
    ("Over time, less blood reaches the heart muscle, which can cause chest pain during exercise.", 2.3),
    ("If a plaque ruptures, a clot can suddenly block the artery. That is a heart attack.", 0.55),
    ("The good news is that we can often open a narrowed artery with angioplasty and a stent.", 0.7),
    ("Uhh.", 0.35),
    ("Exercise alone can completely reverse any blockage.", 0.8),
    ("To protect your arteries, keep your blood pressure below 130 over 80, check your cholesterol, and stay active.", 1.6),
    ("If you have chest pain or shortness of breath, talk to your doctor. Follow for more heart health tips.", 1.0),
]


SCRIPT_AR = [  # Lebanese dialect, the way a cardiologist would actually say it
    ("مرحبا، أنا طبيب قلب، وهيدا موضوع بشرحو لمرضاي كل جمعة.", 0.45),
    ("اليوم بدي إشرحلكن شو عم يصير جوّا شرايين القلب.", 1.9),
    ("لما الكوليسترول يتجمّع جوّا الشرايين التاجية، بيضيّق الشريان شوي شوي.", 0.5),
    ("إمم.", 0.45),
    ("هيدا اسمو تصلّب الشرايين، وفيه", 0.9),
    ("هيدا اسمو تصلّب الشرايين، وفيه يتطوّر بلا عوارض لسنين.", 0.6),
    ("مع الوقت، الدم اللي بيوصل لعضلة القلب بيخفّ، وبيعمل وجع بالصدر وقت الرياضة.", 2.2),
    ("إذا انفجرت اللويحة، بتتشكّل جلطة بتسكّر الشريان. هيدي هي الذبحة القلبية.", 0.55),
    ("الخبر المنيح إنو فينا نفتح الشريان بالبالون ونحطّ دعامة.", 0.7),
    ("يعني.", 0.4),
    ("الرياضة لحالها بتشيل أي تسكير بالشرايين مية بالمية.", 0.8),
    ("حافظوا على ضغط الدم تحت مية وتلاتين على تمانين، وافحصوا الكوليسترول.", 1.5),
    ("إذا عندكن وجع بالصدر أو ضيق نفس، احكوا مع حكيمكن. تابعوني لنصايح أكتر عن صحة القلب.", 1.0),
]
VOICES = {"en": ("vits-piper-en_US-ryan-high", "en_US-ryan-high.onnx"), "ar": ("vits-piper-ar_JO-kareem-medium", "ar_JO-kareem-medium.onnx")}


def tts(out_wav, lang="en"):
    import sherpa_onnx
    vd, vm = VOICES[lang]; d = os.path.join(MODELS, vd)
    cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=f"{d}/{vm}", tokens=f"{d}/tokens.txt", data_dir=f"{d}/espeak-ng-data"),
        num_threads=2))
    tts = sherpa_onnx.OfflineTts(cfg)
    sr = None; chunks = [np.zeros(int(22050 * 0.6), np.float32)]
    rng = np.random.default_rng(0)
    for text, gap in (SCRIPT_AR if lang == "ar" else SCRIPT):
        a = tts.generate(text, sid=0, speed=1.0); sr = a.sample_rate
        chunks.append(np.array(a.samples, np.float32)); chunks.append(np.zeros(int(sr * gap), np.float32))
    y = np.concatenate(chunks)
    y = y + rng.normal(0, 0.004, len(y)).astype(np.float32)  # room tone / hiss to test denoise
    y = y / (np.abs(y).max() + 1e-9) * 0.5
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((y * 32767).astype(np.int16).tobytes())
    return y, sr


def presenter(out_mp4, wav, y, sr, W=1080, H=1920, fps=30):
    n = int(len(y) / sr * fps)
    hop = sr // fps
    env = np.array([np.sqrt(np.mean(y[i * hop:(i + 1) * hop] ** 2)) for i in range(n)])
    env = np.convolve(env / (env.max() + 1e-9), np.ones(3) / 3, "same")
    bg = np.zeros((H, W, 3), np.float32)
    gy = np.linspace(0, 1, H)[:, None]
    bg[:] = (np.array([206, 214, 222]) * (1 - gy) + np.array([168, 180, 192]) * gy)[:, None, :]
    cv2.rectangle(bg, (60, 260), (420, 900), (236, 240, 244), -1)             # window
    cv2.rectangle(bg, (760, 300), (1040, 1150), (120, 92, 70), -1)            # bookshelf
    for k in range(6): cv2.rectangle(bg, (780, 330 + k * 130), (1020, 345 + k * 130), (90, 66, 50), -1)
    cv2.ellipse(bg, (170, 1150), (120, 200), 0, 0, 360, (70, 120, 80), -1)    # plant
    bg = cv2.GaussianBlur(bg, (0, 0), 16).astype(np.uint8)
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                            "-i", wav, "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out_mp4],
                           stdin=subprocess.PIPE)
    skin, hair, coat = (150, 180, 222), (40, 45, 55), (246, 246, 248)
    for i in range(n):
        f = bg.copy(); t = i / fps
        bob = 6 * np.sin(t * 1.3) + 3 * np.sin(t * 2.9); sway = 5 * np.sin(t * 0.7)
        cx, cy = int(540 + sway), int(640 + bob)
        torso = np.array([[150, 1920], [250, 1150], [420, 1000], [660, 1000], [830, 1150], [930, 1920]], np.int32)
        cv2.fillPoly(f, [torso], coat, cv2.LINE_AA)
        cv2.fillPoly(f, [np.array([[440, 1000], [640, 1000], [600, 1300], [480, 1300]], np.int32)], (170, 110, 60), cv2.LINE_AA)  # shirt
        cv2.fillPoly(f, [np.array([[525, 1010], [555, 1010], [570, 1250], [540, 1290], [510, 1250]], np.int32)], (60, 40, 140), cv2.LINE_AA)  # tie
        cv2.line(f, (430, 1040), (400, 1350), (60, 60, 60), 8, cv2.LINE_AA); cv2.circle(f, (400, 1360), 22, (150, 150, 160), -1, cv2.LINE_AA)
        cv2.rectangle(f, (cx - 60, cy + 150), (cx + 60, cy + 380), skin, -1)  # neck
        cv2.ellipse(f, (cx, cy), (165, 215), 0, 0, 360, skin, -1, cv2.LINE_AA)
        cv2.ellipse(f, (cx, cy - 95), (172, 140), 0, 180, 360, hair, -1, cv2.LINE_AA)
        for ex in (-62, 62):
            blink = (t % 3.7) < 0.12
            cv2.ellipse(f, (cx + ex, cy - 10), (24, 3 if blink else 13), 0, 0, 360, (40, 40, 40), -1, cv2.LINE_AA)
            cv2.line(f, (cx + ex - 32, cy - 48), (cx + ex + 32, cy - 52), hair, 9, cv2.LINE_AA)
        cv2.line(f, (cx, cy + 10), (cx - 8, cy + 60), (120, 150, 200), 5, cv2.LINE_AA)
        mo = int(4 + 34 * env[i])
        cv2.ellipse(f, (cx, cy + 110), (44, mo), 0, 0, 360, (60, 50, 140), -1, cv2.LINE_AA)
        enc.stdin.write(f.tobytes())
    enc.stdin.close(); enc.wait()


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "test_recording.mp4"
    lang = sys.argv[2] if len(sys.argv) > 2 else "en"
    wav = out.replace(".mp4", ".wav")
    y, sr = tts(wav, lang); presenter(out, wav, y, sr); os.remove(wav)
    print("wrote", out, f"{len(y)/sr:.1f}s")
