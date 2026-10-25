"""Low-level drawing helpers: fonts, text sprites, rounded masks, alpha blitting, easing."""
import functools, os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONT_DIRS = [os.path.join(os.path.dirname(__file__), "..", "assets", "fonts"),
             "/usr/share/fonts/truetype/google-fonts", "/usr/share/fonts/opentype/inter", "/usr/share/fonts/truetype/dejavu"]
FONT_FILES = {
    "Poppins": {300: "Poppins-Light.ttf", 400: "Poppins-Regular.ttf", 500: "Poppins-Medium.ttf", 700: "Poppins-Bold.ttf"},
    "Inter": {300: "Inter-Light.otf", 400: "Inter-Regular.otf", 500: "Inter-Medium.otf", 600: "Inter-SemiBold.otf", 700: "Inter-Bold.otf", 800: "Inter-ExtraBold.otf", 900: "Inter-Black.otf"},
    "DejaVu": {400: "DejaVuSans.ttf", 700: "DejaVuSans-Bold.ttf"},
    "Tajawal": {500: "Tajawal-Medium.ttf", 700: "Tajawal-Bold.ttf", 800: "Tajawal-ExtraBold.ttf"},  # Arabic + Latin (OFL)
}


def _find(name):
    for d in FONT_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p): return p
    return None


@functools.lru_cache(maxsize=256)
def font(family="Poppins", weight=700, size=48):
    fam = FONT_FILES.get(family, FONT_FILES["Poppins"])
    w = min(fam, key=lambda k: abs(k - weight))
    p = _find(fam[w]) or _find("DejaVuSans-Bold.ttf")
    return ImageFont.truetype(p, size)


def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def ease_out_cubic(x): x = min(1, max(0, x)); return 1 - (1 - x) ** 3
def ease_in_out(x): x = min(1, max(0, x)); return 4 * x ** 3 if x < .5 else 1 - (-2 * x + 2) ** 3 / 2
def ease_out_back(x, s=1.70158):
    x = min(1, max(0, x)); return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2
def lerp(a, b, t): return a + (b - a) * t
def clamp(x, a, b): return max(a, min(b, x))


import re as _re
_AR = _re.compile(r"[\u0600-\u06FF]")


def has_ar(text): return bool(_AR.search(text or ""))


def font_for(text, family="Poppins", weight=700, size=48):
    """Same request, but an Arabic-capable face when the text contains Arabic (shaped + RTL via libraqm)."""
    return font("Tajawal", weight, int(size * 1.08)) if has_ar(text) else font(family, weight, size)


def _weight_of(fnt):
    n = os.path.basename(getattr(fnt, "path", "") or "").lower()
    for k, w in (("black", 900), ("extrabold", 800), ("semibold", 600), ("bold", 700), ("medium", 500), ("light", 300)):
        if k in n: return w
    return 400


def fit_font(text, family, weight, size, max_w):
    f = font_for(text, family, weight, size)
    while size > 10 and f.getlength(text) > max_w:
        size = int(size * 0.92); f = font_for(text, family, weight, size)
    return f


def text_sprite(text, fnt, fill=(255, 255, 255), stroke=0, stroke_fill=(0, 0, 0), shadow=0, pad=8):
    """Render text to an RGBA numpy array (Arabic text is shaped right-to-left in an Arabic face)."""
    if has_ar(text) and "tajawal" not in (getattr(fnt, "path", "") or "").lower():
        fnt = font("Tajawal", _weight_of(fnt), int(fnt.size * 1.08))
    # Arabic lines are always laid out right-to-left (a line that starts with "coroscanner" must not flip to LTR);
    # Latin-only text is forced LTR so "≥300" is never reordered by bidi guessing
    dk = {"direction": "rtl"} if has_ar(text) else {"direction": "ltr"}
    l, t, r, b = fnt.getbbox(text, stroke_width=stroke, **dk)
    w, h = r - l + 2 * pad + shadow * 2, b - t + 2 * pad + shadow * 2
    im = Image.new("RGBA", (max(1, w), max(1, h)), (0, 0, 0, 0))
    if shadow:
        sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).text((pad - l + shadow, pad - t + shadow * 1.5), text, font=fnt, fill=(0, 0, 0, 170), stroke_width=stroke, stroke_fill=(0, 0, 0, 170), **dk)
        im = Image.alpha_composite(im, sh.filter(ImageFilter.GaussianBlur(shadow)))
    ImageDraw.Draw(im).text((pad - l, pad - t), text, font=fnt, fill=fill + (255,) if len(fill) == 3 else fill,
                            stroke_width=stroke, stroke_fill=stroke_fill, **dk)
    return np.array(im)


def rounded_mask(w, h, r):
    w, h, r = int(w), int(h), int(max(0, min(r, w / 2, h / 2)))
    m = np.zeros((h, w), np.uint8)
    if w <= 0 or h <= 0: return m
    if r == 0: m[:] = 255; return m
    cv2.rectangle(m, (r, 0), (w - 1 - r, h - 1), 255, -1); cv2.rectangle(m, (0, r), (w - 1, h - 1 - r), 255, -1)
    for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
        cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return m


def blit(dst, src, x, y, alpha=1.0, mask=None):
    """Alpha-composite src (RGB or RGBA uint8) onto dst (RGB uint8) at (x, y)."""
    x, y = int(round(x)), int(round(y))
    H, W = dst.shape[:2]; h, w = src.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0 or alpha <= 0: return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    if s.shape[2] == 4:
        a = s[..., 3:4].astype(np.float32) / 255.0
        rgb = s[..., :3]
    else:
        a = np.ones(s.shape[:2] + (1,), np.float32); rgb = s
    if mask is not None:
        a = a * (mask[y0 - y:y1 - y, x0 - x:x1 - x, None].astype(np.float32) / 255.0)
    a = a * alpha
    d = dst[y0:y1, x0:x1]
    if d.shape[2] == 4:  # compositing onto an RGBA sprite
        da = d[..., 3:4].astype(np.float32) / 255.0
        oa = a + da * (1 - a)
        d[..., :3] = ((d[..., :3].astype(np.float32) * da * (1 - a) + rgb.astype(np.float32) * a) / np.maximum(oa, 1e-6)).astype(np.uint8)
        d[..., 3:4] = (oa * 255).astype(np.uint8)
        return
    d[:] = (d.astype(np.float32) * (1 - a) + rgb.astype(np.float32) * a).astype(np.uint8)


def shadow_rect(dst, x, y, w, h, r, strength=0.45, blur=28, offset=10):
    pad = blur * 2
    m = np.zeros((int(h) + 2 * pad, int(w) + 2 * pad), np.uint8)
    m[pad:pad + int(h), pad:pad + int(w)] = rounded_mask(w, h, r)
    m = cv2.GaussianBlur(m, (0, 0), blur)
    rgba = np.zeros(m.shape + (4,), np.uint8); rgba[..., 3] = (m * strength).astype(np.uint8)
    blit(dst, rgba, x - pad, y - pad + offset)


def scale_rgba(img, s):
    if abs(s - 1) < 1e-3: return img
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_LINEAR)


def draw_arrow(img, p0, p1, color, thick=6, progress=1.0, head=26):
    x = lerp(p0[0], p1[0], progress); y = lerp(p0[1], p1[1], progress)
    cv2.line(img, tuple(map(int, p0)), (int(x), int(y)), color, thick, cv2.LINE_AA)
    if progress > 0.85:
        ang = np.arctan2(y - p0[1], x - p0[0])
        for da in (2.6, -2.6):
            q = (int(x + head * np.cos(ang + da)), int(y + head * np.sin(ang + da)))
            cv2.line(img, (int(x), int(y)), q, color, thick, cv2.LINE_AA)
