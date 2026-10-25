"""Second-generation medical animations (original procedural art — no third-party rights).

  ct/calcium   — realistic non-contrast cardiac CT slice: reconstruction sweep, zoom to the heart,
                 calcified plaque lights up per artery (LAD/LCx/RCA), then the CAC-DRS scale
  ct/scanner   — side view of a CT scanner: patient slides in, rotating beam, ECG-gated acquisition,
                 axial slices fly out of the gantry
  heart/coronaries — beating heart with its coronary tree; blood pulses along the arteries, a plaque is pointed out
  photo/<id>   — a doctor-supplied photo with a slow Ken Burns move over a blurred fill

All are pure functions of (W, H, t, D) and adapt to the panel shape (split, side card, full screen, PiP).
"""
import functools, math, os
import numpy as np, cv2
from .draw import font, fit_font, text_sprite, blit, rounded_mask, ease_out_cubic, ease_in_out, ease_out_back, clamp, lerp
from . import graphics as G

tr = G.tr
for _lang, _d in {"ar": {"Coronary CT scan": "تصوير الشرايين بالسكانر", "Non-invasive": "بدون تدخّل جراحي", "ECG-gated": "متزامن مع تخطيط القلب",
                         "Coronary arteries": "شرايين القلب التاجية", "Plaque": "لويحة", "Calcified plaque": "ترسّبات كلسية",
                         "Agatston score": "مؤشر أغاتستون", "LAD": "LAD", "LCx": "LCx", "RCA": "RCA", "Right coronary": "الشريان الأيمن",
                         "Left anterior descending": "الشريان الأمامي النازل", "Circumflex": "الشريان المنعطف"},
                  "fr": {"Coronary CT scan": "Coroscanner", "Non-invasive": "Non invasif", "ECG-gated": "Synchronisé à l'ECG",
                         "Coronary arteries": "Artères coronaires", "Plaque": "Plaque", "Calcified plaque": "Plaque calcifiée",
                         "Agatston score": "Score d'Agatston", "Right coronary": "Coronaire droite",
                         "Left anterior descending": "Interventriculaire antérieure", "Circumflex": "Circonflexe"}}.items():
    G.LABELS[_lang].update(_d)

# ------------------------------------------------------------------ shared bits
def _s(W): return max(W / 1080, 0.8 * G.UI_SCALE) if W >= 400 else max(W / 700, 0.45)


def _chip(img, text, x, y, accent, size, rtl=False):
    return G._chip(img, text, x, y, accent, size, rtl)


def _glow(img, mask, color, sigma, gain=1.0):
    ys, xs = np.nonzero(mask > 0.01)
    if not len(ys): return
    pad = int(3 * sigma) + 2; H, W = mask.shape[:2]
    y0, y1, x0, x1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1), max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
    g = cv2.GaussianBlur(mask[y0:y1, x0:x1].astype(np.float32), (0, 0), sigma)
    roi = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = np.clip(roi.astype(np.float32) + g[..., None] * np.array(color, np.float32) * gain, 0, 255).astype(np.uint8)


def _layout(W, H):
    """'land' (split/labels panels), 'port' (full screen), 'card' (square-ish side card)."""
    a = W / H
    return "land" if a > 1.05 else "port" if a < 0.75 else "card"


# ------------------------------------------------------------------ CT slice
# radiological convention: image right = patient left; y grows posteriorly. Units: fraction of half-size.
VESSELS = {"RCA": (-0.30, -0.02), "LAD": (0.10, -0.33), "LCx": (0.47, 0.06), "LM": (-0.02, -0.25)}
VCOL = {"LM": (255, 214, 74), "LAD": (255, 214, 74), "LCx": (90, 220, 140), "RCA": (255, 120, 160)}
# calcified lesions: (vessel, dx, dy, size) around the vessel centre; appearance order
LESIONS = [("LAD", 0.0, 0.0, 0.040), ("LAD", 0.05, 0.035, 0.030), ("RCA", 0.0, 0.0, 0.042), ("LCx", 0.0, 0.0, 0.034),
           ("LAD", -0.045, -0.03, 0.024), ("RCA", 0.02, 0.05, 0.028), ("LCx", 0.03, -0.04, 0.022)]


@functools.lru_cache(maxsize=6)
def _slice(n):
    """n x n float32 greyscale CT slice (mediastinal window) at heart level + the heart centre (px)."""
    rng = np.random.default_rng(11)
    im = np.zeros((n, n), np.float32)
    c = n / 2; u = n / 2 * 0.98
    P = lambda x, y: (int(c + x * u), int(c + y * u))
    A = lambda a, b: (max(1, int(a * u)), max(1, int(b * u)))
    def ell(x, y, a, b, v, ang=0): cv2.ellipse(im, P(x, y), A(a, b), ang, 0, 360, float(v), -1, cv2.LINE_AA)
    ell(0, 0.02, 0.97, 0.70, 128)                                     # skin
    ell(0, 0.02, 0.93, 0.66, 40)                                      # subcutaneous fat
    ell(0, 0.03, 0.86, 0.60, 132)                                     # chest-wall muscle
    for sx in (-1, 1):                                                # pectorals (anterior bulk)
        ell(sx * 0.38, -0.47, 0.34, 0.12, 134, sx * 12)
    ell(0, 0.05, 0.80, 0.54, 44)                                      # inner fat layer
    for k in range(16):                                               # ribs (cortical rings)
        a = math.radians(-160 + k * (320 / 15))
        if abs(math.degrees(a) + 90) < 22: continue                   # sternum gap
        x, y = 0.80 * math.cos(a), 0.05 + 0.55 * math.sin(a)
        ell(x, y, 0.045, 0.028, 235, math.degrees(a) + 90); ell(x, y, 0.025, 0.012, 150, math.degrees(a) + 90)
    ell(0, -0.52, 0.09, 0.045, 225); ell(0, -0.52, 0.06, 0.025, 160)  # sternum
    # lungs (black with faint vessels)
    lung = np.zeros_like(im)
    cv2.ellipse(lung, P(-0.43, 0.06), A(0.33, 0.45), 8, 0, 360, 1.0, -1, cv2.LINE_AA)
    cv2.ellipse(lung, P(0.47, 0.13), A(0.30, 0.41), -10, 0, 360, 1.0, -1, cv2.LINE_AA)
    im = im * (1 - lung) + lung * 6
    for _ in range(int(140)):
        x, y = rng.uniform(-0.75, 0.75), rng.uniform(-0.35, 0.5)
        if lung[P(x, y)[1], P(x, y)[0]] > 0.5:
            cv2.circle(im, P(x, y), max(1, int(rng.uniform(0.004, 0.012) * u)), float(rng.uniform(55, 95)), -1, cv2.LINE_AA)
    # mediastinum + heart (unenhanced: blood ≈ myocardium)
    ell(0.06, 0.02, 0.43, 0.36, 42, -28)                              # pericardial/epicardial fat envelope
    ell(0.08, 0.03, 0.39, 0.31, 126, -28)                             # heart
    cv2.ellipse(im, P(0.08, 0.03), A(0.395, 0.315), -28, 0, 360, 138, max(1, int(0.006 * u)), cv2.LINE_AA)  # pericardium
    ell(-0.16, -0.28, 0.10, 0.09, 128)                                # ascending aorta / RVOT region
    ell(0.10, -0.33, 0.07, 0.05, 40)                                  # fat in anterior IV groove
    for v, (x, y) in VESSELS.items():
        if v == "LM": continue
        ell(x, y, 0.045, 0.040, 40)                                   # fat around each artery
        ell(x, y, 0.022, 0.020, 122)                                  # artery wall/lumen
    ell(0.16, 0.42, 0.085, 0.085, 126)                                # descending aorta
    ell(0.17, 0.39, 0.012, 0.01, 230)                                 # small aortic calcification
    ell(-0.04, 0.36, 0.035, 0.03, 116); ell(-0.04, 0.36, 0.012, 0.01, 0)  # oesophagus (+air)
    ell(0, 0.56, 0.12, 0.11, 225); ell(0, 0.56, 0.095, 0.085, 165)    # vertebral body
    ell(0, 0.70, 0.045, 0.04, 70)                                     # spinal canal
    for sx in (-1, 1): ell(sx * 0.08, 0.72, 0.05, 0.03, 210, sx * 35)  # posterior elements
    for sx in (-1, 1): ell(sx * 0.17, 0.64, 0.10, 0.08, 132)          # paraspinal muscles
    im = cv2.GaussianBlur(im, (0, 0), max(0.6, n / 900))
    noise = cv2.GaussianBlur(rng.normal(0, 1, (n, n)).astype(np.float32), (0, 0), max(0.7, n / 700))
    im += noise / (noise.std() + 1e-6) * 7.0
    outside = np.zeros_like(im); cv2.ellipse(outside, P(0, 0.02), A(0.97, 0.70), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    im = im * outside + (1 - outside) * rng.normal(4, 2, im.shape).astype(np.float32)
    # CT table under the patient
    cv2.ellipse(im, P(0, 0.86), A(0.70, 0.06), 0, 180, 360, 90, max(1, int(0.012 * u)), cv2.LINE_AA)
    return np.clip(im, 0, 255), P


@functools.lru_cache(maxsize=48)
def _slice_rgb(n, lit, scored, u_glow, accent):
    base, P = _slice(n)
    g = base.copy()
    masks = {}
    for k, (v, dx, dy, sz) in enumerate(LESIONS):
        a = clamp(lit - k, 0, 1)
        if a <= 0: continue
        x0, y0 = VESSELS[v]
        m = np.zeros(g.shape, np.float32)
        for j, (ox, oy, f) in enumerate(((0, 0, 1.0), (sz * 0.6, sz * 0.3, 0.7), (-sz * 0.5, sz * 0.4, 0.6))):
            cv2.circle(m, P(x0 + dx + ox, y0 + dy + oy), max(1, int(sz * f * n / 2 * 0.98 * (0.5 + 0.5 * a))), 1.0, -1, cv2.LINE_AA)
        g = np.maximum(g, m * (175 + 70 * a))
        masks.setdefault(v, np.zeros(g.shape, np.float32)); masks[v] = np.maximum(masks[v], m)
    rgb = np.repeat(g[..., None], 3, 2).astype(np.uint8)
    if scored > 0:  # scoring software overlay: lesions >= 130 HU tinted by artery
        for v, m in masks.items():
            col = np.array(VCOL[v], np.float32)
            a = (m * scored)[..., None]
            rgb = (rgb * (1 - a * 0.85) + col * a * 0.85).astype(np.uint8)
            _glow(rgb, m, VCOL[v], max(2, n / 90), 0.9 * scored * (0.7 + 0.3 * u_glow))
    return rgb, P


def _scale_bar(img, x0, y0, w, s, u, lang, accent, vertical=False):
    """CAC-DRS categories light up left→right (RTL for Arabic) with the Agatston ranges."""
    bands = [("0", "None", (60, 200, 120)), ("1–99", "Mild", (240, 200, 60)), ("100–299", "Moderate", (250, 140, 50)), ("≥300", "Severe", (235, 70, 70))]
    head_t = tr("CAC-DRS · Agatston score", lang)
    head = text_sprite(head_t, fit_font(head_t, "Poppins", 700, int(34 * s), w), (215, 228, 245), pad=0)
    blit(img, head, x0 + (w - head.shape[1]) / 2 if not vertical else x0, y0, ease_out_cubic(u / 0.12))
    y = y0 + head.shape[0] + 16 * s
    marker = clamp((u - 0.15) / 0.65, 0, 1) * 4  # sweeps through the four categories
    if vertical:
        bh = int(24 * s); gap = int(18 * s)
        for k, (num, name, col) in enumerate(bands):
            a = ease_out_cubic((u - 0.1 - 0.1 * k) / 0.15)
            if a <= 0: continue
            on = clamp(marker - k, 0, 1)
            row_h = int(max(bh * 3.2, 92 * s))
            card = np.zeros((row_h, int(w), 4), np.uint8); card[..., :3] = (16, 30, 56); card[..., 3] = (rounded_mask(int(w), row_h, int(16 * s)) * 0.9).astype(np.uint8)
            cv2.rectangle(card, (0, 0), (int(10 * s), row_h), col + (255,), -1)
            if on > 0:
                m_ = (rounded_mask(int(w), row_h, int(16 * s)) / 255.0 * 0.55 * on)[..., None]
                card[..., :3] = (card[..., :3] * (1 - m_) + np.array(col) * 0.75 * m_).astype(np.uint8)
            nt = text_sprite(num, fit_font(num, "Poppins", 800, int(48 * s), w * 0.45), (255, 255, 255), pad=0)
            lt = text_sprite(tr(name, lang), fit_font(tr(name, lang), "Poppins", 700, int(34 * s), w * 0.45), (255, 255, 255) if on > 0.5 else col, pad=0)
            if lang == "ar":
                blit(card, nt, int(w - 26 * s - nt.shape[1]), (row_h - nt.shape[0]) / 2); blit(card, lt, int(26 * s), (row_h - lt.shape[0]) / 2)
            else:
                blit(card, nt, int(30 * s), (row_h - nt.shape[0]) / 2); blit(card, lt, int(w - 26 * s - lt.shape[1]), (row_h - lt.shape[0]) / 2)
            blit(img, card, x0 + (1 - a) * 40 * s, y, a)
            y += row_h + gap
        return y
    seg = w / 4; bh = max(10, int(26 * s))
    order = list(range(4))[::-1] if lang == "ar" else list(range(4))
    for slot, k in enumerate(order):
        num, name, col = bands[k]
        a = ease_out_cubic((u - 0.1 - 0.08 * k) / 0.15)
        if a <= 0: continue
        on = clamp(marker - k, 0, 1)
        xa, xb = int(x0 + slot * seg + 4), int(x0 + (slot + 1) * seg - 4)
        bar = np.zeros((bh, xb - xa, 4), np.uint8); bar[..., :3] = col; bar[..., 3] = (rounded_mask(xb - xa, bh, bh // 2) * (0.45 + 0.55 * on)).astype(np.uint8)
        blit(img, bar, xa, y, a)
        nt = text_sprite(num, fit_font(num, "Poppins", 800, int(40 * s), seg - 10), (255, 255, 255), pad=0)
        lt = text_sprite(tr(name, lang), fit_font(tr(name, lang), "Poppins", 600, int(30 * s), seg - 10), col, pad=0)
        mid = (xa + xb) / 2
        blit(img, nt, mid - nt.shape[1] / 2, y + bh + 10 * s, a); blit(img, lt, mid - lt.shape[1] / 2, y + bh + 14 * s + nt.shape[0], a)
    return y + bh + 20 * s + 80 * s


def ct_calcium(W, H, t, D, mode="calcium", accent=(46, 160, 255), title=None, lang="en", **_):
    u = clamp(t / max(D, 0.1), 0, 1); s = _s(W); lay = _layout(W, H)
    img = G._bg(W, H).astype(np.uint8).copy()
    chip_h = int(36 * s) * 2
    top = int(max(0.09 * H, 175 * W / 1080)) if lay == "port" else int(26 * s)  # full screen: below the logo
    _, ch = _chip(img, tr(title or "Coronary calcium score", lang), int(34 * s), top, accent, max(14, int(34 * s)), rtl=lang == "ar")
    # where the slice and the score scale go
    if lay == "land":
        n = int(min(H - top - ch - 40 * s, W * 0.58)); sx = int(W * 0.04) if lang != "ar" else int(W * 0.96 - n)
        sy = int(top + ch + 18 * s + max(0, (H - top - ch - 40 * s - n) / 2))
        bx, bw, by, vertical = (int(W * 0.04 + n + 30 * s), int(W * 0.92 - n - 30 * s), int(sy + max(6 * s, (n - 520 * s) / 2)), True)
        if lang == "ar": bx = int(W * 0.04)
    else:
        if lay == "port":  # full screen (PiP bubble sits top-left/right at 8–28% height): slice below it
            n = int(min(W * 0.84, H * 0.37)); sx, sy = (W - n) // 2, int(H * 0.30)
        else:
            n = int(min(W * 0.86, (H - top - ch) * 0.62)); sx, sy = (W - n) // 2, int(top + ch + 14 * s)
        bx, bw, by, vertical = int(W * 0.06), int(W * 0.88), int(sy + n + 24 * s), False
    n = max(64, n)
    # phases: reconstruction sweep -> zoom to heart -> lesions light up -> scoring overlay -> scale
    sweep = ease_in_out(u / 0.16)
    lit = len(LESIONS) * ease_in_out((u - 0.2) / 0.3)
    scored = ease_out_cubic((u - 0.42) / 0.12)
    zf = clamp(ease_in_out((u - 0.12) / 0.25) - 0.25 * ease_in_out((u - 0.78) / 0.2), 0, 1)
    zoom = 1 + 0.5 * zf
    N = int(n * 1.5) // 2 * 2
    rgb, P = _slice_rgb(N, round(lit * 6) / 6, round(scored * 10) / 10, round(0.5 + 0.5 * math.sin(t * 4), 1), accent)
    hx, hy = P(0.08, 0.0)
    vw = N / zoom  # visible window in slice pixels, panning from the whole slice to the heart
    cxw = lerp(N / 2, hx, zf); cyw = lerp(N / 2, hy, zf)
    x0w, y0w = clamp(cxw - vw / 2, 0, N - vw), clamp(cyw - vw / 2, 0, N - vw)
    view = cv2.resize(rgb[int(y0w):int(y0w + vw), int(x0w):int(x0w + vw)], (n, n), interpolation=cv2.INTER_AREA)
    k_ = n / vw
    cx0, cy0 = 0, 0
    P_view = lambda x, y: ((P(x, y)[0] - x0w) * k_, (P(x, y)[1] - y0w) * k_)
    if sweep < 1:  # reconstruction: image builds top->bottom behind a bright scan line
        yl = int(n * sweep); view[yl:] = (view[yl:] * 0.0).astype(np.uint8)
        cv2.line(view, (0, min(n - 1, yl)), (n, min(n - 1, yl)), accent, max(2, int(3 * s)), cv2.LINE_AA)
    frame = np.zeros((n + 8, n + 8, 4), np.uint8)
    frame[4:-4, 4:-4, :3] = view; frame[4:-4, 4:-4, 3] = rounded_mask(n, n, int(28 * s))
    G.shadow_rect(img, sx, sy, n, n, int(28 * s), 0.5, int(26 * s)) if hasattr(G, "shadow_rect") else None
    blit(img, frame, sx - 4, sy - 4, ease_out_cubic(u / 0.06))
    # corner HUD (window/level like a workstation)
    hud = text_sprite("W 400  L 40", font("Inter", 600, max(10, int(22 * s))), (150, 200, 240), pad=0)
    blit(img, hud, sx + n - hud.shape[1] - 16 * s, sy + 12 * s, 0.8 * ease_out_cubic((u - 0.1) / 0.1))
    # artery labels with leader lines
    if scored > 0.05:
        f = font("Poppins", 700, max(12, int(32 * s)))
        for v in ("LAD", "LCx", "RCA"):
            vx, vy = P_view(*VESSELS[v]); px, py = sx + vx, sy + vy
            if not (sx + 10 < px < sx + n - 10 and sy + 10 < py < sy + n - 10): continue
            side = -1 if v == "RCA" else 1
            lx, ly = px + side * 70 * s, py - (60 if v == "LAD" else 34) * s
            cv2.line(img, (int(px), int(py)), (int(lx), int(ly)), VCOL[v], max(2, int(3 * s)), cv2.LINE_AA)
            tag = text_sprite(v, f, VCOL[v], shadow=3, pad=2)
            tx = lx - tag.shape[1] - 4 * s if side < 0 else lx + 4 * s
            blit(img, tag, clamp(tx, sx + 4, sx + n - tag.shape[1] - 4), ly - tag.shape[0] / 2, scored)
        cap = tr("Calcified plaque", lang) + ("" if lang == "ar" else " ≥ 130 HU")
        ct_ = text_sprite(cap, fit_font(cap, "Poppins", 600, int(28 * s), n - 30 * s), (255, 255, 255), shadow=3, pad=0)
        blit(img, ct_, sx + (n - ct_.shape[1]) / 2, sy + n - ct_.shape[0] - 18 * s, scored)
    _scale_bar(img, bx, by, bw, s, clamp((u - 0.5) / 0.5, 0, 1), lang, accent, vertical=vertical)
    return img


# ------------------------------------------------------------------ CT scanner (side view)
@functools.lru_cache(maxsize=8)
def _mini_slice(n):
    rgb, _ = _slice_rgb(n, len(LESIONS), 0.0, 0.0, (46, 160, 255))
    out = np.zeros((n, n, 4), np.uint8); out[..., :3] = rgb; out[..., 3] = rounded_mask(n, n, max(3, n // 8))
    return out


def ct_scanner(W, H, t, D, mode="scanner", accent=(46, 160, 255), title=None, lang="en", **_):
    u = clamp(t / max(D, 0.1), 0, 1); s = _s(W); lay = _layout(W, H)
    img = G._bg(W, H).astype(np.uint8).copy()
    top = int(max(0.09 * H, 175 * W / 1080)) if lay == "port" else int(26 * s)  # full screen: below the logo
    _, ch = _chip(img, tr(title or "Coronary CT scan", lang), int(34 * s), top, accent, max(14, int(34 * s)), rtl=lang == "ar")
    # scene box
    zx0, zx1 = W * 0.04, W * 0.96
    zy0 = top + ch + 30 * s; zy1 = H * (0.80 if lay == "port" else 0.80)
    if lay == "port": zy0, zy1 = H * 0.30, H * 0.66
    zw, zh = zx1 - zx0, zy1 - zy0
    sc = min(zw / 1000, zh / 560)
    ox, oy = zx0 + (zw - 1000 * sc) / 2, zy0 + (zh - 560 * sc) / 2
    X = lambda x: int(ox + x * sc); Y = lambda y: int(oy + y * sc); L = lambda v: max(1, int(v * sc))
    gx, gcy = 520, 300  # gantry front face centre (3/4 view: the ring faces the viewer, patient enters from the left)
    RX, RY, BX, BY = 120, 235, 62, 118
    yy, xx = np.mgrid[0:H, 0:W]
    def emask(cx, cy, rx, ry):
        m = np.zeros((H, W), np.float32); cv2.ellipse(m, (X(cx), Y(cy)), (L(rx), L(ry)), 0, 0, 360, 1.0, -1, cv2.LINE_AA); return m
    housing = np.maximum(emask(gx + 60, gcy, RX, RY), emask(gx, gcy, RX, RY))
    cv2.rectangle(housing, (X(gx), Y(gcy - RY)), (X(gx + 60), Y(gcy + RY)), 1.0, -1)
    bore = emask(gx, gcy, BX, BY)
    # floor shadow + table pedestal (behind)
    cv2.ellipse(img, (X(560), Y(545)), (L(420), L(16)), 0, 0, 360, (8, 16, 32), -1, cv2.LINE_AA)
    cv2.rectangle(img, (X(150), Y(400)), (X(310), Y(540)), (120, 135, 160), -1, cv2.LINE_AA)
    cv2.rectangle(img, (X(150), Y(400)), (X(310), Y(415)), (170, 182, 200), -1, cv2.LINE_AA)
    # gantry body: side wall darker, front face light with a soft vertical gradient
    side = (np.array([150, 162, 182], np.float32))
    front = np.linspace(1.0, 0.86, H)[:, None, None] * np.array([232, 238, 246], np.float32)
    hm = housing[..., None]; fm = emask(gx, gcy, RX, RY)[..., None]
    img[:] = (img * (1 - hm) + side * hm).astype(np.uint8)
    img[:] = (img * (1 - fm) + front * fm).astype(np.uint8)
    cv2.ellipse(img, (X(gx), Y(gcy)), (L(RX), L(RY)), 0, 0, 360, (175, 188, 206), L(3), cv2.LINE_AA)
    cv2.ellipse(img, (X(gx), Y(gcy)), (L(BX + 22), L(BY + 40)), 0, 0, 360, (205, 214, 228), L(10), cv2.LINE_AA)  # bore trim
    bm = bore[..., None]
    img[:] = (img * (1 - bm) + np.array([20, 30, 48], np.float32) * bm).astype(np.uint8)
    # patient + table on their own layer; visible in front of the gantry (left of it) and through the bore
    slide = ease_in_out((u - 0.05) / 0.35)
    px = -200 + 330 * slide
    lay_ = np.zeros_like(img); la = np.zeros((H, W), np.float32)
    def put(fn, col):
        m = np.zeros((H, W), np.float32); fn(m); lay_[m > 0] = col; np.maximum(la, m, out=la)
    put(lambda m: cv2.rectangle(m, (X(40 + px), Y(355)), (X(1000), Y(392)), 1.0, -1), (205, 215, 230))
    put(lambda m: cv2.ellipse(m, (X(650 + px), Y(334)), (L(170), L(26)), 0, 0, 360, 1.0, -1, cv2.LINE_AA), (90, 145, 205))
    put(lambda m: cv2.ellipse(m, (X(390 + px), Y(325)), (L(175), L(38)), 0, 0, 360, 1.0, -1, cv2.LINE_AA), (110, 170, 230))
    put(lambda m: cv2.ellipse(m, (X(180 + px), Y(318)), (L(46), L(40)), 0, 0, 360, 1.0, -1, cv2.LINE_AA), (233, 196, 170))
    vis = np.where(xx < X(gx), 1.0, 0.0).astype(np.float32)
    vis = np.maximum(vis, bore * 0.75)  # inside the bore: seen through the ring, slightly shaded
    vis = np.maximum(vis, 1 - housing)
    a3 = (la * vis)[..., None]
    img[:] = (img * (1 - a3) + lay_ * a3).astype(np.uint8)
    # rotating X-ray source on the ring + fan beam through the chest (bursts in the ECG window)
    scanning = clamp((u - 0.38) / 0.06, 0, 1) * (1 - clamp((u - 0.86) / 0.06, 0, 1))
    beat = (t * 1.15) % 1.0
    window = 0.55 < beat < 0.85
    if scanning > 0:
        ang = t * 7.0
        sxp, syp = gx + (BX + 30) * math.cos(ang), gcy + (BY + 50) * math.sin(ang)
        ox_, oy_ = gx - (BX + 30) * math.cos(ang), gcy - (BY + 50) * math.sin(ang)
        beam = np.zeros((H, W), np.float32)
        perp = np.array([-(oy_ - syp), ox_ - sxp]); perp = perp / (np.linalg.norm(perp) + 1e-6) * 46
        tri = np.array([[X(sxp), Y(syp)], [X(ox_ + perp[0]), Y(oy_ + perp[1])], [X(ox_ - perp[0]), Y(oy_ - perp[1])]], np.int32)
        cv2.fillPoly(beam, [tri], 1.0, cv2.LINE_AA)
        beam *= bore
        _glow(img, beam, accent, max(2, 8 * sc), (1.3 if window else 0.45) * scanning)
        cv2.circle(img, (X(sxp), Y(syp)), L(12), (255, 255, 255), -1, cv2.LINE_AA)
        ring = np.zeros((H, W), np.float32)
        cv2.ellipse(ring, (X(gx), Y(gcy)), (L(BX + 30), L(BY + 50)), 0, 0, 360, 1.0, L(4), cv2.LINE_AA)
        _glow(img, ring, accent, max(2, 6 * sc), 0.5 * scanning)
    # axial slices fly out of the gantry, top-right
    nshow = int(clamp((u - 0.45) / 0.35, 0, 1) * 5)
    ms = max(24, int(150 * sc))
    for k in range(nshow):
        age = clamp((u - 0.45 - k * 0.07) / 0.12, 0, 1)
        if age <= 0: continue
        ex, ey = X(gx + 230 + k * 44), Y(40 + k * 26)
        sx_, sy_ = lerp(X(gx), ex, ease_out_cubic(age)), lerp(Y(gcy), ey, ease_out_cubic(age))
        sl = _mini_slice(ms)
        G.shadow_rect(img, sx_, sy_, ms, ms, ms // 8, 0.45, max(4, ms // 10)) if hasattr(G, "shadow_rect") else None
        blit(img, sl, sx_, sy_, age)
        cv2.rectangle(img, (int(sx_), int(sy_)), (int(sx_ + ms), int(sy_ + ms)), accent, max(1, int(2 * s)), cv2.LINE_AA)
    # ECG strip with gating windows
    ey0 = int(min(H - 70 * s, zy1 + 30 * s)) if lay != "port" else int(H * 0.72)
    ew = int(W * 0.92); ex0 = int(W * 0.04); eh = int(70 * s)
    xs = np.arange(ew)
    beats = (t * 1.15) - (ew - xs) / ew * 3.0
    ys = ey0 + eh * 0.55 - eh * 0.5 * G._ecg_wave(beats)
    ph = beats % 1.0
    win = (ph > 0.55) & (ph < 0.85)
    band = np.zeros((H, W), np.float32)
    for x in np.where(win)[0][::2]: band[ey0: ey0 + eh, ex0 + x] = 1.0
    img[:] = np.clip(img + band[..., None] * np.array(accent, np.float32) * 0.22 * scanning, 0, 255).astype(np.uint8)
    pts = np.stack([ex0 + xs, ys], 1).astype(np.int32)
    cv2.polylines(img, [pts], False, (40, 255, 160), max(2, int(3 * s)), cv2.LINE_AA)
    lab = text_sprite(tr("ECG-gated", lang), font("Poppins", 600, max(11, int(26 * s))), (150, 230, 200), pad=0)
    blit(img, lab, ex0 + (ew - lab.shape[1] if lang == "ar" else 0), ey0 - lab.shape[0] - 6 * s, scanning)
    if lay != "land" or H > 600 * s:
        ni = tr("Non-invasive", lang)
        nt = text_sprite(ni, font("Poppins", 600, max(11, int(28 * s))), (215, 228, 245), pad=0)
        blit(img, nt, (W - nt.shape[1] - 34 * s) if lang != "ar" else 34 * s, top + 10 * s, ease_out_cubic((u - 0.1) / 0.1))
    return img


# ------------------------------------------------------------------ heart with coronary tree
def _bez(p0, p1, p2, p3, n=40):
    tt = np.linspace(0, 1, n)[:, None]
    return ((1 - tt) ** 3) * p0 + 3 * ((1 - tt) ** 2) * tt * p1 + 3 * (1 - tt) * tt ** 2 * p2 + tt ** 3 * p3


# artery centre-lines in heart units (x right = patient left, y down); (name, control points, width)
ARTERIES = [("RCA", [(-0.22, -0.40), (-0.62, -0.28), (-0.70, 0.18), (-0.25, 0.70)], 0.040),
            ("LM", [(-0.05, -0.44), (0.02, -0.46), (0.08, -0.44), (0.12, -0.41)], 0.044),
            ("LAD", [(0.12, -0.41), (0.08, -0.05), (0.22, 0.45), (0.48, 0.86)], 0.036),
            ("D1", [(0.10, -0.18), (0.28, -0.08), (0.40, 0.10), (0.46, 0.28)], 0.022),
            ("LCx", [(0.12, -0.41), (0.40, -0.42), (0.62, -0.22), (0.72, 0.15)], 0.032),
            ("OM", [(0.58, -0.28), (0.66, -0.02), (0.62, 0.25), (0.58, 0.40)], 0.020)]


@functools.lru_cache(maxsize=8)
def _heart_base(n):
    """RGBA heart (anterior view) of size n x n with shading; arteries drawn separately."""
    c = n / 2; u = n * 0.40
    P = lambda x, y: (c + x * u, c + 0.12 * n + y * u)
    im = np.zeros((n, n, 3), np.float32); a = np.zeros((n, n), np.float32)
    # silhouette (anterior view, apex toward the patient's left = image right)
    pts = [(-0.42, -0.52), (-0.72, -0.30), (-0.80, 0.10), (-0.55, 0.50), (-0.05, 0.80), (0.48, 0.92), (0.72, 0.62),
           (0.80, 0.15), (0.66, -0.30), (0.35, -0.55), (-0.02, -0.58)]
    curve = []
    for k in range(len(pts)):
        p0, p1, p2, p3 = (np.array(pts[(k + j - 1) % len(pts)]) for j in range(4))
        for tt in np.linspace(0, 1, 16, endpoint=False):  # Catmull-Rom
            t2, t3 = tt * tt, tt * tt * tt
            q = 0.5 * ((2 * p1) + (-p0 + p2) * tt + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            curve.append(P(*q))
    cv2.fillPoly(a, [np.array(curve, np.int32)], 1.0, cv2.LINE_AA)
    # great vessels: ascending aorta curving into the arch, pulmonary trunk to its left (image right)
    arch = _bez(np.array(P(-0.18, -0.45)), np.array(P(-0.26, -0.95)), np.array(P(0.20, -1.10)), np.array(P(0.42, -0.86)), 50)
    cv2.polylines(a, [arch.astype(np.int32)], False, 1.0, max(2, int(0.24 * u)), cv2.LINE_AA)
    for bx_, by_ in ((-0.06, -1.00), (0.08, -1.04), (0.22, -1.00)):  # arch branches
        cv2.line(a, tuple(int(v) for v in P(bx_, by_)), tuple(int(v) for v in P(bx_ - 0.02, by_ - 0.18)), 1.0, max(2, int(0.07 * u)), cv2.LINE_AA)
    pa = _bez(np.array(P(0.15, -0.45)), np.array(P(0.20, -0.75)), np.array(P(0.35, -0.80)), np.array(P(0.50, -0.70)), 40)
    cv2.polylines(a, [pa.astype(np.int32)], False, 1.0, max(2, int(0.20 * u)), cv2.LINE_AA)
    vessels = np.zeros((n, n), np.float32)
    cv2.polylines(vessels, [arch.astype(np.int32)], False, 1.0, max(2, int(0.24 * u)), cv2.LINE_AA)
    pav = np.zeros((n, n), np.float32); cv2.polylines(pav, [pa.astype(np.int32)], False, 1.0, max(2, int(0.20 * u)), cv2.LINE_AA)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    lx, ly = c - 0.35 * u, c - 0.45 * u
    shade = np.clip(1.15 - np.sqrt(((xx - lx) / (1.6 * u)) ** 2 + ((yy - ly) / (1.6 * u)) ** 2), 0.35, 1.1)
    base = np.array([168, 46, 52], np.float32)
    im = base * shade[..., None]
    im = im * (1 - vessels[..., None]) + np.array([214, 90, 96], np.float32) * shade[..., None] * vessels[..., None]   # aorta: lighter red
    im = im * (1 - pav[..., None]) + np.array([96, 110, 190], np.float32) * shade[..., None] * pav[..., None]          # pulmonary trunk: bluish
    fat = np.exp(-(((xx - c - 0.0 * u) / (0.2 * u)) ** 2 + ((yy - c + 0.35 * u) / (0.5 * u)) ** 2)) * 0.25
    im = im * (1 - fat[..., None]) + np.array([230, 190, 120], np.float32) * fat[..., None]  # epicardial fat along the grooves
    spec = np.exp(-(((xx - c + 0.38 * u) / (0.22 * u)) ** 2 + ((yy - c + 0.28 * u) / (0.14 * u)) ** 2)) * 70
    im = np.clip(im + spec[..., None], 0, 255)
    a = cv2.GaussianBlur(a, (0, 0), max(0.8, n / 500))
    out = np.zeros((n, n, 4), np.uint8); out[..., :3] = im.astype(np.uint8); out[..., 3] = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return out


def heart(W, H, t, D, mode="coronaries", accent=(46, 160, 255), title=None, lang="en", **_):
    u = clamp(t / max(D, 0.1), 0, 1); s = _s(W); lay = _layout(W, H)
    img = G._bg(W, H).astype(np.uint8).copy()
    top = int(max(0.09 * H, 175 * W / 1080)) if lay == "port" else int(26 * s)  # full screen: below the logo
    _, ch = _chip(img, tr(title or "Coronary arteries", lang), int(34 * s), top, accent, max(14, int(34 * s)), rtl=lang == "ar")
    avail_h = H - top - ch - 40 * s
    n = int(min(W * (0.62 if lay == "land" else 0.86), avail_h * (0.95 if lay != "port" else 0.6)))
    n = max(64, n)
    cx = W * (0.36 if lay == "land" and lang != "ar" else 0.64 if lay == "land" else 0.5); cy = top + ch + 20 * s + n / 2
    if lay == "port": cy = H * 0.45
    bpm = 66; ph = (t * bpm / 60) % 1
    beat = 1 + 0.035 * math.exp(-((ph - 0.12) / 0.07) ** 2) - 0.015 * math.exp(-((ph - 0.45) / 0.12) ** 2)
    hb = _heart_base(n)
    k = beat * ease_out_back(clamp(u / 0.12, 0, 1)) if u < 0.12 else beat
    spr = cv2.resize(hb, (max(8, int(n * k)), max(8, int(n * k))), interpolation=cv2.INTER_LINEAR)
    hx, hy = cx - spr.shape[1] / 2, cy - spr.shape[0] / 2
    glow = np.zeros((H, W), np.float32)
    cv2.circle(glow, (int(cx), int(cy)), int(n * 0.38), 1.0, -1)
    _glow(img, glow, (200, 40, 60), n * 0.12, 0.35)
    blit(img, spr, hx, hy)
    uu = n * k * 0.40
    Pt = lambda x, y: (cx + x * uu, cy + 0.12 * n * k + y * uu)
    reveal = ease_in_out((u - 0.08) / 0.3)
    paths = {}
    for name, cps, wdt in ARTERIES:
        pts = _bez(*[np.array(Pt(*p)) for p in cps], n=60)
        m = max(2, int(len(pts) * reveal))
        paths[name] = pts
        seg = pts[:m].astype(np.int32)
        if len(seg) < 2: continue
        th = max(2, int(wdt * uu))
        cv2.polylines(img, [seg], False, (120, 20, 30), th + max(2, int(th * 0.5)), cv2.LINE_AA)
        cv2.polylines(img, [seg], False, (238, 70, 70), th, cv2.LINE_AA)
        cv2.polylines(img, [seg], False, (255, 160, 150), max(1, th // 3), cv2.LINE_AA)
    # blood pulses travelling down the arteries
    if reveal > 0.95:
        pulse = np.zeros((H, W), np.float32)
        for name, pts in paths.items():
            for q in ((t * 0.9) % 1.0, (t * 0.9 + 0.5) % 1.0):
                p = pts[int(q * (len(pts) - 1))]
                cv2.circle(pulse, (int(p[0]), int(p[1])), max(2, int(0.035 * uu)), 1.0, -1, cv2.LINE_AA)
        _glow(img, pulse, (255, 230, 200), max(2, 0.03 * uu), 1.2)
    # labels
    if reveal > 0.6:
        al = ease_out_cubic((u - 0.32) / 0.1)
        f = font("Poppins", 700, max(12, int(30 * s)))
        for name, at, side in (("RCA", 0.55, -1), ("LAD", 0.62, 1), ("LCx", 0.75, 1)):
            p = paths[name][int(at * 59)]
            lx = p[0] + side * 0.45 * uu; ly = p[1] - 0.1 * uu
            if lay == "land" and side > 0: lx = p[0] + 0.62 * uu
            cv2.line(img, (int(p[0]), int(p[1])), (int(lx), int(ly)), (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
            tag = text_sprite(name, f, (255, 255, 255), shadow=3, pad=2)
            tx = lx - tag.shape[1] - 6 * s if side < 0 else lx + 6 * s
            blit(img, tag, clamp(tx, 4, W - tag.shape[1] - 4), ly - tag.shape[0] / 2, al)
    # a plaque on the proximal LAD, pointed out
    pa = ease_out_cubic((u - 0.55) / 0.12)
    if pa > 0:
        p = paths["LAD"][18]
        r = max(3, int(0.06 * uu * (0.6 + 0.4 * pa)))
        cv2.circle(img, (int(p[0]), int(p[1])), r, (250, 210, 90), -1, cv2.LINE_AA)
        ring_r = int(r * (1.8 + 0.4 * math.sin(t * 5)))
        cv2.circle(img, (int(p[0]), int(p[1])), ring_r, (255, 214, 74), max(2, int(3 * s)), cv2.LINE_AA)
        lab = tr("Plaque", lang)
        tag = text_sprite(lab, font("Poppins", 700, max(12, int(32 * s))), (255, 214, 74), shadow=3, pad=2)
        tx = p[0] + ring_r + 10 * s
        blit(img, tag, clamp(tx, 4, W - tag.shape[1] - 4), p[1] - ring_r - tag.shape[0], pa)
    return img


# ------------------------------------------------------------------ doctor photos
PHOTO_DIR = None  # set by the renderer to <project>/photos


@functools.lru_cache(maxsize=12)
def _photo(path, w, h):
    im = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    ih, iw = im.shape[:2]
    # blurred cover-fill background + contained foreground (never crops a document/scan)
    sc = max(w / iw, h / ih); bg = cv2.resize(im, (int(iw * sc) + 2, int(ih * sc) + 2))
    by, bx = (bg.shape[0] - h) // 2, (bg.shape[1] - w) // 2
    bg = cv2.GaussianBlur(bg[by:by + h, bx:bx + w], (0, 0), max(8, w / 40)) * 0.45
    return im, bg.astype(np.uint8)


def photo(W, H, t, D, mode="", accent=(46, 160, 255), title=None, lang="en", **_):
    path = os.path.join(PHOTO_DIR or "", f"{mode}.jpg")
    if not mode or not os.path.exists(path):
        return G._bg(W, H).astype(np.uint8).copy()
    im, bg = _photo(path, W, H)
    u = clamp(t / max(D, 0.1), 0, 1); s = _s(W)
    out = bg.copy()
    ih, iw = im.shape[:2]
    fit = min(W * 0.94 / iw, H * 0.90 / ih)
    z = fit * (1.0 + 0.06 * ease_in_out(u))  # slow Ken Burns push
    fw, fh = int(iw * z), int(ih * z)
    fg = cv2.resize(im, (fw, fh), interpolation=cv2.INTER_AREA if z < 1 else cv2.INTER_LINEAR)
    vw, vh = min(fw, int(W * 0.94)), min(fh, int(H * 0.90))
    ox = int((fw - vw) * (0.5 + 0.2 * (u - 0.5))); oy = (fh - vh) // 2
    fg = fg[oy:oy + vh, ox:ox + vw]
    x0, y0 = (W - vw) / 2, (H - vh) / 2
    if hasattr(G, "shadow_rect"): G.shadow_rect(out, x0, y0, vw, vh, int(22 * s), 0.55, int(26 * s))
    blit(out, fg, x0, y0, ease_out_cubic(u / 0.08), rounded_mask(vw, vh, int(22 * s)))
    if title:
        cap = text_sprite(title, fit_font(title, "Poppins", 700, int(38 * s), W * 0.8), (255, 255, 255), pad=0)
        pw, ph = cap.shape[1] + int(40 * s), cap.shape[0] + int(24 * s)
        card = np.zeros((ph, pw, 4), np.uint8); card[..., :3] = (10, 20, 38); card[..., 3] = (rounded_mask(pw, ph, ph // 2) * 0.85).astype(np.uint8)
        blit(card, cap, int(20 * s), int(12 * s))
        blit(out, card, (W - pw) / 2, max(10 * s, y0 + 18 * s), ease_out_cubic((u - 0.1) / 0.15))  # top: subtitles live at the bottom
    return out


def ct(W, H, t, D, mode="calcium", **kw):
    return ct_scanner(W, H, t, D, **kw) if mode == "scanner" else ct_calcium(W, H, t, D, **kw)


def register():
    from .draw import shadow_rect
    G.shadow_rect = shadow_rect
    G.ASSETS.update({"ct": ct, "heart": heart, "photo": photo, "broll": lambda *a, **k: broll(*a, **k)})
    G.ASSET_INFO.update({
        "ct/calcium": "Cardiac CT — calcium lights up in each artery, then the CAC-DRS score scale",
        "ct/scanner": "CT scanner — patient slides in, ECG-gated scan, slices appear",
        "heart/coronaries": "Beating heart with its coronary arteries (RCA, LAD, LCx) and a plaque",
    })


register()


# ------------------------------------------------------------------ icon cards (any specialty)
ICON_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "icons")


@functools.lru_cache(maxsize=128)
def _icon_mask(icon_id, size):
    m = cv2.imread(os.path.join(ICON_DIR, icon_id + ".png"), cv2.IMREAD_GRAYSCALE)
    if m is None: m = np.zeros((256, 256), np.uint8)
    return cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA)


def _icon_badge(icon_id, size, accent, k=1.0):
    """Accent disc with a white icon (RGBA)."""
    d = max(8, int(size * k))
    out = np.zeros((d, d, 4), np.uint8)
    yy, xx = np.mgrid[0:d, 0:d].astype(np.float32)
    g = np.clip(1.15 - ((xx - d * .35) ** 2 + (yy - d * .3) ** 2) ** 0.5 / d, 0.75, 1.15)
    out[..., :3] = np.clip(np.array(accent, np.float32) * g[..., None], 0, 255).astype(np.uint8)
    disc = np.zeros((d, d), np.uint8); cv2.circle(disc, (d // 2, d // 2), d // 2 - 1, 255, -1, cv2.LINE_AA)
    out[..., 3] = disc
    ic = _icon_mask(icon_id, int(d * 0.62)); o = (d - ic.shape[0]) // 2
    a = ic.astype(np.float32)[..., None] / 255
    out[o:o + ic.shape[0], o:o + ic.shape[1], :3] = (out[o:o + ic.shape[0], o:o + ic.shape[1], :3] * (1 - a) + 255 * a).astype(np.uint8)
    return out


def icons_card(W, H, t, D, items=(), accent=(46, 160, 255), lang="en", **_):
    """RGBA card: one big tile (icon + label) or a row of 2–3 tiles, each popping in when it is said.
    items: [{"key", "at"}] with `at` seconds after the card starts."""
    from . import concepts as CO
    s = _s(W) if W >= 400 else W / 950
    items = [it for it in items if it.get("key") in CO.C][:3]
    card = np.zeros((H, W, 4), np.uint8)
    if not items: return card
    out_a = min(1.0, (D - t) / 0.3) if D - t < 0.3 else 1.0
    base = ease_out_cubic(t / 0.25) * out_a
    card[..., :3] = (12, 24, 46); card[..., 3] = (rounded_mask(W, H, int(min(H * 0.28, 40 * s))).astype(np.float32) * 0.9 * base).astype(np.uint8)
    rtl = lang == "ar"
    if len(items) == 1:
        it = items[0]; a = clamp((t - it.get("at", 0)) / 0.35, 0, 1)
        d = int(H * 0.72); k = 0.55 + 0.45 * ease_out_back(a)
        b = _icon_badge(CO.icon(it["key"]), d, accent, k)
        bx = W - int(H * 0.14) - d if rtl else int(H * 0.14)
        blit(card, b, bx + (d - b.shape[1]) / 2, (H - b.shape[0]) / 2, min(1, a * 3) * out_a)
        lab = CO.label(it["key"], lang)
        f = fit_font(lab, "Poppins", 800, int(H * 0.30), W - d - H * 0.5)
        ts = text_sprite(lab, f, (255, 255, 255), pad=0)
        tx = bx - int(H * 0.16) - ts.shape[1] if rtl else bx + d + int(H * 0.16)
        la = ease_out_cubic((t - it.get("at", 0) - 0.1) / 0.3) * out_a
        blit(card, ts, tx + (1 - la) * (-30 if rtl else 30) * s, (H - ts.shape[0]) / 2, la)
        return card
    n = len(items); cw = W / n
    order = list(range(n))[::-1] if rtl else list(range(n))
    for slot, idx in enumerate(order):
        it = items[idx]; a = clamp((t - it.get("at", 0)) / 0.35, 0, 1)
        if a <= 0: continue
        d = int(min(H * 0.46, cw * 0.5)); k = 0.55 + 0.45 * ease_out_back(a)
        b = _icon_badge(CO.icon(it["key"]), d, accent, k)
        cx = cw * slot + cw / 2
        blit(card, b, cx - b.shape[1] / 2, H * 0.08 + (d - b.shape[0]) / 2, min(1, a * 3) * out_a)
        lab = CO.label(it["key"], lang)
        ts = text_sprite(lab, fit_font(lab, "Poppins", 700, int(H * 0.20), cw * 0.92), (255, 255, 255), pad=0)
        blit(card, ts, cx - ts.shape[1] / 2, H * 0.08 + d + H * 0.05, ease_out_cubic((t - it.get("at", 0) - 0.1) / 0.3) * out_a)
        if slot < n - 1:
            x = int(cw * (slot + 1)); cv2.line(card, (x, int(H * .2)), (x, int(H * .8)), (60, 80, 110, int(255 * base)), max(1, int(2 * s)), cv2.LINE_AA)
    return card


# ------------------------------------------------------------------ stock B-roll clips
BROLL_DIR = None  # set by the renderer to <project>/broll
_READERS = {}


def broll(W, H, t, D, mode="", **_):
    path = os.path.join(BROLL_DIR or "", f"{mode}.mp4")
    if not mode or not os.path.exists(path): return G._bg(W, H).astype(np.uint8).copy()
    r = _READERS.get(path)
    if r is None:
        cap = cv2.VideoCapture(path)
        r = _READERS[path] = {"cap": cap, "i": -1, "frame": None, "n": max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), "fps": cap.get(cv2.CAP_PROP_FPS) or 30}
    want = int(t * r["fps"]) % r["n"]
    if want < r["i"]:  # scene restarted (or looped): rewind
        r["cap"].set(cv2.CAP_PROP_POS_FRAMES, 0); r["i"] = -1
    while r["i"] < want:
        ok, fr = r["cap"].read()
        if not ok: break
        r["i"] += 1; r["frame"] = fr
    fr = r["frame"]
    if fr is None: return G._bg(W, H).astype(np.uint8).copy()
    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    fh, fw = fr.shape[:2]
    z = 1.0 + 0.05 * clamp(t / max(D, 0.1), 0, 1)  # gentle push-in
    sc = max(W / fw, H / fh) * z
    rw, rh = max(W, int(fw * sc)), max(H, int(fh * sc))
    big = cv2.resize(fr, (rw, rh), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
    x0, y0 = (rw - W) // 2, (rh - H) // 2
    return np.ascontiguousarray(big[y0:y0 + H, x0:x0 + W])
