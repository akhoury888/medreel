"""Procedural, timeline-synchronised medical animations (original assets — no third-party rights).

Every asset is a pure function  render(W, H, t, D, **params) -> RGB uint8 frame,
so it is resolution-independent, deterministic (seekable to any frame) and can be
re-timed by the editor. Assets:
  coronary/narrowing  — LDL plaque grows in the wall, lumen narrows, % stenosis counts up,
                        red cells speed up through the stenosis (continuity equation)
  coronary/occlusion  — fibrous cap ruptures, thrombus forms, flow stops, downstream ischaemia
  coronary/stent      — guidewire, balloon inflation, stent deployment, flow restored
  ecg/sinus           — sweeping monitor trace with heart-rate readout
  bp/gauge            — RGBA card: animated dial + counting systolic/diastolic numbers
"""
import functools, re
import numpy as np, cv2
from .draw import font, fit_font, text_sprite, blit, rounded_mask, ease_out_cubic, ease_in_out, clamp, lerp, draw_arrow, hex2rgb

RNG = np.random.default_rng(7)
UI_SCALE = 1.0  # set by the renderer: output width / 1080 (0.5 for previews)

# on-graphic labels per language (medical graphics use clear Modern Standard Arabic)
LABELS = {"ar": {"Atherosclerosis": "تصلّب الشرايين", "Cholesterol plaque": "لويحة الكوليسترول", "Narrowed lumen": "تضيّق الشريان",
                 "Heart attack": "الذبحة القلبية", "Plaque ruptures": "تمزّق اللويحة", "Clot blocks blood flow": "الجلطة تسدّ الشريان",
                 "Heart muscle starved of oxygen": "عضلة القلب بلا أكسجين", "Angioplasty & stent": "توسيع الشريان والدعامة",
                 "Guidewire": "السلك الموجّه", "Balloon opens the artery": "البالون يفتح الشريان", "Stent keeps it open": "الدعامة تُبقيه مفتوحاً",
                 "Blood flow restored": "عودة تدفّق الدم", "narrowing": "تضيّق", "blocked": "انسداد", "ECG": "تخطيط القلب", "bpm": "نبضة/د",
                 "BLOOD PRESSURE": "ضغط الدم", "BLOOD PRESSURE TARGET": "هدف ضغط الدم", "Goal: below": "الهدف: أقل من", "mmHg": "ملم زئبق",
                 "Coronary calcium score": "مؤشر الكالسيوم في الشرايين", "Calcium ≥ 130 HU": "كثافة الكالسيوم ≥ 130",
                 "None": "لا يوجد", "Mild": "خفيف", "Moderate": "متوسط", "Severe": "شديد", "CAC-DRS · Agatston score": "مؤشر أغاتستون"},
          "fr": {"Atherosclerosis": "Athérosclérose", "Cholesterol plaque": "Plaque de cholestérol", "Narrowed lumen": "Artère rétrécie",
                 "Heart attack": "Infarctus", "Plaque ruptures": "Rupture de plaque", "Clot blocks blood flow": "Le caillot bloque le flux",
                 "Heart muscle starved of oxygen": "Le muscle cardiaque manque d'oxygène", "Angioplasty & stent": "Angioplastie et stent",
                 "Guidewire": "Guide", "Balloon opens the artery": "Le ballon ouvre l'artère", "Stent keeps it open": "Le stent la maintient ouverte",
                 "Blood flow restored": "Flux sanguin rétabli", "narrowing": "sténose", "blocked": "occlusion", "ECG": "ECG", "bpm": "bpm",
                 "BLOOD PRESSURE": "TENSION ARTÉRIELLE", "BLOOD PRESSURE TARGET": "OBJECTIF TENSIONNEL", "Goal: below": "Objectif : moins de", "mmHg": "mmHg",
                 "Coronary calcium score": "Score calcique coronaire", "Calcium ≥ 130 HU": "Calcium ≥ 130 UH",
                 "None": "Aucun", "Mild": "Léger", "Moderate": "Modéré", "Severe": "Sévère", "CAC-DRS · Agatston score": "CAC-DRS · score d'Agatston"}}


def tr(s, lang): return LABELS.get(lang, {}).get(s, s)


@functools.lru_cache(maxsize=16)
def _bg(w, h, top=(14, 30, 60), bot=(5, 12, 28)):
    y = np.linspace(0, 1, h)[:, None, None]
    img = (np.array(top) * (1 - y) + np.array(bot) * y) * np.ones((1, w, 1))
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((xx - w / 2) / (w * .75)) ** 2 + ((yy - h / 2) / (h * .75)) ** 2)
    img *= (1 - 0.35 * np.clip(r, 0, 1))[..., None]
    n = cv2.GaussianBlur(RNG.normal(0, 1, (h, w)).astype(np.float32), (0, 0), max(2, w / 90))
    img += (n / (n.std() + 1e-6) * 3)[..., None]
    return np.clip(img, 0, 255).astype(np.float32)


@functools.lru_cache(maxsize=16)
def _noise(w, h, s):
    n = cv2.GaussianBlur(RNG.normal(0, 1, (h, w)).astype(np.float32), (0, 0), s)
    return n / (n.std() + 1e-6)


_CELLS = RNG.random((70, 3))  # phase, lateral position, size jitter


def _vessel(w, h, t, sev_top, sev_bot, flow, cy_frac=0.5, clot=0.0, ischemia=0.0, stent_r=None, balloon=None, wire_tip=None, rupture=0.0):
    S = w
    cy = h * cy_frac
    R, r0 = 0.165 * S, 0.118 * S
    xs = np.arange(w, dtype=np.float32)
    xp, sig = 0.52 * w, 0.115 * w
    yc = cy + 0.035 * S * np.sin(xs / w * 2.4 + 0.6)
    gt = np.exp(-((xs - xp) / sig) ** 2)
    gb = np.exp(-((xs - xp - 0.025 * w) / (sig * .8)) ** 2)
    top_l = yc - r0 + sev_top * r0 * 1.25 * gt
    bot_l = yc + r0 - sev_bot * r0 * 0.6 * gb
    by0, by1 = int(max(0, yc.min() - R * 1.5)), int(min(h, yc.max() + R * 1.5))
    Y = np.arange(by0, by1, dtype=np.float32)[:, None]
    dy = Y - yc[None, :]
    nd = np.clip(dy / R, -1, 1)
    shade = np.sqrt(1 - nd ** 2)
    full = _bg(w, h).copy()
    img = full[by0:by1]
    a_wall = np.clip(R - np.abs(dy) + 0.5, 0, 1)
    a_lum = np.clip(np.minimum(Y - top_l[None, :], bot_l[None, :] - Y) + 0.5, 0, 1)
    in_nom = np.clip(r0 + 1.0 - np.abs(dy), 0, 1)
    a_plq = np.clip(in_nom - a_lum, 0, 1) * a_wall
    # vessel wall (tunica) with cylindrical shading + specular
    wall = np.array([196, 66, 78], np.float32)[None, None] * (0.32 + 0.68 * shade[..., None])
    spec = np.exp(-((nd + 0.58) / 0.12) ** 2)[..., None] * 70
    wall = wall + spec
    glow = np.exp(-((np.abs(dy) - R) / (0.03 * S)) ** 2)[..., None] * np.array([150, 40, 60]) * 0.25
    img = img + glow
    img = img * (1 - a_wall[..., None]) + wall * a_wall[..., None]
    # plaque: lipid core with texture + bright fibrous cap toward lumen
    nz = _noise(w, h, max(1.5, w / 160))[by0:by1]
    plq = np.array([236, 192, 92], np.float32)[None, None] * (0.78 + 0.22 * shade[..., None]) + nz[..., None] * 9
    core = np.clip(1 - np.minimum(np.abs(Y - top_l[None, :]), np.abs(Y - bot_l[None, :])) / (0.06 * S), 0, 1)
    cap = (core > 0.72)[..., None] * np.array([18, 30, 50], np.float32)
    plq = plq + cap - (1 - core)[..., None] * np.array([10, 40, 50], np.float32) * 0.6
    if rupture > 0:
        crack = np.exp(-((xs - xp) / (0.02 * w)) ** 2)[None, :] * np.exp(-((Y - top_l[None, :]) / (0.02 * S)) ** 2)
        plq = plq + (crack * rupture)[..., None] * np.array([90, 40, -20], np.float32)
    img = img * (1 - a_plq[..., None]) + plq * a_plq[..., None]
    # lumen (blood)
    lum_n = np.clip((Y - (top_l + bot_l)[None, :] / 2) / ((bot_l - top_l)[None, :] / 2 + 1e-3), -1, 1)
    blood = np.array([120, 10, 26], np.float32)[None, None] * (0.55 + 0.45 * np.sqrt(1 - lum_n ** 2))[..., None]
    img = img * (1 - a_lum[..., None]) + blood * a_lum[..., None]
    # endothelium highlight line
    edge = (np.exp(-((Y - top_l[None, :]) / 1.2) ** 2) + np.exp(-((Y - bot_l[None, :]) / 1.2) ** 2)) * (1 - a_plq) * a_wall
    img = img + edge[..., None] * np.array([70, 60, 60], np.float32)
    if ischemia > 0:
        isc = np.clip((xs - (xp + sig * 0.9)) / (0.12 * w), 0, 1)[None, :] * a_wall * ischemia
        grey = img.mean(2, keepdims=True) * np.array([0.85, 0.8, 1.15])
        img = img * (1 - 0.65 * isc[..., None]) + grey * 0.65 * isc[..., None]
    full[by0:by1] = img
    img = np.clip(full, 0, 255).astype(np.uint8)
    width = np.maximum(bot_l - top_l, 1.0)
    # red cells: transit-time mapping so cells speed up through the stenosis
    if flow > 0.001 or clot < 0.999:
        u = (2 * r0) / width
        if clot > 0: u = u / (1 - 0.95 * clot)
        tau = np.concatenate([[0], np.cumsum(1 / u)])[:-1]
        T = tau[-1] + 1e-6
        speed = 0.9 * w
        for ph, lat, sz in _CELLS:
            tt = (ph * T + (t * speed * flow)) % T
            x = float(np.interp(tt, tau, xs))
            xi = int(clamp(x, 0, w - 1))
            if clot > 0 and abs(x - xp) < sig * 1.1 and clot > 0.5: continue
            yy = top_l[xi] + (0.15 + 0.7 * lat) * width[xi]
            rx = (0.020 + 0.006 * sz) * S; ry = rx * 0.55
            ang = float(np.degrees(np.arctan2(yc[min(w - 1, xi + 2)] - yc[max(0, xi - 2)], 4)))
            c = (int(x), int(yy))
            cv2.ellipse(img, c, (int(rx), max(1, int(ry))), ang, 0, 360, (214, 38, 54), -1, cv2.LINE_AA)
            cv2.ellipse(img, c, (max(1, int(rx * .5)), max(1, int(ry * .45))), ang, 0, 360, (160, 18, 34), -1, cv2.LINE_AA)
    if clot > 0:
        cw = sig * 0.95 * clot
        for k in range(14):
            px = xp + (k / 13 - 0.5) * 2 * cw
            xi = int(clamp(px, 0, w - 1))
            ry = (width[xi] / 2) * min(1, clot * 1.2)
            cv2.ellipse(img, (int(px), int((top_l[xi] + bot_l[xi]) / 2)), (int(cw * 0.35 + 1), max(1, int(ry))), 0, 0, 360, (88, 6, 18), -1, cv2.LINE_AA)
        rs = np.random.default_rng(3)
        ov = img.copy()
        for k in range(int(18 * clot)):  # fibrin mesh + platelets
            px, py = xp + (rs.random() - .5) * 1.6 * cw, cy + (rs.random() - .5) * r0 * 1.1 * clot
            ang = rs.random() * np.pi
            q = (int(px + np.cos(ang) * cw * .35), int(py + np.sin(ang) * r0 * .25))
            cv2.line(ov, (int(px), int(py)), q, (200, 160, 140), 1, cv2.LINE_AA)
            cv2.circle(ov, (int(px), int(py)), max(1, int(S * .004)), (230, 200, 190), -1, cv2.LINE_AA)
        img = cv2.addWeighted(ov, 0.45, img, 0.55, 0)
    if wire_tip is not None:
        pts = np.stack([xs[: int(wire_tip)], yc[: int(wire_tip)] + 0.01 * S], 1).astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, (220, 225, 235), max(1, int(S * 0.005)), cv2.LINE_AA)
            cv2.polylines(img, [pts[: int(len(pts) * 0.55)]], False, (150, 170, 200), max(2, int(S * 0.016)), cv2.LINE_AA)
    if balloon is not None and balloon[1] > 0:
        bl, br = balloon
        ov = img.copy()
        cv2.ellipse(ov, (int(xp), int(cy)), (int(bl / 2), max(1, int(br))), 0, 0, 360, (225, 232, 245), -1, cv2.LINE_AA)
        img = cv2.addWeighted(ov, 0.42, img, 0.58, 0)
        cv2.ellipse(img, (int(xp), int(cy)), (int(bl / 2), max(1, int(br))), 0, 0, 360, (240, 245, 255), max(1, int(S * .004)), cv2.LINE_AA)
    if stent_r is not None:
        L = 0.34 * w; n = 12
        col = (205, 215, 230)
        for sgn in (-1, 1):
            pts = []
            for k in range(n + 1):
                x = xp - L / 2 + k * L / n
                xi = int(clamp(x, 0, w - 1))
                pts.append((int(x), int(cy + 0.035 * S * np.sin(x / w * 2.4 + .6) - (cy - yc[xi]) * 0 + sgn * stent_r * (1 if k % 2 else 0.86))))
            cv2.polylines(img, [np.array(pts, np.int32)], False, col, max(1, int(S * .006)), cv2.LINE_AA)
        ov = img.copy()
        for k in range(0, n, 2):
            x0 = xp - L / 2 + k * L / n
            cv2.line(ov, (int(x0), int(cy - stent_r)), (int(x0 + 2 * L / n), int(cy + stent_r)), col, max(1, int(S * .003)), cv2.LINE_AA)
            cv2.line(ov, (int(x0 + 2 * L / n), int(cy - stent_r)), (int(x0), int(cy + stent_r)), col, max(1, int(S * .003)), cv2.LINE_AA)
        img = cv2.addWeighted(ov, 0.28, img, 0.72, 0)
    sten = 1 - float(width.min()) / (2 * r0)
    return img, {"xp": xp, "cy": cy, "R": R, "r0": r0, "top_min": float(top_l[int(xp)]), "stenosis": sten}


def _chip(img, text, x, y, accent, size, rtl=False):
    f = fit_font(text, "Poppins", 700, size, img.shape[1] * 0.5 - size)  # leaves room for a PiP bubble
    spr = text_sprite(text, f, (255, 255, 255), pad=0)
    pw, ph = spr.shape[1] + size, spr.shape[0] + int(size * .7)
    card = np.zeros((ph, pw, 4), np.uint8); card[..., :3] = accent; card[..., 3] = rounded_mask(pw, ph, ph // 2)
    if rtl: x = img.shape[1] - x - pw  # right-aligned for Arabic
    blit(img, card, x, y, 0.95); blit(img, spr, x + size // 2, y + (ph - spr.shape[0]) // 2)
    return pw, ph


def coronary(W, H, t, D, mode="narrowing", accent=(46, 160, 255), title=None, safe_bottom=0.0, lang="en"):
    f = 0.5
    w, h = max(64, int(W * f)), max(64, int(H * f))
    u = t / max(D, 0.1)
    portrait = H > W * 1.3
    cy_frac = 0.47 if portrait else 0.55
    kw = {}
    if mode == "narrowing":
        sev = 0.72 * ease_in_out((u - 0.08) / 0.75)
        kw = dict(sev_top=sev, sev_bot=sev, flow=1.0)
        labels = [("Cholesterol plaque", 0.22), ("Narrowed lumen", 0.55)]
        title = title or "Atherosclerosis"
    elif mode == "occlusion":
        rup = clamp((u - 0.2) / 0.15, 0, 1)
        cl = ease_in_out((u - 0.33) / 0.42)
        kw = dict(sev_top=0.62, sev_bot=0.62, flow=max(0.0, 1 - cl), clot=cl, ischemia=ease_in_out((u - 0.6) / 0.3), rupture=rup * (1 - cl * .5))
        labels = [("Plaque ruptures", 0.18), ("Clot blocks blood flow", 0.5), ("Heart muscle starved of oxygen", 0.75)]
        title = title or "Heart attack"
    else:  # stent
        S = w; r0 = 0.118 * S
        wire = ease_out_cubic(u / 0.2) * w
        inf = ease_in_out((u - 0.28) / 0.17); defl = ease_in_out((u - 0.55) / 0.12)
        sev = 0.72 * (1 - 0.9 * inf)
        br = (0.03 * S + (r0 * 0.93 - 0.03 * S) * inf) * (1 - defl) if u > 0.2 else 0
        bl = 0.36 * w if u > 0.2 else 0
        bx = 0 if defl < 1 else 1
        stent_r = (0.04 * S + (r0 * 0.92 - 0.04 * S) * inf) if u > 0.22 else None
        flow = 0.35 + 0.65 * ease_in_out((u - 0.6) / 0.25)
        kw = dict(sev_top=sev, sev_bot=sev, flow=flow, wire_tip=wire if u < 0.75 else None,
                  balloon=(bl, br) if (u > 0.2 and defl < 1) else None, stent_r=stent_r)
        labels = [("Guidewire", 0.05), ("Balloon opens the artery", 0.3), ("Stent keeps it open", 0.58), ("Blood flow restored", 0.78)]
        title = title or "Angioplasty & stent"
    title = tr(title, lang); labels = [(tr(a, lang), b) for a, b in labels]
    small, info = _vessel(w, h, t, cy_frac=cy_frac, **kw)
    img = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)
    s = max(W / 1080, 0.8 * UI_SCALE)  # keep labels legible on small cards
    # title chip
    chip_y = int(max(0.09 * H, 175 * W / 1080) if portrait else 0.06 * H + (175 * W / 1080 if W > 900 else 0))
    _, chip_h = _chip(img, title, int(40 * W / 1080), chip_y, accent, max(16, int(36 * s)), rtl=lang == "ar")
    chip_bottom = chip_y + chip_h
    # stage label with arrow (one active label at a time)
    active = [l for l in labels if u >= l[1]]
    if active:
        txt, t0 = active[-1]
        a = ease_out_cubic((u - t0) / 0.08)
        fnt = font("Poppins", 600, max(16, int(42 * s)))
        spr = text_sprite(txt, fnt, (255, 255, 255), shadow=max(2, int(4 * s)))
        xp, ty = info["xp"] / f, info["top_min"] / f
        lx = clamp(xp - spr.shape[1] / 2, 20 * s, W - spr.shape[1] - 20 * s)
        ly = max((info["cy"] - info["R"]) / f - spr.shape[0] - 110 * s, chip_bottom + 12 * s)
        blit(img, spr, lx, ly + (1 - a) * 30 * s, a)
        draw_arrow(img, (xp, ly + spr.shape[0] + 6 * s), (xp, ty - 10 * s), (255, 255, 255), max(2, int(5 * s)), progress=a, head=int(20 * s))
    # stenosis / flow readout
    if mode in ("narrowing", "occlusion"):
        pct = 100 if (mode == "occlusion" and kw.get("clot", 0) > 0.98) else int(round(max(0, info["stenosis"]) * 100 if mode == "narrowing" else (62 + 38 * kw.get("clot", 0))))
        big = text_sprite(f"{pct}%", font("Poppins", 800, max(24, int(92 * s))), (255, 255, 255), shadow=3)
        cap = text_sprite(tr("blocked" if (mode == "occlusion" and kw.get("clot", 0) > 0.5) else "narrowing", lang), font("Poppins", 500, max(14, int(34 * s))), (180, 200, 230))
        by = (info["cy"] + info["R"]) / f + 50 * s
        bx = W - big.shape[1] - 50 * s
        blit(img, big, bx, by); blit(img, cap, bx + (big.shape[1] - cap.shape[1]) / 2, by + big.shape[0] - 10 * s)
    return img


def _ecg_wave(x):
    """x in beats (float) -> mV-like amplitude of a sinus PQRST complex."""
    p = x % 1.0
    v = 0.12 * np.exp(-((p - 0.16) / 0.035) ** 2)
    v += -0.10 * np.exp(-((p - 0.285) / 0.010) ** 2) + 1.0 * np.exp(-((p - 0.30) / 0.012) ** 2) - 0.22 * np.exp(-((p - 0.318) / 0.012) ** 2)
    v += 0.28 * np.exp(-((p - 0.55) / 0.055) ** 2)
    return v


def ecg(W, H, t, D, mode="sinus", accent=(46, 160, 255), title="ECG", bpm=72, lang="en", **_):
    title = tr(title, lang)
    img = _bg(W, H).astype(np.uint8).copy()
    s = W / 1080
    g = max(8, int(40 * s))
    ov = img.copy()
    for x in range(0, W, g): cv2.line(ov, (x, 0), (x, H), (40, 70, 110) if x % (g * 5) else (60, 100, 150), 1)
    for y in range(0, H, g): cv2.line(ov, (0, y), (W, y), (40, 70, 110) if y % (g * 5) else (60, 100, 150), 1)
    img = cv2.addWeighted(ov, 0.5, img, 0.5, 0)
    cy, amp = H * 0.5, H * (0.16 if H > W * 1.3 else 0.26)
    sweep_w = W; beats_per_px = (bpm / 60) / (W / 4.0)  # 4 s across the screen
    head = (t * W / 4.0) % sweep_w
    xs = np.arange(W)
    tt = np.where(xs <= head, t - (head - xs) / (W / 4.0), t - (head - xs + W) / (W / 4.0))
    ys = cy - amp * _ecg_wave(tt * bpm / 60)
    pts = np.stack([xs, ys], 1).astype(np.int32)
    gap = int(30 * s)
    seg1 = pts[: int(head)]; seg2 = pts[int(head) + gap:]
    for seg in (seg2, seg1):
        if len(seg) > 1:
            cv2.polylines(img, [seg], False, (40, 255, 160), max(2, int(5 * s)), cv2.LINE_AA)
    if len(seg1):
        glow = np.zeros_like(img); cv2.circle(glow, tuple(seg1[-1]), int(14 * s), (120, 255, 200), -1, cv2.LINE_AA)
        img = cv2.add(img, cv2.GaussianBlur(glow, (0, 0), 6 * s))
    _chip(img, title, int(40 * s), int(max(0.09 * H, 175 * s) if H > W * 1.3 else 0.06 * H + (175 * s if W > 900 else 0)), accent, max(16, int(36 * s)), rtl=lang == "ar")
    beat_phase = (t * bpm / 60) % 1
    pulse = 1 + 0.12 * np.exp(-((beat_phase - 0.32) / 0.06) ** 2)
    hr = text_sprite(f"{bpm}", font("Poppins", 800, int(96 * s * pulse)), (40, 255, 160), shadow=3)
    unit = text_sprite(tr("bpm", lang), font("Poppins", 500, int(36 * s)), (160, 220, 200))
    blit(img, hr, W - hr.shape[1] - unit.shape[1] - 40 * s, H * 0.72); blit(img, unit, W - unit.shape[1] - 36 * s, H * 0.72 + hr.shape[0] - unit.shape[0] - 14 * s)
    return img


def bp_card(W, H, t, D, text="", accent=(46, 160, 255), lang="en", **_):
    """Transparent-background RGBA card with an animated BP dial."""
    from .knowledge import BP_RE, ar_norm as _an
    m = BP_RE.search(_an(text or ""))
    sys_v, dia_v = (int(m.group(1)), int(m.group(2))) if m else (120, 80)
    if not m:  # French convention in cmHg: "13/8", "14 sur 9"
        mf = re.search(r"\b(1[0-9])\s*(?:/|sur)\s*([5-9])\b", (text or "").lower())
        if mf: sys_v, dia_v = int(mf.group(1)) * 10, int(mf.group(2)) * 10
    from .knowledge import ar_norm
    target = bool(re.search(r"\b(below|under|less than|lower than)\b|تحت|اقل من|ما يطلع فوق|en dessous|moins de", ar_norm((text or "").lower())))
    card = np.zeros((H, W, 4), np.uint8)
    card[..., :3] = (12, 24, 46); card[..., 3] = (rounded_mask(W, H, int(48 * W / 900)).astype(np.float32) * 0.92).astype(np.uint8)
    s = W / 900
    a = ease_out_cubic(t / 1.2)
    cx, cy, r = int(W * 0.21), int(H * 0.56), int(min(W * .15, H * .3))
    rgb = card[..., :3].copy()
    th = max(4, int(r * .16))
    cv2.ellipse(rgb, (cx, cy), (r, r), 0, 150, 390, (50, 70, 100), th, cv2.LINE_AA)
    frac = clamp(sys_v / 200, 0, 1) * a
    for lo, hi, col in [(0, .6, (60, 200, 120)), (.6, .65, (240, 200, 60)), (.65, .7, (250, 140, 50)), (.7, 1, (235, 70, 70))]:
        e0, e1 = 150 + 240 * lo, 150 + 240 * min(hi, frac)
        if e1 > e0: cv2.ellipse(rgb, (cx, cy), (r, r), 0, e0, e1, col, th, cv2.LINE_AA)
    ang = np.radians(150 + 240 * frac)
    cv2.line(rgb, (cx, cy), (int(cx + r * .74 * np.cos(ang)), int(cy + r * .74 * np.sin(ang))), (255, 255, 255), max(2, th // 3), cv2.LINE_AA)
    cv2.circle(rgb, (cx, cy), max(3, th // 2), (255, 255, 255), -1, cv2.LINE_AA)
    card[..., :3] = rgb
    x0 = int(W * .42); mw = int(W * .54)
    ht = tr("BLOOD PRESSURE TARGET" if target else "BLOOD PRESSURE", lang)
    head = text_sprite(ht, fit_font(ht, "Poppins", 700, int(34 * s), mw), accent, pad=0)
    num = text_sprite(f"{int(sys_v * a)}/{int(dia_v * a)}", fit_font(f"{sys_v}/{dia_v}", "Inter", 900, int(120 * s), mw), (255, 255, 255), pad=0)
    unit = text_sprite(tr("mmHg", lang), font("Poppins", 500, int(36 * s)), (160, 185, 220), pad=0)
    y = int(H * .2)
    blit(card[..., :3], head, x0, y); y += head.shape[0] + int(18 * s)
    blit(card[..., :3], num, x0, y); y += num.shape[0] + int(8 * s)
    blit(card[..., :3], unit, x0, y); y += unit.shape[0] + int(26 * s)
    if target:
        ct = f"{tr('Goal: below', lang)} {sys_v}/{dia_v}"
        chip = text_sprite(ct, fit_font(ct, "Poppins", 700, int(34 * s), mw - 30 * s), (255, 255, 255), pad=0)
        pw, ph = chip.shape[1] + int(36 * s), chip.shape[0] + int(22 * s)
        c = np.zeros((ph, pw, 4), np.uint8); c[..., :3] = accent; c[..., 3] = rounded_mask(pw, ph, ph // 2)
        blit(card[..., :3], c, x0, y); blit(card[..., :3], chip, x0 + int(18 * s), y + int(11 * s))
    return card



# (x, y, radius, vessel) in heart-relative units; order = appearance order
_CA = [(0.18, -0.42, 0.055, "LAD"), (0.30, -0.25, 0.045, "LAD"), (0.05, -0.55, 0.04, "LM"), (-0.55, -0.05, 0.05, "RCA"),
       (-0.62, 0.22, 0.04, "RCA"), (0.62, -0.05, 0.045, "LCx"), (0.40, -0.10, 0.035, "LAD"), (0.68, 0.18, 0.035, "LCx")]
_VCOL = {"LM": (255, 214, 74), "LAD": (255, 214, 74), "LCx": (90, 220, 140), "RCA": (255, 110, 150)}


def ct_calcium(W, H, t, D, mode="calcium", accent=(46, 160, 255), title=None, lang="en", **_):
    """Non-contrast cardiac CT slice: calcified plaque appears vessel by vessel, then CAC-DRS categories."""
    u = t / max(D, 0.1); s = max(W / 1080, 0.8 * UI_SCALE)
    img = _bg(W, H).astype(np.uint8).copy()
    portrait = H > W * 1.3
    chip_y = int(max(0.09 * H, 175 * W / 1080) if portrait else 0.06 * H + (175 * W / 1080 if W > 900 else 0))
    _, ch = _chip(img, tr(title or "Coronary calcium score", lang), int(40 * W / 1080), chip_y, accent, max(16, int(36 * s)), rtl=lang == "ar")
    bar_h = int(215 * s)
    r = int(min(W * 0.36, (H - chip_y - ch - bar_h - 60 * s) * 0.5, H * 0.24))
    cx, cy = W // 2, int(chip_y + ch + 24 * s + r)
    # --- CT slice (greyscale, circular field of view)
    sl = np.zeros((2 * r, 2 * r), np.float32)
    c = (r, r)
    cv2.ellipse(sl, c, (int(r * .98), int(r * .74)), 0, 0, 360, 95, -1, cv2.LINE_AA)        # body wall / fat
    cv2.ellipse(sl, c, (int(r * .90), int(r * .66)), 0, 0, 360, 120, -1, cv2.LINE_AA)       # muscle
    for sx in (-1, 1): cv2.ellipse(sl, (int(r + sx * r * .48), int(r * 1.0)), (int(r * .34), int(r * .52)), 0, 0, 360, 22, -1, cv2.LINE_AA)  # lungs
    hx, hy, hr = int(r * 1.12), int(r * .98), int(r * .48)
    cv2.ellipse(sl, (hx, hy), (hr, int(hr * .85)), -25, 0, 360, 128, -1, cv2.LINE_AA)       # heart (unenhanced blood ~ muscle)
    cv2.ellipse(sl, (hx, hy), (hr, int(hr * .85)), -25, 0, 360, 104, max(1, int(r * .03)), cv2.LINE_AA)  # epicardial fat
    cv2.circle(sl, (int(r * .86), int(r * .56)), int(r * .11), 132, -1, cv2.LINE_AA)        # ascending aorta
    cv2.circle(sl, (int(r * .78), int(r * 1.50)), int(r * .10), 132, -1, cv2.LINE_AA)       # descending aorta
    cv2.circle(sl, (r, int(r * 1.62)), int(r * .12), 235, -1, cv2.LINE_AA)                  # vertebra
    cv2.circle(sl, (r, int(r * 1.62)), int(r * .07), 150, -1, cv2.LINE_AA)
    cv2.ellipse(sl, (r, int(r * .30)), (int(r * .10), int(r * .04)), 0, 0, 360, 220, -1, cv2.LINE_AA)  # sternum
    sl += _noise(2 * r, 2 * r, max(1.0, r / 160)) * 6
    rgb = np.repeat(np.clip(sl, 0, 255)[..., None], 3, 2).astype(np.uint8)
    # calcium deposits
    n_show = len(_CA) * ease_in_out((u - 0.12) / 0.45)
    for k, (dx, dy, rr, v) in enumerate(_CA):
        a = clamp(n_show - k, 0, 1)
        if a <= 0: continue
        p = (int(hx + dx * hr), int(hy + dy * hr)); rad = max(2, int(rr * hr * (0.6 + 0.4 * a)))
        cv2.circle(rgb, p, rad, (250, 250, 250), -1, cv2.LINE_AA)
        if u > 0.35:  # scoring overlay: lesions >= 130 HU coloured by vessel
            cv2.circle(rgb, p, rad + max(2, int(r * .022)), _VCOL[v], max(2, int(r * .018)), cv2.LINE_AA)
    fov = rounded_mask(2 * r, 2 * r, r)
    ring = np.zeros((2 * r + 8, 2 * r + 8, 4), np.uint8)
    cv2.circle(ring, (r + 4, r + 4), r + 2, accent + (255,), max(2, int(4 * s)), cv2.LINE_AA)
    blit(img, rgb, cx - r, cy - r, 1.0, fov); blit(img, ring, cx - r - 4, cy - r - 4)
    # vessel tags
    if u > 0.35:
        al = ease_out_cubic((u - 0.35) / 0.1)
        for v, (dx, dy), side in (("LAD", (0.30, -0.25), 1), ("LCx", (0.68, 0.18), 1), ("RCA", (-0.62, 0.22), -1)):
            px, py = cx - r + hx + dx * hr, cy - r + hy + dy * hr
            tag = text_sprite(v, font("Poppins", 700, max(12, int(30 * s))), _VCOL[v], shadow=2, pad=2)
            tx = px + side * (r * .22) - (tag.shape[1] if side < 0 else 0)
            tx = clamp(tx, 6, W - tag.shape[1] - 6)
            blit(img, tag, tx, py - tag.shape[0] / 2, al)
    hu = text_sprite(tr("Calcium ≥ 130 HU", lang), font("Poppins", 500, max(12, int(28 * s))), (180, 200, 230), pad=0)
    blit(img, hu, cx - hu.shape[1] / 2, cy + r + 10 * s, ease_out_cubic((u - 0.3) / 0.1))
    hu_bottom = cy + r + 10 * s + hu.shape[0]
    # --- CAC-DRS category bar
    bands = [("0", "None", (60, 200, 120)), ("1–99", "Mild", (240, 200, 60)), ("100–299", "Moderate", (250, 140, 50)), ("≥300", "Severe", (235, 70, 70))]
    if lang == "ar": bands = bands[::-1]  # read right-to-left
    bx0, bw = int(W * 0.06), int(W * 0.88); seg = bw / 4; bh = max(10, int(22 * s))
    head = text_sprite(tr("CAC-DRS · Agatston score", lang), fit_font(tr("CAC-DRS · Agatston score", lang), "Poppins", 600, int(28 * s), bw), (200, 215, 235), pad=0)
    by = int(hu_bottom + 18 * s + head.shape[0] + 8 * s)
    blit(img, head, bx0 + (bw - head.shape[1]) / 2, by - head.shape[0] - 8 * s, ease_out_cubic((u - 0.55) / 0.1))
    for k, (num, name, col) in enumerate(bands):
        kk = (3 - k) if lang == "ar" else k
        a = ease_out_cubic((u - 0.58 - 0.07 * kk) / 0.1)
        if a <= 0: continue
        x0 = int(bx0 + k * seg + 3); x1 = int(bx0 + (k + 1) * seg - 3)
        bar = np.zeros((bh, x1 - x0, 4), np.uint8); bar[..., :3] = col; bar[..., 3] = rounded_mask(x1 - x0, bh, bh // 2)
        blit(img, bar, x0, by, a)
        nt = text_sprite(num, fit_font(num, "Poppins", 700, int(34 * s), seg - 8), (255, 255, 255), pad=0)
        lt = text_sprite(tr(name, lang), fit_font(tr(name, lang), "Poppins", 500, int(28 * s), seg - 8), col, pad=0)
        mid = (x0 + x1) / 2
        blit(img, nt, mid - nt.shape[1] / 2, by + bh + 8 * s, a); blit(img, lt, mid - lt.shape[1] / 2, by + bh + 12 * s + nt.shape[0], a)
    return img


ASSETS = {"coronary": coronary, "ecg": ecg, "bp": bp_card, "ct": ct_calcium}
ASSET_INFO = {
    "coronary/narrowing": "Coronary artery — plaque build-up and narrowing",
    "coronary/occlusion": "Coronary artery — plaque rupture, clot and heart attack",
    "coronary/stent": "Coronary artery — balloon angioplasty and stent",
    "ecg/sinus": "ECG monitor trace with heart rate",
    "bp/gauge": "Blood pressure dial with animated numbers (overlay card)",
    "ct/calcium": "Cardiac CT — coronary calcium appears, then CAC-DRS score categories",
}
