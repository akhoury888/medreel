"""Frame compositor + audio pipeline: EditPlan JSON -> 1080x1920 H.264/AAC MP4.

Video: ffmpeg decodes the (never modified) source -> numpy frames -> per-frame
composition (reframe/zoom, scene layouts with animated transitions, procedural
medical animations, captions, titles, branding, outro) -> ffmpeg/libx264.
Audio: ffmpeg filtergraph (segment concat with micro-fades, high-pass, FFT denoise,
compression, EBU R128 loudness -14 LUFS, music bed with side-chain ducking)."""
import math, os, re, subprocess, time
import numpy as np, cv2
from PIL import Image, ImageDraw
from .draw import (font, font_for, fit_font, text_sprite, blit, rounded_mask, shadow_rect, scale_rgba, ease_out_cubic, ease_in_out,
                   ease_out_back, lerp, clamp, hex2rgb, draw_arrow)
from . import graphics as G, planner as P, knowledge as K, music as M
from . import graphics_pro as GP
from . import sfx as SFX

# ---------------- layouts (fractions of W,H) ----------------
# doc: rect, r(adius frac of W), fx/fy (face position inside rect), face (face-height fraction; None=use zoom), alpha
# gfx: rect, r, alpha
LAYOUT = {
    "full":         dict(doc=(0, 0, 1, 1), dr=0, fx=.5, fy=.38, face=None, zm=1, da=1, gfx=(.06, .30, .88, .36), gr=.04, ga=0),
    "side":         dict(doc=(0, 0, 1, 1), dr=0, fx=.76, fy=.28, face=None, zm=1.6, da=1, gfx=(.03, .40, .53, .32), gr=.04, ga=1),
    "pip":          dict(doc=(.60, .085, .34, .34 * 9 / 16), dr=.17, fx=.5, fy=.47, face=.46, zm=1, da=1, gfx=(0, 0, 1, 1), gr=0, ga=1),
    "full_graphic": dict(doc=(.66, .10, .22, .22 * 9 / 16), dr=.11, fx=.5, fy=.47, face=.46, zm=1, da=0, gfx=(0, 0, 1, 1), gr=0, ga=1),
    "split":        dict(doc=(0, .5, 1, .5), dr=0, fx=.5, fy=.47, face=.33, zm=1, da=1, gfx=(0, 0, 1, .5), gr=0, ga=1),
    "labels":       dict(doc=(0, 0, 1, 1), dr=0, fx=.5, fy=.30, face=None, zm=1, da=1, gfx=(.05, .47, .90, .26), gr=.035, ga=1),
}
GRADES = {
    "natural": "eq=contrast=1.04:saturation=1.06,unsharp=5:5:0.4",
    "bright": "eq=contrast=1.07:brightness=0.03:saturation=1.16,colorbalance=rh=0.03:bh=-0.02,unsharp=5:5:0.5",
    "punchy": "eq=contrast=1.18:saturation=1.34,colorbalance=rm=0.03:bm=-0.02,unsharp=5:5:0.8",
    "cinematic": "eq=contrast=1.12:saturation=0.88,colorbalance=rs=-0.07:bs=0.08:rh=0.08:bh=-0.07,vignette=PI/4",
    "soft": "eq=contrast=0.98:brightness=0.025:saturation=0.90,unsharp=5:5:0.3",
}


def _mirror(lay):
    m = dict(lay)
    x, y, w, h = lay["doc"]; m["doc"] = (1 - x - w, y, w, h); m["fx"] = 1 - lay["fx"]
    x, y, w, h = lay["gfx"]; m["gfx"] = (1 - x - w, y, w, h)
    return m


LAYOUT_RTL = {k: _mirror(v) for k, v in LAYOUT.items()}


def _lerp_layout(a, b, m):
    out = {}
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, tuple): out[k] = tuple(lerp(x, y, m) for x, y in zip(va, vb))
        elif va is None or vb is None: out[k] = vb if m > 0.5 else va
        else: out[k] = lerp(va, vb, m)
    return out


def _face_at(track, t):
    if not track: return [0.36, 0.16, 0.28, 0.17]
    ts = [p["t"] for p in track]
    i = int(np.searchsorted(ts, t))
    if i <= 0: return track[0]["box"]
    if i >= len(track): return track[-1]["box"]
    a, b = track[i - 1], track[i]; m = (t - a["t"]) / max(1e-6, b["t"] - a["t"])
    return [lerp(x, y, m) for x, y in zip(a["box"], b["box"])]


MAX_FACE = 0.60  # face box may fill at most this share of the crop height


def _crop(frame, face, rect_wh, fx, fy, face_frac, zoom, zm=1.0):
    SH, SW = frame.shape[:2]; rw, rh = rect_wh
    a = rw / rh
    ch_max = min(SH, SW / a)
    fh = face[3] * SH
    ch = clamp(fh / face_frac, ch_max * 0.25, ch_max) if face_frac else ch_max / max(1.0, zoom * zm)
    ch = min(ch_max, max(ch, fh / MAX_FACE))  # never zoom a close selfie down to nose and mouth
    cw = ch * a
    fcx, fcy = (face[0] + face[2] / 2) * SW, (face[1] + face[3] / 2) * SH
    x0 = clamp(fcx - fx * cw, 0, SW - cw); y0 = clamp(fcy - fy * ch, 0, SH - ch)
    return x0, y0, cw, ch


# ---------------- captions ----------------
# subtitle designs: words per line, characters per line, Latin font (family, weight, size), Arabic weight
CAPCFG = {"pop": (3, 18, ("Poppins", 700, 84), 800), "highlight": (4, 26, ("Poppins", 700, 66), 700), "minimal": (6, 38, ("Inter", 600, 54), 500),
          "boxed": (5, 30, ("Poppins", 700, 58), 700), "outline": (3, 20, ("Poppins", 700, 80), 800), "karaoke": (4, 24, ("Poppins", 700, 70), 800),
          "neon": (4, 24, ("Poppins", 700, 70), 800), "bubble": (4, 24, ("Poppins", 700, 60), 800), "typewriter": (5, 30, ("Inter", 700, 62), 700)}


class Captions:
    """source='original' -> spoken words (word-synced); source='translated' -> plan.captions.translations
    (per sentence, words timed across the sentence's kept span). Arabic is laid out right-to-left."""
    def __init__(self, plan, tl, W, H, accent, source="original", style=None, scale=1.0):
        self.style = style or plan["captions"]["style"]; self.W, self.H = W, H; self.accent = accent
        if self.style not in CAPCFG: self.style = "highlight"
        self.s = W / 1080 * scale
        cuts = [(c["start"], c["end"]) for c in plan["cuts"] if c["enabled"]]
        kept = lambda w: not w.get("deleted") and not K.is_hesitation(w["text"]) and not any(a <= (w["start"] + w["end"]) / 2 <= b for a, b in cuts) and tl.is_kept((w["start"] + w["end"]) / 2)
        words = []
        if source == "original":
            kw = set()
            for t in K.detect_terms(plan["transcript"]):
                if t["label"] not in K.LOW_VALUE_TERMS: kw.update(range(t["start_idx"], t["end_idx"] + 1))
            ends = {sen["i1"] for sen in plan["analysis"]["sentences"]}
            for w in plan["transcript"]:
                if kept(w): words.append({"text": w["text"], "s": tl.to_out(w["start"]), "e": tl.to_out(w["end"]), "kw": w["i"] in kw, "eos": w["i"] in ends})
        else:
            trs = plan["captions"].get("translations", {})
            for idx, sen in enumerate(plan["analysis"]["sentences"]):
                text = (trs.get(str(idx)) or "").strip()
                spans = [(tl.to_out(w["start"]), tl.to_out(w["end"])) for w in plan["transcript"][sen["i0"]: sen["i1"] + 1] if kept(w)]
                if not text or not spans: continue
                o0, o1 = min(a for a, _ in spans), max(b for _, b in spans)
                toks = text.split(); wts = [len(t) + 1 for t in toks]; tot = float(sum(wts)); t0 = o0
                kwi = set()
                for t in K.detect_terms([{"text": x} for x in toks]):
                    if t["label"] not in K.LOW_VALUE_TERMS: kwi.update(range(t["start_idx"], t["end_idx"] + 1))
                for i, (tk, wt) in enumerate(zip(toks, wts)):
                    d = (o1 - o0) * wt / tot
                    words.append({"text": tk, "s": t0, "e": t0 + d, "kw": i in kwi, "eos": i == len(toks) - 1}); t0 += d
        self.rtl = bool(words) and sum(K.is_ar(w["text"]) for w in words) > 0.3 * len(words)
        maxw = plan["captions"].get("max_words") or CAPCFG[self.style][0]
        maxc = CAPCFG[self.style][1]
        groups, cur = [], []
        for i, w in enumerate(words):
            cur.append(w)
            nxt = words[i + 1] if i + 1 < len(words) else None
            chars = sum(len(x["text"]) + 1 for x in cur)
            if (len(cur) >= maxw or chars >= maxc or re.search(r"[.!?,;:،؟]$", w["text"]) or w.get("eos") or nxt is None or nxt["s"] - w["e"] > 0.4):
                groups.append(cur); cur = []
        self.groups = []
        for gi, g in enumerate(groups):
            nxt_s = groups[gi + 1][0]["s"] if gi + 1 < len(groups) else g[-1]["e"] + 1
            self.groups.append({"words": g, "s": g[0]["s"] - 0.04, "e": min(nxt_s - 0.02, g[-1]["e"] + 0.6), "y": None})
        self.cache = {}

    def srt(self):
        def ts(x):
            x = max(0, x); return f"{int(x//3600):02d}:{int(x%3600//60):02d}:{int(x%60):02d},{int(x*1000%1000):03d}"
        return "\n".join(f"{i+1}\n{ts(g['s'])} --> {ts(g['e'])}\n{' '.join(w['text'] for w in g['words'])}\n" for i, g in enumerate(self.groups))

    def _fonts(self):
        s = self.s; st = self.style; _, _, (fam, wt, size), arw = CAPCFG[st]
        if self.rtl:
            f = font("Tajawal", arw, int(size * 1.06 * s))
            return f, (font("Tajawal", arw, int(size * 1.17 * s)) if st in ("pop", "outline") else None)
        return font(fam, wt, int(size * s)), (font(fam, wt, int(size * 1.12 * s)) if st in ("pop", "outline") else None)

    def sprite(self, gi, active):
        key = (gi, active)
        if key in self.cache: return self.cache[key]
        g = self.groups[gi]; st = self.style; s = self.s
        f, fa = self._fonts()
        texts = [w["text"].upper() if st in ("pop", "outline") and not self.rtl else w["text"] for w in g["words"]]
        if st in ("pop", "outline"): texts = [re.sub(r"[.,;:،؟]$", "", t) for t in texts]
        space = f.getlength(" ")
        fonts = [fa if (fa is not None and i == active) else f for i in range(len(texts))]
        widths = [fo.getlength(t) for fo, t in zip(fonts, texts)]
        maxw = self.W * 0.86
        lines, cur, cw = [], [], 0
        for i, wd in enumerate(widths):
            if cur and cw + space + wd > maxw: lines.append(cur); cur, cw = [], 0
            cw += (space if cur else 0) + wd; cur.append(i)
        if cur: lines.append(cur)
        lh = int(f.size * 1.32); pad = int(28 * s)
        W_ = int(max(sum(widths[i] for i in ln) + space * (len(ln) - 1) for ln in lines)) + 2 * pad
        H_ = lh * len(lines) + 2 * pad
        im = Image.new("RGBA", (W_, H_), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        acc = self.accent; kwcol = (255, 214, 74)
        if st == "boxed":  # each line on a dark bar
            for li, ln in enumerate(lines):
                lw = sum(widths[i] for i in ln) + space * (len(ln) - 1); x0b = (W_ - lw) / 2 - 22 * s; y0b = pad + li * lh - 6 * s
                d.rounded_rectangle((x0b, y0b, x0b + lw + 44 * s, y0b + lh - 2 * s), radius=int(12 * s), fill=(10, 16, 32, 215))
        elif st == "bubble":  # one white speech bubble behind the whole line
            d.rounded_rectangle((pad * 0.35, pad * 0.45, W_ - pad * 0.35, H_ - pad * 0.45), radius=int(min(H_ * 0.3, 40 * s)), fill=(255, 255, 255, 245))
        for li, ln in enumerate(lines):
            lw = sum(widths[i] for i in ln) + space * (len(ln) - 1)
            x = (W_ - lw) / 2; y = pad + li * lh
            if self.rtl:  # Arabic line: right-to-left, but a run of English words ("chest CT") keeps its own order
                runs = []
                for i in ln:
                    latin = not K.is_ar(texts[i]) and bool(re.search(r"[A-Za-z0-9]", texts[i]))
                    if runs and runs[-1][0] == latin: runs[-1][1].append(i)
                    else: runs.append([latin, [i]])
                order = [i for latin, ids in reversed(runs) for i in (ids if latin else reversed(ids))]
            else:
                order = ln
            xpos, xx = {}, x
            for i in order: xpos[i] = xx; xx += widths[i] + space
            for i in ln:
                t, fo = texts[i], fonts[i]
                x = xpos[i]
                yy = y - (fo.size - f.size) * 0.6
                is_kw = g["words"][i]["kw"]
                if st == "pop":
                    col = acc if i == active else (kwcol if is_kw else (255, 255, 255))
                    d.text((x, yy), t, font=fo, fill=col + (255,), stroke_width=int(8 * s), stroke_fill=(0, 0, 0, 255))
                elif st == "highlight":
                    if i == active:
                        bb = d.textbbox((x, yy), t, font=fo)
                        d.rounded_rectangle((bb[0] - 12 * s, bb[1] - 8 * s, bb[2] + 12 * s, bb[3] + 10 * s), radius=int(14 * s), fill=acc + (255,))
                    col = (255, 255, 255) if (i == active or not is_kw) else kwcol
                    d.text((x, yy), t, font=fo, fill=col + (255,), stroke_width=int(4 * s), stroke_fill=(0, 0, 0, 230))
                elif st == "outline":  # classic social subtitles: white, thick black outline, the spoken word a bit bigger
                    col = kwcol if is_kw else (255, 255, 255)
                    d.text((x, yy), t, font=fo, fill=col + (255,), stroke_width=int(10 * s), stroke_fill=(0, 0, 0, 255))
                elif st == "karaoke":  # words fill with colour as they are spoken
                    col = acc if i <= active else (255, 255, 255)
                    d.text((x, yy), t, font=fo, fill=col + (255,), stroke_width=int(6 * s), stroke_fill=(0, 0, 0, 255))
                elif st == "neon":
                    col = (255, 255, 255) if i != active else _tint(acc, 0.55)
                    d.text((x, yy), t, font=fo, fill=col + (255,), stroke_width=max(1, int(2 * s)), stroke_fill=acc + (255,))
                elif st == "bubble":
                    col = acc if i == active else (27, 36, 56)
                    d.text((x, yy), t, font=fo, fill=col + (255,))
                elif st == "typewriter":
                    if i > active: continue  # not said yet: space is kept so the line doesn't jump
                    col = acc if i == active else (kwcol if is_kw else (255, 255, 255))
                    d.text((x, yy), t, font=fo, fill=col + (255,), stroke_width=int(5 * s), stroke_fill=(0, 0, 0, 255))
                elif st == "boxed":
                    col = acc if i == active else (kwcol if is_kw else (255, 255, 255))
                    d.text((x, yy), t, font=fo, fill=_tint(col, 0.15) + (255,))
                else:
                    col = kwcol if is_kw else (255, 255, 255)
                    d.text((x, yy), t, font=fo, fill=col + (255,), stroke_width=max(1, int(2 * s)), stroke_fill=(0, 0, 0, 160))
        arr = np.array(im)
        if st == "neon":  # coloured glow under the letters
            a = arr[..., 3].astype(np.float32)
            gl = cv2.GaussianBlur(a, (0, 0), 9 * s) * 1.6 + cv2.GaussianBlur(a, (0, 0), 3 * s)
            out = np.zeros_like(arr, np.float32); out[..., :3] = acc; out[..., 3] = np.clip(gl, 0, 255)
            fa_ = arr[..., 3:4] / 255.0
            out[..., :3] = arr[..., :3] * fa_ + out[..., :3] * (1 - fa_); out[..., 3] = np.maximum(out[..., 3], arr[..., 3])
            arr = out.astype(np.uint8)
        if st in ("minimal", "highlight", "bubble", "boxed"):  # soft drop shadow
            a = arr[..., 3].astype(np.float32)
            sh = cv2.GaussianBlur(a, (0, 0), 6 * s)
            sh = np.roll(sh, int(4 * s), axis=0)
            base = np.zeros_like(arr); base[..., 3] = np.clip(sh * (0.75 if st == "minimal" else 0.5), 0, 255).astype(np.uint8)
            out = base.astype(np.float32); fa_ = arr[..., 3:4] / 255.0
            out[..., :3] = arr[..., :3] * fa_; out[..., 3] = np.maximum(base[..., 3], arr[..., 3])
            arr = out.astype(np.uint8)
            arr[..., :3] = np.where(arr[..., 3:4] > 0, arr[..., :3], 0)
        self.cache[key] = arr
        return arr

    def draw(self, canvas, T, avoid, force_top=None, layout_key=None):
        """Draws the current group; returns its (top, bottom) in px, or None. The position is re-chosen
        when the layout changes mid-line (a graphic appearing moves the face)."""
        H, W = canvas.shape[:2]
        for gi, g in enumerate(self.groups):
            if not (g["s"] <= T < g["e"]): continue
            if layout_key != "keep" and g.get("lk", "∅") != layout_key: g["y"] = None; g["lk"] = layout_key
            active = max([i for i, w in enumerate(g["words"]) if w["s"] <= T + 0.03] or [0])
            spr = self.sprite(gi, active)
            if force_top is not None:
                al = ease_out_cubic((T - g["s"]) / 0.2)
                blit(canvas, spr, (W - spr.shape[1]) / 2, force_top, al)
                return (force_top, force_top + spr.shape[0])
            if g["y"] is None:
                h = spr.shape[0] / H; wf = spr.shape[1] / W
                def ov(b, r): return max(0, min(b[2], r[2]) - max(b[0], r[0])) * max(0, min(b[3], r[3]) - max(b[1], r[1]))
                best = None
                for k, yc in enumerate((0.775, 0.70, 0.83, 0.62, 0.89, 0.92, 0.50, 0.54, 0.46, 0.42, 0.38, 0.30)):
                    box = (0.5 - wf / 2, yc - h / 2, 0.5 + wf / 2, yc + h / 2)
                    if box[1] < 0.02 or box[3] > 0.97: continue
                    pen = sum(ov(box, r[:4]) * (r[4] if len(r) > 4 else 1.0) for r in avoid) + k * 1e-4
                    if best is None or pen < best[0]: best = (pen, yc)
                g["y"] = best[1] if best else 0.775
            age = T - g["s"]
            st = self.style
            if st in ("pop", "outline"):
                sc = 0.78 + 0.22 * ease_out_back(age / 0.16); spr2 = scale_rgba(spr, sc); al = 1
            elif st == "bubble":
                sc = 0.9 + 0.1 * ease_out_back(age / 0.2); spr2 = scale_rgba(spr, sc); al = ease_out_cubic(age / 0.12)
            elif st in ("highlight", "boxed"):
                spr2 = spr; al = ease_out_cubic(age / 0.12)
            elif st == "typewriter":
                spr2 = spr; al = 1
            elif st == "neon":
                spr2 = spr; al = ease_out_cubic(age / 0.2) * (0.75 + 0.25 * (age > 0.08 or (age * 40) % 2 > 1))  # tiny flicker as it lights up
            else:
                spr2 = spr; al = ease_out_cubic(age / 0.2)
            dy = (1 - ease_out_cubic(age / 0.15)) * 18 * self.s if st in ("highlight", "boxed", "karaoke") else 0
            top = g["y"] * H - spr2.shape[0] / 2 + dy
            blit(canvas, spr2, (W - spr2.shape[1]) / 2, top, al)
            return (top, top + spr2.shape[0])
        return None


def _tint(c, k): return tuple(int(v + (255 - v) * k) for v in c)


# ---------------- branding sprites ----------------
def _initials(name):
    if K.is_ar(name):
        parts = re.sub(r"^(د\.?|الدكتور|دكتور)\s*", "", name.strip()).split()
        last = parts[-1] if parts else name
        return re.sub(r"^ال", "", last)[:1] or last[:1]
    parts = [p for p in re.sub(r"^(dr\.?|pr\.?|prof\.?|د\.?|الدكتور|دكتور)\s+", "", name.strip(), flags=re.I).split() if p[0].isalpha()]
    return ((parts[0][0] + parts[-1][0]) if len(parts) > 1 else (parts[0][:2] if parts else "MD")).upper()


def _logo(brand, size, accent, sec):
    if brand.get("logo") and os.path.exists(brand["logo"]):
        im = Image.open(brand["logo"]).convert("RGBA"); im.thumbnail((size, size))
        return np.array(im)
    c = np.zeros((size, size, 4), np.uint8)
    cv2.circle(c, (size // 2, size // 2), size // 2 - 1, accent + (255,), -1, cv2.LINE_AA)
    t = text_sprite(_initials(brand["name"]), font("Poppins", 700, int(size * 0.4)), (255, 255, 255), pad=0)
    blit(c, t, (size - t.shape[1]) / 2, (size - t.shape[0]) / 2 - size * 0.02)
    return c


def _lower_third(brand, s, accent, rtl=False):
    n = text_sprite(brand["name"], font(brand.get("font", "Poppins"), 700, int(52 * s)), (255, 255, 255), pad=0)
    sub = brand["specialty"] + (f"  ·  {brand['clinic']}" if brand.get("clinic") else "")
    sp = text_sprite(sub, fit_font(sub, "Poppins", 500, int(32 * s), 900 * s), (200, 215, 235), pad=0)
    pw, ph = max(n.shape[1], sp.shape[1]) + int(76 * s), n.shape[0] + sp.shape[0] + int(58 * s)
    card = np.zeros((ph, pw, 4), np.uint8); card[..., :3] = (10, 20, 38); card[..., 3] = (rounded_mask(pw, ph, int(22 * s)) * 0.86).astype(np.uint8)
    if rtl:
        cv2.rectangle(card, (pw - int(30 * s), int(24 * s)), (pw - int(22 * s), ph - int(24 * s)), accent + (255,), -1)
        blit(card[..., :3], n, pw - int(50 * s) - n.shape[1], int(22 * s)); blit(card[..., :3], sp, pw - int(50 * s) - sp.shape[1], int(30 * s) + n.shape[0])
    else:
        cv2.rectangle(card, (int(22 * s), int(24 * s)), (int(30 * s), ph - int(24 * s)), accent + (255,), -1)
        blit(card[..., :3], n, int(50 * s), int(22 * s)); blit(card[..., :3], sp, int(50 * s), int(30 * s) + n.shape[0])
    return card


def _title_lines(text, s, W, accent, kind="classic"):
    """Opening title sprites in one of the title designs."""
    size = {"big": 104, "boxed": 74, "underline": 72}.get(kind, 68)
    f = fit_font(text, "Poppins", 700, int(size * s), W * 1.7)
    words, lines, cur = text.split(), [], ""
    lim = W * (0.86 if kind == "big" else 0.82)
    for w in words:
        if f.getlength((cur + " " + w).strip()) > lim and cur: lines.append(cur); cur = w
        else: cur = (cur + " " + w).strip()
    if cur: lines.append(cur)
    lines = lines[:4 if kind == "big" else 3]
    out = []
    if kind == "boxed":  # one solid block
        ts = [text_sprite(ln, f, (255, 255, 255), pad=0) for ln in lines]
        pw = max(t.shape[1] for t in ts) + int(70 * s); ph = sum(t.shape[0] for t in ts) + int(14 * s) * (len(ts) - 1) + int(56 * s)
        c = np.zeros((ph, pw, 4), np.uint8); c[..., :3] = accent; c[..., 3] = rounded_mask(pw, ph, int(26 * s))
        cv2.rectangle(c, (int(26 * s), ph - int(18 * s)), (pw - int(26 * s), ph - int(12 * s)), (255, 255, 255, 200), -1)
        yy = int(26 * s)
        for t in ts: blit(c, t, (pw - t.shape[1]) / 2, yy); yy += t.shape[0] + int(14 * s)
        return [c]
    for i, ln in enumerate(lines):
        if kind in ("big", "underline"):  # bare words with a soft shadow; the first line in the accent colour for 'big'
            col = accent if (kind == "big" and i == 0) else (255, 255, 255)
            out.append(text_sprite(ln.upper() if kind == "big" and not K.is_ar(ln) else ln, f, col, stroke=int(3 * s) if kind == "big" else 0, shadow=max(3, int(7 * s))))
            continue
        t = text_sprite(ln, f, (255, 255, 255), pad=0)
        pw, ph = t.shape[1] + int(40 * s), t.shape[0] + int(26 * s)
        c = np.zeros((ph, pw, 4), np.uint8)
        c[..., :3] = accent if i == 0 else (10, 20, 38); c[..., 3] = (rounded_mask(pw, ph, int(16 * s)) * (1 if i == 0 else .9)).astype(np.uint8)
        blit(c[..., :3], t, int(20 * s), int(12 * s)); out.append(c)
    return out


def _frame_layouts(kind):
    """Layouts adjusted to the frame design: 'card' shows the doctor in a rounded card over the background."""
    L = {k: dict(v) for k, v in LAYOUT.items()}
    if kind == "card":
        for k in ("full", "side", "labels"):
            L[k]["doc"] = (0.045, 0.035, 0.91, 0.885); L[k]["dr"] = 0.055
    return L


_BG = {}


def _background(W, H, sec, accent, kind):
    key = (W, H, tuple(sec), tuple(accent), kind)
    if key in _BG: return _BG[key]
    c1 = np.array(sec, np.float32)
    if kind == "plain":
        img = np.ones((H, W, 3), np.float32) * c1
    else:
        y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        img = (c1 * (1 - y * 0.55)) * np.ones((1, W, 1), np.float32)
        yy, xx = np.mgrid[0:H:4, 0:W:4]
        glow = cv2.resize(np.exp(-(((xx - W * .5) / (W * .6)) ** 2 + ((yy - H * .2) / (H * .3)) ** 2)).astype(np.float32), (W, H))
        img += glow[..., None] * np.array(accent, np.float32) * 0.22
    s = W / 1080
    if kind == "grid":
        step = int(64 * s); ov = img.copy()
        for x in range(0, W, step): cv2.line(ov, (x, 0), (x, H), accent, 1)
        for y_ in range(0, H, step): cv2.line(ov, (0, y_), (W, y_), accent, 1)
        img = img * 0.9 + ov * 0.1
    elif kind == "dots":
        step = int(46 * s); ov = img.copy()
        for y_ in range(step // 2, H, step):
            for x in range(step // 2, W, step): cv2.circle(ov, (x, y_), max(1, int(3 * s)), (255, 255, 255), -1, cv2.LINE_AA)
        img = img * 0.88 + ov * 0.12
    out = np.clip(img, 0, 255).astype(np.uint8)
    if len(_BG) > 6: _BG.clear()
    _BG[key] = out
    return out


def _glitch(canvas, T, s):
    """A few frames of digital glitch: RGB split and shifted slices."""
    H, W = canvas.shape[:2]; rng = np.random.default_rng(int(T * 1000))
    k = int(14 * s)
    canvas[..., 0] = np.roll(canvas[..., 0], k, axis=1); canvas[..., 2] = np.roll(canvas[..., 2], -k, axis=1)
    for _ in range(5):
        y0 = int(rng.integers(0, H - 20)); h = int(rng.integers(int(10 * s), int(70 * s)))
        canvas[y0:y0 + h] = np.roll(canvas[y0:y0 + h], int(rng.integers(-40, 40) * s), axis=1)


def _outro(brand, W, H, T, accent, sec, logo, lang="en"):
    s = W / 1080
    y = np.linspace(0, 1, H)[:, None, None]
    c1 = np.array(sec, np.float32); c2 = c1 * 0.45
    img = (c1 * (1 - y) + c2 * y) * np.ones((1, W, 1))
    yy, xx = np.mgrid[0:H:4, 0:W:4]
    glow = np.exp(-(((xx - W * .5) / (W * .55)) ** 2 + ((yy - H * .36) / (H * .25)) ** 2)).astype(np.float32)
    glow = cv2.resize(glow, (W, H))
    img = np.clip(img + glow[..., None] * np.array(accent) * 0.35, 0, 255).astype(np.uint8)
    items = []
    lg = scale_rgba(logo, (220 * s) / logo.shape[1]); items.append((lg, 0.0))
    items.append((text_sprite(brand["name"], fit_font(brand["name"], "Poppins", 700, int(72 * s), W * .86), (255, 255, 255), pad=0), 0.1))
    items.append((text_sprite(brand["specialty"], fit_font(brand["specialty"], "Poppins", 500, int(38 * s), W * .86), (190, 210, 235), pad=0), 0.18))
    if brand.get("clinic"): items.append((text_sprite(brand["clinic"], fit_font(brand["clinic"], "Poppins", 500, int(36 * s), W * .86), accent, pad=0), 0.24))
    if brand.get("booking") or brand.get("website"):
        cta = P.UI_TEXT.get(lang, P.UI_TEXT["en"])["cta"]
        t = text_sprite(cta, font_for(cta, "Poppins", 700, int(44 * s)), (255, 255, 255), pad=0)
        pw, ph = t.shape[1] + int(90 * s), t.shape[0] + int(48 * s)
        b = np.zeros((ph, pw, 4), np.uint8); b[..., :3] = accent; b[..., 3] = rounded_mask(pw, ph, ph // 2)
        blit(b[..., :3], t, int(45 * s), int(24 * s)); items.append((b, 0.34))
        link = brand.get("booking") or brand.get("website")
        items.append((text_sprite(link, fit_font(link, "Inter", 500, int(34 * s), W * .86), (220, 230, 245), pad=0), 0.4))
    if brand.get("phone"): items.append((text_sprite(brand["phone"], font("Inter", 500, int(34 * s)), (220, 230, 245), pad=0), 0.44))
    total_h = sum(i[0].shape[0] for i in items) + int(36 * s) * (len(items) - 1)
    yv = H * 0.40 - total_h / 2
    for spr, d in items:
        a = ease_out_cubic((T - d) / 0.45)
        blit(img, spr, (W - spr.shape[1]) / 2, yv + (1 - a) * 40 * s, a)
        yv += spr.shape[0] + int(36 * s)
    if brand.get("disclaimer"):
        dl = text_sprite(brand["disclaimer"], fit_font(brand["disclaimer"], "Inter", 400, int(28 * s), W * .86), (150, 165, 190), pad=0)
        blit(img, dl, (W - dl.shape[1]) / 2, H * 0.86, ease_out_cubic((T - 0.6) / 0.5))
    return img


# ---------------- audio ----------------
def sfx_events(plan, tl):
    """When sounds happen, in output seconds: a whoosh as each visual scene enters, a pop for icon cards and
    keyword pop-ups, an impact under the opening title, a shimmer on the end card."""
    ev = []
    if plan["title"].get("enabled") and plan["title"].get("text"): ev.append((0.08, "impact"))
    last = -9.0
    for sc in sorted(plan["scenes"], key=lambda x: x["start"]):
        if not sc.get("enabled", True): continue
        a, b = tl.to_out(sc["start"]), tl.to_out(sc["end"])
        if b - a < 0.8: continue
        if sc.get("layout") == "overlay" or sc["graphic"]["asset"] == "icons":
            for it in sc["graphic"].get("items", [])[:3]: ev.append((a + it.get("at", 0) + 0.02, "pop"))
        elif a - last > 1.2:
            ev.append((max(0.0, a - 0.22), "whoosh")); last = a
    for e in plan.get("emphasis", []):
        if e.get("enabled", True): ev.append((tl.to_out(e["start"]), "pop"))
    if plan["branding"].get("outro", True): ev.append((tl.duration + 0.05, "shimmer"))
    return sorted(ev)


def build_audio(plan, keeps, total, workdir, log):
    src = plan["source"]["path"]; out = os.path.join(workdir, "audio.m4a")
    au = plan["audio"]; parts = []
    for i, (a, b) in enumerate(keeps):
        d = b - a; fd = min(0.012, d / 4)
        parts.append(f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS,afade=t=in:d={fd:.3f},afade=t=out:st={d-fd:.3f}:d={fd:.3f}[a{i}]")
    chain = f"highpass=f={au.get('highpass', 80)}"
    if au.get("denoise", True): chain += ",afftdn=nr=12:nf=-35:tn=1"
    chain += ",acompressor=threshold=-21dB:ratio=3:attack=5:release=90:makeup=2"
    chain += f",loudnorm=I={au.get('loudnorm', -14)}:TP=-1.5:LRA=11,aresample=44100,aformat=channel_layouts=stereo"
    fg = ";".join(parts) + ";" + "".join(f"[a{i}]" for i in range(len(keeps))) + f"concat=n={len(keeps)}:v=0:a=1,{chain},apad=whole_dur={total:.3f}[voice]"
    inputs = ["-i", src]
    mus = plan["music"]
    fg += ";[voice]asplit=3[v1][vsc][vsc2]"
    if mus.get("enabled") and mus.get("track") in M.TRACKS:
        mp = os.path.join(workdir, f"music_v2_{mus['track']}_{int(total)+2}.wav")
        if not os.path.exists(mp): M.generate(mus["track"], total + 2, mp)
        inputs += ["-i", mp]
        # music bed clearly audible but under the voice (-14 LUFS): db -24 -> about -25 LUFS, ducked while speaking
        target = clamp(mus.get("db", -24) - 1, -36, -18)
        fg += (f";[1:a]atrim=0:{total:.3f},loudnorm=I={target:.1f}:TP=-3:LRA=7,aresample=44100,aformat=channel_layouts=stereo,"
               f"afade=t=in:d=0.6,afade=t=out:st={max(0, total-2.5):.3f}:d=2.5[m];"
               f"[m][vsc]sidechaincompress=threshold=0.03:ratio=2.5:attack=30:release=500[md];[v1][md]amix=inputs=2:normalize=0:duration=first[mixA]")
    else:
        fg += ";[vsc]anullsink;[v1]anull[mixA]"
    ev = sfx_events(plan, P.Timeline(keeps)) if plan.get("effects", {}).get("sfx", True) else []
    if ev:  # sound design: whooshes, pops, title impact (procedural, royalty-free), ducked under the voice so no word is masked
        sp = os.path.join(workdir, "sfx.wav"); SFX.render_track(ev, total, sp)
        k = len(inputs) // 2; inputs += ["-i", sp]
        fg += (f";[{k}:a]atrim=0:{total:.3f},aresample=44100,aformat=channel_layouts=stereo[fx0];"
               f"[fx0][vsc2]sidechaincompress=threshold=0.02:ratio=6:attack=2:release=250[fx];[mixA][fx]amix=inputs=2:normalize=0:duration=first[aout]")
    else:
        fg += ";[vsc2]anullsink;[mixA]anull[aout]"
    cmd = ["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", fg, "-map", "[aout]", "-t", f"{total:.3f}", "-c:a", "aac", "-b:a", "192k", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: raise RuntimeError("audio: " + r.stderr[-800:])
    return out


# ---------------- presenter-less background (AI-voice videos) ----------------
_AMB = {}


def _ambient(w, h, T, accent, sec):
    """Slowly moving brand-coloured glows on a dark gradient; cheap (computed at 1/8 size)."""
    key = (w, h, accent, sec)
    if key not in _AMB:
        y = np.linspace(0, 1, h // 8)[:, None, None]
        base = (np.array(sec, np.float32) * (1 - y) + np.array(sec, np.float32) * 0.35 * y) * np.ones((1, w // 8, 1))
        yy, xx = np.mgrid[0:h // 8, 0:w // 8].astype(np.float32)
        _AMB[key] = (base, yy / (h // 8), xx / (w // 8))
    base, yy, xx = _AMB[key]
    img = base.copy()
    for k, (ax, ay, sp, rad, amp) in enumerate(((0.25, 0.30, 0.11, 0.35, 0.55), (0.75, 0.65, 0.07, 0.42, 0.40), (0.5, 0.95, 0.05, 0.30, 0.30))):
        cx = ax + 0.12 * np.sin(T * sp * 2 * np.pi + k); cy = ay + 0.08 * np.cos(T * sp * 1.6 * np.pi + k * 2)
        g = np.exp(-(((xx - cx) / rad) ** 2 + ((yy - cy) / (rad * 0.56)) ** 2))
        img += g[..., None] * np.array(accent, np.float32) * amp
    return cv2.resize(np.clip(img, 0, 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)


# ---------------- main render ----------------
def render(plan, out_path, workdir, preview=False, progress=None, log=print):
    t_start = time.time()
    os.makedirs(workdir, exist_ok=True)
    W, H, fps = plan["output"]["width"], plan["output"]["height"], plan["output"]["fps"]
    if preview: W, H, fps = W // 2, H // 2, 24
    s = W / 1080
    G.UI_SCALE = s
    brand = plan["branding"]; _acc, _sec = P.colors(plan); accent = hex2rgb(_acc); sec = hex2rgb(_sec)
    DS = plan.get("design", {}); frame_kind = DS.get("frame", "full"); tstyle = DS.get("transition", "flash")
    try:  # never cut through a word, even when the transcript timing is a little off
        from . import analyze as AN, asr as ASR
        n = AN.snap_cuts(plan["cuts"], ASR.load_audio(plan["source"]["path"]))
        if n: log(f"Moved {n} cut edge(s) to real silence so no word is clipped.")
    except Exception as e: log(f"cut snapping skipped: {e}")
    keeps = P.keep_segments(plan); tl = P.Timeline(keeps)
    _sp = P.STYLES.get(plan.get("style"), {})
    fxk = float(plan["style_params"].get("fx", _sp.get("fx", 1.0)))  # motion-graphics strength of this style
    fxflash = 1.0 if plan["style_params"].get("flash", _sp.get("flash", True)) else 0.0
    outro_d = 3.0 if brand.get("outro", True) else 0.0
    total = tl.duration + outro_d
    nframes = int(total * fps)
    audio = build_audio(plan, keeps, total, workdir, log)

    src = plan["source"]; SW, SH = src["width"], src["height"]
    dscale = min(1.0, (H * 1.0) / SH) if SH >= SW else min(1.0, (H * 1.0) / SH)
    DW, DH = int(SW * dscale) // 2 * 2, int(SH * dscale) // 2 * 2
    grade = GRADES.get(plan.get("grade", "natural"), GRADES["natural"])
    luma = plan["analysis"]["qa"].get("luma_mean", 110)
    if luma < 85: grade = f"eq=gamma={min(1.35, 1 + (85 - luma) / 100):.2f}," + grade
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src["path"], "-vf", f"fps={fps},scale={DW}:{DH},{grade}",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=DW * DH * 3 * 4)
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                            "-i", audio, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast" if not preview else "ultrafast",
                            "-crf", "20" if not preview else "26", "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart",
                            "-t", f"{total:.3f}", out_path], stdin=subprocess.PIPE)
    fsize = DW * DH * 3
    cur_idx, cur_frame = -1, None

    def read_until(idx):
        nonlocal cur_idx, cur_frame
        while cur_idx < idx:
            buf = dec.stdout.read(fsize)
            if len(buf) < fsize: break
            cur_frame = np.frombuffer(buf, np.uint8).reshape(DH, DW, 3); cur_idx += 1
        return cur_frame

    cp = plan["captions"]; mode = cp.get("mode", "original")
    lang = plan.get("language", "en")
    has_tr = any((v or "").strip() for v in cp.get("translations", {}).values())
    if mode != "original" and not has_tr:
        log("caption translation requested but no translations in the plan — using original-language captions"); mode = "original"
    caps = caps2 = None
    if cp["enabled"]:
        caps = Captions(plan, tl, W, H, accent, source="translated" if mode == "translated" else "original")
        if mode == "both": caps2 = Captions(plan, tl, W, H, accent, source="translated", style="minimal", scale=0.72)
        base = os.path.splitext(out_path)[0]
        open(base + ".srt", "w").write(caps.srt())
        if caps2: open(base + "." + cp.get("target", "en") + ".srt", "w").write(caps2.srt())
    glang = cp.get("target", "en") if mode == "translated" else lang
    rtl = glang == "ar"
    scenes = []
    for sc in plan["scenes"]:
        if not sc.get("enabled", True): continue
        a, b = tl.to_out(sc["start"]), tl.to_out(sc["end"])
        if b - a > 0.8: scenes.append((a, b, sc))
    emph = [(tl.to_out(e["start"]), tl.to_out(e["end"])) for e in plan.get("emphasis", []) if e.get("enabled", True)]
    kw_pills = [(tl.to_out(e["start"]), e["term"]) for e in plan.get("emphasis", []) if e.get("enabled", True) and e.get("term")]
    pill_cache = {}
    GP.PHOTO_DIR = os.path.join(os.path.dirname(os.path.abspath(src["path"])), "photos")
    from . import motion as MO
    MO.GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(src["path"])), "anims")
    GP.BROLL_DIR = os.path.join(os.path.dirname(os.path.abspath(src["path"])), "broll"); GP._READERS.clear()
    ov_y = {}
    zp = plan["zoom"]; tr = plan["style_params"].get("transition", 0.4)
    logo = _logo(brand, int(150 * s), accent, sec)
    small_logo = scale_rgba(logo, (92 * s) / logo.shape[1])
    lower = _lower_third(brand, s, accent, K.is_ar(brand["name"]) or rtl) if brand.get("lower_third", True) else None
    tkind = DS.get("title", "classic")
    title = _title_lines(plan["title"]["text"], s, W, accent, tkind) if plan["title"].get("enabled") and plan["title"].get("text") else []
    bg = _background(W, H, sec, accent, DS.get("bg", "plain"))
    LYL = _frame_layouts(frame_kind); LYR = {k: _mirror(v) for k, v in LYL.items()}
    presenter = plan.get("presenter", True)
    wm = text_sprite("Made with MedReel", font("Poppins", 700, int(30 * s)), (255, 255, 255), shadow=max(2, int(4 * s))) if plan["output"].get("watermark") else None
    last_doc = None; lower_y = None
    for n in range(nframes):
        T = n / fps
        if T >= tl.duration:  # outro
            o = _outro(brand, W, H, T - tl.duration, accent, sec, logo, glang)
            if wm is not None: blit(o, wm, (W - wm.shape[1]) / 2, H - wm.shape[0] - 120 * s, 0.9)
            if last_doc is not None and T - tl.duration < 0.35:
                m = (T - tl.duration) / 0.35; o = (last_doc * (1 - m) + o * m).astype(np.uint8)
            enc.stdin.write(o.tobytes()); continue
        src_t, seg = tl.to_src(T)
        frame = read_until(int(src_t * fps))
        if frame is None: frame = np.zeros((DH, DW, 3), np.uint8)
        face = _face_at(plan["face_track"], src_t)
        if not presenter:
            frame = _ambient(DW, DH, T, accent, sec).copy(); face = [0.35, 0.30, 0.30, 0.17]
            xs = np.arange(0, DW, 3); ph = (T * 0.9 + xs / DW * 3.2)  # faint ECG trace drifting behind the captions
            ys = DH * 0.40 - DH * 0.07 * G._ecg_wave(ph)
            ov = frame.copy(); cv2.polylines(ov, [np.stack([xs, ys], 1).astype(np.int32)], False, accent, max(2, int(4 * s)), cv2.LINE_AA)
            frame = cv2.addWeighted(ov, 0.28, frame, 0.72, 0)
        # active scene + transition mix
        LY = LYR if rtl else LYL
        lay, scn, m, ovs = LY["full"], None, 0.0, None
        for a, b, sc in scenes:
            if a - 0.001 <= T < b:
                if sc["layout"] == "overlay" or sc["graphic"]["asset"] == "icons":  # floating card: the layout stays as it is
                    ovs = ovs or (a, b, sc); continue
                if scn is not None: continue
                scn = sc; lt = sc["layout"] if sc["layout"] in LAYOUT else "side"
                m = ease_in_out(min((T - a) / tr, (b - T) / tr, 1.0))
                lay = _lerp_layout(LY["full"], LY[lt], m)
        # zoom: alternate per jump-cut segment + slow push + keyword punch
        seg_a, seg_b = keeps[seg]; seg_out0 = tl.offsets[seg]
        z = 1 + (zp["intensity"] if seg % 2 and presenter else 0)
        z *= 1 + zp["push"] * clamp((T - seg_out0) / max(2.0, seg_b - seg_a), 0, 1)
        for ea, eb in emph:
            if ea <= T < eb: z *= 1 + zp["punch"] * min(ease_out_cubic((T - ea) / 0.18), ease_in_out((eb - T) / 0.3))
        fxon = plan.get("effects", {}).get("transitions", True)
        if fxon and scn is not None and tstyle in ("flash", "glitch"):  # camera punch as a visual arrives / leaves (strength depends on the style)
            sa, sb = next((a, b) for a, b, sc in scenes if sc is scn)
            z *= 1 + 0.06 * fxk * max(np.exp(-((T - sa) / 0.18) ** 2), np.exp(-((T - sb) / 0.18) ** 2))
        canvas = bg.copy()
        dx, dy, dw, dh = lay["doc"][0] * W, lay["doc"][1] * H, lay["doc"][2] * W, lay["doc"][3] * H
        gx, gy, gw, gh = lay["gfx"][0] * W, lay["gfx"][1] * H, lay["gfx"][2] * W, lay["gfx"][3] * H
        gfx_img, gfx_rgba = None, None
        if scn is not None and lay["ga"] > 0.01:
            g = scn["graphic"]; ga_, gb_ = next((a, b) for a, b, sc in scenes if sc is scn)
            gt, gd = T - ga_, gb_ - ga_
            if g["asset"] == "bp":
                gfx_rgba = G.bp_card(int(gw), int(gh), gt, gd, text=scn.get("text", ""), accent=accent, lang=glang)
            else:
                fn = G.ASSETS[g["asset"]]
                kw = {"title": g["title"]} if g.get("title") else {}
                if g["asset"] == "photo" and scn.get("caption"): kw = {"title": scn["caption"]}
                if g.get("fill"): kw["fill"] = True
                if g["asset"] == "gen": kw["panel_bg"] = sec
                gfx_img = fn(max(32, int(gw)), max(32, int(gh)), gt, gd, mode=g["mode"], accent=accent, lang=glang, **kw)
        doc_full = dw >= W * 0.99 and dh >= H * 0.99
        def draw_doc():
            if lay["da"] < 0.01 or dw < 8 or dh < 8: return
            x0, y0, cw, ch = _crop(frame, face, (dw, dh), lay["fx"], lay["fy"], lay["face"], z, lay["zm"])
            crop = frame[int(y0):int(y0 + ch), int(x0):int(x0 + cw)]
            im = cv2.resize(crop, (int(round(dw)), int(round(dh))), interpolation=cv2.INTER_LINEAR if cw < dw else cv2.INTER_AREA)
            r = lay["dr"] * W
            if doc_full and r < 1:
                canvas[int(dy):int(dy) + im.shape[0], int(dx):int(dx) + im.shape[1]] = im[: H - int(dy), : W - int(dx)]
            else:
                mask = rounded_mask(im.shape[1], im.shape[0], r)
                if r > 1: shadow_rect(canvas, dx, dy, im.shape[1], im.shape[0], r, 0.5, int(18 * s))
                blit(canvas, im, dx, dy, lay["da"], mask)
                if r > im.shape[1] * 0.4:
                    cv2.circle(canvas, (int(dx + im.shape[1] / 2), int(dy + im.shape[0] / 2)), int(im.shape[1] / 2), accent, max(2, int(7 * s)), cv2.LINE_AA)
        def draw_gfx():
            if gfx_img is None and gfx_rgba is None: return
            r = lay["gr"] * W
            if gfx_rgba is not None:
                shadow_rect(canvas, gx, gy, gfx_rgba.shape[1], gfx_rgba.shape[0], r, 0.5, int(24 * s))
                blit(canvas, gfx_rgba, gx, gy + (1 - m) * 60 * s, lay["ga"] * m)
                return
            full = gw >= W * 0.99 and gh >= H * 0.99
            img_, x_, y_, extra = gfx_img, gx, gy, None
            if fxon and scn is not None and tstyle in ("slide", "zoom", "wipe"):
                sa_, sb_ = next((a, b) for a, b, sc in scenes if sc is scn)
                ph_ = min(ease_out_cubic((T - sa_) / 0.5), ease_in_out((sb_ - T) / 0.35), 1.0)
                if tstyle == "slide":
                    x_ += (1 - ph_) * W * 0.7 * (-1 if rtl else 1)
                elif tstyle == "zoom" and ph_ < 0.999:
                    k_ = 0.8 + 0.2 * ph_; nw, nh = max(8, int(img_.shape[1] * k_)), max(8, int(img_.shape[0] * k_))
                    img_ = cv2.resize(img_, (nw, nh), interpolation=cv2.INTER_AREA)
                    x_ += (gfx_img.shape[1] - nw) / 2; y_ += (gfx_img.shape[0] - nh) / 2; r = r * k_
                elif tstyle == "wipe" and ph_ < 0.999:
                    extra = np.zeros(img_.shape[:2], np.uint8)
                    cv2.circle(extra, (img_.shape[1] // 2, img_.shape[0] // 2), int(ph_ * 0.6 * math.hypot(*img_.shape[:2])), 255, -1, cv2.LINE_AA)
            if not full and r > 0.5:
                shadow_rect(canvas, x_, y_, img_.shape[1], img_.shape[0], r, 0.55, int(26 * s))
                mask = rounded_mask(img_.shape[1], img_.shape[0], r)
                if extra is not None: mask = np.minimum(mask, extra)
                blit(canvas, img_, x_, y_, lay["ga"], mask)
            else:
                blit(canvas, img_, x_, y_, lay["ga"], extra)
        if gw * gh > dw * dh: draw_gfx(); draw_doc()
        else: draw_doc(); draw_gfx()
        if scn is not None and scn["layout"] == "split" and m > 0.01:
            cv2.line(canvas, (0, int(H * .5)), (int(W * m), int(H * .5)), accent, max(2, int(8 * s)))
        if fxon and scn is not None and (gfx_img is not None or gfx_rgba is not None):
            sa, sb = next((a, b) for a, b, sc in scenes if sc is scn)
            u_ = (T - sa - 0.3) / 0.7  # diagonal light sweep across the visual, once, after it lands
            if 0 < u_ < 1 and gw > 8 and gh > 8:
                x0_, y0_, x1_, y1_ = int(max(0, gx)), int(max(0, gy)), int(min(W, gx + gw)), int(min(H, gy + gh))
                if x1_ > x0_ and y1_ > y0_:
                    yy_, xx_ = np.mgrid[y0_:y1_, x0_:x1_].astype(np.float32)
                    pos = (xx_ - x0_) + (yy_ - y0_) * 0.45
                    c_ = -0.3 * (x1_ - x0_) + u_ * 1.9 * (x1_ - x0_)
                    band = np.exp(-((pos - c_) / (0.07 * (x1_ - x0_))) ** 2) * 0.22 * min(1.3, fxk) * np.sin(np.pi * u_)
                    roi = canvas[y0_:y1_, x0_:x1_].astype(np.float32)
                    canvas[y0_:y1_, x0_:x1_] = np.clip(roi + band[..., None] * 255, 0, 255).astype(np.uint8)
            if tstyle == "flash":
                fl = 0.30 * fxk * fxflash * np.exp(-((T - sa) / 0.07) ** 2)  # quick flash on the cut
                if fl > 0.01: canvas[:] = np.clip(canvas.astype(np.float32) * (1 - fl) + 255 * fl, 0, 255).astype(np.uint8)
            elif tstyle == "glitch" and (0 <= T - sa < 0.22 or 0 <= sb - T < 0.12):
                _glitch(canvas, T, s)
        if frame_kind == "border":  # thin accent frame around the whole video
            mg = int(22 * s); cv2.rectangle(canvas, (mg, mg), (W - mg, H - mg), accent, max(2, int(5 * s)), cv2.LINE_AA)
            for cx_, cy_ in ((mg, mg), (W - mg, mg), (mg, H - mg), (W - mg, H - mg)): cv2.circle(canvas, (cx_, cy_), int(9 * s), accent, -1, cv2.LINE_AA)
        # avoid-zones for captions: face (when visible) + floating graphic card
        avoid = []; face_box = None
        if frame_kind == "card": avoid.append((0, 0.905, 1, 1, 6.0))  # keep subtitles inside the card
        if not presenter:
            if scn is None: avoid.append((0, 0.6, 1, 1))  # captions sit mid-screen as kinetic type
        elif lay["da"] > 0.5:
            x0, y0, cw, ch = _crop(frame, face, (dw, dh), lay["fx"], lay["fy"], lay["face"], z, lay["zm"])
            fx0 = dx + (face[0] * DW - x0) / cw * dw; fy0 = dy + (face[1] * DH - y0) / ch * dh
            fx1 = fx0 + face[2] * DW / cw * dw; fy1 = fy0 + face[3] * DH / ch * dh
            face_box = (fx0 / W - .03, fy0 / H - .035, fx1 / W + .03, fy1 / H + .05)
            avoid.append(face_box + (12.0,))  # covering the face costs far more than covering a graphic
            if not doc_full: avoid.append((dx / W, dy / H, (dx + dw) / W, (dy + dh) / H))
        if (gfx_img is not None or gfx_rgba is not None) and not (gw >= W * .99 and gh >= H * .99):
            avoid.append((gx / W, gy / H, (gx + gw) / W, (gy + gh) / H))
        if gw >= W * .99 and gh >= H * .99 and gfx_img is not None:
            avoid.append((0, 0.12, 1, 0.80))  # main band of full-screen animations (slice, scale, scanner)
        # title (hook)
        if title and T < 3.4:
            ty = H * (0.085 if tkind == "big" else 0.115)
            if tkind == "big":  # darken the top so huge words read over any background
                k_ = 0.55 * ease_out_cubic(T / 0.3) * min(1, (3.4 - T) / 0.3)
                yb = int(H * 0.42); grad = np.linspace(1, 0, yb, dtype=np.float32)[:, None, None] * k_
                canvas[:yb] = (canvas[:yb].astype(np.float32) * (1 - grad)).astype(np.uint8)
            for i, spr in enumerate(title):
                a = ease_out_cubic((T - 0.15 - i * 0.12) / 0.35) * min(1, (3.4 - T) / 0.3)
                if tkind == "big":
                    d_ = 0.1 + i * 0.09; spr2 = scale_rgba(spr, 0.6 + 0.4 * ease_out_back((T - d_) / 0.28)); dyy = 0
                    a = clamp((T - d_) / 0.08, 0, 1) * min(1, (3.4 - T) / 0.3)
                elif plan["title"]["style"] == "pop" or tkind == "boxed": spr2 = scale_rgba(spr, 0.7 + 0.3 * ease_out_back((T - 0.15 - i * .1) / 0.3)); dyy = 0
                else: spr2 = spr; dyy = (1 - a) * 30 * s
                blit(canvas, spr2, (W - spr2.shape[1]) / 2, ty + dyy + (spr.shape[0] - spr2.shape[0]) / 2, a); ty += spr.shape[0] + 10 * s
            if plan.get("effects", {}).get("transitions", True) and tkind in ("classic", "underline", "big"):  # accent bar sweeps in under the title
                bw_ = int(W * 0.34 * ease_out_cubic((T - 0.45) / 0.4) * min(1, (3.4 - T) / 0.3))
                if bw_ > 4: cv2.rectangle(canvas, (int((W - bw_) / 2), int(ty + 4 * s)), (int((W + bw_) / 2), int(ty + 12 * s)), accent, -1)
                ty += 16 * s
            avoid.append((0, 0.1, 1, ty / H))
        # lower third
        if lower is not None and 0.6 <= T < 5.0 and (scn is None or m < 0.99):
            a = ease_out_cubic((T - 0.6) / 0.45) * min(1, (5.0 - T) / 0.35) * (1 - (m if scn is not None else 0))
            if lower_y is None:
                lh, lw = lower.shape[0] / H, lower.shape[1] / W
                lower_y = 0.86 if not presenter else 0.585
                if face_box is not None:
                    x0f = (1 - lw - 50 * s / W) if rtl else 50 * s / W
                    for yc in (0.585, 0.66, 0.72, 0.78, 0.50):
                        b = (x0f, yc, x0f + lw, yc + lh)
                        if not (b[0] < face_box[2] and face_box[0] < b[2] and b[1] < face_box[3] and face_box[1] < b[3]): lower_y = yc; break
            ly = H * lower_y
            lx = (W - 50 * s - lower.shape[1] + (1 - a) * 120 * s) if rtl else (50 * s - (1 - a) * 120 * s)
            blit(canvas, lower, lx, ly, a)
            avoid.append((0, ly / H, 1, (ly + lower.shape[0]) / H))
        if ovs is not None:  # icon card (any specialty), placed where it covers neither the face nor the title
            oa, ob, osc = ovs; items = osc["graphic"].get("items", [])
            cw_, ch_ = int(W * 0.88), int(H * (0.085 if len(items) <= 1 else 0.13))
            if not presenter: cw_, ch_ = int(W * 0.90), int(H * (0.17 if len(items) <= 1 else 0.24))  # no face: the card is the visual
            if osc["id"] not in ov_y:
                yy_ = 0.14
                if presenter:
                    for yc in (0.60, 0.66, 0.70, 0.13, 0.18, 0.715):
                        bx = (0.06, yc, 0.94, yc + ch_ / H)
                        if face_box is None or not (bx[0] < face_box[2] and face_box[0] < bx[2] and bx[1] < face_box[3] and face_box[1] < bx[3]):
                            yy_ = yc; break
                ov_y[osc["id"]] = yy_
            card = GP.icons_card(cw_, ch_, T - oa, ob - oa, items=items, accent=accent, lang=glang)
            yv = ov_y[osc["id"]] * H + (1 - ease_out_cubic((T - oa) / 0.3)) * 40 * s
            shadow_rect(canvas, (W - cw_) / 2, yv, cw_, ch_, int(min(ch_ * 0.28, 40 * s)), 0.35 * min(1, (T - oa) / 0.3), int(20 * s))
            blit(canvas, card, (W - cw_) / 2, yv)
            avoid.append((0.06, ov_y[osc["id"]], 0.94, ov_y[osc["id"]] + ch_ / H, 4.0))
        for ka, term in kw_pills:  # kinetic keyword: the medical term pops up as the doctor says it
            age = T - ka
            if not (0 <= age < 2.2) or scn is not None or ovs is not None: continue
            if term not in pill_cache:
                f_ = font_for(term, "Poppins", 800, int(56 * s))
                t_ = text_sprite(term, f_, (255, 255, 255), pad=0)
                pw, ph = t_.shape[1] + int(64 * s), t_.shape[0] + int(34 * s)
                pc = np.zeros((ph, pw, 4), np.uint8); pc[..., :3] = accent; pc[..., 3] = rounded_mask(pw, ph, ph // 2)
                blit(pc, t_, int(32 * s), int(17 * s)); pill_cache[term] = pc
            pc = pill_cache[term]
            al = ease_out_cubic(age / 0.18) * min(1, (2.2 - age) / 0.3)
            sc_ = 0.6 + 0.4 * ease_out_back(age / 0.28)
            spr_ = scale_rgba(pc, sc_)
            yk = None
            for yc in (0.15, 0.22, 0.64, 0.70):
                b = (0.5 - pc.shape[1] / W / 2, yc, 0.5 + pc.shape[1] / W / 2, yc + pc.shape[0] / H)
                if face_box is None or not (b[0] < face_box[2] and face_box[0] < b[2] and b[1] < face_box[3] and face_box[1] < b[3]):
                    if not (title and T < 3.6 and yc < 0.3): yk = yc; break
            if yk is None: continue
            blit(canvas, spr_, (W - spr_.shape[1]) / 2, yk * H + (pc.shape[0] - spr_.shape[0]) / 2, al)
            avoid.append((0, yk, 1, yk + pc.shape[0] / H)); break
        if caps:
            lk = None if (scn is None or m < 0.02) else scn["id"] if m > 0.97 else "keep"  # re-place once the layout has settled
            if lk != "keep" and ovs is not None: lk = f"{lk}|{ovs[2]['id']}"  # an icon card appearing also moves the subtitles
            placed = caps.draw(canvas, T, avoid, layout_key=lk)
            if caps2 and placed: caps2.draw(canvas, T, avoid, force_top=placed[1] - 6 * s)
        blit(canvas, small_logo, 40 * s, 52 * s, 0.85)
        if wm is not None: blit(canvas, wm, W - wm.shape[1] - 30 * s, H - wm.shape[0] - 40 * s, 0.8)
        if plan["style_params"].get("progress_bar"):
            cv2.rectangle(canvas, (0, 0), (int(W * T / tl.duration), int(8 * s)), accent, -1)
        last_doc = canvas
        enc.stdin.write(canvas.tobytes())
        if progress and n % 15 == 0: progress(n / nframes)
    enc.stdin.close(); enc.wait(); dec.stdout.close(); dec.kill()
    if enc.returncode: raise RuntimeError("encoder failed")
    if progress: progress(1.0)
    log(f"rendered {nframes} frames in {time.time()-t_start:.1f}s -> {out_path}")
    return {"path": out_path, "duration": total, "frames": nframes, "seconds": time.time() - t_start,
            "removed": round(src["duration"] - tl.duration, 2)}
