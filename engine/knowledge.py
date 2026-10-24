"""Cardiology knowledge layer (English + Lebanese/Levantine Arabic): term lexicon, graphic
matching rules, fillers, claim-risk patterns and ASR medical-term correction.
Specialty/language packs are plain data so new ones can be added without touching the engine."""
import re, unicodedata

# ------------------------------------------------------------------ text normalisation
AR_CHARS = re.compile(r"[؀-ۿ]")
AR_DIAC = re.compile(r"[ً-ْٰـ]")
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
CLITICS = ["وبال", "وال", "بال", "فال", "كال", "لل", "ال", "و", "ب", "ل", "ف"]


def is_ar(text: str) -> bool:
    return bool(AR_CHARS.search(text or ""))


def ar_norm(s: str) -> str:
    """Normalise Arabic for matching: no diacritics/tatweel, unified alef/ya/ta-marbuta, ASCII digits."""
    s = AR_DIAC.sub("", s.translate(AR_DIGITS))
    s = re.sub("[إأآٱ]", "ا", s).replace("ى", "ي").replace("ة", "ه").replace("ئ", "ي").replace("ؤ", "و")
    return s


def strip_clitic(w: str) -> str:
    for p in CLITICS:
        if w.startswith(p) and len(w) - len(p) >= 3:
            return w[len(p):]
    return w


def norm(w: str) -> str:
    """Matching key for one word (Latin or Arabic)."""
    if is_ar(w):
        return re.sub(r"[^؀-ۿ0-9%/]", "", ar_norm(w))
    w = "".join(c for c in unicodedata.normalize("NFKD", w.lower()) if not unicodedata.combining(c))  # é -> e
    return re.sub(r"[^a-z0-9%/']", "", w)


def core(w: str) -> str:
    """norm() minus Arabic clitics (و/ب/ل/ف/ال…): 'بالشرايين' -> 'شرايين'."""
    n = norm(w)
    return strip_clitic(n) if is_ar(n) else n


# ------------------------------------------------------------------ lexicons
FILLERS = {"um", "umm", "uh", "uhh", "erm", "er", "hmm", "mm", "ah", "eh",
           "امم", "اممم", "مم", "ممم", "ام", "اا", "ااا", "اه", "اهه", "ايه"}
FILLERS |= {"euh", "heu", "euhh", "hum", "bah"}
SOFT_FILLERS = {"يعني", "ya3ne", "ya3ni", "yaane", "so", "okay", "well", "oh", "ah", "ben", "bon", "voila", "enfin"}  # only when isolated
FILLER_PHRASES = [("you", "know"), ("i", "mean"), ("sort", "of"), ("kind", "of")]

# key (space-separated core tokens) -> display label
CARDIO_TERMS = {
    "cholesterol": "Cholesterol", "ldl": "LDL", "plaque": "Plaque", "plaques": "Plaque",
    "atherosclerosis": "Atherosclerosis", "coronary": "Coronary", "arteries": "Arteries",
    "artery": "Artery", "angioplasty": "Angioplasty", "stent": "Stent", "stents": "Stents",
    "heart attack": "Heart attack", "myocardial infarction": "Myocardial infarction",
    "angina": "Angina", "chest pain": "Chest pain", "blood pressure": "Blood pressure",
    "hypertension": "Hypertension", "clot": "Clot", "rupture": "Rupture", "ruptures": "Rupture",
    "ecg": "ECG", "ekg": "ECG", "electrocardiogram": "ECG", "arrhythmia": "Arrhythmia",
    "atrial fibrillation": "Atrial fibrillation", "heart muscle": "Heart muscle",
    "shortness of breath": "Shortness of breath", "statin": "Statin", "statins": "Statins",
    "diabetes": "Diabetes", "smoking": "Smoking", "exercise": "Exercise", "heart": "Heart",
    "valve": "Valve", "aortic stenosis": "Aortic stenosis", "tavr": "TAVR", "tavi": "TAVI",
    "echocardiography": "Echocardiography", "echo": "Echo", "blockage": "Blockage",
    # Lebanese / Levantine Arabic (normalised core forms)
    "كوليسترول": "الكوليسترول", "كولسترول": "الكوليسترول", "كولستيرول": "الكوليسترول",
    "شرايين تاجيه": "الشرايين التاجية", "شرايين": "الشرايين", "شريان": "الشريان",
    "تصلب شرايين": "تصلّب الشرايين", "تصلب": "التصلّب", "لويحه": "اللويحة", "ترسبات": "الترسبات",
    "جلطه": "الجلطة", "ذبحه قلبيه": "الذبحة القلبية", "ذبحه": "الذبحة", "سكته قلبيه": "الذبحة القلبية",
    "ستنت": "الستنت", "رسور": "الرسور", "ريسور": "الرسور", "دعامه": "الدعامة", "بالون": "البالون",
    "تمييل": "التمييل", "قسطره": "القسطرة", "ضغط دم": "ضغط الدم", "ضغط": "الضغط",
    "سكري": "السكري", "تدخين": "التدخين", "رياضه": "الرياضة", "وجع صدر": "وجع الصدر",
    "ضيق نفس": "ضيق النفس", "عضله قلب": "عضلة القلب", "تخطيط قلب": "تخطيط القلب",
    "رجفان": "الرجفان", "صمام": "الصمام", "قلب": "القلب",
    "calcium": "Calcium", "calcium score": "Calcium score", "agatston": "Agatston", "ct": "CT", "scanner": "CT scanner",
    "coronary ct": "Coronary CT", "ct scan": "CT scan", "كالسيوم": "الكالسيوم", "كلس": "الكلس", "تكلس": "التكلّس",
    "سكانر": "السكانر", "سكانير": "السكانر", "صوره مقطعيه": "الصورة المقطعية", "تضيق": "التضيّق",
}
# French pack (keys are normalised below, so accents are fine here)
_FR_TERMS = {"cholestérol": "Cholestérol", "plaque": "Plaque", "athérosclérose": "Athérosclérose", "artère": "Artère",
             "artères": "Artères", "artères coronaires": "Artères coronaires", "coronaire": "Coronaire", "sténose": "Sténose",
             "rétrécissement": "Rétrécissement", "caillot": "Caillot", "infarctus": "Infarctus", "crise cardiaque": "Crise cardiaque",
             "angioplastie": "Angioplastie", "ballon": "Ballon", "tension artérielle": "Tension artérielle",
             "pression artérielle": "Pression artérielle", "hypertension": "Hypertension", "douleur thoracique": "Douleur thoracique",
             "essoufflement": "Essoufflement", "muscle cardiaque": "Muscle cardiaque", "électrocardiogramme": "ECG",
             "fibrillation auriculaire": "Fibrillation auriculaire", "arythmie": "Arythmie", "diabète": "Diabète", "tabac": "Tabac",
             "statine": "Statine", "statines": "Statines", "calcium": "Calcium", "score calcique": "Score calcique",
             "coroscanner": "Coroscanner", "scanner coronaire": "Scanner coronaire", "valve": "Valve", "rétrécissement aortique": "Rétrécissement aortique"}
for _k, _v in _FR_TERMS.items():
    CARDIO_TERMS.setdefault(" ".join(norm(x) for x in _k.split()), _v)
LOW_VALUE_TERMS = {"Artère", "Artères", "Coronaire", "Heart", "القلب", "Exercise", "الرياضة", "Arteries", "Artery", "الشرايين", "الشريان", "الضغط"}

# canonical Arabic medical words used to repair ASR misspellings (normalised core forms)
AR_CORRECTIONS = {"كوليسترول": "كوليسترول", "شرايين": "شرايين", "تصلب": "تصلّب", "لويحه": "لويحة", "جلطه": "جلطة",
                  "ذبحه": "ذبحة", "ستنت": "ستنت", "دعامه": "دعامة", "بالون": "بالون", "قسطره": "قسطرة",
                  "تخطيط": "تخطيط", "رجفان": "رجفان", "انجيوبلاستي": "أنجيوبلاستي", "تاجيه": "تاجية",
                  "ترسبات": "ترسبات", "شريان": "شريان", "كالسيوم": "كالسيوم", "سكانر": "سكانر"}

# Graphic matching rules: (asset, mode, patterns, priority, default layout, label)
# patterns run on lower-cased English AND on ar_norm() Arabic text.
GRAPHIC_RULES = [
    ("coronary", "stent", [r"\bangioplast", r"\bstent", r"\bballoon", r"open (?:a|the) (?:narrowed |blocked )?arter",
                           r"ستنت|رسور|ريسور|دعام|بالون|تمييل|قسطر|نفتح الشريان|نفتح الشرايين",
                           r"angioplastie|\bballon|deboucher l'artere|ouvrir l'artere"], 10, "pip", "Angioplasty & stent"),
    ("coronary", "occlusion", [r"\bclot\b", r"\bheart attack", r"\bmyocardial infarction", r"\bplaque ruptur", r"\bruptures?\b",
                               r"\bblock(?:s|ed)? the artery", r"جلط|ذبح|انسد|بتسكر|بيسكر|سكر الشريان|انفجر|تمزق",
                               r"caillot|infarctus|crise cardiaque|rupture de (?:la )?plaque|bouche l'artere|obstru"], 9, "split", "Plaque rupture → clot"),
    ("coronary", "narrowing", [r"\bcholesterol", r"\bplaque", r"\batherosclero", r"\bnarrow", r"\bcoronary arter",
                               r"كوليسترول|كولسترول|لويح|ترسب|تضيق|بيضيق|يضيق|بضيق الشريان|تصلب|شرايين التاجي|الشرايين التاجي",
                               r"cholesterol|atherosclerose|retreci|stenose|arteres? coronaires?|depots? de graisse"], 8, "side", "Plaque narrows the artery"),
    ("ct", "calcium", [r"\bcalcium", r"\bagatston", r"\bct scan", r"\bcoronary ct", r"\bscanner",
                       r"كالسيوم|كلس|تكلس|سكانر|سكانير|مقطعي|سيتي سكان|كالسيوم سكور",
                       r"score calcique|coroscanner|scanner (?:coronaire|cardiaque)"], 8.5, "split", "Coronary calcium score"),
    ("ecg", "sinus", [r"\becg\b", r"\bekg\b", r"\belectrocardiogram", r"\bheart rhythm", r"\barrhythmi", r"\bpalpitation",
                      r"\batrial fibrillation", r"تخطيط|خفقان|رجفان|دقات القلب",
                      r"electrocardiogramme|rythme cardiaque|arythmie|palpitation|fibrillation"], 7, "side", "ECG"),
    ("bp", "gauge", [r"\b\d{2,3}\s*(?:over|/)\s*\d{2,3}\b", r"\bblood pressure", r"ضغط الدم|ضغط دم|\d{2,3}\s*(?:علي|على|ع)\s*\d{2,3}",
                 r"tension arterielle|pression arterielle|hypertension|\b1[0-9]\s*/\s*[5-9]\b|\b1[0-9] sur [5-9]\b"], 6, "labels", "Blood pressure"),
]

# Claim-risk heuristics — flags for PHYSICIAN REVIEW, not a fact-check.
CLAIM_PATTERNS = [
    (r"\bguarantee|مضمون|بضمن|بنضمن|ضمان|\bgaranti", "Absolute guarantee of outcome"),
    (r"\bguerit\b|\bguerir (?:definitivement|totalement)|guerison (?:totale|assuree)", "Claims a cure"),
    (r"elimine (?:toute|tous les)|fait disparaitre (?:toute|completement)|inverse completement", "Overstated reversal claim"),
    (r"\b100 ?% (?:des cas|efficace|garanti)|cent pour cent", "100% claim"),
    (r"sans (?:aucun )?effets? secondaires?", "Claims no side effects"),
    (r"arretez? (?:vos|votre|de prendre)|a la place de (?:vos )?medicaments", "May discourage prescribed therapy"),
    (r"\bmiracle|\bsecret\b|\bmagique", "Sensational language"),
    (r"a (?:lui|elle) seule? (?:peut|suffit)|suffit a (?:lui|elle) seul", "Single-intervention overstatement"),
    (r"meilleur (?:cardiologue|medecin|docteur)", "Superlative self-promotion (advertising rules)"),
    (r"\b(?:cure|cures|cured)\b|بيشفي|بتشفي|شفاء تام|بيشيل نهائي", "Claims a cure"),
    (r"\bcompletely reverse|\breverse any\b|\breverses? (?:all|any)\b|بتشيل اي|بيشيل اي|بتزيل اي|بيزيل اي", "Overstated reversal claim"),
    (r"\bnever (?:have|get|develop)\b|ما رح (?:يصير|تصير) (?:معك|معكن) ابدا", "Absolute 'never' outcome claim"),
    (r"\b100 ?%|\bhundred percent\b|ميه بالميه|100 بال", "100% claim"),
    (r"\bno side effects\b|ما في (?:اي )?(?:اعراض|اثار) جانبي", "Claims no side effects"),
    (r"\bstop taking\b|\binstead of (?:your )?medication|وقف(?:وا)? (?:ال)?دوا|بدل (?:ال)?دوا", "May discourage prescribed therapy"),
    (r"\bmiracle\b|\bsecret\b|معجز|سحري", "Sensational language"),
    (r"\b(?:alone|only) can\b|\bcan .{0,20}\balone\b|لحالها|لحالو|لوحدها", "Single-intervention overstatement"),
    (r"\bbest (?:doctor|cardiologist|clinic)\b|افضل (?:دكتور|حكيم|طبيب)", "Superlative self-promotion (advertising rules)"),
]

BP_RE = re.compile(r"(\d{2,3})\s*(?:over|/|على|علي|ع)\s*(\d{2,3})", re.I)


def _match_text(text):
    t = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c)) if not is_ar(text) else text.lower()
    return ar_norm(t)


# ------------------------------------------------------------------ functions
def detect_terms(words):
    """Return list of {start_idx,end_idx,term,label} for medical terms (multi-word, clitic aware)."""
    toks = [core(w["text"]) for w in words]
    out, i = [], 0
    keys = sorted(CARDIO_TERMS, key=lambda k: -len(k.split()))
    while i < len(toks):
        hit = None
        for k in keys:
            parts = k.split()
            if toks[i:i + len(parts)] == parts:
                hit = (k, len(parts)); break
        if hit:
            out.append({"start_idx": i, "end_idx": i + hit[1] - 1, "term": hit[0], "label": CARDIO_TERMS[hit[0]]})
            i += hit[1]
        else:
            i += 1
    return out


def match_graphic(text: str):
    best = None
    t = _match_text(text)
    for asset, mode, pats, prio, layout, label in GRAPHIC_RULES:
        for p in pats:
            if re.search(p, t):
                if asset == "bp" and BP_RE.search(t): prio = 11
                if not best or prio > best["priority"]:
                    best = {"asset": asset, "mode": mode, "priority": prio, "layout": layout, "label": label}
                break
    return best


def flag_claims(text: str):
    t = _match_text(text)
    return list(dict.fromkeys(reason for pat, reason in CLAIM_PATTERNS if re.search(pat, t)))


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _best_fix(c):
    best = None
    for key, fixed in AR_CORRECTIONS.items():
        if c == key: continue
        cands = [c] if abs(len(c) - len(key)) <= 2 else []
        if len(key) >= 8 and len(c) > len(key): cands += [c[: len(key)], c[: len(key) + 1]]  # agglutinated tail ("…يتن")
        for cc in cands:
            d = _lev(cc, key)
            lim = round(len(key) * 0.34) if len(key) >= 8 else max(1, round(len(key) * 0.22))
            if d <= lim and (best is None or d < best[0]): best = (d, fixed)
    return best


def correct_medical_terms(words):
    """Repair ASR misspellings of Arabic medical words, incl. one word split in two
    ('التولي سترولت' -> 'الكوليسترول'). Keeps the clitic prefix; records the original."""
    out, i = [], 0
    while i < len(words):
        w = words[i]
        if is_ar(w["text"]):
            n = norm(w["text"]); c = strip_clitic(n); pre = n[: len(n) - len(c)]
            if i + 1 < len(words) and is_ar(words[i + 1]["text"]) and len(c) >= 2:  # split word?
                joined = c + norm(words[i + 1]["text"])
                fix = _best_fix(joined)
                if fix and len(joined) >= 6 and not _best_fix(c) and core(words[i + 1]["text"]) not in AR_CORRECTIONS:
                    out.append({**w, "text": pre + fix[1], "end": words[i + 1]["end"], "asr": w["text"] + " " + words[i + 1]["text"]})
                    i += 2; continue
            if len(c) >= 5:
                fix = _best_fix(c)
                if fix and fix[0] > 0:
                    out.append({**w, "text": pre + fix[1], "asr": w["text"]}); i += 1; continue
        out.append(w); i += 1
    return out
