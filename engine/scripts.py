"""Script writing for physicians: AI scripts from a prompt (Claude), plus reviewed starter scripts.

Every script is educational, makes no outcome guarantees, does not promise patient acquisition,
and ends with a "talk to your doctor" style call to action. The physician must review it before
recording or publishing — the app shows it as an editable draft, never publishes it directly."""
import json, os, re, urllib.request

LANG_NAMES = {"en": "English", "fr": "French (France/Quebec neutral)", "ar": "Lebanese Arabic dialect (spoken, natural; medical terms may stay in English or French as Lebanese doctors say them)"}
WPS = {"en": 2.5, "fr": 2.4, "ar": 2.0}  # spoken words per second, for length targets


class ScriptUnavailable(RuntimeError):
    pass


SYSTEM = """You write short educational social-media video scripts for a practising physician who will read them on camera.
Rules: medically accurate and current; plain language for patients; no guarantees of outcome; no promises of results or patient numbers;
no fabricated statistics (only well-established figures, stated cautiously); no superlatives about the physician; no product endorsements;
encourage seeing one's own doctor; respect physician-advertising rules in Canada, France and Lebanon.
Structure: a strong first-sentence hook, 3-5 short spoken sentences of explanation, one practical takeaway, a soft call to action.
Write sentences that sound natural when spoken. Mention concrete anatomy/procedures (e.g. plaque, coronary arteries, stent, blood pressure)
when relevant, because the editing software adds matching medical animations when those words are spoken."""


def generate(prompt, lang="en", seconds=45, specialty="cardiology", name=None):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ScriptUnavailable("AI script writing needs ANTHROPIC_API_KEY on the server. Pick a starter script or write your own.")
    words = int(seconds * WPS.get(lang, 2.4))
    user = (f"Specialty: {specialty}. Presenter: {name or 'the physician'}.\nTopic / request: {prompt}\n"
            f"Language: {LANG_NAMES.get(lang, lang)}. Target length: about {words} spoken words ({seconds} seconds).\n"
            'Return ONLY JSON: {"title": "...", "hook": "...", "script": "...", "caption": "...", "hashtags": ["..."]} '
            "where script is the full text to read aloud (including the hook), caption is the Instagram caption.")
    body = json.dumps({"model": os.environ.get("MEDREEL_LLM", "claude-sonnet-5-5"), "max_tokens": 1500, "system": SYSTEM,
                       "messages": [{"role": "user", "content": user}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", body,
                                 {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    txt = json.loads(urllib.request.urlopen(req, timeout=90).read())["content"][0]["text"]
    out = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
    out["script"] = re.sub(r"\s+\n", "\n", out.get("script", "")).strip()
    out["source"] = "ai"
    return out


# Starter scripts (reviewed wording, ~35-50 s). Placeholders: none — safe to record as-is after the physician's own review.
TEMPLATES = {
    "calcium": {
        "en": {"title": "What is a coronary calcium score?",
               "script": "Do you know what a coronary calcium score is? It's a quick CT scan of the heart, with no injection. "
                         "It shows whether calcium has built up in the walls of your coronary arteries. "
                         "That calcium is a sign of plaque, which can slowly narrow the arteries. "
                         "A score of zero is reassuring. A higher score helps your doctor decide how strongly to treat your cholesterol and other risk factors. "
                         "Ask your doctor whether this test makes sense for you."},
        "fr": {"title": "C'est quoi le score calcique ?",
               "script": "Savez-vous ce qu'est le score calcique coronaire ? C'est un scanner du cœur rapide, sans injection. "
                         "Il montre si du calcium s'est déposé dans la paroi de vos artères coronaires. "
                         "Ce calcium est un signe de plaque, qui peut rétrécir les artères avec le temps. "
                         "Un score à zéro est rassurant. Un score plus élevé aide votre médecin à décider jusqu'où traiter votre cholestérol et vos autres facteurs de risque. "
                         "Demandez à votre médecin si cet examen est utile dans votre cas."},
        "ar": {"title": "شو هو الكالسيوم سكور؟",
               "script": "بتعرفوا شو هو الكالسيوم سكور؟ هيدا سكانر سريع للقلب، بلا إبرة وبلا صبغة. "
                         "بيورجينا إذا في كالسيوم متجمّع بحيطان الشرايين التاجية. "
                         "هيدا الكالسيوم علامة إنو في ترسّبات، وهالترسّبات فيها تضيّق الشريان مع الوقت. "
                         "إذا السكور صفر، هيدا شي مطمّن. وإذا أعلى، بيساعد حكيمك يقرّر قدّيش لازم يعالج الكوليسترول وعوامل الخطر التانية. "
                         "اسألوا حكيمكن إذا هالفحص مناسب إلكن."},
    },
    "cholesterol": {
        "en": {"title": "How cholesterol narrows your arteries",
               "script": "Here's what cholesterol actually does to your heart. "
                         "When LDL cholesterol stays high for years, it slips into the wall of the coronary arteries and forms plaque. "
                         "Over time, that plaque narrows the artery and reduces blood flow to the heart muscle. "
                         "If a plaque ruptures, a clot can suddenly block the artery. That is a heart attack. "
                         "The good news: lowering LDL slows this process. Ask your doctor what your target should be."},
        "fr": {"title": "Comment le cholestérol bouche les artères",
               "script": "Voici ce que le cholestérol fait vraiment à votre cœur. "
                         "Quand le mauvais cholestérol reste élevé pendant des années, il s'infiltre dans la paroi des artères coronaires et forme une plaque. "
                         "Avec le temps, cette plaque rétrécit l'artère et diminue le sang qui arrive au muscle cardiaque. "
                         "Si la plaque se rompt, un caillot peut boucher l'artère d'un coup. C'est l'infarctus. "
                         "La bonne nouvelle : baisser le LDL ralentit ce processus. Demandez à votre médecin quelle est votre cible."},
        "ar": {"title": "كيف الكوليسترول بيسكّر الشرايين",
               "script": "خلّيني إشرحلكن شو بيعمل الكوليسترول بقلبكن. "
                         "لما الكوليسترول الضار يضل عالي لسنين، بيفوت عحيط الشرايين التاجية وبيعمل لويحة. "
                         "مع الوقت، هاللويحة بتضيّق الشريان وبتخفّف الدم اللي واصل لعضلة القلب. "
                         "وإذا انفجرت اللويحة، بتتشكّل جلطة فيها تسكّر الشريان فجأة. هيدي هي الذبحة القلبية. "
                         "الخبر المنيح إنو تنزيل الكوليسترول الضار بيبطّئ هالشي. اسألوا حكيمكن شو الرقم المناسب إلكن."},
    },
    "blood_pressure": {
        "en": {"title": "Your blood pressure target",
               "script": "Is your blood pressure really under control? "
                         "High blood pressure usually causes no symptoms, but over the years it damages your arteries, your heart and your kidneys. "
                         "For many adults, doctors aim for a blood pressure below 130/80. "
                         "Check it at home, at rest, a few times a week, and write the numbers down. "
                         "Bring them to your next appointment so your doctor can adjust your treatment."},
        "fr": {"title": "Votre objectif de tension",
               "script": "Votre tension est-elle vraiment contrôlée ? "
                         "L'hypertension ne donne souvent aucun symptôme, mais au fil des années elle abîme les artères, le cœur et les reins. "
                         "Pour beaucoup d'adultes, on vise une tension artérielle en dessous de 13/8. "
                         "Mesurez-la à la maison, au repos, plusieurs fois par semaine, et notez les chiffres. "
                         "Apportez-les à votre prochain rendez-vous pour que votre médecin ajuste le traitement."},
        "ar": {"title": "قدّيش لازم يكون ضغطك؟",
               "script": "ضغطك فعلاً مضبوط؟ "
                         "الضغط العالي عادةً ما بيعطي عوارض، بس مع السنين بيأذي الشرايين والقلب والكلاوي. "
                         "لكتير ناس، منحاول يكون ضغط الدم تحت 130/80. "
                         "قيسوه بالبيت وإنتو مرتاحين، كذا مرة بالجمعة، واكتبوا الأرقام. "
                         "جيبوها معكن عالموعد الجاي تيقدر حكيمكن يظبط العلاج."},
    },
    "heart_attack_signs": {
        "en": {"title": "Heart attack warning signs",
               "script": "Would you recognise a heart attack? "
                         "The classic sign is chest pain or pressure that lasts more than a few minutes, sometimes spreading to the arm, jaw or back. "
                         "It can also come with shortness of breath, sweating or nausea, especially in women and people with diabetes. "
                         "A heart attack happens when a clot blocks a coronary artery, and every minute counts. "
                         "If you have these symptoms, call emergency services right away. Don't drive yourself."},
        "fr": {"title": "Les signes d'un infarctus",
               "script": "Sauriez-vous reconnaître un infarctus ? "
                         "Le signe classique, c'est une douleur ou une pression dans la poitrine qui dure plus de quelques minutes, parfois vers le bras, la mâchoire ou le dos. "
                         "Il peut aussi y avoir un essoufflement, des sueurs ou des nausées, surtout chez les femmes et les diabétiques. "
                         "L'infarctus survient quand un caillot bouche une artère coronaire, et chaque minute compte. "
                         "Si vous avez ces symptômes, appelez immédiatement les urgences. Ne conduisez pas vous-même."},
        "ar": {"title": "علامات الذبحة القلبية",
               "script": "بتعرفوا كيف بتكون الذبحة القلبية؟ "
                         "العلامة المعروفة هي وجع أو ضغط بالصدر بيطوّل أكتر من كم دقيقة، وأحياناً بيروح عالإيد أو الفك أو الضهر. "
                         "فيه كمان يجي مع ضيق نفس أو عرق أو لعيان، خصوصاً عند الستات ومرضى السكري. "
                         "الذبحة بتصير لما جلطة تسكّر شريان تاجي، وكل دقيقة بتفرق. "
                         "إذا عندكن هالعوارض، اتصلوا بالطوارئ فوراً، وما تسوقوا لحالكن."},
    },
}


def templates(lang):
    return [{"id": k, **v[lang]} for k, v in TEMPLATES.items() if lang in v]
