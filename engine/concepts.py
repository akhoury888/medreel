"""Specialty-independent visual vocabulary: spoken concept -> icon (Health Icons, MIT) + stock-video query.

Patterns run on knowledge._match_text(): lower-case; Latin accents folded (é->e); Arabic normalised
(أإآ->ا, ة->ه, ى->ي, no diacritics). Arabic patterns are substrings so clitics (و/ب/ل/ال) don't matter.
Each entry: key: (icon, {en, fr, ar labels}, [patterns], stock query or None)."""
import re
from . import knowledge as K

C = {
    # --- lungs / respiratory
    "lungs": ("lungs", {"en": "Lungs", "fr": "Poumons", "ar": "الرئتين"}, [r"\blungs?\b", r"\bpoumons?\b", r"رئه|الرئتين|رئتين|الريه|رييه|الرية"], "lungs breathing doctor"),
    "breathing": ("respirology", {"en": "Breathing", "fr": "Respiration", "ar": "التنفّس"}, [r"\bbreath(e|ing)\b", r"\brespir(ation|er)\b", r"تنفس|بتنفس|نفس عميق"], "deep breath woman"),
    "short_breath": ("coughing-alt", {"en": "Shortness of breath", "fr": "Essoufflement", "ar": "ضيق النفس"}, [r"short(ness)? of breath|breathless", r"essouffl|souffle court", r"ضيق نفس|ضيقه نفس|ضيق بالنفس|ما عم يقدر يتنفس|نفسو مقطوع|بيلهث"], "out of breath man stairs"),
    "asthma": ("asthma", {"en": "Asthma", "fr": "Asthme", "ar": "الربو"}, [r"\basthma", r"\basthme", r"(?:^|\s)(?:و|ب|ل)?(?:ال)?ربو\b|الازمه الصدريه|ازمه صدريه"], "asthma inhaler"),
    "inhaler": ("asthma-inhaler", {"en": "Inhaler", "fr": "Inhalateur", "ar": "البخّاخ"}, [r"\binhaler|\bpuffer|ventolin", r"inhalateur|ventoline|bouffee", r"بخاخ|فنتولين|البخاخه"], "using inhaler"),
    "cough": ("coughing-alt", {"en": "Cough", "fr": "Toux", "ar": "السعلة"}, [r"\bcough", r"\btoux\b|\btousse", r"سعله|سعال|قحه|بيقح|بتقح|عم يسعل"], "man coughing"),
    "smoking": ("smoking", {"en": "Smoking", "fr": "Tabac", "ar": "التدخين"}, [r"\bsmok(e|ing|er)|cigarett|\bvap(e|ing)", r"\btabac|cigarette|\bfum(er|eur|e)\b|vapot", r"دخان|تدخين|سيكاره|سجاير|سيجاره|بيدخن|بتدخن|دخن"], "cigarette smoke"),
    "shisha": ("smoking", {"en": "Shisha", "fr": "Narguilé", "ar": "الأركيلة"}, [r"shisha|hookah|narghil|argil", r"narguil|chicha", r"اركيله|نرجيله|ارجيله|شيشه"], "hookah smoke"),
    "quit_smoking": ("smoking-cessation", {"en": "Quit smoking", "fr": "Arrêter de fumer", "ar": "ترك التدخين"}, [r"quit(ting)? smok|stop(ping)? smok|smok\w*[^.]{0,25}\bquit", r"arret\w* de fumer|sevrage tabagique", r"ترك التدخين|وقف التدخين|يوقف دخان|توقف عن التدخين|تترك الدخان|يترك الدخان"], "breaking cigarette"),
    "copd": ("lungs", {"en": "COPD", "fr": "BPCO", "ar": "الانسداد الرئوي المزمن"}, [r"\bcopd\b|emphysema|chronic bronchitis", r"\bbpco\b|emphyseme|bronchite chronique", r"انسداد (?:ال)?رئوي|انتفاخ الرئه|التهاب القصبات المزمن"], None),
    "pneumonia": ("pneumonia", {"en": "Pneumonia", "fr": "Pneumonie", "ar": "الالتهاب الرئوي"}, [r"pneumonia|chest infection", r"pneumonie|infection pulmonaire", r"التهاب رئوي|التهاب بالرئه|ذات الرئه"], None),
    "tb": ("tb", {"en": "Tuberculosis", "fr": "Tuberculose", "ar": "السلّ"}, [r"tubercul|\btb\b", r"tubercul", r"السل\b|درن"], None),
    "oxygen": ("oxygen-tank", {"en": "Oxygen", "fr": "Oxygène", "ar": "الأكسجين"}, [r"\boxygen", r"oxygen", r"اكسجين|اوكسجين"], "oxygen mask hospital"),
    "saturation": ("pulse-oximeter", {"en": "Oxygen saturation", "fr": "Saturation", "ar": "نسبة الأكسجين"}, [r"saturation|oximeter|\bsats?\b|spo2", r"saturation|oxymetre|spo2", r"نسبه الاكسجين|نسبه الاوكسجين|ساتوراسيون|اوكسيميتر"], "pulse oximeter finger"),
    "sleep_apnea": ("cpap-machine", {"en": "Sleep apnea", "fr": "Apnée du sommeil", "ar": "انقطاع النفس أثناء النوم"}, [r"sleep apn|\bcpap\b|snor", r"apnee du sommeil|apnee|ronfl|\bppc\b", r"انقطاع النفس|شخير|بيشخر|بتشخر|سي باب"], "man snoring bed"),
    "spirometry": ("chart-line", {"en": "Breathing test", "fr": "Spirométrie", "ar": "فحص وظائف الرئة"}, [r"spirometr|lung function test|pulmonary function", r"spirometr|epreuves? fonctionnelles? respiratoires|\befr\b", r"وظائف الرئه|سبيرومتري|فحص التنفس"], None),
    "lung_cancer": ("tumour", {"en": "Lung cancer", "fr": "Cancer du poumon", "ar": "سرطان الرئة"}, [r"lung cancer|lung nodule", r"cancer du poumon|nodule pulmonaire", r"سرطان الرئه|سرطان بالرئه|عقده بالرئه|كتله بالرئه"], None),
    "xray": ("xray", {"en": "X-ray", "fr": "Radiographie", "ar": "صورة الأشعة"}, [r"x-?ray|radiograph", r"\bradio\b|radiographie", r"صوره اشعه|اشعه|صوره صدر|راديو"], "doctor looking xray"),
    "ventilator": ("ventilator", {"en": "Ventilator", "fr": "Respirateur", "ar": "جهاز التنفّس"}, [r"ventilator|intubat", r"respirateur|intub", r"جهاز تنفس|ماكينه تنفس|تنبيب"], None),
    "allergy": ("allergies", {"en": "Allergy", "fr": "Allergie", "ar": "الحساسية"}, [r"\ballerg|hay fever|pollen", r"allerg|pollen", r"حساسيه|الرشح التحسسي|غبار الطلع"], "pollen allergy sneezing"),
    "virus": ("virus", {"en": "Virus", "fr": "Virus", "ar": "الفيروس"}, [r"\bvirus|covid|\bflu\b|influenza|\bcold\b", r"\bvirus|covid|grippe|rhume", r"فيروس|كورونا|كوفيد|انفلونزا|رشح|كريب"], "virus 3d"),
    "vaccine": ("syringe-vaccine", {"en": "Vaccine", "fr": "Vaccin", "ar": "اللقاح"}, [r"vaccin|immuni[sz]", r"vaccin", r"لقاح|تلقيح|تطعيم"], "vaccine injection arm"),
    "mask": ("ppe-face_mask", {"en": "Mask", "fr": "Masque", "ar": "الكمامة"}, [r"\bmasks?\b", r"\bmasques?\b", r"كمامه|كمامات"], None),
    # --- heart & circulation (complements the procedural cardiology animations)
    "heart": ("heart-organ", {"en": "Heart", "fr": "Cœur", "ar": "القلب"}, [r"\bheart\b", r"\bcoeur\b", r"القلب|قلبك|قلبو"], None),
    "blood_pressure": ("blood-pressure", {"en": "Blood pressure", "fr": "Tension", "ar": "ضغط الدم"}, [r"blood pressure|hypertens", r"\btension|hypertens", r"ضغط الدم|ضغط دم|الضغط"], "blood pressure measurement"),
    "cholesterol": ("blood-vessel", {"en": "Cholesterol", "fr": "Cholestérol", "ar": "الكوليسترول"}, [r"cholesterol|\bldl\b", r"cholesterol", r"كوليسترول|كولسترول"], None),
    "palpitations": ("heartbeat", {"en": "Palpitations", "fr": "Palpitations", "ar": "الخفقان"}, [r"palpitation|racing heart", r"palpitation", r"خفقان|دقات قلب سريعه|قلبي عم يدق"], None),
    # --- general medicine
    "medication": ("pills-3", {"en": "Medication", "fr": "Médicaments", "ar": "الدواء"}, [r"\bmedic(ation|ine)s?\b|\bpills?\b|\btablets?\b|\bdrugs?\b", r"medicament|traitement|comprime|cachet", r"(?:^|\s)(?:ب|و|ل)?(?:ال)?دوا\b|ادويه|حبوب|(?:^|\s)حبه\b|دواء|علاج"], "pills in hand"),
    "prescription": ("prescription-document", {"en": "Prescription", "fr": "Ordonnance", "ar": "الوصفة الطبية"}, [r"prescri", r"ordonnance|prescri", r"وصفه|روشته"], None),
    "injection": ("syringe", {"en": "Injection", "fr": "Injection", "ar": "الإبرة"}, [r"inject|\bneedle|\bshot\b", r"injection|piqure", r"ابره|ابر|حقنه"], None),
    "antibiotic": ("bacteria", {"en": "Antibiotics", "fr": "Antibiotiques", "ar": "المضاد الحيوي"}, [r"antibiotic|bacteri", r"antibiot|bacteri", r"انتيبيوتيك|مضاد حيوي|مضادات حيويه|بكتيريا|جراثيم"], None),
    "blood_test": ("medical-sample", {"en": "Blood test", "fr": "Prise de sang", "ar": "فحص الدم"}, [r"blood test|lab test|blood work", r"prise de sang|bilan sanguin|analyse", r"فحص دم|تحليل دم|تحاليل|فحوصات|تحليل"], "blood sample lab"),
    "lab": ("test-tubes", {"en": "Laboratory", "fr": "Laboratoire", "ar": "المختبر"}, [r"\blab(oratory)?\b", r"laboratoire|\blabo\b", r"مختبر|مختبرات"], "laboratory test tubes"),
    "ultrasound": ("sonogram", {"en": "Ultrasound", "fr": "Échographie", "ar": "الإيكو"}, [r"ultrasound|sonograph|\becho(cardiogra)?", r"echograph|\becho\b", r"ايكو|صوره صوتيه|سونار"], "ultrasound scan"),
    "scan": ("radiology", {"en": "Imaging", "fr": "Imagerie", "ar": "التصوير"}, [r"\bmri\b|\bct\b|\bscan\b|imaging", r"\birm\b|scanner|imagerie", r"رنين|تصوير|صوره مقطعيه|سكانر"], None),
    "surgery": ("general-surgery", {"en": "Surgery", "fr": "Chirurgie", "ar": "العملية"}, [r"surger|operation|\bsurgeon", r"chirurg|operation|intervention", r"عمليه|جراحه|عمليات"], "surgeons operating room"),
    "hospital": ("hospital", {"en": "Hospital", "fr": "Hôpital", "ar": "المستشفى"}, [r"\bhospital", r"hopital|clinique", r"مستشفى|مستشفي"], "hospital corridor"),
    "emergency": ("accident-and_emergency", {"en": "Emergency", "fr": "Urgences", "ar": "الطوارئ"}, [r"emergenc|\bthe er\b|ambulance|\b911\b", r"urgence|samu|ambulance", r"طوارئ|طوارىء|اسعاف|الصليب الاحمر"], "ambulance lights"),
    "checkup": ("stethoscope", {"en": "Check-up", "fr": "Bilan", "ar": "الفحص الدوري"}, [r"check-?up|screening|follow-?up|see (a|your) doctor", r"depistage|bilan|consult|suivi", r"فحص دوري|كشف|معاينه|متابعه|راجع الطبيب|راجعوا الطبيب|زوروا"], "doctor patient consultation"),
    "appointment": ("calendar", {"en": "Appointment", "fr": "Rendez-vous", "ar": "الموعد"}, [r"appointment|book (a|your)", r"rendez-vous|\brdv\b", r"موعد|مواعيد"], None),
    "telemedicine": ("telemedicine", {"en": "Online consultation", "fr": "Téléconsultation", "ar": "استشارة أونلاين"}, [r"telemedic|online consult|video call", r"teleconsult", r"اونلاين|عن بعد"], None),
    "pain": ("pain", {"en": "Pain", "fr": "Douleur", "ar": "الوجع"}, [r"\bpain|\bache\b|hurts?\b", r"douleur|\bmal (?:au|a la|aux|de)\b", r"وجع|اوجاع|(?:^|\s)(?:و|ب)?الم\b|بيوجع|بتوجع"], None),
    "headache": ("headache", {"en": "Headache", "fr": "Mal de tête", "ar": "وجع الراس"}, [r"headache|migraine", r"mal de tete|migraine|cephalee", r"وجع راس|صداع|شقيقه"], "woman headache"),
    "fever": ("thermometer-digital", {"en": "Fever", "fr": "Fièvre", "ar": "الحرارة"}, [r"\bfever|temperature", r"fievre|temperature", r"حراره|سخونه|سخن"], "thermometer fever"),
    "nausea": ("nausea", {"en": "Nausea", "fr": "Nausées", "ar": "الغثيان"}, [r"nause|vomit", r"nausee|vomi", r"غثيان|لعيان|استفراغ|تطريش"], None),
    "fatigue": ("face-sleepy", {"en": "Fatigue", "fr": "Fatigue", "ar": "التعب"}, [r"tired|fatigue|exhaust", r"fatigu|epuis", r"تعب|تعبان|ارهاق|منهك"], "tired man desk"),
    "dizziness": ("face-dizzy", {"en": "Dizziness", "fr": "Vertiges", "ar": "الدوخة"}, [r"dizz|vertigo|faint", r"vertige|malaise|etourdi", r"دوخه|دايخ|بيدوخ|غشيان"], None),
    "infection": ("virus-patient", {"en": "Infection", "fr": "Infection", "ar": "الالتهاب"}, [r"infect|inflamm", r"infection|inflamm", r"التهاب|عدوى|انتان"], None),
    "genetics": ("dna", {"en": "Genetics", "fr": "Génétique", "ar": "الوراثة"}, [r"genetic|hereditar|family history|\bdna\b|\bgenes?\b", r"genetique|hereditaire|antecedents familiaux", r"وراثه|وراثي|جينات|بالعيله"], None),
    "cancer": ("ribbon", {"en": "Cancer", "fr": "Cancer", "ar": "السرطان"}, [r"\bcancer|tumou?r|oncolog", r"cancer|tumeur", r"سرطان|ورم|خبيث"], None),
    # --- lifestyle
    "exercise": ("running", {"en": "Exercise", "fr": "Activité physique", "ar": "الرياضة"}, [r"exercis|physical activit|\bworkout|\bsport", r"activite physique|\bsport|exercice", r"رياضه|تمارين|تمرين"], "people jogging park"),
    "walking": ("walking", {"en": "Walking", "fr": "Marche", "ar": "المشي"}, [r"\bwalk(ing)?\b|steps a day", r"\bmarch(e|er)\b|\bpas par jour", r"مشي|نمشي|تمشي|يمشي|امشوا"], "people walking outdoors"),
    "swimming": ("swim", {"en": "Swimming", "fr": "Natation", "ar": "السباحة"}, [r"swim", r"natation|nager", r"سباحه|سبح"], "swimming pool"),
    "cycling": ("exercise-bicycle", {"en": "Cycling", "fr": "Vélo", "ar": "الدرّاجة"}, [r"cycling|bike|bicycle", r"\bvelo\b|cyclisme", r"بسكليت|دراجه"], "cycling road"),
    "yoga": ("exercise-yoga", {"en": "Yoga & stretching", "fr": "Yoga", "ar": "اليوغا"}, [r"\byoga|stretch", r"\byoga|etirement", r"يوغا|تمطيط"], "yoga woman"),
    "weight": ("weight", {"en": "Weight", "fr": "Poids", "ar": "الوزن"}, [r"\bweight|obes|overweight|\bbmi\b", r"\bpoids|obesit|surpoids|\bimc\b", r"وزن|سمنه|رجيم|تخسيس|تنحيف"], "bathroom scale feet"),
    "diet": ("vegetables", {"en": "Healthy diet", "fr": "Alimentation", "ar": "الأكل الصحي"}, [r"\bdiet\b|healthy (food|eating)|vegetable|fruit|nutrition", r"alimentation|regime|legume|fruit|nutrition", r"اكل صحي|خضره|خضار|فواكه|غذاء صحي|تغذيه"], "healthy food vegetables"),
    "fast_food": ("unhealthy-food", {"en": "Junk food", "fr": "Malbouffe", "ar": "الأكل السريع"}, [r"junk food|fast food|fried|processed food", r"malbouffe|fast-?food|friture", r"اكل سريع|مقالي|نواشف|معجنات|فاست فود"], "burger fries"),
    "salt": ("sodium-free", {"en": "Less salt", "fr": "Moins de sel", "ar": "تخفيف الملح"}, [r"\bsalt|sodium", r"\bsel\b|sodium", r"ملح|مالح"], None),
    "sugar": ("sugar", {"en": "Sugar", "fr": "Sucre", "ar": "السكر"}, [r"\bsugar|sweets|soda", r"\bsucre|sucrer|soda", r"(?:^|\s)(?:و|ب)?(?:ال)?سكر\b|حلويات|السكريات|مشروبات غازيه"], None),
    "alcohol": ("alcohol", {"en": "Alcohol", "fr": "Alcool", "ar": "الكحول"}, [r"alcohol|\bdrinking\b|\bwine\b|\bbeer\b", r"alcool|\bvin\b|biere", r"كحول|مشروب|عرق|بيره|نبيذ"], None),
    "water": ("running-water", {"en": "Drink water", "fr": "Hydratation", "ar": "شرب المي"}, [r"\bwater\b|hydrat", r"\beau\b|hydrat", r"(?:^|\s)(?:و|ب)?(?:ال)?مي\b|مياه|ماء|اشربوا مي"], "glass of water"),
    "sleep": ("face-sleepy", {"en": "Sleep", "fr": "Sommeil", "ar": "النوم"}, [r"\bsleep|insomnia", r"sommeil|dormir|insomnie", r"(?:^|\s)(?:ب|و)?(?:ال)?نوم\b|ننام|تنام|(?:^|\s)(?:ال)?ارق\b"], "woman sleeping bed"),
    "stress": ("mental-health", {"en": "Stress", "fr": "Stress", "ar": "التوتر"}, [r"\bstress|anxiety|depress|mental health", r"\bstress|anxiet|depress|sante mentale", r"توتر|قلق|اكتئاب|ضغط نفسي|نفسيه"], "stressed man office"),
    # --- organs & specialties
    "kidney": ("kidneys", {"en": "Kidneys", "fr": "Reins", "ar": "الكلى"}, [r"kidney|renal", r"\breins?\b|renal", r"كلى|كلاوي|كليه"], None),
    "dialysis": ("dialysis", {"en": "Dialysis", "fr": "Dialyse", "ar": "غسيل الكلى"}, [r"dialys", r"dialyse", r"غسيل كلى|غسيل الكلي|دياليز"], None),
    "liver": ("liver", {"en": "Liver", "fr": "Foie", "ar": "الكبد"}, [r"\bliver|hepat", r"\bfoie|hepat", r"كبد|تشمع"], None),
    "stomach": ("stomach", {"en": "Stomach", "fr": "Estomac", "ar": "المعدة"}, [r"stomach|gastr|reflux|heartburn", r"estomac|gastr|reflux|brulures", r"معده|حرقه|ارتجاع|معدي"], None),
    "bowel": ("intestine", {"en": "Bowel", "fr": "Intestin", "ar": "الأمعاء"}, [r"bowel|intestin|colon|constipat|diarrh", r"intestin|colon|constip|diarrh", r"امعاء|مصارين|قولون|امساك|اسهال"], None),
    "diabetes": ("diabetes-measure", {"en": "Diabetes", "fr": "Diabète", "ar": "السكري"}, [r"diabet|blood sugar|glucose|insulin|hba1c", r"diabet|glycemi|insuline", r"سكري|السكر بالدم|نسبه السكر|انسولين"], "glucose meter finger"),
    "thyroid": ("thyroid", {"en": "Thyroid", "fr": "Thyroïde", "ar": "الغدّة الدرقية"}, [r"thyroid", r"thyroid", r"غده|درقيه"], None),
    "bones": ("bone", {"en": "Bones", "fr": "Os", "ar": "العضم"}, [r"\bbones?\b|osteopor|fracture", r"\bos\b|osteopor|fracture", r"(?:^|\s)(?:ب|و)?(?:ال)?(?:عضم|عظم)|عضام|ترقق العظم|ترقق|كسر بالعضم"], None),
    "joints": ("joints", {"en": "Joints", "fr": "Articulations", "ar": "المفاصل"}, [r"\bjoints?\b|arthrit|\bknee", r"articulation|arthrose|genou", r"مفاصل|مفصل|ركبه|روماتيزم"], None),
    "back": ("back-pain", {"en": "Back pain", "fr": "Mal de dos", "ar": "وجع الضهر"}, [r"back pain|\bspine|lower back", r"mal de dos|colonne|lombaire", r"وجع ضهر|وجع الضهر|(?:^|\s)(?:ب|و)?(?:ال)?ضهر|ضهري|الم الظهر|عمود فقري|ديسك"], "back pain office"),
    "brain": ("neurology", {"en": "Brain", "fr": "Cerveau", "ar": "الدماغ"}, [r"\bbrain|stroke|neurolog|memory", r"cerveau|\bavc\b|neurolog|memoire", r"دماغ|جلطه دماغيه|اعصاب|ذاكره|مخ"], None),
    "eyes": ("eye", {"en": "Eyes", "fr": "Yeux", "ar": "العيون"}, [r"\beyes?\b|vision|glaucom|cataract", r"\byeux|\boeil|vision|glaucom|cataract", r"عيون|(?:^|\s)(?:ب|و)?(?:ال)?عين\b|ضعف النظر|مي زرقا|ماء ابيض"], None),
    "ears": ("ear", {"en": "Ears", "fr": "Oreilles", "ar": "الأذن"}, [r"\bears?\b|hearing", r"oreille|audition", r"دينه|(?:^|\s)(?:ب|و)?(?:ال)?اذن\b|ضعف السمع|السمع"], None),
    "teeth": ("tooth", {"en": "Teeth", "fr": "Dents", "ar": "الأسنان"}, [r"\bteeth|\btooth|dental|gum", r"\bdents?\b|dentaire|gencive", r"سنان|اسنان|ضرس|لثه"], None),
    "skin": ("skin-layers", {"en": "Skin", "fr": "Peau", "ar": "البشرة"}, [r"\bskin|sun ?screen|eczema|acne", r"\bpeau|ecran solaire|eczema|acne", r"بشره|جلد|حب الشباب|اكزيما|كريم شمس"], "sunscreen skin"),
    "pregnancy": ("pregnant", {"en": "Pregnancy", "fr": "Grossesse", "ar": "الحمل"}, [r"pregnan", r"grossesse|enceinte", r"(?:^|\s)(?:ب|و)?(?:ال)?حمل\b|حامل"], "pregnant woman"),
    "baby": ("baby-0306m", {"en": "Baby", "fr": "Bébé", "ar": "الطفل"}, [r"\bbab(y|ies)|newborn|infant", r"\bbebe|nourrisson", r"طفل|اطفال|ولاد|رضيع|بيبي"], None),
    "elderly": ("elderly", {"en": "Older adults", "fr": "Personnes âgées", "ar": "كبار السن"}, [r"elderly|older (adults|people)|seniors?", r"personnes? agees?|seniors?", r"كبار السن|ختياريه|المسنين"], "elderly couple walking"),
    "women": ("woman", {"en": "Women", "fr": "Femmes", "ar": "النساء"}, [r"\bwomen\b|\bwoman\b", r"\bfemmes?\b", r"نسوان|النساء|سيدات"], None),
    "men": ("man", {"en": "Men", "fr": "Hommes", "ar": "الرجال"}, [r"\bmen\b|\bman\b", r"\bhommes?\b", r"(?:^|\s)(?:ال)?رجال\b"], None),
    "prostate": ("prostate", {"en": "Prostate", "fr": "Prostate", "ar": "البروستات"}, [r"prostat", r"prostat", r"بروستات"], None),
    "breast": ("breast", {"en": "Breast", "fr": "Sein", "ar": "الثدي"}, [r"\bbreast|mammogra", r"\bsein|mammograph", r"ثدي|صدر المراه|ماموغرام"], None),
    "bladder": ("bladder", {"en": "Bladder", "fr": "Vessie", "ar": "المثانة"}, [r"\bbladder|\burin", r"vessie|\burin", r"مثانه|(?:^|\s)(?:ال)?بول\b|بوله"], None),
    "blood": ("blood-drop", {"en": "Blood", "fr": "Sang", "ar": "الدم"}, [r"\bblood\b|anemi", r"\bsang\b|anemi", r"\bدم\b|الدم|فقر دم"], None),
}

_RX = {k: [re.compile(K.ar_norm(p)) for p in v[2]] for k, v in C.items()}  # patterns normalised like the text (ئ->ي, ة->ه…)
SUPERSEDES = {"quit_smoking": {"smoking"}, "lung_cancer": {"cancer", "lungs"}, "sleep_apnea": {"sleep"}, "blood_test": {"blood", "lab"},
              "headache": {"pain"}, "back": {"pain"}, "short_breath": {"breathing", "lungs"}, "dialysis": {"kidney"}, "copd": {"lungs"},
              "pneumonia": {"lungs", "infection"}, "asthma": {"breathing"}, "saturation": {"oxygen"}, "diabetes": {"sugar", "blood"}}
GENERIC = {"heart", "pain", "blood", "women", "men"}  # shown only when nothing more specific is said


def find(words):
    """-> [(word_index, key)] in speaking order: every concept mentioned, at the word where it's said."""
    if not words: return []
    toks = [K._match_text(w["text"]) for w in words]
    # char offsets of each word in the joined text so multi-word patterns map back to a word index
    starts, pos = [], 0
    for t in toks: starts.append(pos); pos += len(t) + 1
    text = " ".join(toks)
    import bisect
    hits = {}
    for k, rxs in _RX.items():
        for rx in rxs:
            for m in rx.finditer(text):
                i = bisect.bisect_right(starts, m.start()) - 1
                if k not in hits or i < hits[k]: hits[k] = i
    drop = set().union(*(SUPERSEDES.get(k, set()) for k in hits)) if hits else set()
    out = sorted(((i, k) for k, i in hits.items() if k not in drop), key=lambda x: x[0])
    specific = [x for x in out if x[1] not in GENERIC]
    return specific or out


def label(key, lang):
    return C[key][1].get(lang) or C[key][1]["en"]


def icon(key): return C[key][0]


def stock_query(key): return C[key][3]
