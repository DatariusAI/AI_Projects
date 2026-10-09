"""المساعد العربي الذكي 2025: دردشة عربية مع أدوات NLP ونموذج لغوي اختياري.

Arabic chat assistant. Commands route to local NLP tools (summary, sentiment,
dialect, retrieval). When GROQ_API_KEY or OPENAI_API_KEY is set (st.secrets or
environment), translation and open questions are answered by an LLM grounded
on the retrieved knowledge-base passages (RAG). Without a key everything still
works offline.
"""
import os
import re

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer

st.set_page_config(page_title="المساعد العربي الذكي 2025", page_icon="🤖", layout="wide")
st.markdown(
    """<style>
[data-testid="stMain"], [data-testid="stSidebar"] { direction: rtl; text-align: right; }
[data-testid="stMain"] input, [data-testid="stMain"] textarea { direction: rtl; text-align: right; }
[data-testid="stChatMessage"] { direction: rtl; text-align: right; }
</style>""",
    unsafe_allow_html=True,
)

MAX_CHARS = 5000


# ---------------------------------------------------------------- text utils
def normalize(text: str) -> str:
    """Light Arabic normalisation: drop diacritics/tatweel, unify alef/yaa/taa marbuta."""
    text = re.sub(r"[ً-ْـ]", "", text)
    text = re.sub(r"[إأآ]", "ا", text)
    return text.replace("ى", "ي").replace("ة", "ه")


class ArabicTokenizer:
    """Regex tokenizer for sumy, so no NLTK download is needed."""

    language = "arabic"

    def to_sentences(self, paragraph):
        return [s.strip() for s in re.split(r"(?<=[.!?؟؛\n])\s*", paragraph) if len(s.strip()) > 1]

    def to_words(self, sentence):
        return re.findall(r"\w+", sentence)


def summarize(text: str, n_sentences: int):
    parser = PlaintextParser.from_string(text, ArabicTokenizer())
    sentences = parser.document.sentences
    if len(sentences) <= n_sentences:
        return [str(s) for s in sentences], len(sentences)
    picked = LexRankSummarizer()(parser.document, n_sentences)
    return [str(s) for s in picked], len(sentences)


# ---------------------------------------------------------------- sentiment
POSITIVE = ["سعيد", "ممتاز", "جميل", "رائع", "مبسوط", "جيد", "مسرور", "احب", "حلو", "مفيد", "ناجح", "نجاح", "شكرا", "افضل", "مريح", "سريع", "فرحان", "عظيم", "متميز", "انصح"]
NEGATIVE = ["حزين", "سيء", "سيئ", "كئيب", "ممل", "غاضب", "مضطرب", "زعلان", "فاشل", "بطيء", "مزعج", "اكره", "خايب", "مشكله", "تعبان", "متعب", "محبط", "رديء", "خساره", "صعب"]
NEGATIONS = {"لا", "ليس", "ما", "مش", "مو", "غير", "لم", "لن", "مب"}
INTENSIFIERS = {"جدا", "كثير", "كتير", "وايد", "اوي", "مره", "للغايه"}


def _polarity(word: str) -> int:
    candidates = [word]
    for prefix in ("وال", "بال", "فال", "لل", "ال", "و", "ف", "ب", "ل"):
        if word.startswith(prefix) and len(word) - len(prefix) >= 3:
            candidates.append(word[len(prefix):])
    for c in candidates:
        if any(c.startswith(p) for p in POSITIVE):
            return 1
        if any(c.startswith(n) for n in NEGATIVE):
            return -1
    return 0


def sentiment(text: str):
    originals = re.findall(r"\w+", text)
    tokens = [normalize(t) for t in originals]
    score, hits = 0.0, []
    for i, tok in enumerate(tokens):
        polarity = _polarity(tok)
        if not polarity:
            continue
        if i > 0 and tokens[i - 1] in NEGATIONS:
            polarity = -polarity
        if i + 1 < len(tokens) and tokens[i + 1] in INTENSIFIERS:
            polarity *= 1.5
        score += polarity
        hits.append((originals[i], polarity))
    norm = max(-1.0, min(1.0, score / 3))
    label = "إيجابي 😊" if norm > 0.15 else "سلبي 😞" if norm < -0.15 else "محايد 😐"
    return label, norm, hits


# ---------------------------------------------------------------- dialects
DIALECT_DATA = {
    "مصرية": ["إزيك عامل إيه", "انا مش فاهم حاجة", "عايز اروح البيت دلوقتي", "فين الحاجات بتاعتي", "ده كويس اوي",
              "مفيش مشكلة خالص", "انت بتعمل ايه النهارده", "يلا بينا نتمشى شوية", "الجو حر اوي النهارده",
              "معلش مش هقدر اجي", "هو فين مش لاقيه", "ايه الأخبار يا باشا"],
    "خليجية": ["شلونك اليوم", "شخبارك وش مسوي", "أبي أروح البيت الحين", "وايد زين", "شنو تبي", "وين رايح الحين",
               "ترى ما عندي وقت", "هالشي حلو وايد", "يالله نروح السوق باجر", "ليش ما جيت أمس", "شفيك زعلان",
               "الجو حار وايد اليوم"],
    "شامية": ["كيفك شو الأخبار", "شو بدك تعمل هلق", "بدي روح عالبيت", "كتير منيح", "وين رايح هلأ", "ما في مشكلة",
              "شو عم تعمل", "هيدا الشي حلو كتير", "يلا نروح نتمشى شوي", "ليش ما إجيت مبارح", "شو صاير معك",
              "الطقس حلو كتير اليوم"],
    "فصحى": ["كيف حالك اليوم", "ماذا تريد أن تفعل الآن", "أريد الذهاب إلى المنزل", "هذا جيد جدا", "إلى أين أنت ذاهب",
             "لا توجد مشكلة", "ماذا تفعل الآن", "هذا الشيء جميل جدا", "لنذهب للتنزه قليلا", "لماذا لم تحضر أمس",
             "ما الذي حدث معك", "الطقس جميل جدا اليوم"],
}


@st.cache_resource(show_spinner=False)
def dialect_model():
    texts = [normalize(t) for v in DIALECT_DATA.values() for t in v]
    labels = [k for k, v in DIALECT_DATA.items() for _ in v]
    model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)), MultinomialNB(alpha=0.3))
    return model.fit(texts, labels)


# ---------------------------------------------------------------- knowledge base
KB = [
    "الذكاء الاصطناعي فرع من علوم الحاسوب يهتم ببناء أنظمة تحاكي التفكير والتعلم البشري.",
    "تعلم الآلة جزء من الذكاء الاصطناعي يطوّر خوارزميات تتعلم من البيانات وتتحسن مع الوقت.",
    "التعلم العميق يستخدم شبكات عصبية متعددة الطبقات لتحليل الصور والنصوص والصوت.",
    "معالجة اللغة الطبيعية تمكّن الحاسوب من فهم النصوص واللغة البشرية وتحليلها.",
    "النماذج اللغوية الكبيرة تتدرّب على كميات ضخمة من النصوص لتوليد الإجابات والترجمة والتلخيص.",
    "الاسترجاع المعزّز بالتوليد (RAG) يبحث عن مقاطع ذات صلة ثم يمرّرها للنموذج اللغوي ليجيب بالاعتماد عليها.",
    "الرؤية الحاسوبية تمكّن الآلات من التعرف على الأشياء في الصور والفيديو.",
    "البيانات الضخمة مجموعات كبيرة جدًا من البيانات تُحلَّل لكشف الأنماط والعلاقات.",
    "الحوسبة السحابية توفر خوادم وتخزينًا وخدمات عبر الإنترنت بدل الأجهزة المحلية.",
    "الأمن السيبراني يحمي الأنظمة والشبكات والبيانات من الهجمات الرقمية.",
    "إنترنت الأشياء شبكة من الأجهزة المتصلة التي تجمع البيانات وتتبادلها.",
    "اللهجات العربية تختلف عن الفصحى في المفردات والنطق، مثل المصرية والخليجية والشامية والمغاربية.",
]


STOPWORDS = set(normalize(" ".join([
    "ما", "ماذا", "من", "هو", "هي", "هل", "كيف", "لماذا", "متى", "اين", "في", "على", "الى", "عن", "مع", "او", "و",
    "ثم", "هذا", "هذه", "ذلك", "التي", "الذي", "ان", "كان", "يمكن", "عبر", "بين", "مثل", "اشرح", "عرف", "اخبرني",
])).split())


def clean_for_search(text: str) -> str:
    return " ".join(w for w in re.findall(r"\w+", normalize(text)) if w not in STOPWORDS)


@st.cache_resource(show_spinner=False)
def kb_index():
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    return vec, vec.fit_transform([clean_for_search(t) for t in KB])


def retrieve(query: str, top_k: int = 3):
    vec, matrix = kb_index()
    scores = cosine_similarity(vec.transform([clean_for_search(query)]), matrix)[0]
    order = scores.argsort()[::-1][:top_k]
    return [(KB[i], float(scores[i])) for i in order]


# ---------------------------------------------------------------- glossary translation
GLOSSARY = {
    "مرحبا": "hello", "اهلا": "welcome", "كيف حالك": "how are you", "شكرا": "thank you", "صباح الخير": "good morning",
    "مساء الخير": "good evening", "مع السلامه": "goodbye", "من فضلك": "please", "نعم": "yes", "لا": "no",
    "الذكاء الاصطناعي": "artificial intelligence", "تعلم الاله": "machine learning", "التعلم العميق": "deep learning",
    "البيانات": "data", "الحاسوب": "computer", "اللغه": "language", "العربيه": "Arabic", "انا": "I", "انت": "you",
    "احب": "I love", "اريد": "I want", "اليوم": "today", "غدا": "tomorrow", "كتاب": "book", "مدرسه": "school",
    "جامعه": "university", "عمل": "work", "بيت": "house", "سياره": "car", "جميل": "beautiful", "كبير": "big",
    "صغير": "small", "جديد": "new", "و": "and", "في": "in", "على": "on", "من": "from", "الى": "to", "هذا": "this",
}


def glossary_translate(text: str):
    words = re.findall(r"\w+", normalize(text))
    out, known, i = [], 0, 0
    while i < len(words):
        two = " ".join(words[i:i + 2])
        if i + 1 < len(words) and two in GLOSSARY:
            out.append(GLOSSARY[two]); known += 2; i += 2; continue
        w = words[i]
        base = w[2:] if w.startswith("ال") and w not in GLOSSARY and ("ال" + w[2:]) not in GLOSSARY else w
        if w in GLOSSARY or base in GLOSSARY:
            out.append(GLOSSARY.get(w) or GLOSSARY[base]); known += 1
        elif w.startswith("و") and w[1:] in GLOSSARY:
            out.append("and " + GLOSSARY[w[1:]]); known += 1
        else:
            out.append(f"[{w}]")
        i += 1
    return " ".join(out), (known / len(words) if words else 0)


# ---------------------------------------------------------------- optional LLM
def get_setting(name: str) -> str:
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return str(value or os.getenv(name, "")).strip()


def llm_provider():
    if get_setting("GROQ_API_KEY"):
        return "Groq"
    if get_setting("OPENAI_API_KEY"):
        return "OpenAI"
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def llm_chat(system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if get_setting("GROQ_API_KEY"):
        from groq import Groq

        client = Groq(api_key=get_setting("GROQ_API_KEY"))
        model = get_setting("GROQ_MODEL") or "llama-3.3-70b-versatile"
    else:
        from openai import OpenAI

        client = OpenAI(api_key=get_setting("OPENAI_API_KEY"))
        model = get_setting("OPENAI_MODEL") or "gpt-4o-mini"
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.3, max_tokens=500)
    return resp.choices[0].message.content.strip()


def safe_llm(system: str, user: str):
    try:
        return llm_chat(system, user), None
    except Exception as exc:
        return None, f"تعذّر الوصول إلى النموذج اللغوي حاليًا ({type(exc).__name__})."


# ---------------------------------------------------------------- router
COMMANDS = ("ترجم", "لخص", "ملخص", "مشاعر", "شعور", "حلل", "لهجة")


def strip_command(prompt: str, words) -> str:
    for w in words:
        if prompt.startswith(w):
            return prompt[len(w):].lstrip(" :：-")
    return prompt


def respond(prompt: str) -> str:
    prompt = prompt.strip()[:MAX_CHARS]
    provider = llm_provider()

    if prompt.startswith("ترجم"):
        text = strip_command(prompt, ["ترجم"])
        if not text:
            return "اكتب النص بعد كلمة «ترجم». مثال: ترجم صباح الخير"
        if provider:
            answer, err = safe_llm(
                "Translate the user's text. Arabic goes to English, any other language goes to Arabic. Reply with the translation only.",
                text,
            )
            if answer:
                return f"**الترجمة ({provider}):**\n\n{answer}"
            note = err
        else:
            note = "لا يوجد مفتاح نموذج لغوي، لذلك استخدمت القاموس المصغّر."
        translation, coverage = glossary_translate(text)
        return f"**الترجمة (قاموس، تغطية {coverage:.0%}):**\n\n{translation}\n\n_{note}_"

    if prompt.startswith(("لخص", "ملخص")):
        text = strip_command(prompt, ["لخص", "ملخص"])
        if len(text.split()) < 8:
            return "أرسل نصًا أطول بعد كلمة «لخص» (عدة جمل) حتى أستطيع تلخيصه."
        picked, total = summarize(text, 2)
        return f"**الملخص (جملتان من {total}):**\n\n{' '.join(picked)}"

    if prompt.startswith(("مشاعر", "شعور", "حلل")):
        text = strip_command(prompt, ["مشاعر", "شعور", "حلل"]) or prompt
        label, score, hits = sentiment(text)
        words = "، ".join(w for w, _ in hits) or "لا توجد"
        return f"**تحليل المشاعر:** {label} (الدرجة {score:+.2f})\n\nالكلمات المؤثرة: {words}"

    if prompt.startswith("لهجة"):
        text = strip_command(prompt, ["لهجة"])
        if not text:
            return "اكتب الجملة بعد كلمة «لهجة». مثال: لهجة شو عم تعمل هلق"
        model = dialect_model()
        probs = model.predict_proba([normalize(text)])[0]
        ranked = sorted(zip(model.classes_, probs), key=lambda x: -x[1])
        detail = " · ".join(f"{k} {p:.0%}" for k, p in ranked)
        return f"**اللهجة المرجّحة:** {ranked[0][0]}\n\n{detail}"

    # Open question: retrieve, then (optionally) generate.
    results = retrieve(prompt)
    context = [t for t, s in results if s >= 0.15]
    if provider:
        system = (
            "أنت مساعد عربي ودود. أجب بالعربية الفصحى المبسطة وباختصار (أقل من 120 كلمة). "
            "استعن بالمقاطع المرفقة إن كانت ذات صلة، وإن لم تكن كافية فأجب من معرفتك العامة وقل ذلك بوضوح."
        )
        user = "المقاطع:\n" + "\n".join(f"- {c}" for c in context) + f"\n\nالسؤال: {prompt}"
        answer, err = safe_llm(system, user)
        if answer:
            sources = "\n".join(f"- {c}" for c in context)
            return answer + (f"\n\n**المصادر من قاعدة المعرفة:**\n{sources}" if context else "")
        prefix = f"_{err} هذه أقرب إجابة من قاعدة المعرفة:_\n\n"
    else:
        prefix = "**من قاعدة المعرفة:**\n\n"
    if not context:
        return "لا توجد معلومات كافية في قاعدة المعرفة. جرّب سؤالًا عن الذكاء الاصطناعي أو البيانات أو الحوسبة السحابية."
    return prefix + "\n\n".join(context[:2])


# ================================================================ UI
EXAMPLES = [
    "ما هو التعلم العميق؟",
    "ترجم صباح الخير، أحب الذكاء الاصطناعي",
    "مشاعر الخدمة ممتازة جدا لكن التطبيق بطيء",
    "لهجة شلونك وش مسوي اليوم",
    "لخص " + "يشهد العالم العربي اهتمامًا متزايدًا بالذكاء الاصطناعي. تستثمر الحكومات في مراكز البيانات. "
    "وتطلق الجامعات برامج في تعلم الآلة. ويرى الخبراء أن نقص البيانات العربية ما زال تحديًا رئيسيًا.",
]

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "مرحبًا! أنا مساعد عربي. اسألني سؤالًا، أو ابدأ رسالتك بـ «ترجم» أو «لخص» أو «مشاعر» أو «لهجة»."}
    ]

with st.sidebar:
    st.header("🌟 ميزات المساعد")
    provider = llm_provider()
    if provider:
        st.success(f"النموذج اللغوي مفعّل عبر {provider}.")
    else:
        st.info("وضع بدون إنترنت: لا يوجد مفتاح GROQ_API_KEY أو OPENAI_API_KEY، لذلك الترجمة بالقاموس والإجابات من قاعدة المعرفة فقط.")
    st.markdown(
        """
- 🔤 **ترجم** نص ← ترجمة
- 📝 **لخص** نص طويل ← ملخص
- 💬 **مشاعر** نص ← إيجابي / سلبي / محايد
- 🗣️ **لهجة** جملة ← مصرية / خليجية / شامية / فصحى
- 📚 أي سؤال آخر ← بحث في قاعدة المعرفة (RAG)
"""
    )
    st.subheader("جرّب مثالًا")
    clicked = None
    for i, ex in enumerate(EXAMPLES):
        label = ex if len(ex) < 45 else ex[:42] + "…"
        if st.button(label, key=f"ex_{i}", width="stretch"):
            clicked = ex
    if st.button("🗑️ مسح المحادثة", width="stretch"):
        st.session_state.messages = st.session_state.messages[:1]
        st.rerun()

st.title("🤖 المساعد العربي الذكي 2025")
st.caption("دردشة عربية تجمع أدوات معالجة اللغة المحلية مع نموذج لغوي اختياري.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("اكتب رسالتك هنا...") or clicked
if prompt and prompt.strip():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("جاري التفكير..."):
            response = respond(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

with st.expander("كيف يعمل المساعد؟"):
    st.markdown(
        """
- الرسالة تُوجَّه حسب الكلمة الأولى: «ترجم»، «لخص»، «مشاعر»، «لهجة»، وإلا تُعامل كسؤال.
- **الأسئلة:** بحث TF-IDF على مستوى الحروف في قاعدة معرفة صغيرة، ثم (عند توفر مفتاح) يُمرَّر أفضل مقطعين للنموذج اللغوي ليجيب بالاعتماد عليهما (RAG).
- **التلخيص:** LexRank عبر مكتبة sumy. **المشاعر:** قاموس مع معالجة النفي والتوكيد. **اللهجة:** Naive Bayes على مقاطع الحروف.
- مفاتيح النماذج تُقرأ من st.secrets أو متغيرات البيئة فقط، ولا تُخزَّن في الكود.
"""
    )
