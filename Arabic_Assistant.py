"""المساعد العربي الذكي: أدوات معالجة لغة عربية خفيفة تعمل بدون إنترنت.

Offline Arabic NLP toolkit: extractive summary (LexRank), lexicon sentiment,
dialect identification (char n-gram Naive Bayes), TF-IDF retrieval and a
small glossary translator. No API keys and no model downloads.
"""
import re

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer

st.set_page_config(page_title="المساعد العربي الذكي", page_icon="🤖", layout="wide")
st.markdown(
    """<style>
[data-testid="stMain"], [data-testid="stSidebar"] { direction: rtl; text-align: right; }
[data-testid="stMain"] input, [data-testid="stMain"] textarea { direction: rtl; text-align: right; }
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] { direction: ltr; }
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


# ================================================================ UI
st.title("🤖 المساعد العربي الذكي")
st.write("مجموعة أدوات لمعالجة النصوص العربية تعمل مباشرة في المتصفح، بدون مفاتيح API وبدون تحميل نماذج. اختر أداة وجرّب النص الجاهز أو اكتب نصك.")

tab_sum, tab_sent, tab_dial, tab_kb, tab_tr = st.tabs(["📝 التلخيص", "💬 المشاعر", "🗣️ اللهجة", "📚 قاعدة المعرفة", "🔤 الترجمة"])

SAMPLE_ARTICLE = (
    "يشهد العالم العربي اهتمامًا متزايدًا بالذكاء الاصطناعي. تستثمر الحكومات في مراكز البيانات والحوسبة السحابية. "
    "وتطلق الجامعات برامج جديدة في تعلم الآلة وعلوم البيانات. كما تعمل الشركات الناشئة على نماذج لغوية تفهم اللهجات العربية. "
    "ويرى الخبراء أن نقص البيانات العربية عالية الجودة ما زال تحديًا رئيسيًا. لذلك تبرز مبادرات لجمع النصوص العربية وتنظيمها. "
    "ومن المتوقع أن يسهم ذلك في تحسين الترجمة الآلية والمساعدات الذكية باللغة العربية."
)

with tab_sum:
    text = st.text_area("النص المراد تلخيصه", SAMPLE_ARTICLE, height=180, max_chars=MAX_CHARS)
    n = st.slider("عدد الجمل في الملخص", 1, 5, 2)
    if not text.strip():
        st.info("اكتب نصًا من عدة جمل للتلخيص.")
    else:
        picked, total = summarize(text, n)
        st.markdown("**الملخص:**")
        st.success(" ".join(picked) or "لم أجد جملًا كافية.")
        c1, c2 = st.columns(2)
        c1.metric("جمل النص الأصلي", total)
        c2.metric("نسبة الاختصار", f"{1 - len(' '.join(picked)) / max(len(text), 1):.0%}")

with tab_sent:
    text = st.text_area("النص", "الخدمة كانت ممتازة جدا والموظفين رائعين، لكن التطبيق بطيء شوي", height=100, max_chars=MAX_CHARS)
    if text.strip():
        label, score, hits = sentiment(text)
        c1, c2 = st.columns(2)
        c1.metric("التصنيف", label)
        c2.metric("الدرجة (من -1 إلى 1)", f"{score:+.2f}")
        if hits:
            st.write("**الكلمات المؤثرة:** " + "، ".join(f"{w} ({'+' if p > 0 else ''}{p:g})" for w, p in hits))
        else:
            st.caption("لم أجد كلمات من القاموس العاطفي في النص.")

with tab_dial:
    text = st.text_input("جملة باللهجة", "شلونك؟ وش مسوي اليوم")
    if text.strip():
        model = dialect_model()
        probs = model.predict_proba([normalize(text)])[0]
        df = pd.DataFrame({"اللهجة": model.classes_, "الاحتمال": probs}).sort_values("الاحتمال", ascending=False)
        st.metric("اللهجة المرجّحة", df.iloc[0]["اللهجة"], f"ثقة {df.iloc[0]['الاحتمال']:.0%}")
        fig = px.bar(df, x="الاحتمال", y="اللهجة", orientation="h", range_x=[0, 1])
        fig.update_layout(height=260, margin=dict(t=10, b=10), yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")
    st.caption("جرّب: «عايز اروح دلوقتي» أو «شو عم تعمل هلق» أو «ماذا تفعل الآن».")

with tab_kb:
    query = st.text_input("سؤالك", "ما هو التعلم العميق؟")
    if query.strip():
        results = retrieve(query)
        best_text, best_score = results[0]
        if best_score < 0.15:
            st.warning("لا توجد معلومات كافية في قاعدة المعرفة عن هذا السؤال.")
        else:
            st.success(best_text)
        with st.expander("أقرب المقاطع ودرجة التشابه"):
            for t, s in results:
                st.write(f"`{s:.2f}` — {t}")

with tab_tr:
    text = st.text_input("نص عربي قصير", "شكرا، أحب الذكاء الاصطناعي")
    if text.strip():
        translation, coverage = glossary_translate(text)
        st.success(translation)
        st.progress(coverage, text=f"تغطية القاموس: {coverage:.0%}")
        st.caption("ترجمة كلمة بكلمة من قاموس مصغّر للعرض فقط. الكلمات بين [ ] غير موجودة في القاموس. للترجمة الكاملة جرّب «المساعد العربي 2025».")

with st.expander("كيف يعمل التطبيق؟"):
    st.markdown(
        """
- **التلخيص:** خوارزمية LexRank (مكتبة sumy) ترتّب الجمل حسب تشابهها مع باقي النص وتختار الأهم.
- **المشاعر:** قاموس كلمات إيجابية وسلبية مع معالجة النفي («مش»، «لا»، «ليس») والتوكيد («جدا»، «وايد»).
- **اللهجة:** مصنّف Naive Bayes على مقاطع حروف (2-4) مدرّب على جمل قصيرة لأربع فئات: مصرية، خليجية، شامية، فصحى.
- **قاعدة المعرفة:** بحث TF-IDF على مستوى الحروف مع تشابه جيب التمام (cosine) لإيجاد أقرب مقطع.
- **الترجمة:** قاموس مصغّر للعرض. كل شيء يعمل محليًا ولا يرسل النص لأي خدمة.
"""
    )
