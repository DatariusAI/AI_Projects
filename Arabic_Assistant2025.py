import streamlit as st
import requests
import json
import nltk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from langdetect import detect
import time

# Download NLTK punkt tokenizer
try:
    nltk.download('punkt', quiet=True)
except:
    pass

# ---------------------------
# Knowledge Base
# ---------------------------
kb = [
    "الذكاء الاصطناعي هو فرع من علوم الحاسوب يهتم بإنشاء آلات ذكية يمكنها التفكير والتعلم كالبشر",
    "تعلم الآلة هو جزء من الذكاء الاصطناعي يركز على تطوير خوارزميات تتعلم من البيانات وتتحسن مع الوقت",
    "المعالجة الطبيعية للغة تتعامل مع النصوص واللغة البشرية لفهمها وتحليلها بواسطة الحواسيب",
    "التعلم العميق هو فرع من تعلم الآلة يستخدم شبكات عصبية متعددة الطبقات لتحليل البيانات المعقدة",
    "الروبوتات هي آلات مبرمجة للقيام بمهام محددة وقد تكون مدعومة بالذكاء الاصطناعي",
    "البيانات الضخمة هي مجموعات كبيرة جدًا من البيانات التي يمكن تحليلها لكشف الأنماط والعلاقات",
    "الحوسبة السحابية توفر خدمات حوسبية عبر الإنترنت بدلاً من الاعتماد على الأجهزة المحلية",
    "الأمن السيبراني يهتم بحماية الأنظمة والشبكات والبرامج من الهجمات الرقمية",
    "إنترنت الأشياء يشير إلى شبكة من الأجهزة المتصلة التي تجمع وتبادل البيانات",
    "الواقع الافتراضي يخلق بيئة محاكاة يمكن للمستخدمين التفاعل معها بطريقة تبدو حقيقية"
]

# Cache the TF-IDF vectorizer and vectors
@st.cache_resource(show_spinner=False)
def load_vectorizer_and_kb():
    vectorizer = TfidfVectorizer()
    kb_vectors = vectorizer.fit_transform(kb)
    return vectorizer, kb_vectors

vectorizer, kb_vectors = load_vectorizer_and_kb()

# ---------------------------
# Translation API with Auto-Detect
# ---------------------------
def translate_text(text, target_language="en"):
    try:
        source_lang = detect(text)
        if source_lang == target_language:
            return text  # No need to translate
        
        url = "https://libretranslate.de/translate"
        payload = {
            "q": text,
            "source": source_lang,
            "target": target_language,
            "format": "text"
        }
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            return response.json()["translatedText"]
        else:
            return f"Translation failed: {response.status_code}"
    except Exception as e:
        return f"Translation error: {e}"

# ---------------------------
# Summarization (TextRank and fallback)
# ---------------------------
def summarize_text(text, num_sentences=3):
    try:
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.lex_rank import LexRankSummarizer

        parser = PlaintextParser.from_string(text, Tokenizer("arabic"))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, num_sentences)

        return " ".join([str(sentence) for sentence in summary]) or basic_summarizer(text)

    except Exception:
        return basic_summarizer(text)

def basic_summarizer(text, max_words=30):
    words = text.split()
    return " ".join(words[:max_words]) + "..." if len(words) > max_words else text

# ---------------------------
# Sentiment Analysis
# ---------------------------
positive_words = ["سعيد", "ممتاز", "جميل", "رائع", "مبتهج", "حب", "نجاح"]
negative_words = ["حزين", "سيء", "غاضب", "ممل", "قلق", "محبط", "متعب"]

def analyze_sentiment(text):
    text = text.lower()
    pos = sum(word in text for word in positive_words)
    neg = sum(word in text for word in negative_words)

    if pos > neg:
        return f"إيجابي (ثقة: {int((pos / (pos + neg + 0.1)) * 100)}%)"
    elif neg > pos:
        return f"سلبي (ثقة: {int((neg / (pos + neg + 0.1)) * 100)}%)"
    return "محايد"

# ---------------------------
# Dialect Detection
# ---------------------------
dialect_texts = ["إزيك عامل إيه؟", "شلونك اليوم؟", "كيفك شو الأخبار؟"]
dialect_labels = ["مصرية", "خليجية", "شامية"]

@st.cache_resource(show_spinner=False)
def train_dialect_model():
    vec = TfidfVectorizer(ngram_range=(1, 3))
    X = vec.fit_transform(dialect_texts)
    clf = MultinomialNB()
    clf.fit(X, dialect_labels)
    return vec, clf

dialect_vectorizer, dialect_clf = train_dialect_model()

def detect_dialect(text):
    try:
        text_vec = dialect_vectorizer.transform([text])
        pred = dialect_clf.predict(text_vec)[0]
        confidence = max(dialect_clf.predict_proba(text_vec)[0])
        return f"{pred} (ثقة: {int(confidence * 100)}%)"
    except:
        return "غير معروفة"

# ---------------------------
# Knowledge Retrieval (RAG)
# ---------------------------
def knowledge_retrieval(query, top_k=2):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, kb_vectors)[0]
    top_indices = scores.argsort()[-top_k:][::-1]

    if scores[top_indices[0]] > 0.1:
        return "\n\n".join([kb[i] for i in top_indices])
    return "لا يوجد معلومات كافية."

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="المساعد العربي الذكي", page_icon="🤖", layout="wide")
st.title("🤖 المساعد العربي الذكي")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("اكتب رسالتك هنا..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("⏳ جاري التفكير...")

        # Handle commands
        response = ""
        if prompt.startswith("ترجم"):
            text = prompt.replace("ترجم", "").strip()
            response = f"**الترجمة:**\n\n{translate_text(text)}"
        elif prompt.startswith("لخص") or prompt.startswith("ملخص"):
            text = prompt.replace("لخص", "").replace("ملخص", "").strip()
            response = f"**الملخص:**\n\n{summarize_text(text)}"
        elif any(word in prompt for word in ["شعور", "مشاعر", "حلل"]):
            response = f"**تحليل المشاعر:**\n\n{analyze_sentiment(prompt)}"
        elif "لهجة" in prompt:
            response = f"**اللهجة المكتشفة:**\n\n{detect_dialect(prompt)}"
        else:
            response = f"**إجابة من قاعدة المعرفة:**\n\n{knowledge_retrieval(prompt)}"

        # Simulate typing
        for i in range(0, len(response), 10):
            placeholder.markdown(response[:i+10])
            time.sleep(0.05)

        placeholder.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

with st.sidebar:
    st.header("🌟 ميزات المساعد")
    st.markdown("""
    **يمكنني القيام بـ:**
    - 🔤 الترجمة (تلقائي)
    - 📝 تلخيص النصوص
    - 💬 تحليل المشاعر
    - 🗣️ كشف اللهجة
    - 📚 استرجاع المعرفة
    """)
    
    st.header("🧠 هل كانت الاجابة مفيدة؟")
    st.button("👍 نعم")
    st.button("👎 لا")
