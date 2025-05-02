import streamlit as st
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB

# Try to download nltk data, but with a fallback
try:
    nltk.download('punkt', quiet=True)
except:
    pass  # Silently continue if download fails

# ---------------------------
# Knowledge Base
# ---------------------------
kb = [
    "الذكاء الاصطناعي هو فرع من علوم الحاسوب",
    "تعلم الآلة هو جزء من الذكاء الاصطناعي",
    "المعالجة الطبيعية للغة تتعامل مع النصوص واللغة"
]

# ---------------------------
# Simple RAG (TF-IDF)
# ---------------------------
try:
    vectorizer = TfidfVectorizer()
    kb_vectors = vectorizer.fit_transform(kb)
    
    def simple_rag(query):
        query_vec = vectorizer.transform([query])
        similarity = cosine_similarity(query_vec, kb_vectors)
        best_idx = similarity.argmax()
        return kb[best_idx]
except:
    def simple_rag(query):
        return "عذراً، لا يمكنني الإجابة على هذا السؤال."

# ---------------------------
# Simple Sentiment Analysis (Rule-based)
# ---------------------------
positive_words = ["سعيد", "ممتاز", "جميل", "رائع", "مبسوط", "جيد", "مسرور"]
negative_words = ["حزين", "سيء", "كئيب", "ممل", "غاضب", "مضطرب"]

def simple_sentiment(text):
    text = text.lower()
    if any(word in text for word in positive_words):
        return "إيجابي"
    elif any(word in text for word in negative_words):
        return "سلبي"
    else:
        return "محايد"

# ---------------------------
# Dialect Identification
# ---------------------------
try:
    dialect_texts = ["إزايك عامل إيه؟", "شلونك اليوم؟", "كيفك شو الأخبار؟"]
    dialect_labels = ["Egyptian", "Gulf", "Levantine"]
    dialect_vectorizer = TfidfVectorizer()
    X = dialect_vectorizer.fit_transform(dialect_texts)
    dialect_clf = MultinomialNB()
    dialect_clf.fit(X, dialect_labels)
    
    def detect_dialect(text):
        try:
            return dialect_clf.predict(dialect_vectorizer.transform([text]))[0]
        except:
            return "غير معروفة"
except:
    def detect_dialect(text):
        return "غير معروفة"

# ---------------------------
# Simple summarization (word count based)
# ---------------------------
def basic_summarizer(text, max_words=20):
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."

# ---------------------------
# Simple translation dictionary
# ---------------------------
translations = {
    "مرحبا": "Hello",
    "كيف حالك": "How are you",
    "شكرا": "Thank you",
    "الذكاء الاصطناعي": "Artificial Intelligence"
}

def simple_translate(text):
    return translations.get(text.strip(), "Translation not available")

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🤖 المساعد العربي الذكي")
st.write("اكتب طلبك أو سؤالك:")

user_input = st.text_input("اكتب هنا:")

if st.button("أرسل"):
    if not user_input.strip():
        st.warning("الرجاء إدخال نص")
    elif "ترجم" in user_input:
        text = user_input.replace("ترجم", "").strip()
        st.success(f"الترجمة: {simple_translate(text)}")
    elif "ملخص" in user_input:
        text = user_input.replace("ملخص", "").strip()
        st.success(f"الملخص: {basic_summarizer(text)}")
    elif "شعور" in user_input or "مشاعر" in user_input:
        st.success(f"تحليل المشاعر: {simple_sentiment(user_input)}")
    elif "لهجة" in user_input:
        text = user_input.replace("لهجة", "").strip()
        st.success(f"اللهجة: {detect_dialect(text)}")
    else:
        st.success(f"إجابة: {simple_rag(user_input)}")

st.markdown("---")
st.markdown("""
**تعليمات الاستخدام:**
- اكتب 'ترجم' متبوعًا بالنص لترجمة النص من العربية إلى الإنجليزية
- اكتب 'ملخص' متبوعًا بالنص لتلخيص النص
- اكتب 'شعور' أو 'مشاعر' متبوعًا بالنص لتحليل المشاعر
- اكتب 'لهجة' متبوعًا بالنص لاكتشاف اللهجة العربية
- أو اكتب أي استعلام للبحث في قاعدة المعرفة
""")
