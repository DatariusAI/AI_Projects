import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk

# Download NLTK resources (with quiet=True to avoid printing output)
try:
    nltk.download('punkt', quiet=True)
except:
    st.warning("NLTK resources could not be downloaded, summarization may not work correctly.")

# ---------------------------
# Knowledge Base + TF-IDF (RAG)
# ---------------------------
kb = [
    "الذكاء الاصطناعي هو فرع من علوم الحاسوب",
    "تعلم الآلة هو جزء من الذكاء الاصطناعي",
    "المعالجة الطبيعية للغة تتعامل مع النصوص واللغة"
]
vectorizer = TfidfVectorizer()
kb_vectors = vectorizer.fit_transform(kb)

def simple_rag(query):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, kb_vectors)
    best_idx = similarity.argmax()
    return kb[best_idx]

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

# ---------------------------
# Summarization (TextRank via sumy)
# ---------------------------
def simple_summarizer(text, num_sentences=2):
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("arabic"))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, num_sentences)
        return " ".join([str(sentence) for sentence in summary])
    except Exception as e:
        return f"خطأ في التلخيص: {str(e)}"

# ---------------------------
# Simple translation simulator (without using heavy models)
# ---------------------------
translation_examples = {
    "مرحبا": "Hello",
    "كيف حالك": "How are you",
    "شكرا": "Thank you",
    "الذكاء الاصطناعي": "Artificial Intelligence"
}

def simple_translate(text):
    return translation_examples.get(text.strip(), "Translation not available for this text")

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="المساعد العربي الذكي", page_icon="🤖")
st.title("🤖 المساعد العربي الذكي")
st.write("اكتب طلبك أو سؤالك:")

user_input = st.text_input("اكتب هنا:")

if st.button("أرسل"):
    if user_input.strip() == "":
        st.write("من فضلك أدخل نصاً.")
    
    elif "ترجم" in user_input:
        text_to_translate = user_input.replace("ترجم", "").strip()
        result = simple_translate(text_to_translate)
        st.write("**الترجمة:**", result)
    
    elif "ملخص" in user_input:
        text_to_summarize = user_input.replace("ملخص", "").strip()
        result = simple_summarizer(text_to_summarize)
        st.write("**الملخص:**", result)
    
    elif "شعور" in user_input or "مشاعر" in user_input:
        result = simple_sentiment(user_input)
        st.write("**تحليل المشاعر:**", result)
    
    elif "لهجة" in user_input:
        text_to_analyze = user_input.replace("لهجة", "").strip()
        result = detect_dialect(text_to_analyze)
        st.write("**اللهجة المكتشفة:**", result)
    
    else:
        result = simple_rag(user_input)
        st.write("**من قاعدة المعرفة:**", result)

st.markdown("---")
st.write("""
**تعليمات الاستخدام:**
- اكتب 'ترجم' متبوعًا بالنص لترجمة النص من العربية إلى الإنجليزية
- اكتب 'ملخص' متبوعًا بالنص لتلخيص النص
- اكتب 'شعور' أو 'مشاعر' متبوعًا بالنص لتحليل المشاعر
- اكتب 'لهجة' متبوعًا بالنص لاكتشاف اللهجة العربية
- أو اكتب أي استعلام للبحث في قاعدة المعرفة
""")
