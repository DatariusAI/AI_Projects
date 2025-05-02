import streamlit as st
from transformers import pipeline, MarianMTModel, MarianTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download required nltk packages
nltk.download('punkt')

# ---------------------------
# Knowledge Base and TF-IDF
# ---------------------------
kb = ["الذكاء الاصطناعي هو فرع من علوم الحاسوب",
      "تعلم الآلة هو جزء من الذكاء الاصطناعي",
      "المعالجة الطبيعية للغة تتعامل مع النصوص واللغة"]

vectorizer = TfidfVectorizer()
kb_vectors = vectorizer.fit_transform(kb)

def simple_rag(query):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, kb_vectors)
    best_idx = similarity.argmax()
    return kb[best_idx]

# ---------------------------
# Machine Translation (Arabic → English)
# ---------------------------
mt_model_name = "Helsinki-NLP/opus-mt-ar-en"
mt_tokenizer = MarianTokenizer.from_pretrained(mt_model_name)
mt_model = MarianMTModel.from_pretrained(mt_model_name)

def translate(text):
    tokens = mt_tokenizer(text, return_tensors="pt", padding=True)
    translated = mt_model.generate(**tokens)
    return mt_tokenizer.decode(translated[0], skip_special_tokens=True)

# ---------------------------
# Sentiment Analysis
# ---------------------------
sentiment_pipeline = pipeline("sentiment-analysis", model="CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment")

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
    return dialect_clf.predict(dialect_vectorizer.transform([text]))[0]

# ---------------------------
# Summarization
# ---------------------------
summarizer = pipeline("summarization", model="csebuetnlp/mT5_multilingual_XLSum")

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="المساعد العربي الذكي", page_icon="🤖")
st.title("🤖 المساعد العربي الذكي")

st.write("اكتب سؤالك أو طلبك في المربع أدناه:")

user_input = st.text_input("اكتب هنا:")

if st.button("أرسل"):

    if user_input.strip() == "":
        st.write("من فضلك أدخل نصاً.")
    
    elif "ترجم" in user_input:
        result = translate(user_input.replace("ترجم", ""))
        st.write("**الترجمة:**", result)

    elif "ملخص" in user_input:
        result = summarizer(user_input, max_length=40, min_length=10)[0]['summary_text']
        st.write("**الملخص:**", result)

    elif "شعور" in user_input:
        result = sentiment_pipeline(user_input)
        st.write("**المشاعر:**", result)

    elif "لهجة" in user_input:
        result = detect_dialect(user_input)
        st.write("**اللهجة:**", result)

    else:
        result = simple_rag(user_input)
        st.write("**إجابة المعرفة:**", result)
