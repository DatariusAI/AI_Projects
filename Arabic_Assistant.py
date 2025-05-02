import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline, MarianMTModel, MarianTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# Load Models and Prepare Data
# -----------------------------

# Knowledge base
kb = ["الذكاء الاصطناعي هو فرع من علوم الحاسوب",
      "تعلم الآلة هو جزء من الذكاء الاصطناعي",
      "المعالجة الطبيعية للغة تتعامل مع النصوص واللغة"]

# Sentence Transformer for RAG
embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
kb_embeddings = embed_model.encode(kb).astype("float32")

index = faiss.IndexFlatL2(kb_embeddings.shape[1])
index.add(kb_embeddings)

# Translation
mt_model_name = "Helsinki-NLP/opus-mt-ar-en"
mt_tokenizer = MarianTokenizer.from_pretrained(mt_model_name)
mt_model = MarianMTModel.from_pretrained(mt_model_name)

def translate(text):
    tokens = mt_tokenizer(text, return_tensors="pt", padding=True)
    translated = mt_model.generate(**tokens)
    return mt_tokenizer.decode(translated[0], skip_special_tokens=True)

# Sentiment Analysis
sentiment_pipeline = pipeline("sentiment-analysis", model="CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment")

# Dialect Identification
dialect_texts = ["إزايك عامل إيه؟", "شلونك اليوم؟", "كيفك شو الأخبار؟"]
dialect_labels = ["Egyptian", "Gulf", "Levantine"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dialect_texts)
dialect_clf = MultinomialNB()
dialect_clf.fit(X, dialect_labels)

# Summarization
summarizer = pipeline("summarization", model="csebuetnlp/mT5_multilingual_XLSum")


# -----------------------------
# Streamlit Interface
# -----------------------------

st.set_page_config(page_title="المساعد العربي الذكي", page_icon="🤖")
st.title("🤖 المساعد العربي الذكي")

st.write("أهلاً بك في المساعد. اكتب استفسارك وسأقوم بمساعدتك!")

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
        result = dialect_clf.predict(vectorizer.transform([user_input]))[0]
        st.write("**اللهجة:**", result)

    else:
        query_embedding = embed_model.encode([user_input]).astype("float32")
        D, I = index.search(query_embedding, k=1)
        result = kb[I[0][0]]
        st.write("**إجابة المعرفة:**", result)
