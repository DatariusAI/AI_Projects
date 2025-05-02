import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk

# Download NLTK resources
nltk.download('punkt', quiet=True)

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
# Machine Translation (Arabic -> English)
# ---------------------------
@st.cache_resource
def load_translation_model():
    try:
        mt_model_name = "Helsinki-NLP/opus-mt-ar-en"
        mt_tokenizer = MarianTokenizer.from_pretrained(mt_model_name)
        mt_model = MarianMTModel.from_pretrained(mt_model_name)
        return mt_tokenizer, mt_model
    except Exception as e:
        st.error(f"Error loading translation model: {e}")
        return None, None

mt_tokenizer, mt_model = load_translation_model()

def translate(text):
    if mt_tokenizer is None or mt_model is None:
        return "فشل تحميل نموذج الترجمة"
    
    try:
        tokens = mt_tokenizer(text, return_tensors="pt", padding=True)
        translated = mt_model.generate(**tokens)
        return mt_tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception as e:
        return f"خطأ في الترجمة: {str(e)}"

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
# Streamlit UI
# ---------------------------
def main():
    st.set_page_config(page_title="المساعد العربي الذكي", page_icon="🤖")
    
    st.title("🤖 المساعد العربي الذكي")
    st.write("اكتب طلبك أو سؤالك:")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    user_input = st.chat_input("اكتب هنا...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message in chat
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Process the input and generate a response
        response = process_input(user_input)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display assistant response in chat
        with st.chat_message("assistant"):
            st.markdown(response)

def process_input(user_input):
    if user_input.strip() == "":
        return "من فضلك أدخل نصاً."
    
    elif "ترجم" in user_input:
        text_to_translate = user_input.replace("ترجم", "").strip()
        result = translate(text_to_translate)
        return f"**الترجمة:**\n{result}"
    
    elif "ملخص" in user_input:
        text_to_summarize = user_input.replace("ملخص", "").strip()
        result = simple_summarizer(text_to_summarize)
        return f"**الملخص:**\n{result}"
    
    elif "شعور" in user_input or "مشاعر" in user_input:
        result = simple_sentiment(user_input)
        return f"**تحليل المشاعر:**\n{result}"
    
    elif "لهجة" in user_input:
        text_to_analyze = user_input.replace("لهجة", "").strip()
        result = detect_dialect(text_to_analyze)
        return f"**اللهجة المكتشفة:**\n{result}"
    
    else:
        result = simple_rag(user_input)
        return f"**من قاعدة المعرفة:**\n{result}"

if __name__ == "__main__":
    main()
