import streamlit as st
import requests
import json
import nltk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
import time

# Try to download nltk data, but with a fallback
try:
    nltk.download('punkt', quiet=True)
except:
    pass  # Silently continue if download fails

# ---------------------------
# Knowledge Base (Expand this with more entries for better coverage)
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

# Initialize TF-IDF for RAG
try:
    vectorizer = TfidfVectorizer()
    kb_vectors = vectorizer.fit_transform(kb)
except Exception as e:
    st.error(f"Error initializing TF-IDF: {e}")

# ---------------------------
# External Translation API
# ---------------------------
def translate_text(text, target_language="en"):
    """
    Translate text using LibreTranslate API which is free and open-source
    """
    try:
        # Use LibreTranslate's public API (free but may have limitations)
        url = "https://libretranslate.de/translate"
        
        payload = {
            "q": text,
            "source": "ar",
            "target": target_language,
            "format": "text"
        }
        
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            return response.json()["translatedText"]
        else:
            # Fallback to dictionary for common phrases
            if text in translations:
                return translations[text]
            return f"Translation API error: {response.status_code}"
    except Exception as e:
        # Fallback to dictionary
        if text in translations:
            return translations[text]
        return f"Translation error: {str(e)}"

# Fallback translation dictionary
translations = {
    "مرحبا": "Hello",
    "كيف حالك": "How are you",
    "شكرا": "Thank you",
    "الذكاء الاصطناعي": "Artificial Intelligence",
    "تعلم الآلة": "Machine Learning",
    "معالجة اللغة الطبيعية": "Natural Language Processing",
    "البيانات الضخمة": "Big Data",
    "إنترنت الأشياء": "Internet of Things",
    "الحوسبة السحابية": "Cloud Computing",
    "الأمن السيبراني": "Cybersecurity"
}

# ---------------------------
# Advanced Summarization Using TextRank Algorithm
# ---------------------------
def summarize_text(text, num_sentences=3):
    try:
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.lex_rank import LexRankSummarizer
        
        parser = PlaintextParser.from_string(text, Tokenizer("arabic"))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, num_sentences)
        
        if summary:
            return " ".join([str(sentence) for sentence in summary])
        else:
            return basic_summarizer(text)
    except Exception as e:
        # Fallback to basic summarizer
        return basic_summarizer(text)

# Basic summarizer as fallback
def basic_summarizer(text, max_words=30):
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."

# ---------------------------
# Enhanced Sentiment Analysis
# ---------------------------
# Expanded word lists for better sentiment detection
positive_words = [
    "سعيد", "ممتاز", "جميل", "رائع", "مبسوط", "جيد", "مسرور", "رضا", "فرح", "حب",
    "نجاح", "إنجاز", "متفائل", "أمل", "مستمتع", "مبتهج", "متحمس", "ناجح", "محبوب",
    "مقدر", "متألق", "مثير", "ودود", "مريح", "مرضي", "عظيم", "مشجع", "محفز"
]

negative_words = [
    "حزين", "سيء", "كئيب", "ممل", "غاضب", "مضطرب", "قلق", "خائف", "محبط", "متعب",
    "فاشل", "مؤلم", "مخيف", "صعب", "مزعج", "مرهق", "مضايق", "متضايق", "محرج",
    "مهموم", "مشوش", "متوتر", "يائس", "مخيب", "مهدد", "مستاء", "متضرر", "محزن"
]

def analyze_sentiment(text):
    """Enhanced sentiment analysis with more terms and better scoring"""
    text = text.lower()
    
    # Count occurrences of positive and negative words
    pos_count = sum(1 for word in positive_words if word in text)
    neg_count = sum(1 for word in negative_words if word in text)
    
    # Determine sentiment based on count
    if pos_count > neg_count:
        confidence = min(100, int((pos_count / (pos_count + neg_count + 0.1)) * 100))
        return f"إيجابي (ثقة: {confidence}%)"
    elif neg_count > pos_count:
        confidence = min(100, int((neg_count / (pos_count + neg_count + 0.1)) * 100))
        return f"سلبي (ثقة: {confidence}%)"
    else:
        return "محايد"

# ---------------------------
# Improved Dialect Identification
# ---------------------------
dialect_texts = [
    # Egyptian dialect examples
    "إزيك عامل إيه؟", "أنا مش عارف", "دلوقتي", "عايز", "بص كدا", "ماشي", 
    # Gulf dialect examples
    "شلونك اليوم؟", "يبغالي", "وش فيك", "وينك؟", "من وين؟", "طيب",
    # Levantine dialect examples
    "كيفك شو الأخبار؟", "بدي روح", "هيك", "منيح", "حكي", "بكرا"
]

dialect_labels = [
    "مصرية", "مصرية", "مصرية", "مصرية", "مصرية", "مصرية",
    "خليجية", "خليجية", "خليجية", "خليجية", "خليجية", "خليجية",
    "شامية", "شامية", "شامية", "شامية", "شامية", "شامية"
]

try:
    dialect_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    X = dialect_vectorizer.fit_transform(dialect_texts)
    dialect_clf = MultinomialNB()
    dialect_clf.fit(X, dialect_labels)
except Exception as e:
    st.error(f"Error initializing dialect classifier: {e}")

def detect_dialect(text):
    """Detect Arabic dialect with confidence score"""
    try:
        text_vec = dialect_vectorizer.transform([text])
        predicted_dialect = dialect_clf.predict(text_vec)[0]
        proba = max(dialect_clf.predict_proba(text_vec)[0])
        confidence = int(proba * 100)
        return f"{predicted_dialect} (ثقة: {confidence}%)"
    except:
        # If classification fails, do basic keyword matching
        text = text.lower()
        if any(word in text for word in ["إزاي", "إزيك", "كده", "دلوقتي", "عايز"]):
            return "مصرية (تخمين بسيط)"
        elif any(word in text for word in ["شلون", "يبغى", "وش", "وين"]):
            return "خليجية (تخمين بسيط)"
        elif any(word in text for word in ["كيف", "هيك", "بدي", "منيح"]):
            return "شامية (تخمين بسيط)"
        return "غير معروفة"

# ---------------------------
# Enhanced RAG for Knowledge Retrieval
# ---------------------------
def knowledge_retrieval(query, top_k=2):
    """Retrieve the most relevant information from knowledge base"""
    try:
        query_vec = vectorizer.transform([query])
        similarity_scores = cosine_similarity(query_vec, kb_vectors)[0]
        
        # Get the indices of top_k most similar documents
        top_indices = similarity_scores.argsort()[-top_k:][::-1]
        
        if similarity_scores[top_indices[0]] > 0.1:  # Threshold for relevance
            results = [kb[i] for i in top_indices]
            return "\n\n".join(results)
        else:
            # If no good match, generate a response
            return "لا يوجد معلومات كافية في قاعدة المعرفة. يرجى إعادة صياغة سؤالك."
    except Exception as e:
        return f"حدث خطأ في البحث: {str(e)}"

# ---------------------------
# Streamlit UI with Chat Interface
# ---------------------------
def main():
    st.set_page_config(
        page_title="المساعد العربي الذكي", 
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 المساعد العربي الذكي")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("أكتب رسالتك هنا..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response in chat container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("⏳ جاري التفكير...")
            
            # Process the request
            response = process_request(prompt)
            
            # Update with the full response
            message_placeholder.markdown(response)
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Sidebar with examples and information
    with st.sidebar:
        st.header("🌟 ميزات المساعد")
        st.markdown("""
        **المساعد العربي الذكي** يمكنه:
        
        * 🔤 الترجمة من العربية إلى الإنجليزية
        * 📝 تلخيص النصوص العربية
        * 🧠 تحليل المشاعر في النص
        * 🗣️ التعرف على اللهجات العربية
        * 💡 الإجابة على الأسئلة من قاعدة المعرفة
        """)
        
        st.header("🔍 أمثلة للاستخدام")
        example1 = "ترجم الذكاء الاصطناعي يغير العالم بسرعة كبيرة"
        example2 = "ما هو الذكاء الاصطناعي؟"
        example3 = "حلل مشاعر أنا سعيد جدا بهذا التطبيق الرائع"
        example4 = "ما هي لهجة كيفك شو الأخبار؟"
        
        if st.button("ترجمة نص"):
            st.session_state.messages.append({"role": "user", "content": example1})
        
        if st.button("سؤال عن الذكاء الاصطناعي"):
            st.session_state.messages.append({"role": "user", "content": example2})
            
        if st.button("تحليل مشاعر"):
            st.session_state.messages.append({"role": "user", "content": example3})
            
        if st.button("كشف لهجة"):
            st.session_state.messages.append({"role": "user", "content": example4})

def process_request(user_input):
    """Process user input and return appropriate response"""
    user_input = user_input.strip()
    
    # Check for empty input
    if not user_input:
        return "يرجى إدخال نص للمعالجة."
    
    # Process translation requests
    if user_input.startswith("ترجم"):
        text_to_translate = user_input.replace("ترجم", "", 1).strip()
        if text_to_translate:
            with st.spinner("جاري الترجمة..."):
                translation = translate_text(text_to_translate)
            return f"**الترجمة:** \n\n{translation}"
        else:
            return "يرجى تقديم نص للترجمة بعد كلمة 'ترجم'."
    
    # Process summarization requests
    elif user_input.startswith("لخص") or user_input.startswith("ملخص"):
        text_to_summarize = user_input.replace("لخص", "", 1).replace("ملخص", "", 1).strip()
        if text_to_summarize:
            with st.spinner("جاري التلخيص..."):
                summary = summarize_text(text_to_summarize)
            return f"**الملخص:** \n\n{summary}"
        else:
            return "يرجى تقديم نص للتلخيص بعد كلمة 'لخص' أو 'ملخص'."
    
    # Process sentiment analysis requests
    elif any(word in user_input for word in ["شعور", "مشاعر", "حلل"]):
        cleaned_input = user_input
        for term in ["شعور", "مشاعر", "حلل"]:
            cleaned_input = cleaned_input.replace(term, "")
        
        text_to_analyze = cleaned_input.strip()
        if text_to_analyze:
            sentiment = analyze_sentiment(text_to_analyze)
            return f"**تحليل المشاعر:** \n\n{sentiment}"
        else:
            return "يرجى تقديم نص لتحليل المشاعر."
    
    # Process dialect identification requests
    elif "لهجة" in user_input:
        text_to_analyze = user_input.replace("لهجة", "").strip()
        if text_to_analyze:
            dialect = detect_dialect(text_to_analyze)
            return f"**اللهجة المكتشفة:** \n\n{dialect}"
        else:
            return "يرجى تقديم نص لتحديد اللهجة."
    
    # For all other queries, use the knowledge base
    else:
        with st.spinner("جاري البحث في قاعدة المعرفة..."):
            response = knowledge_retrieval(user_input)
        return f"**إجابة من قاعدة المعرفة:** \n\n{response}"

if __name__ == "__main__":
    main()
