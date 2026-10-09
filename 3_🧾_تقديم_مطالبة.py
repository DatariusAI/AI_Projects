import streamlit as st
from dotenv import load_dotenv
from utils import faq_router, places_search_enabled

load_dotenv()

st.set_page_config(page_title="Med Assist", page_icon="🏥", layout="wide")

import os
if os.path.exists("assets/style.css"):  # optional custom styling
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

col1, col2 = st.columns([2,1], vertical_alignment="center")
with col1:
    st.markdown(
        '''
        <div class="big-hero">
          <div class="badge">MVP • Healthcare Helpdesk</div>
          <h1 style="margin-top:.5rem;">مرحبًا 👋 — كيف فينيساعدك اليوم؟</h1>
          <p>اسأل عن تنزيل التطبيق، إيجاد مستشفى قريب، طريقة تقديم مطالبة، أو حالة المطالبة.</p>
        </div>
        ''',
        unsafe_allow_html=True
    )
with col2:
    st.metric("جاهزية", "MVP", delta="Streamlit")

st.divider()

left, right = st.columns([1.6, 1])
with left:
    st.subheader("الدردشة")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    # Chat transcript
    for role, text in st.session_state.chat:
        css_class = "user" if role == "user" else ""
        st.markdown(f'<div class="chat-bubble {css_class}"><b>{"أنت" if role=="user" else "المساعد"}:</b> {text}</div>', unsafe_allow_html=True)

    with st.form("chat-form", clear_on_submit=True):
        user_msg = st.text_input("اكتب سؤالك بالعربي…", placeholder="مثال: حالة المطالبة 1001")
        submitted = st.form_submit_button("أرسل")
    if submitted and user_msg.strip():
        st.session_state.chat.append(("user", user_msg.strip()))
        reply = faq_router(user_msg.strip())
        st.session_state.chat.append(("assistant", reply))
        st.experimental_rerun()

with right:
    st.subheader("اختصارات سريعة")
    if st.button("كيف أنزّل التطبيق؟"):
        st.session_state.chat.append(("user", "كيف أنزّل التطبيق؟"))
        from utils import faq_router as fr; st.session_state.chat.append(("assistant", fr("كيف أنزّل التطبيق؟"))); st.experimental_rerun()
    if st.button("كيف أقدّم مطالبة؟"):
        st.session_state.chat.append(("user", "كيف أقدّم مطالبة؟"))
        from utils import faq_router as fr; st.session_state.chat.append(("assistant", fr("كيف أقدّم مطالبة؟"))); st.experimental_rerun()
    if st.button("أريد أقرب مستشفى"):
        st.switch_page("pages/2_🏥_إيجاد_مستشفى.py")
    if st.button("استعلام عن حالة مطالبة"):
        st.switch_page("pages/4_📄_حالة_مطالبة.py")

st.divider()
st.markdown('<p class="footer-note">تنويه: هذا نموذج أولي تعليمي. لا يقدّم نصيحة طبية حقيقية. للمطالبات وبيانات المرضى الفعلية، اربطه بواجهات نظامك الداخلي مع ضوابط الحماية والخصوصية.</p>', unsafe_allow_html=True)
