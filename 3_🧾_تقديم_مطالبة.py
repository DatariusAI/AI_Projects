"""Arabic insurance-claims helpdesk (entry point of the multipage app).

This file builds the navigation with st.navigation, so the four pages
work without a pages/ folder. The home page is the chat assistant below.
"""
import os

import streamlit as st

from utils import apply_rtl, faq_router, load_data

st.set_page_config(page_title="مساعد المطالبات", page_icon="🧾", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_QUESTIONS = [
    "كيف أنزّل التطبيق؟",
    "كيف أقدّم مطالبة؟",
    "ما هي المستندات المطلوبة للمطالبة؟",
    "حالة المطالبة C00016",
]


def ask(question: str):
    st.session_state.chat.append(("user", question))
    st.session_state.chat.append(("assistant", faq_router(question)))


def home():
    apply_rtl()
    if os.path.exists(os.path.join(HERE, "assets", "style.css")):  # optional custom styling
        with open(os.path.join(HERE, "assets", "style.css"), encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    st.title("🧾 مساعد مطالبات التأمين الصحي")
    st.write(
        "مرحبًا 👋 اسألني عن تنزيل التطبيق، إيجاد مستشفى، طريقة تقديم مطالبة، أو حالة مطالبتك برقمها. "
        "هذا نموذج أولي تعليمي يعمل على بيانات تجريبية."
    )

    claims = load_data()["claims"]
    if not claims.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("مطالبات تجريبية", len(claims))
        c2.metric("موافق عليها", int((claims["status"] == "Approved").sum()))
        c3.metric("بحاجة لمعلومات", int((claims["status"] == "Need Info").sum()))

    st.divider()
    if "chat" not in st.session_state:
        st.session_state.chat = []

    left, right = st.columns([1.6, 1])
    with right:
        st.subheader("اختصارات سريعة")
        for i, q in enumerate(SAMPLE_QUESTIONS):
            if st.button(q, key=f"quick_{i}", width="stretch"):
                ask(q)
        if st.button("🏥 أريد أقرب مستشفى", width="stretch"):
            st.switch_page(PAGES["hospital"])
        if st.button("📄 استعلام عن حالة مطالبة", width="stretch"):
            st.switch_page(PAGES["status"])
        if st.session_state.chat and st.button("🗑️ مسح المحادثة", width="stretch"):
            st.session_state.chat = []
            st.rerun()
        with st.expander("كيف يعمل المساعد؟"):
            st.write(
                "المساعد مبني على قواعد بسيطة: يتعرّف على نية السؤال بتعابير نمطية (regex)، "
                "ثم يبحث في ملف الأسئلة المتكررة، أو في ملف المطالبات إذا ذكرت رقم مطالبة. "
                "لا يستخدم نموذجًا لغويًا ولا يرسل بياناتك إلى أي خدمة خارجية."
            )

    with left:
        st.subheader("الدردشة")
        if not st.session_state.chat:
            st.caption("ابدأ بكتابة سؤالك في الأسفل أو اختر سؤالًا جاهزًا.")
        for role, text in st.session_state.chat:
            with st.chat_message("user" if role == "user" else "assistant"):
                st.write(text)

    user_msg = st.chat_input("اكتب سؤالك بالعربي… مثال: حالة المطالبة 4")
    if user_msg and user_msg.strip():
        ask(user_msg.strip())
        st.rerun()

    st.caption(
        "تنويه: هذا نموذج أولي تعليمي ولا يقدّم نصيحة طبية. للربط ببيانات حقيقية يلزم الاتصال بأنظمة "
        "المطالبات الداخلية مع ضوابط الحماية والخصوصية."
    )


PAGES = {
    "home": st.Page(home, title="تقديم مطالبة", icon="🧾", default=True),
    "faq": st.Page(os.path.join(HERE, "1_❓_الأسئلة_المتكررة.py"), title="الأسئلة المتكررة", icon="❓", url_path="faq"),
    "hospital": st.Page(os.path.join(HERE, "2_🏥_إيجاد_مستشفى.py"), title="إيجاد مستشفى", icon="🏥", url_path="hospital"),
    "status": st.Page(os.path.join(HERE, "4_📄_حالة_مطالبة.py"), title="حالة مطالبة", icon="📄", url_path="status"),
}

st.navigation(list(PAGES.values())).run()
