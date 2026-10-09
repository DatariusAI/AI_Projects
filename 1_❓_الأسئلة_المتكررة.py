import streamlit as st

from utils import apply_rtl, faq_examples

st.set_page_config(page_title="الأسئلة المتكررة", page_icon="❓", layout="wide")
apply_rtl()

st.title("❓ الأسئلة المتكررة")
st.caption("أجوبة سريعة لأكثر الأسئلة شيوعًا حول التطبيق والمطالبات. البيانات تجريبية لأغراض العرض.")

faqs = faq_examples()
if not faqs:
    st.warning("لم يتم العثور على ملف الأسئلة (faq.csv).")
    st.stop()

query = st.text_input("ابحث في الأسئلة", placeholder="مثال: مطالبة، مستشفى، التطبيق")
q = query.strip()
shown = {k: v for k, v in faqs.items() if not q or q in k or q in v}

st.caption(f"عدد النتائج: {len(shown)} من {len(faqs)}")
if not shown:
    st.info("لا توجد نتائج مطابقة. جرّب كلمة أخرى، أو اسأل المساعد في صفحة «تقديم مطالبة».")
for question, answer in shown.items():
    with st.expander(question, expanded=bool(q)):
        st.write(answer)
