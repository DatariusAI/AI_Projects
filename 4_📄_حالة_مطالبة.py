import streamlit as st
from utils import claim_status_lookup

st.set_page_config(page_title="حالة مطالبة", page_icon="📄", layout="wide")
st.title("الاستعلام عن حالة مطالبة")

claim_text = st.text_input("اكتب رقم المطالبة (مثال: 1001 أو C001001):", "")
if st.button("استعلام"):
    if not claim_text.strip():
        st.error("الرجاء إدخال رقم مطالبة.")
    else:
        st.success(claim_status_lookup(claim_text))
