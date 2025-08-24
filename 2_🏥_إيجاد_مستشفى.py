import os, httpx
import streamlit as st
from dotenv import load_dotenv
from utils import hospital_card, places_search_enabled

load_dotenv()
st.set_page_config(page_title="إيجاد مستشفى", page_icon="🏥", layout="wide")
st.title("إيجاد مستشفى")

api_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
if not places_search_enabled():
    st.info("لتفعيل البحث، أضف GOOGLE_MAPS_API_KEY كمتغيّر بيئي (Environment Variable).")
    st.stop()

col1, col2 = st.columns([1.2,1])

with col1:
    query = st.text_input("اكتب اسم منطقة/مدينة أو اسم المستشفى:", placeholder="مثال: حولي، الكويت")
    if st.button("ابحث"):
        if not query.strip():
            st.error("الرجاء إدخال كلمة بحث.")
        else:
            with st.spinner("جاري البحث عن مستشفيات…"):
                url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
                params = {"query": f"{query} مستشفى", "type": "hospital", "key": api_key}
                r = httpx.get(url, params=params, timeout=20)
                data = r.json()
                results = data.get("results", [])[:10]
                if not results:
                    st.warning("لم يتم العثور على نتائج مناسبة.")
                for item in results:
                    hospital_card(item)

with col2:
    st.caption("نصائح: أدخل اسم الحي/المدينة. لاحقًا سنضيف فلترة حسب شبكة التأمين.")
