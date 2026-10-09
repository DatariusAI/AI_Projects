from urllib.parse import quote_plus

import pandas as pd
import requests
import streamlit as st

from utils import apply_rtl, get_setting, hospital_card

st.set_page_config(page_title="إيجاد مستشفى", page_icon="🏥", layout="wide")
apply_rtl()

st.title("🏥 إيجاد مستشفى")
st.caption("ابحث عن مستشفى حسب المنطقة أو الاسم.")

# A small built-in list so the page is useful without any API key.
DEMO_HOSPITALS = pd.DataFrame(
    [
        ("المستشفى الأميري", "العاصمة", "الشرق، مدينة الكويت"),
        ("مستشفى مبارك الكبير", "حولي", "الجابرية، حولي"),
        ("مستشفى الفروانية", "الفروانية", "الفروانية"),
        ("مستشفى العدان", "الأحمدي", "هدية، الأحمدي"),
        ("مستشفى الجهراء", "الجهراء", "الجهراء"),
        ("مستشفى جابر الأحمد", "العاصمة", "جنوب السرة"),
        ("مستشفى الصباح", "العاصمة", "منطقة الصباح الصحية، الشويخ"),
        ("مستشفى الأمراض الصدرية", "العاصمة", "منطقة الصباح الصحية، الشويخ"),
    ],
    columns=["name", "area", "address"],
)


def maps_link(text: str) -> str:
    return "https://www.google.com/maps/search/?api=1&query=" + quote_plus(text)


@st.cache_data(ttl=3600, show_spinner=False)
def google_places(query: str, key: str):
    r = requests.get(
        "https://maps.googleapis.com/maps/api/place/textsearch/json",
        params={"query": f"{query} مستشفى", "type": "hospital", "key": key, "language": "ar"},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("status") not in ("OK", "ZERO_RESULTS"):
        raise RuntimeError(data.get("error_message") or data.get("status"))
    return data.get("results", [])[:10]


api_key = get_setting("GOOGLE_MAPS_API_KEY")
if api_key:
    st.success("البحث المباشر عبر Google Places مفعّل.")
else:
    st.info("وضع العرض: النتائج من قائمة تجريبية. لتفعيل البحث المباشر أضف GOOGLE_MAPS_API_KEY في st.secrets أو كمتغيّر بيئي.")

col1, col2 = st.columns([1.4, 1])

with col1:
    with st.form("search"):
        query = st.text_input("اسم المنطقة أو المستشفى", placeholder="مثال: حولي")
        submitted = st.form_submit_button("ابحث", type="primary")

    if submitted and not query.strip():
        st.error("الرجاء إدخال كلمة بحث.")
    elif submitted or not api_key:
        q = query.strip()
        if api_key and q:
            try:
                with st.spinner("جاري البحث عن مستشفيات…"):
                    results = google_places(q, api_key)
            except Exception as exc:
                st.error(f"تعذّر الاتصال بخدمة الخرائط حاليًا ({type(exc).__name__}). حاول لاحقًا.")
                results = None
            if results is not None:
                if not results:
                    st.warning("لم يتم العثور على نتائج مناسبة.")
                for item in results:
                    hospital_card(item)
        else:
            df = DEMO_HOSPITALS
            if q:
                df = df[df.apply(lambda r: q in r["name"] or q in r["area"] or q in r["address"], axis=1)]
            st.caption(f"عدد النتائج: {len(df)}")
            if df.empty:
                st.warning("لا توجد نتائج في القائمة التجريبية. جرّب البحث على الخريطة مباشرة.")
            for _, row in df.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{row['name']}**")
                    st.caption(f"{row['area']} — {row['address']}")
                    st.link_button("افتح على الخريطة", maps_link(row["name"]))
        if q:
            st.link_button(f"ابحث عن «{q}» في خرائط Google", maps_link(f"{q} مستشفى"))

with col2:
    st.subheader("نصائح")
    st.markdown(
        "- اكتب اسم الحي أو المحافظة للحصول على نتائج أدق.\n"
        "- في الحالات الطارئة اتصل بالطوارئ مباشرة.\n"
        "- تأكّد من أن المستشفى ضمن شبكة التأمين قبل الزيارة."
    )
    with st.expander("كيف تعمل الصفحة؟"):
        st.write(
            "عند توفّر مفتاح Google Maps تستدعي الصفحة واجهة Places Text Search وتعرض أول 10 نتائج "
            "(مع تخزين مؤقت لمدة ساعة). بدون مفتاح تعرض قائمة تجريبية ثابتة مع روابط مباشرة إلى خرائط Google."
        )
