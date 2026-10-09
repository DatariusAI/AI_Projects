"""Shared helpers for the Arabic insurance-claims assistant (pages 1-4).

All data is synthetic and lives in small CSV files next to this module
(or in an optional ``data/`` folder). Nothing here talks to a real
claims system.
"""
import html
import os
import re

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIRS = [os.path.join(BASE_DIR, "data"), BASE_DIR]

STATUS_AR = {
    "Submitted": "تم الإرسال",
    "Pending Review": "قيد المراجعة",
    "Approved": "موافق عليها",
    "Denied": "مرفوضة",
    "Need Info": "بحاجة لمعلومات إضافية",
}

RTL_CSS = """
<style>
[data-testid="stMain"], [data-testid="stSidebar"] { direction: rtl; text-align: right; }
[data-testid="stMain"] input, [data-testid="stMain"] textarea { direction: rtl; text-align: right; }
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"], code, pre { direction: ltr; text-align: left; }
</style>
"""


def apply_rtl():
    """Right-to-left layout for the Arabic pages."""
    st.markdown(RTL_CSS, unsafe_allow_html=True)


def get_setting(name: str) -> str:
    """Read a key from st.secrets first, then the environment. Never raises."""
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return str(value or os.getenv(name, "")).strip()


def _read_csv(name):
    for folder in DATA_DIRS:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return pd.read_csv(path, dtype=str).fillna("")
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_data():
    data = {
        "patients": _read_csv("patients.csv"),
        "encounters": _read_csv("encounters.csv"),
        "medications": _read_csv("medications.csv"),
        "claims": _read_csv("claims.csv"),
        "faq": _read_csv("faq.csv"),
    }
    claims = data["claims"]
    if not claims.empty:
        for col in ("amount_billed", "amount_approved"):
            claims[col] = pd.to_numeric(claims[col], errors="coerce").fillna(0.0)
        claims["status_ar"] = claims["status"].map(STATUS_AR).fillna(claims["status"])
    return data


def places_search_enabled() -> bool:
    return bool(get_setting("GOOGLE_MAPS_API_KEY"))


def faq_examples():
    df = load_data()["faq"]
    if df.empty:
        return {}
    return dict(zip(df["question"], df["answer"]))


FAQ_PATTERNS = [
    (re.compile(r"(نز(?:ل|ول)|تحميل|تطبيق)"), "تقدر تنزّل التطبيق من App Store أو Google Play. ابحث باسم الشركة، ثم حمّل وثبّت التطبيق."),
    (re.compile(r"(مستشف[ىي])"), "أكيد! افتح صفحة «إيجاد مستشفى» من القائمة الجانبية، واكتب منطقتك."),
    (re.compile(r"(مستند|مستندات|أوراق)"), "عادةً الفاتورة الأصلية، تقرير الطبيب، وبطاقة التأمين. قد نطلب مستندات إضافية."),
    (re.compile(r"(أوقات|دوام|مركز الاتصال)"), "يعمل مركز الاتصال يوميًا من 8 صباحًا حتى 8 مساءً."),
    (re.compile(r"(أقد[ّ]?م|تقديم|طريقة).{0,20}مطالب"), "لتقديم مطالبة: افتح التطبيق > المطالبات > إنشاء مطالبة جديدة، وارفع المستندات المطلوبة."),
]

CLAIM_ID_PATTERN = re.compile(r"(?:C\s*)?(\d{1,8})", re.I)


def find_claim(text: str):
    """Return the claim row (a pandas Series) matching the number in text, or None.

    Accepts "C00004", "c4", "4" or "حالة المطالبة 4".
    """
    m = CLAIM_ID_PATTERN.search((text or "")[:100])
    if not m:
        return None
    claims = load_data()["claims"]
    if claims.empty:
        return None
    wanted = int(m.group(1))
    numbers = pd.to_numeric(claims["claim_id"].str.extract(r"(\d+)")[0], errors="coerce")
    hit = claims[numbers == wanted]
    return None if hit.empty else hit.iloc[0]


def claim_status_lookup(text: str) -> str:
    if not CLAIM_ID_PATTERN.search((text or "")[:100]):
        return "إذا كان لديك رقم مطالبة، اكتب مثل: «حالة المطالبة C00004»."
    if load_data()["claims"].empty:
        return "لا توجد بيانات مطالبات في هذا النموذج."
    row = find_claim(text)
    if row is None:
        return "لم أجد مطالبة بهذا الرقم. الأرقام المتاحة في النموذج من C00001 إلى C00020."
    extra = f" — السبب: {row['denial_reason']}" if str(row.get("denial_reason", "")).strip() else ""
    return (
        f"رقم المطالبة: {row['claim_id']} — الحالة: {row['status_ar']}"
        f" — آخر تحديث: {row['last_update']}"
        f" — المبلغ المطلوب: {row['amount_billed']:.2f} — المبلغ الموافق عليه: {row['amount_approved']:.2f}{extra}"
    )


# Cap user-supplied text before evaluating any regex. The router only
# needs to recognise short conversational inputs; capping also keeps the
# regexes cheap on very long inputs.
MAX_ROUTER_INPUT = 500


def faq_router(text: str) -> str:
    text = (text or "")[:MAX_ROUTER_INPUT]
    if re.search(r"(حال[ةه].{0,20}مطالب|وين.{0,20}مطالب|claim|رقم.{0,10}مطالب|C\d{3,})", text, re.I):
        return claim_status_lookup(text)
    for pat, ans in FAQ_PATTERNS:
        if pat.search(text):
            return ans
    df = load_data()["faq"]
    if not df.empty:
        words = {w for w in re.findall(r"\w{3,}", text)}
        best, best_score = None, 0
        for _, row in df.iterrows():
            score = len(words & set(re.findall(r"\w{3,}", row["question"])))
            if score > best_score:
                best, best_score = row["answer"], score
        if best:
            return best
    return (
        "لم أفهم سؤالك تمامًا. جرّب: «كيف أنزّل التطبيق؟»، «كيف ألقى مستشفى؟» أو «كيف أقدّم مطالبة؟». "
        "وإذا عندك رقم مطالبة اكتب: «حالة المطالبة C00004»."
    )


def star_rating(rating: float) -> str:
    r = max(0, min(5, int(round(rating or 0))))
    return "★" * r + "☆" * (5 - r)


def hospital_card(place: dict):
    name = html.escape(str(place.get("name", "—")))
    addr = html.escape(str(place.get("formatted_address") or place.get("vicinity", "—")))
    try:
        rating = float(place.get("rating", 0) or 0)
    except (TypeError, ValueError):
        rating = 0.0
    with st.container(border=True):
        st.markdown(f"**{name}**")
        st.caption(addr)
        if rating:
            st.markdown(f"{star_rating(rating)} ({rating:.1f})")
