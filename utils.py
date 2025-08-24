import os, re, html, pandas as pd
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def _read_csv(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_data():
    patients = _read_csv("patients.csv")
    encounters = _read_csv("encounters.csv")
    meds = _read_csv("medications.csv")
    claims = _read_csv("claims.csv")
    faq = _read_csv("faq.csv")
    return {"patients": patients, "encounters": encounters, "medications": meds, "claims": claims, "faq": faq}

def places_search_enabled() -> bool:
    return bool(os.getenv("GOOGLE_MAPS_API_KEY", "").strip())

def faq_examples():
    df = load_data()["faq"]
    if df.empty:
        return {}
    return {row["question"]: row["answer"] for _, row in df.iterrows()}

FAQ_PATTERNS = [
    (re.compile(r"(نز(?:ل|ول)|تحميل|تحميل.*تطبيق|كيف.*(أنز|نز|حمّل).*(تطبيق|اب))", re.I), "تقدر تنزّل التطبيق من App Store أو Google Play. ابحث باسم شركتنا، ثم حمّل وثبّت التطبيق."),
    (re.compile(r"(مستشفى|مستشفي|أقرب.*مستشفى|كيف.*ألقى.*مستشفى)", re.I), "أكيد! افتح صفحة 'إيجاد مستشفى' من القائمة وحدّد منطقتك، وأنا بساعدك بالبحث."),
    (re.compile(r"(مطالب(?:ة|اتي)|أقدّم.*مطالب|طريقة.*تقديم.*مطالب)", re.I), "لتقديم مطالبة: افتح التطبيق > المطالبات > إنشاء مطالبة جديدة، وارفع المستندات المطلوبة."),
]

CLAIM_ID_PATTERN = re.compile(r"(?:(?:claim|رقم|مطالب(?:تي)?)[^\d]{0,6})?(\d{3,8})", re.I)

def claim_status_lookup(text: str) -> str:
    m = CLAIM_ID_PATTERN.search(text)
    if not m:
        return "إذا كان لديك رقم مطالبة، اكتب مثل: «حالة المطالبة 12345»."
    claim_num = m.group(1)
    df = load_data()["claims"]
    if df.empty:
        return "لا توجد بيانات مطالبات في هذا النموذج."
    candidates = df[df["claim_id"].str.contains(claim_num)].head(1)
    if candidates.empty:
        return f"لم أجد مطالبة برقم {claim_num}. تأكد من الرقم."
    row = candidates.iloc[0]
    status_ar = {
        "Submitted": "تم الإرسال",
        "Pending Review": "قيد المراجعة",
        "Approved": "موافق عليها",
        "Denied": "مرفوضة",
        "Need Info": "بحاجة لمعلومات إضافية"
    }.get(row["status"], row["status"])
    extra = ""
    if str(row.get("denial_reason","")).strip():
        extra = f" — السبب: {row['denial_reason']}"
    return (f"رقم المطالبة: {row['claim_id']} — الحالة: {status_ar}"
            f" — تاريخ آخر تحديث: {row['last_update']}"
            f" — المبلغ المطلوب: {row['amount_billed']} — المبلغ الموافق عليه: {row['amount_approved']}{extra}")

def faq_router(text: str) -> str:
    # Claims first
    if re.search(r"(حالة.*مطالب|شو.*صار.*مطالب|وين.*وصلت.*مطالب|claim|رقم.*مطالب)", text, re.I):
        return claim_status_lookup(text)

    t = html.escape(text)
    for pat, ans in FAQ_PATTERNS:
        if pat.search(t):
            return ans
    # fallback to FAQ dataset by simple contains
    df = load_data()["faq"]
    if not df.empty:
        for _, row in df.iterrows():
            if any(tok in t for tok in row["question"].split()[:2]):
                return row["answer"]
    return "لم أفهم تمامًا سؤالك. جرّب: «كيف أنزّل التطبيق؟»، «كيف ألقى مستشفى؟» أو «كيف أقدّم مطالبة؟». وإذا عندك رقم مطالبة اكتب: «حالة المطالبة 12345»."

def star_rating(rating: float) -> str:
    full = "★" * int(rating)
    empty = "☆" * (5 - int(rating))
    return f"{full}{empty}"

def hospital_card(place: dict):
    name = place.get("name", "—")
    addr = place.get("formatted_address") or place.get("vicinity", "—")
    rating = place.get("rating", 0)
    html_block = f"""
    <div style="border:1px solid #E2E8F0;border-radius:14px;padding:1rem;margin-bottom:.75rem;background:#fff">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div>
          <h4 style="margin:0 0 .25rem 0;">{name}</h4>
          <div style="color:#486581">{addr}</div>
        </div>
        <div style="text-align:right;min-width:120px;">
          <div style="font-size:1.1rem;">التقييم</div>
          <div style="font-size:1.25rem;">{star_rating(float(rating))} <span style="font-size:.95rem;color:#486581">({rating})</span></div>
        </div>
      </div>
    </div>
    """
    st.markdown(html_block, unsafe_allow_html=True)
