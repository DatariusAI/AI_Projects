import plotly.express as px
import streamlit as st

from utils import apply_rtl, find_claim, load_data

st.set_page_config(page_title="حالة مطالبة", page_icon="📄", layout="wide")
apply_rtl()

st.title("📄 الاستعلام عن حالة مطالبة")
st.caption("بيانات مطالبات تجريبية (20 مطالبة). جرّب الرقم C00004 أو C00016.")

data = load_data()
claims, encounters = data["claims"], data["encounters"]
if claims.empty:
    st.warning("لم يتم العثور على ملف المطالبات (claims.csv).")
    st.stop()

claim_text = st.text_input("رقم المطالبة", value="C00004", help="يمكن كتابة C00004 أو 4 فقط.")

if not claim_text.strip():
    st.error("الرجاء إدخال رقم مطالبة.")
else:
    row = find_claim(claim_text)
    if row is None:
        st.error("لم أجد مطالبة بهذا الرقم. الأرقام المتاحة من C00001 إلى C00020.")
    else:
        status = row["status"]
        box = {"Approved": st.success, "Denied": st.error, "Need Info": st.warning}.get(status, st.info)
        box(f"المطالبة {row['claim_id']}: {row['status_ar']}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("المبلغ المطلوب", f"{row['amount_billed']:,.2f}")
        c2.metric("المبلغ الموافق عليه", f"{row['amount_approved']:,.2f}")
        ratio = row["amount_approved"] / row["amount_billed"] if row["amount_billed"] else 0
        c3.metric("نسبة التغطية", f"{ratio:.0%}")
        c4.metric("آخر تحديث", row["last_update"])

        if str(row.get("denial_reason", "")).strip():
            st.write(f"**سبب الرفض:** {row['denial_reason']}")
        if status == "Need Info":
            st.write("**الخطوة التالية:** ارفع المستندات الناقصة (الفاتورة الأصلية أو تقرير الطبيب) من التطبيق.")

        if not encounters.empty:
            enc = encounters[encounters["encounter_id"] == row["encounter_id"]]
            if not enc.empty:
                e = enc.iloc[0]
                with st.expander("تفاصيل الزيارة الطبية", expanded=True):
                    st.write(f"**التاريخ:** {e['date']}")
                    st.write(f"**التشخيص:** {e['diagnosis_desc']} ({e['diagnosis_code']})")
                    st.write(f"**الإجراء:** {e['procedure_desc']} ({e['procedure_code']})")
                    st.write(f"**الطبيب / المنشأة:** {e['provider']} — {e['facility']}")

st.divider()
st.subheader("نظرة عامة على المطالبات")
counts = claims.groupby("status_ar").size().reset_index(name="count").sort_values("count", ascending=False)
col_a, col_b = st.columns([1, 1.3])
with col_a:
    fig = px.bar(counts, x="status_ar", y="count", labels={"status_ar": "الحالة", "count": "العدد"}, text="count")
    fig.update_layout(height=320, margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")
with col_b:
    st.dataframe(
        claims[["claim_id", "status_ar", "submitted_date", "amount_billed", "amount_approved"]].rename(
            columns={
                "claim_id": "رقم المطالبة",
                "status_ar": "الحالة",
                "submitted_date": "تاريخ التقديم",
                "amount_billed": "المطلوب",
                "amount_approved": "الموافق عليه",
            }
        ),
        hide_index=True,
        width="stretch",
        height=320,
    )
