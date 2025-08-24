import streamlit as st
from utils import faq_examples

st.set_page_config(page_title="الأسئلة المتكررة", page_icon="❓", layout="wide")

st.title("الأسئلة المتكررة (FAQ)")
st.write("أمثلة جاهزة يمكنك تعديلها لاحقًا:")
for q, a in faq_examples().items():
    with st.expander(q, expanded=False):
        st.write(a)
