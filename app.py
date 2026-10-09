"""Nutrition Disorder Agent.

One file, two ways to run it:
  * streamlit run app.py              -> chat UI (Streamlit Community Cloud)
  * uvicorn app:app --port 7860       -> REST API, POST /ask  (Docker / Hugging Face)

The LLM is served by Groq. Set GROQ_API_KEY as an environment variable or a
Streamlit secret. Without a key the app still loads and explains what is missing.
"""
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
SYSTEM = (
    "You are a careful nutrition assistant. You explain nutrition-related disorders "
    "(for example iron-deficiency anaemia, vitamin D or B12 deficiency, malnutrition, "
    "obesity, coeliac disease, lactose intolerance), typical dietary approaches and the "
    "evidence behind them. Be concise and practical. You do not diagnose, and you tell "
    "the user to see a doctor or registered dietitian for personal medical decisions."
)


def groq_key():
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


def ask(messages):
    """messages: list of {"role": "user"|"assistant", "content": str}."""
    key = groq_key()
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set.")
    from groq import Groq
    client = Groq(api_key=key)
    out = client.chat.completions.create(model=MODEL, temperature=0.3, max_tokens=700,
                                         messages=[{"role": "system", "content": SYSTEM}] + messages)
    return out.choices[0].message.content


# ---------- REST API (uvicorn app:app) ----------
app = FastAPI(title="Nutrition Disorder Agent", description=f"Powered by Groq ({MODEL})")


class QueryInput(BaseModel):
    query: str


@app.get("/")
def root():
    return {"message": "Nutrition Disorder Agent is live.", "model": MODEL, "llm_configured": bool(groq_key())}


@app.post("/ask")
def ask_question(data: QueryInput):
    if not groq_key():
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured on the server.")
    try:
        return {"response": ask([{"role": "user", "content": data.query}])}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {exc}")


# ---------- Streamlit UI (streamlit run app.py) ----------
def streamlit_ui():
    import streamlit as st

    st.set_page_config(page_title="Nutrition Disorder Agent", page_icon="🥗")
    st.title("🥗 Nutrition Disorder Agent")
    st.caption(f"LLM assistant for nutrition-related disorders · Groq `{MODEL}` · educational use only, not medical advice")

    if not groq_key():
        st.warning("This demo needs a Groq API key. Add `GROQ_API_KEY` in the app's Settings → Secrets.")

    examples = ["What foods help with iron-deficiency anaemia?",
                "How is coeliac disease managed through diet?",
                "Signs of vitamin D deficiency and how to raise levels safely?"]
    if "chat" not in st.session_state:
        st.session_state.chat = []
    cols = st.columns(len(examples))
    picked = None
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            picked = ex

    for m in st.session_state.chat:
        st.chat_message(m["role"]).markdown(m["content"])

    prompt = st.chat_input("Ask about a nutrition-related condition…") or picked
    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking…"):
                    reply = ask(st.session_state.chat[-10:])
            except Exception as exc:
                reply = f"Sorry, the model is not available right now ({exc})."
            st.markdown(reply)
        st.session_state.chat.append({"role": "assistant", "content": reply})

    if st.session_state.chat and st.sidebar.button("Clear chat"):
        st.session_state.chat = []
        st.rerun()


try:
    from streamlit.runtime import exists as _running_in_streamlit
except Exception:
    def _running_in_streamlit():
        return False

if _running_in_streamlit():
    streamlit_ui()
