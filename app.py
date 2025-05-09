import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.chat_models import ChatGroq

app = FastAPI(title="Nutrition Disorder Agent", description="Powered by Groq + Mixtral")

# Load the Groq API key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Validate API key presence
if not GROQ_API_KEY:
    raise EnvironmentError("❌ GROQ_API_KEY not found in environment variables.")

# Initialize the Groq-based LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="mixtral-8x7b"
)

# Input schema
class QueryInput(BaseModel):
    query: str

# Health check
@app.get("/")
def root():
    return {"message": "✅ Nutrition Disorder Agent is live and ready!"}

# Main inference endpoint
@app.post("/ask")
def ask_question(data: QueryInput):
    try:
        response = llm.invoke(data.query)
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM invocation failed: {str(e)}")
