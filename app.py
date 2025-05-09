import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.chat_models import ChatGroq  # ✅ Corrected import

app = FastAPI()

# Load the Groq API key from environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Initialize the LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="mixtral-8x7b"
)

# Request schema
class QueryInput(BaseModel):
    query: str

# Root endpoint
@app.get("/")
def root():
    return {"message": "Nutrition Disorder Agent is live!"}

# Ask endpoint
@app.post("/ask")
def ask_question(data: QueryInput):
    response = llm.invoke(data.query)
    return {"response": response.content}
