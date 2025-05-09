import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chat_models import ChatGroq

app = FastAPI()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="mixtral-8x7b"
)

class QueryInput(BaseModel):
    query: str

@app.get("/")
def root():
    return {"message": "Nutrition Disorder Agent is live!"}

@app.post("/ask")
def ask_question(data: QueryInput):
    response = llm.invoke(data.query)
    return {"response": response.content}