from langchain_groq import ChatGroq

from config import GROQ_MODEL, GROQ_API_KEY

llm = ChatGroq(model= GROQ_MODEL, temperature=0, max_tokens=2048)