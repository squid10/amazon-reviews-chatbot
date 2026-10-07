import os

from dotenv import load_dotenv

load_dotenv()

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
CATEGORIA = "All_Beauty"
REVIEW_COLS = ["rating", "title", "text", "asin", "parent_asin",
               "user_id", "timestamp", "helpful_vote", "verified_purchase"]
META_COLS = ["parent_asin", "title", "main_category",
             "average_rating", "rating_number", "price", "store"]

# Aquisição: uma categoria, um ano inteiro, sem amostragem.
ANO_ALVO = 2021
MIN_CARACTERES = 20

DB_PATH = "data/reviews.db"
EMBEDDING_DIR = "data/embeddings"
MODELS_DIR = "models"

SENTIMENT_LABELS = {0: "negativo", 1: "neutro", 2: "positivo"}

# NLP e modelagem (módulo 2)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MAX_CHARS = 1000
SEED = 42
TEST_SIZE = 0.2

# Módulo 3: chatbot (LLM da Groq, chave gratuita em console.groq.com)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "openai/gpt-oss-120b"
RAG_TOP_K = 5

GROQ_MODEL = "openai/gpt-oss-120b"
