"""Índice, busca por cosseno e RAG puro.

É o mesmo caminho que o notebook 05 percorre passo a passo, empacotado para o app.py.
"""

import sqlite3

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from config import DB_PATH, EMBEDDING_DIR, RAG_TOP_K, SENTIMENT_LABELS
from utils.embedding import embed
from utils.sql import query_sql

RAG_SYSTEM = ("Você é um analista de reviews. Responda em português, de forma objetiva, usando "
              "SOMENTE os trechos fornecidos. Se eles não bastarem, diga isso claramente.")
RAG_USER = "Trechos de reviews:\n{contexto}\n\nPergunta: {pergunta}"


def carregar_indice():
    """Junta os vetores do módulo 2 com o texto e o sentimento, alinhados pelo review_id."""
    emb = np.load(f"{EMBEDDING_DIR}/emb.npy")
    ids = np.load(f"{EMBEDDING_DIR}/review_ids.npy")

    conn = sqlite3.connect(DB_PATH)
    docs = query_sql(conn, """
        SELECT r.review_id, r.rating, r.title, r.text, s.sentiment_label
        FROM reviews r
        LEFT JOIN review_sentiment s ON s.review_id = r.review_id
    """).set_index("review_id")
    conn.close()

    docs = docs.loc[ids].reset_index()
    docs["sentimento"] = docs["sentiment_label"].map(SENTIMENT_LABELS)
    return emb, docs


def formatar(trechos: pd.DataFrame) -> str:
    """Os reviews recuperados no formato que vai para o prompt."""
    return "\n".join(
        f"- [nota {int(r.rating)} | {r.sentimento}] {r.title}: "
        f"{' '.join(str(r.text).split())[:220]}"
        for r in trechos.itertuples()
    )


def buscar(emb, docs, pergunta: str, k: int = RAG_TOP_K) -> pd.DataFrame:
    """Os vetores são normalizados, então o produto interno já é a similaridade de cosseno."""
    similaridades = emb @ embed([pergunta], show_progress=False)[0]
    return docs.iloc[np.argsort(-similaridades)[:k]]


def responder_rag(llm, emb, docs, pergunta: str, k: int = RAG_TOP_K) -> str:
    """RAG puro: recupera os trechos e responde fundamentado somente neles."""
    contexto = formatar(buscar(emb, docs, pergunta, k))
    return llm.invoke([
        SystemMessage(RAG_SYSTEM),
        HumanMessage(RAG_USER.format(contexto=contexto, pergunta=pergunta)),
    ]).content
