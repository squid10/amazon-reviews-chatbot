"""O agente: um LLM que escolhe entre consultar o banco em SQL e buscar nos textos."""

import sqlite3

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from chatbot.rag import buscar, carregar_indice, formatar
from config import ANO_ALVO, CATEGORIA, DB_PATH, GROQ_API_KEY, GROQ_MODEL
from utils.sql import query_sql

SYSTEM = (
    "Você é o assistente de uma plataforma de análise de reviews da Amazon. Responda sempre em "
    f"português, de forma objetiva e fundamentada nos dados. O banco tem todos os reviews da "
    f"categoria {CATEGORIA} em {ANO_ALVO}. "
    "Para contagens, médias e agregações, use consultar_sql com UMA consulta só, que responda a "
    "pergunta inteira: para séries por período use GROUP BY strftime('%m', review_date), nunca "
    "uma consulta por mês. "
    "Para o que os clientes dizem (reclamações, elogios, temas), use busca_semantica UMA vez, "
    "nunca LIKE sobre text ou title. Não repita a mesma chamada de ferramenta. "
    "Tabelas: reviews(review_id, product_id, user_id, rating, title, text, review_date, "
    "helpful_vote, verified_purchase); products(product_id, title, main_category, average_rating, "
    "rating_number, price, store); review_sentiment(review_id, sentiment_label[0=negativo, "
    "1=neutro, 2=positivo], prob_positivo)."
)


def construir_agente():
    """Devolve o agente, o LLM e o índice: o app usa os três."""
    if not GROQ_API_KEY:
        raise RuntimeError("Defina GROQ_API_KEY no .env (chave gratuita em console.groq.com).")

    emb, docs = carregar_indice()
    llm = ChatGroq(model=GROQ_MODEL, temperature=0, max_tokens=2048)

    @tool
    def busca_semantica(pergunta: str) -> str:
        """Busca semântica nos textos dos reviews. Use para perguntas sobre o que os clientes
        dizem: reclamações, elogios e temas recorrentes."""
        return formatar(buscar(emb, docs, pergunta))

    @tool
    def consultar_sql(query: str) -> str:
        """Executa uma consulta SQL de leitura e devolve as primeiras linhas do resultado."""
        with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:   # só leitura
            return query_sql(conn, query).head(30).to_string(index=False)

    return create_agent(llm, [busca_semantica, consultar_sql], system_prompt=SYSTEM), llm, (emb, docs)


def perguntar(agente, historico) -> tuple[str, list[dict]]:
    """Responde e devolve as ferramentas usadas com seus argumentos, para a interface mostrar
    qual caminho o agente escolheu e qual consulta SQL ele escreveu."""
    saida = agente.invoke({"messages": historico})
    ferramentas = [{"nome": c["name"], "args": c.get("args", {})}
                   for m in saida["messages"] for c in getattr(m, "tool_calls", []) or []]
    return saida["messages"][-1].content, ferramentas
