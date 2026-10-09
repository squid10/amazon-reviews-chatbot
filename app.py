"""Interface do chatbot do módulo 3. Rode com: python -m streamlit run app.py"""

import streamlit as st

from chatbot import construir_agente, perguntar, responder_rag
from config import ANO_ALVO, CATEGORIA, GROQ_API_KEY

EXEMPLOS = [
    "Quantos reviews negativos, neutros e positivos existem?",
    "Como a nota média variou mês a mês?",
    "Quais são as reclamações mais comuns dos clientes?",
    "Qual a probabilidade positiva média dos reviews de nota 3?",
]

st.set_page_config(page_title="Amazon Reviews Chatbot", page_icon="💬")
st.title("Amazon Reviews Chatbot")
st.caption(f"{CATEGORIA.replace('_', ' ')} · {ANO_ALVO}")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY não encontrada. Crie um `.env` na raiz do projeto com "
             "`GROQ_API_KEY=sua_chave` (chave gratuita em console.groq.com).")
    st.stop()


@st.cache_resource(show_spinner="Carregando índice e agente (só na primeira vez)...")
def carregar():
    return construir_agente()


def mostrar_ferramentas(ferramentas):
    """Qual ferramenta o agente chamou e, quando é SQL, a consulta que ele escreveu."""
    if not ferramentas:
        return
    nomes = ", ".join(dict.fromkeys(f["nome"] for f in ferramentas))
    with st.expander(f"🔧 {len(ferramentas)} chamada(s): {nomes}",
                     expanded=len(ferramentas) == 1):
        for f in ferramentas:
            if consulta := f["args"].get("query"):
                st.code(consulta, language="sql")
            else:
                termo = " · ".join(str(v) for v in f["args"].values())
                st.caption(f"**{f['nome']}**: {termo}")


with st.sidebar:
    modo = st.radio("Modo", ["Agente (SQL + busca)", "RAG puro (só busca)"])
    if st.button("Limpar conversa"):
        st.session_state.historico = []
        st.rerun()
    st.caption("Experimente perguntar:")
    for exemplo in EXEMPLOS:
        st.caption(f"• {exemplo}")

st.session_state.setdefault("historico", [])
for mensagem in st.session_state.historico:
    with st.chat_message(mensagem["role"]):
        st.markdown(mensagem["content"])
        mostrar_ferramentas(mensagem.get("ferramentas", []))

pergunta = st.chat_input("Pergunte sobre os reviews...")
if pergunta:
    st.session_state.historico.append({"role": "user", "content": pergunta})
    st.chat_message("user").markdown(pergunta)

    with st.chat_message("assistant"), st.spinner("Analisando os reviews..."):
        agente, llm, (emb, docs) = carregar()
        ferramentas = []
        try:
            if modo.startswith("Agente"):
                # o agente só recebe papel e texto: as ferramentas são registro da interface
                conversa = [{"role": m["role"], "content": m["content"]}
                            for m in st.session_state.historico]
                resposta, ferramentas = perguntar(agente, conversa)
            else:
                resposta = responder_rag(llm, emb, docs, pergunta)
        except Exception as erro:
            resposta = f"Erro ao responder ({type(erro).__name__}): {erro}"
        st.markdown(resposta)
        mostrar_ferramentas(ferramentas)

    st.session_state.historico.append(
        {"role": "assistant", "content": resposta, "ferramentas": ferramentas})
