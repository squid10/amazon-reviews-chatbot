"""Núcleo do chatbot do módulo 3 (LangChain + Groq).

O notebook 05 escreve este mesmo caminho passo a passo; aqui ele fica empacotado
para o app.py consumir.
"""

from chatbot.agent import construir_agente, perguntar
from chatbot.rag import buscar, carregar_indice, formatar, responder_rag

__all__ = ["construir_agente", "perguntar", "carregar_indice", "buscar", "formatar",
           "responder_rag"]
