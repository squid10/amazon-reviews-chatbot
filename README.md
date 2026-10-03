# Análise de reviews da Amazon

Projeto do curso, em três módulos que se encaixam num pipeline só: sair do dado bruto e chegar a um
chatbot que responde perguntas sobre ele.

O conjunto são **todos os reviews da categoria `All_Beauty` publicados em 2021**, cerca de 112 mil,
vindos do dataset *Amazon Reviews 2023*. Um ano e uma categoria, sem amostragem: assim não há
desenho amostral para justificar, e a comparação entre meses é direta.

- **Módulo 1, dados e EDA.** Baixa os reviews, aplica filtros de qualidade e monta o banco SQLite.
  A exploração mostra o que decide o resto: as notas se distribuem em J (muito 5, muito 1) e o
  rótulo de sentimento derivado delas fica desbalanceado, com 64,5% de positivo contra 8,6% de
  neutro.
- **Módulo 2, NLP e modelagem.** Cada review vira um vetor de 384 dimensões com um
  *sentence-transformer* local, e um XGBoost classifica o sentimento em negativo, neutro ou
  positivo. A métrica é F1-macro, não acurácia, porque a classe neutra é minoria e é justamente a
  difícil. Um notebook à parte compara com um LLM e estima o custo de usá-lo no lugar.
- **Módulo 3, RAG e agente.** Os mesmos embeddings viram um índice de busca semântica. Sobre ele,
  duas formas de responder: RAG puro, que fundamenta a resposta nos trechos recuperados, e um
  agente, que escolhe entre consultar o banco em SQL e buscar nos textos. A entrega final é o
  dashboard em Streamlit.

## Estrutura

```
Projeto/
├── chatbot/                            RAG e agente, empacotados para o app
├── data/                               banco SQLite, embeddings e cache do download
├── models/                             o classificador treinado (.joblib)
├── utils/                              embedding.py (vetores) e sql.py (consultas)
├── .env                                chaves (GROQ_API_KEY)
├── .gitignore
├── config.py                           todos os parâmetros num lugar só
├── app.py                              dashboard Streamlit
├── requirements.txt                    dependências
├── 01_get_data.ipynb                   monta data/reviews.db
├── 02_m1_eda.ipynb                     análise exploratória
├── 03_m2_text_and_embeddings.ipynb     trata o texto e gera os embeddings
├── 04_m2_sentiment_model.ipynb         treina o modelo e grava review_sentiment
├── 04b_m2_sentiment_llm.ipynb          compara com um LLM e estima o custo
└── 05_m3_rag_chatbot.ipynb             busca semântica, RAG e agente
```

Os notebooks ficam na raiz de propósito: os caminhos do `config.py` são relativos a ela, então
nada precisa ajustar diretório para funcionar.

Depois que o `data/reviews.db` estiver montado, a pasta `data/hf_cache` (cerca de 1 GB de arquivos
brutos) pode ser apagada. Ela só é necessária para reconstruir o banco do zero.

## Como rodar

**1.** Criar o ambiente virtual:

```
python3.11 -m venv .venv
```

**2.** Ativar o `.venv`:

```
source .venv/bin/activate          # macOS e Linux
./.venv/Scripts/activate.ps1       # Windows
```

**3.** Instalar as dependências:

```
pip install -r requirements.txt
```

**4.** Criar um `.env` na raiz com a chave da Groq (gratuita em console.groq.com), usada só no
módulo 3:

```
GROQ_API_KEY=sua_chave
```

**5.** Rodar os notebooks na ordem (01, 02, 03, 04, 04b e 05) e, no fim, abrir a interface:

```
python -m streamlit run app.py
```

O `python -m` garante que a interface rode no Python do `.venv`. Chamar `streamlit run` direto pode
pegar uma instalação do sistema, com outras versões do langchain, e quebrar já na importação.
