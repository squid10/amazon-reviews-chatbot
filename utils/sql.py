import pandas as pd


def query_sql(conn, query:str) -> pd.DataFrame:
    """Executa uma query SQL e devolve um DataFrame com o resultado."""
    return pd.read_sql_query(query, conn)