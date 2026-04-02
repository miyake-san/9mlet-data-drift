"""
Pré-processamento de dados para monitoramento de drift em escala.

Implementa ingestão, limpeza e preparação de dados de transações
financeiras para o caso FinBank, conforme discutido na Aula 7.
O pré-processamento é pensado para suportar cenários de streaming
e janelas temporais (Sculley et al., 2015).

Referências:
    Sculley, D. et al. (2015). Hidden Technical Debt in Machine
    Learning Systems. NeurIPS 2015.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class DataPreprocessor:
    """
    Classe para ingestão e pré-processamento de transações FinBank.

    Implementa pipeline de dados para monitoramento de drift,
    incluindo separação temporal por janelas (meses) para
    comparação baseline vs. corrente, conforme discutido
    na seção 'Saiba Mais' (janelas, amostragem e trade-offs).
    """

    NUMERIC_FEATURES: List[str] = [
        "amount",
        "num_items",
        "hour_of_day",
        "day_of_week",
        "customer_age",
        "account_age_days",
        "transaction_count_30d",
        "avg_amount_30d",
    ]

    CATEGORICAL_FEATURES: List[str] = ["channel", "is_international"]

    TARGET: str = "is_fraud"

    def __init__(self, random_state: int = 42) -> None:
        """Inicializa o preprocessador com seed fixa para reprodutibilidade."""
        self.random_state = random_state
        self._data: Optional[pd.DataFrame] = None

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Carrega dataset de transações a partir de CSV.

        Args:
            filepath: Caminho para o arquivo CSV.

        Returns:
            DataFrame com os dados carregados.

        Raises:
            FileNotFoundError: Se o arquivo não existir.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

        df = pd.read_csv(filepath, parse_dates=["timestamp"])
        self._data = df
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza limpeza básica dos dados.

        Remove duplicatas, trata valores ausentes e garante tipos
        corretos para cada coluna, preparando para o cálculo de
        métricas de drift (PSI, K–S) discutido no material.

        Args:
            df: DataFrame bruto.

        Returns:
            DataFrame limpo.
        """
        df = df.copy()

        # Remover duplicatas por transaction_id
        df = df.drop_duplicates(subset=["transaction_id"])

        # Preencher numéricos ausentes com mediana
        for col in self.NUMERIC_FEATURES:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        # Preencher categóricos com moda
        for col in self.CATEGORICAL_FEATURES:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode()[0])

        # Garantir tipos
        df["month"] = df["month"].astype(int)
        df["is_fraud"] = df["is_fraud"].astype(int)
        df["is_international"] = df["is_international"].astype(int)

        return df

    def prepare_features(
        self, df: pd.DataFrame, encode_categorical: bool = True
    ) -> pd.DataFrame:
        """
        Prepara features para análise de drift e modelagem.

        Aplica encoding de variáveis categóricas e normalização
        opcional, seguindo boas práticas para comparação de
        distribuições (Gama et al., 2014).

        Args:
            df: DataFrame limpo.
            encode_categorical: Se True, aplica one-hot encoding.

        Returns:
            DataFrame com features preparadas.
        """
        df = df.copy()

        if encode_categorical and "channel" in df.columns:
            channel_dummies = pd.get_dummies(df["channel"], prefix="channel")
            df = pd.concat([df, channel_dummies], axis=1)
            df = df.drop(columns=["channel"])

        return df

    def split_data(
        self,
        df: pd.DataFrame,
        baseline_months: Tuple[int, ...] = (1, 2, 3),
        current_months: Optional[Tuple[int, ...]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Separa dados em baseline e corrente por janelas temporais.

        Implementa a separação por janelas conforme discutido na
        seção 'Saiba Mais' — janelas, amostragem e trade-offs de
        custo. O baseline corresponde ao período estável e o
        corrente ao período sob monitoramento.

        Args:
            df: DataFrame completo.
            baseline_months: Tupla de meses para baseline.
            current_months: Tupla de meses para período corrente.
                Se None, usa meses restantes.

        Returns:
            Dicionário com 'baseline' e 'current' DataFrames.
        """
        if current_months is None:
            all_months = set(df["month"].unique())
            current_months = tuple(sorted(all_months - set(baseline_months)))

        baseline = df[df["month"].isin(baseline_months)].copy()
        current = df[df["month"].isin(current_months)].copy()

        return {"baseline": baseline, "current": current}

    def sample_data(
        self, df: pd.DataFrame, fraction: float = 0.1, stratify_col: str = "month"
    ) -> pd.DataFrame:
        """
        Realiza amostragem estratificada para redução de custo computacional.

        Conforme discutido na seção 'Saiba Mais' — estratégia de
        amostragem para Big Data: mantém representatividade estatística
        com custo menor (Sculley et al., 2015).

        Args:
            df: DataFrame completo.
            fraction: Fração de amostragem (0, 1].
            stratify_col: Coluna para estratificação.

        Returns:
            DataFrame amostrado.
        """
        if fraction <= 0 or fraction > 1:
            raise ValueError("fraction deve estar em (0, 1]")

        return (
            df.groupby(stratify_col, group_keys=False)
            .apply(lambda x: x.sample(
                frac=fraction, random_state=self.random_state
            ))
            .reset_index(drop=True)
        )

    def get_monthly_windows(
        self, df: pd.DataFrame
    ) -> Dict[int, pd.DataFrame]:
        """
        Retorna dicionário com dados separados por mês (janelas temporais).

        Facilita a análise de drift mês a mês, conforme pipeline
        de janelas discutido na seção Hands On.

        Args:
            df: DataFrame completo.

        Returns:
            Dicionário {mês: DataFrame}.
        """
        return {
            month: group.copy()
            for month, group in df.groupby("month")
        }
