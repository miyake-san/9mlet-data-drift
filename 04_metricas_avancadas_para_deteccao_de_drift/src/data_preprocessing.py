# -*- coding: utf-8 -*-
"""
data_preprocessing.py – Geração e pré-processamento de dados sintéticos
para a Aula 04: Métricas Avançadas para Detecção de Drift.

Implementa o cenário da fintech de crédito descrito no Documento 04,
onde um drift multivariado sutil (inversão de correlação renda–idade)
não é detectado por testes univariados (KS, PSI), mas é capturado
por métricas multivariadas como MMD e Energy Distance.

Referências:
    Gama, J. et al. (2014). A survey on concept drift adaptation.
        ACM Computing Surveys, 46(4), art. 44.
    Gretton, A. et al. (2012). A Kernel Two-Sample Test. JMLR, 13, 723–773.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


class DataPreprocessor:
    """Gera e pré-processa datasets sintéticos para detecção de drift.

    O dataset simula o cenário da fintech de crédito (Documento 04,
    seção "O Que Vem Por Aí"): clientes com atributos (idade, renda,
    dívida, score_crédito, tempo_emprego, num_parcelas) em dois períodos,
    onde a correlação entre renda e idade se inverte no período atual,
    configurando um drift multivariado sutil.

    Args:
        n_samples: Número de amostras por período. Default: 5000.
        seed: Semente para reprodutibilidade. Default: 42.
    """

    def __init__(self, n_samples: int = 5000, seed: int = 42) -> None:
        self.n_samples = n_samples
        self.seed = seed
        self._rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Geração de dados
    # ------------------------------------------------------------------
    def generate_dataset(self) -> pd.DataFrame:
        """Gera dataset completo com períodos de referência e atual.

        Implementa dados sintéticos conforme a seção "Hands On" do
        Documento 04, com inversão de correlação renda–idade no
        período atual.

        Returns:
            DataFrame com colunas: idade, renda, divida, score_credito,
            tempo_emprego, num_parcelas, periodo.
        """
        df_ref = self._generate_period(period="referencia", invert_corr=False)
        df_cur = self._generate_period(period="atual", invert_corr=True)
        return pd.concat([df_ref, df_cur], ignore_index=True)

    def _generate_period(
        self, period: str, invert_corr: bool
    ) -> pd.DataFrame:
        """Gera dados para um período específico.

        Conforme Documento 04, Snippet 2: a correlação renda–idade
        é positiva no período de referência e negativa (invertida)
        no período atual, simulando um drift multivariado sutil.

        Args:
            period: Rótulo do período ("referencia" ou "atual").
            invert_corr: Se True, inverte a correlação renda–idade.

        Returns:
            DataFrame com dados do período.
        """
        n = self.n_samples

        # Idade: distribuição normal μ=40, σ=10
        idade = self._rng.normal(loc=40, scale=10, size=n)
        idade = np.clip(idade, 18, 80)

        # Renda: correlacionada com idade (positiva ou negativa)
        # Conforme Documento 04, Snippet 2 - Hands On
        coef = -500.0 if invert_corr else 500.0
        renda = 3000 + coef * idade + self._rng.normal(0, 10000, size=n)
        renda = np.clip(renda, 500, 100000)

        # Dívida: parcialmente correlacionada com renda
        divida = 0.3 * renda + self._rng.normal(0, 5000, size=n)
        divida = np.clip(divida, 0, 200000)

        # Score de crédito: inversamente proporcional a dívida/renda
        ratio = np.where(renda > 0, divida / renda, 1.0)
        score_credito = 800 - 200 * ratio + self._rng.normal(0, 50, size=n)
        score_credito = np.clip(score_credito, 0, 1000)

        # Tempo no emprego: correlacionado com idade
        tempo_emprego = 0.3 * (idade - 18) + self._rng.normal(0, 3, size=n)
        tempo_emprego = np.clip(tempo_emprego, 0, 40)

        # Número de parcelas ativas
        num_parcelas = self._rng.poisson(lam=4, size=n).astype(float)

        return pd.DataFrame(
            {
                "idade": idade,
                "renda": renda,
                "divida": divida,
                "score_credito": score_credito,
                "tempo_emprego": tempo_emprego,
                "num_parcelas": num_parcelas,
                "periodo": period,
            }
        )

    # ------------------------------------------------------------------
    # Carregamento e limpeza
    # ------------------------------------------------------------------
    def load_data(self, path: Optional[str] = None) -> pd.DataFrame:
        """Carrega dataset de um arquivo CSV.

        Args:
            path: Caminho para o CSV. Se None, usa caminho padrão.

        Returns:
            DataFrame carregado.

        Raises:
            FileNotFoundError: Se o arquivo não existir.
        """
        if path is None:
            base = Path(__file__).resolve().parent.parent
            path = str(base / "data" / "raw" / "dataset.csv")

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Dataset não encontrado: {path}")

        return pd.read_csv(path)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpa o dataset removendo valores nulos e duplicatas.

        Args:
            df: DataFrame a ser limpo.

        Returns:
            DataFrame limpo.
        """
        df = df.dropna()
        df = df.drop_duplicates()
        return df.reset_index(drop=True)

    def prepare_features(
        self, df: pd.DataFrame, normalize: bool = True
    ) -> pd.DataFrame:
        """Prepara features numéricas para os detectores de drift.

        Conforme Documento 04, Seção "Saiba Mais": para métricas
        como MMD com kernel RBF, é importante normalizar os dados
        para que o parâmetro γ do kernel funcione adequadamente.

        Args:
            df: DataFrame com features.
            normalize: Se True, aplica Z-score normalização.

        Returns:
            DataFrame apenas com colunas numéricas (normalizadas).
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        result = df[numeric_cols].copy()

        if normalize:
            for col in result.columns:
                mean = result[col].mean()
                std = result[col].std()
                if std > 0:
                    result[col] = (result[col] - mean) / std

        return result

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Divide o dataset nos períodos de referência e atual.

        Args:
            df: DataFrame completo com coluna 'periodo'.

        Returns:
            Tupla (df_referencia, df_atual).
        """
        df_ref = df[df["periodo"] == "referencia"].copy()
        df_cur = df[df["periodo"] == "atual"].copy()
        return df_ref, df_cur

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------
    def generate_and_save(self, output_dir: Optional[str] = None) -> str:
        """Gera o dataset sintético e salva em CSV.

        Args:
            output_dir: Diretório de saída. Se None, usa data/raw/.

        Returns:
            Caminho do arquivo salvo.
        """
        if output_dir is None:
            base = Path(__file__).resolve().parent.parent
            output_dir = str(base / "data" / "raw")

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "dataset.csv")

        df = self.generate_dataset()
        df.to_csv(path, index=False)
        return path
