"""
Módulo de pré-processamento e geração de dados sintéticos para a Aula 1.

Implementa a geração de dados que simulam o cenário motivador da loja online
descrito no material da aula (seção 'O Que Vem Por Aí?'), onde um modelo de
recomendação perde acurácia ao longo do tempo devido a mudanças na distribuição
dos dados dos clientes.

Referências:
    Moreno-Torres, J.G. et al. (2012). A unifying view on dataset shift
    in classification. Pattern Recognition, 45(1), 521–530.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """Gera, carrega e pré-processa dados sintéticos com drift injetado.

    O dataset simula atributos demográficos e comportamentais de clientes de
    uma loja online, conforme o cenário motivador da aula.  Em períodos
    posteriores, a distribuição de P(X) é alterada para representar data drift
    (covariate shift) — P_treino(X) ≠ P_produção(X).

    Attributes:
        random_state: Semente para reprodutibilidade.
        scaler: StandardScaler ajustado nos dados de treino.
    """

    # Nomes descritivos das features que simulam atributos de clientes
    FEATURE_NAMES: list[str] = [
        "idade",
        "renda_mensal",
        "tempo_no_site_min",
        "paginas_visitadas",
        "itens_carrinho",
        "ticket_medio",
        "frequencia_visitas_mes",
        "dias_desde_ultima_compra",
        "score_engajamento",
        "num_categorias_visitadas",
    ]

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.scaler: Optional[StandardScaler] = None

    # ------------------------------------------------------------------
    # Geração de dados sintéticos
    # ------------------------------------------------------------------

    def generate_synthetic_data(
        self,
        n_samples: int = 10_000,
        n_periods: int = 5,
        drift_start_period: int = 3,
        drift_magnitude: float = 1.5,
    ) -> pd.DataFrame:
        """Gera dataset sintético com drift injetado a partir de um período.

        Implementa a simulação de data drift (covariate shift) conforme
        discutido na seção 'Saiba Mais — Definindo Data Drift e Concept Drift'.
        Nos períodos anteriores a ``drift_start_period``, os dados seguem a
        distribuição de treinamento P_treino(X).  A partir desse período, a
        média de algumas features é deslocada por ``drift_magnitude`` desvios-
        padrão, simulando P_produção(X) ≠ P_treino(X).

        Args:
            n_samples: Número total de amostras.
            n_periods: Quantidade de períodos temporais.
            drift_start_period: Período a partir do qual o drift é injetado
                (1-indexed).
            drift_magnitude: Magnitude do deslocamento em desvios-padrão.

        Returns:
            DataFrame com features, target e metadados de período.
        """
        rng = np.random.RandomState(self.random_state)
        samples_per_period = n_samples // n_periods
        frames: list[pd.DataFrame] = []

        for period in range(1, n_periods + 1):
            n = samples_per_period if period < n_periods else (
                n_samples - samples_per_period * (n_periods - 1)
            )

            # Distribuição base (período estável)
            mean_base = np.zeros(len(self.FEATURE_NAMES))
            std_base = np.ones(len(self.FEATURE_NAMES))

            # Injeção de drift: desloca a média de metade das features
            # Simula mudança gradual — cada período após drift_start_period
            # aumenta o deslocamento proporcionalmente.
            if period >= drift_start_period:
                shift_factor = (period - drift_start_period + 1) * drift_magnitude
                # Desloca features 0..4 (demográficas)
                mean_base[:5] += shift_factor * 0.5

            X = rng.normal(loc=mean_base, scale=std_base, size=(n, len(self.FEATURE_NAMES)))

            # Gera target com regra simples + ruído
            # P(Y|X) permanece constante → data drift puro (covariate shift)
            logit = 0.8 * X[:, 0] - 0.6 * X[:, 1] + 0.4 * X[:, 2] + rng.normal(0, 0.3, n)
            y = (logit > 0).astype(int)

            df_period = pd.DataFrame(X, columns=self.FEATURE_NAMES)
            df_period["target"] = y
            df_period["periodo"] = period
            frames.append(df_period)

        df = pd.concat(frames, ignore_index=True)
        return df

    # ------------------------------------------------------------------
    # Carga e persistência
    # ------------------------------------------------------------------

    def load_data(self, filepath: str | Path) -> pd.DataFrame:
        """Carrega dataset CSV.

        Args:
            filepath: Caminho para o arquivo CSV.

        Returns:
            DataFrame carregado.

        Raises:
            FileNotFoundError: Se o arquivo não existir.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        return pd.read_csv(filepath)

    def save_data(self, df: pd.DataFrame, filepath: str | Path) -> Path:
        """Salva DataFrame em CSV.

        Args:
            df: DataFrame a ser salvo.
            filepath: Caminho de destino.

        Returns:
            Path do arquivo salvo.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        return filepath

    # ------------------------------------------------------------------
    # Limpeza e preparação
    # ------------------------------------------------------------------

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove valores ausentes e duplicatas.

        Args:
            df: DataFrame bruto.

        Returns:
            DataFrame limpo.
        """
        df = df.dropna().drop_duplicates().reset_index(drop=True)
        return df

    def prepare_features(
        self,
        df: pd.DataFrame,
        fit: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Separa features e target, e aplica padronização (StandardScaler).

        Conforme boas práticas de pré-processamento discutidas no material,
        a padronização garante que todas as features contribuam igualmente
        para métricas de distância e gradientes.

        Args:
            df: DataFrame com colunas de features e 'target'.
            fit: Se True, ajusta o scaler nos dados (usar apenas no treino).

        Returns:
            Tupla (X_scaled, y).
        """
        feature_cols = [c for c in self.FEATURE_NAMES if c in df.columns]
        X = df[feature_cols].values
        y = df["target"].values

        if fit or self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return X_scaled, y

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Divide dados em treino e teste mantendo proporção de classes.

        Args:
            X: Matriz de features.
            y: Vetor de targets.
            test_size: Fração para teste.

        Returns:
            Tupla (X_train, X_test, y_train, y_test).
        """
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y,
        )

    def split_by_period(
        self,
        df: pd.DataFrame,
        train_periods: list[int] | None = None,
        test_periods: list[int] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Divide dataset por períodos temporais.

        Essa divisão é mais realista que a aleatória, pois simula o cenário
        de produção onde o modelo é treinado com dados passados e avaliado
        com dados futuros (potencialmente com drift).

        Args:
            df: DataFrame completo com coluna 'periodo'.
            train_periods: Lista de períodos para treino (default: [1, 2]).
            test_periods: Lista de períodos para teste (default: [3, 4, 5]).

        Returns:
            Tupla (df_train, df_test).
        """
        if train_periods is None:
            train_periods = [1, 2]
        if test_periods is None:
            test_periods = [3, 4, 5]

        df_train = df[df["periodo"].isin(train_periods)].copy()
        df_test = df[df["periodo"].isin(test_periods)].copy()
        return df_train, df_test


# ------------------------------------------------------------------
# Execução direta para geração rápida
# ------------------------------------------------------------------
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    preprocessor = DataPreprocessor(random_state=42)
    df = preprocessor.generate_synthetic_data()
    out_path = preprocessor.save_data(df, base_dir / "data" / "raw" / "dataset.csv")
    print(f"Dataset salvo em {out_path}  ({len(df)} linhas)")
