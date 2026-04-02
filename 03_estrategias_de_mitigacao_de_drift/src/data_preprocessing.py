"""
Pré-processamento e geração do dataset de churn telecom com drift.

Implementa a geração do dataset sintético descrito na seção 'Hands On'
do documento acadêmico, simulando o cenário da operadora de telecomunicações
em que uma oferta disruptiva da concorrência altera hábitos de consumo
e desloca a taxa de cancelamento (covariate shift + concept drift).

Referências:
    Moreno-Torres, J. G., et al. (2012). A unifying view on dataset shift
    in classification. Pattern Recognition, 45(1), 521-530.

    Sugiyama, M., & Kawanabe, M. (2012). Machine Learning in Non-Stationary
    Environments. MIT Press.
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
    """Geração e pré-processamento do dataset de churn com drift simulado.

    Simula o cenário discutido no documento acadêmico da Aula 03:
    uma operadora de telecomunicações enfrenta drift quando um concorrente
    lança oferta disruptiva, alterando distribuições de features e a
    relação entre atributos e target.

    Attributes:
        seed: Semente para reprodutibilidade.
        n_per_period: Número de amostras por período temporal.
        data_dir: Diretório para salvar/carregar o dataset.
    """

    def __init__(
        self,
        seed: int = 42,
        n_per_period: int = 3000,
        data_dir: Optional[str] = None,
    ) -> None:
        self.seed = seed
        self.n_per_period = n_per_period
        self.rng = np.random.default_rng(seed)
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)

    def generate_dataset(self) -> pd.DataFrame:
        """Gera dataset sintético de churn em telecomunicações com drift.

        Implementa três períodos temporais conforme discutido na seção
        'Saiba Mais' — linha do tempo T0 a T3:
          - Período 1: Regime estável pré-competição (~12% churn)
          - Período 2: Choque competitivo — drift abrupto (~28% churn)
          - Período 3: Adaptação gradual — drift incremental (~20% churn)

        Returns:
            DataFrame com 9.000 amostras e 15 colunas.
        """
        dfs = []
        for period in [1, 2, 3]:
            df_period = self._generate_period(period)
            dfs.append(df_period)

        df = pd.concat(dfs, ignore_index=True)
        df["customer_id"] = range(1, len(df) + 1)
        return df

    def _generate_period(self, period: int) -> pd.DataFrame:
        """Gera amostras para um período temporal específico.

        O drift é controlado por parâmetros que mudam entre períodos,
        refletindo o deslocamento de distribuição discutido no documento
        acadêmico (covariate shift, prior probability shift e concept drift).

        Args:
            period: Número do período (1, 2 ou 3).

        Returns:
            DataFrame com n_per_period amostras para o período especificado.
        """
        n = self.n_per_period
        rng = self.rng

        # --- Parâmetros de drift por período ---
        # Conforme discutido na seção 'Saiba Mais': o concorrente altera
        # hábitos de consumo, desloca churn e inverte relações estatísticas.
        drift_params = {
            1: {  # Regime estável
                "data_usage_mean": 8.0,
                "data_usage_std": 3.0,
                "charges_mean": 70.0,
                "charges_std": 25.0,
                "complaint_rate": 0.15,
                "churn_base": 0.12,
                "competitor_exposure": 0.05,
                "concept_flip": 0.0,
            },
            2: {  # Choque competitivo (drift abrupto)
                "data_usage_mean": 4.5,  # Covariate shift: queda de uso
                "data_usage_std": 3.5,
                "charges_mean": 65.0,
                "charges_std": 30.0,
                "complaint_rate": 0.35,
                "churn_base": 0.28,  # Prior probability shift
                "competitor_exposure": 0.65,
                "concept_flip": 0.4,  # Concept drift parcial
            },
            3: {  # Adaptação gradual
                "data_usage_mean": 5.5,
                "data_usage_std": 3.2,
                "charges_mean": 60.0,
                "charges_std": 28.0,
                "complaint_rate": 0.25,
                "churn_base": 0.20,
                "competitor_exposure": 0.40,
                "concept_flip": 0.2,
            },
        }
        p = drift_params[period]

        # --- Geração de features ---
        tenure_months = rng.integers(1, 72, size=n)
        monthly_charges = np.clip(
            rng.normal(p["charges_mean"], p["charges_std"], size=n), 20, 150
        )
        total_charges = monthly_charges * tenure_months * (1 + rng.normal(0, 0.05, n))
        data_usage_gb = np.clip(
            rng.normal(p["data_usage_mean"], p["data_usage_std"], size=n), 0, 25
        )
        call_duration_min = np.clip(
            rng.normal(300, 120, size=n), 0, 800
        )
        num_complaints = rng.poisson(p["complaint_rate"] * 3, size=n)
        payment_delay_days = np.clip(
            rng.exponential(5 + period * 2, size=n).astype(int), 0, 90
        )
        contract_type = rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])
        has_premium_support = rng.binomial(1, 0.3, size=n)
        competitor_offer_exposure = np.clip(
            rng.beta(2, 5, size=n) * (1 + p["competitor_exposure"] * 3), 0, 1
        )
        customer_value_segment = rng.choice([0, 1, 2], size=n, p=[0.4, 0.35, 0.25])

        # --- Geração do target (churn) com concept drift ---
        # Implementa a lógica de dependência não-estacionária discutida
        # na seção 'Saiba Mais': a relação entre features e churn muda.
        logit = (
            -2.0
            + 0.02 * (72 - tenure_months)  # menos tempo = mais churn
            + 0.01 * monthly_charges * (1 - p["concept_flip"])
            - 0.015 * monthly_charges * p["concept_flip"]  # concept drift
            - 0.1 * data_usage_gb
            + 0.3 * num_complaints
            + 0.02 * payment_delay_days
            - 0.5 * contract_type  # contratos longos retêm
            - 0.3 * has_premium_support
            + 2.0 * competitor_offer_exposure
        )
        # Ajustar para atingir churn_base
        prob_churn = 1 / (1 + np.exp(-logit))
        # Recalibrar para taxa alvo
        scale_factor = p["churn_base"] / np.mean(prob_churn)
        prob_churn = np.clip(prob_churn * scale_factor, 0.01, 0.99)
        churn = rng.binomial(1, prob_churn)

        # Timestamp simulado (meses)
        month_start = (period - 1) * 12 + 1
        timestamp_month = rng.integers(month_start, month_start + 12, size=n)

        return pd.DataFrame(
            {
                "customer_id": 0,  # será sobrescrito
                "tenure_months": tenure_months,
                "monthly_charges": np.round(monthly_charges, 2),
                "total_charges": np.round(total_charges, 2),
                "data_usage_gb": np.round(data_usage_gb, 2),
                "call_duration_min": np.round(call_duration_min, 1),
                "num_complaints": num_complaints,
                "payment_delay_days": payment_delay_days,
                "contract_type": contract_type,
                "has_premium_support": has_premium_support,
                "competitor_offer_exposure": np.round(competitor_offer_exposure, 4),
                "customer_value_segment": customer_value_segment,
                "period": period,
                "timestamp_month": timestamp_month,
                "churn": churn,
            }
        )

    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """Carrega o dataset de churn a partir de CSV.

        Args:
            filepath: Caminho para o CSV. Se None, usa data/raw/dataset.csv.

        Returns:
            DataFrame carregado.
        """
        if filepath is None:
            filepath = str(self.data_dir / "raw" / "dataset.csv")
        return pd.read_csv(filepath)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpa o dataset removendo valores ausentes e corrigindo tipos.

        Implementa etapas de limpeza para preparar os dados para
        treinamento, conforme discutido na seção 'Saiba Mais' sobre
        a importância da qualidade da janela de dados.

        Args:
            df: DataFrame bruto.

        Returns:
            DataFrame limpo.
        """
        df = df.copy()
        df = df.dropna()
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
        df = df.dropna()
        # Garantir tipos corretos
        int_cols = [
            "tenure_months", "num_complaints", "payment_delay_days",
            "contract_type", "has_premium_support", "customer_value_segment",
            "period", "churn",
        ]
        for col in int_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)
        return df.reset_index(drop=True)

    def prepare_features(
        self,
        df: pd.DataFrame,
        scale: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepara features e target para modelagem.

        Separa features numéricas, aplica normalização (StandardScaler)
        e retorna X, y prontos para treinamento.

        Args:
            df: DataFrame limpo.
            scale: Se True, aplica StandardScaler nas features numéricas.

        Returns:
            Tupla (X, y) com features e target.
        """
        feature_cols = [
            "tenure_months", "monthly_charges", "total_charges",
            "data_usage_gb", "call_duration_min", "num_complaints",
            "payment_delay_days", "contract_type", "has_premium_support",
            "competitor_offer_exposure", "customer_value_segment",
        ]
        X = df[feature_cols].copy()
        y = df["churn"].copy()

        if scale:
            scaler = StandardScaler()
            num_cols = [
                "tenure_months", "monthly_charges", "total_charges",
                "data_usage_gb", "call_duration_min",
                "competitor_offer_exposure",
            ]
            X[num_cols] = scaler.fit_transform(X[num_cols])

        return X, y

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        stratify: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Divide dados em treino e teste.

        Args:
            X: Features.
            y: Target.
            test_size: Proporção do conjunto de teste.
            stratify: Se True, mantém proporção de classes.

        Returns:
            Tupla (X_train, X_test, y_train, y_test).
        """
        stratify_col = y if stratify else None
        return train_test_split(
            X, y, test_size=test_size, random_state=self.seed, stratify=stratify_col
        )

    def split_by_period(
        self, df: pd.DataFrame
    ) -> dict[int, pd.DataFrame]:
        """Divide o dataset por período temporal.

        Essencial para avaliar estratégias de mitigação conforme discutido
        na seção 'Saiba Mais': treinar no período 1 (estável) e avaliar
        degradação nos períodos 2 e 3 (com drift).

        Args:
            df: DataFrame completo.

        Returns:
            Dicionário {período: DataFrame}.
        """
        return {period: df[df["period"] == period].copy() for period in [1, 2, 3]}

    def generate_and_save(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """Gera e salva o dataset em CSV.

        Args:
            filepath: Caminho de saída. Se None, salva em data/raw/dataset.csv.

        Returns:
            DataFrame gerado.
        """
        df = self.generate_dataset()
        if filepath is None:
            filepath = str(self.data_dir / "raw" / "dataset.csv")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Dataset salvo em {filepath} ({len(df)} amostras)")
        return df


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.generate_and_save()
