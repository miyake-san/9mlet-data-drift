"""
Testes unitários para o módulo data_preprocessing da Aula 03.

Valida a geração do dataset sintético de churn em telecomunicações
com drift simulado entre períodos, conforme discutido na seção
'Saiba Mais' do documento acadêmico.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import DataPreprocessor


@pytest.fixture
def preprocessor() -> DataPreprocessor:
    """Cria instância do DataPreprocessor com seed fixa."""
    return DataPreprocessor(seed=42, n_per_period=500)


@pytest.fixture
def dataset(preprocessor: DataPreprocessor) -> pd.DataFrame:
    """Gera dataset sintético reduzido para testes."""
    return preprocessor.generate_dataset()


class TestGenerateDataset:
    """Testes para generate_dataset()."""

    def test_generate_dataset_shape(self, dataset: pd.DataFrame) -> None:
        """Verifica que o dataset tem 1500 amostras (3 períodos × 500)."""
        assert dataset.shape[0] == 1500

    def test_generate_dataset_columns(self, dataset: pd.DataFrame) -> None:
        """Verifica presença de todas as colunas esperadas."""
        expected_cols = {
            "customer_id", "tenure_months", "monthly_charges", "total_charges",
            "data_usage_gb", "call_duration_min", "num_complaints",
            "payment_delay_days", "contract_type", "has_premium_support",
            "competitor_offer_exposure", "customer_value_segment",
            "period", "timestamp_month", "churn",
        }
        assert expected_cols == set(dataset.columns)

    def test_generate_dataset_periods(self, dataset: pd.DataFrame) -> None:
        """Verifica que existem exatamente 3 períodos."""
        assert set(dataset["period"].unique()) == {1, 2, 3}

    def test_generate_dataset_churn_binary(self, dataset: pd.DataFrame) -> None:
        """Verifica que churn é binário (0 ou 1)."""
        assert set(dataset["churn"].unique()).issubset({0, 1})

    def test_generate_dataset_churn_rate_increases(
        self, dataset: pd.DataFrame
    ) -> None:
        """Verifica que taxa de churn aumenta do P1 para P2 (drift)."""
        rate_p1 = dataset[dataset["period"] == 1]["churn"].mean()
        rate_p2 = dataset[dataset["period"] == 2]["churn"].mean()
        assert rate_p2 > rate_p1, (
            f"Churn P2 ({rate_p2:.2f}) deveria ser > P1 ({rate_p1:.2f})"
        )

    def test_generate_dataset_reproducibility(self) -> None:
        """Verifica reprodutibilidade com mesma seed."""
        df1 = DataPreprocessor(seed=123, n_per_period=100).generate_dataset()
        df2 = DataPreprocessor(seed=123, n_per_period=100).generate_dataset()
        pd.testing.assert_frame_equal(df1, df2)


class TestCleanData:
    """Testes para clean_data()."""

    def test_clean_data_no_nulls(
        self, preprocessor: DataPreprocessor, dataset: pd.DataFrame
    ) -> None:
        """Verifica que os dados limpos não contêm NaN."""
        cleaned = preprocessor.clean_data(dataset)
        assert cleaned.isna().sum().sum() == 0

    def test_clean_data_types(
        self, preprocessor: DataPreprocessor, dataset: pd.DataFrame
    ) -> None:
        """Verifica tipos corretos após limpeza."""
        cleaned = preprocessor.clean_data(dataset)
        assert cleaned["churn"].dtype in (np.int32, np.int64)
        assert cleaned["period"].dtype in (np.int32, np.int64)


class TestPrepareFeatures:
    """Testes para prepare_features()."""

    def test_prepare_features_shape(
        self, preprocessor: DataPreprocessor, dataset: pd.DataFrame
    ) -> None:
        """Verifica shape de X e y."""
        cleaned = preprocessor.clean_data(dataset)
        X, y = preprocessor.prepare_features(cleaned)
        assert X.shape[0] == len(cleaned)
        assert X.shape[1] == 11  # 11 features
        assert len(y) == len(cleaned)

    def test_prepare_features_scaled(
        self, preprocessor: DataPreprocessor, dataset: pd.DataFrame
    ) -> None:
        """Verifica que features numéricas são escaladas (média ~0)."""
        cleaned = preprocessor.clean_data(dataset)
        X, _ = preprocessor.prepare_features(cleaned, scale=True)
        assert abs(X["monthly_charges"].mean()) < 0.5


class TestSplitData:
    """Testes para split_data()."""

    def test_split_data_proportions(
        self, preprocessor: DataPreprocessor, dataset: pd.DataFrame
    ) -> None:
        """Verifica proporções de treino/teste."""
        cleaned = preprocessor.clean_data(dataset)
        X, y = preprocessor.prepare_features(cleaned)
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.2)
        assert abs(len(X_test) / len(X) - 0.2) < 0.05


class TestSplitByPeriod:
    """Testes para split_by_period()."""

    def test_split_by_period_keys(
        self, preprocessor: DataPreprocessor, dataset: pd.DataFrame
    ) -> None:
        """Verifica que retorna dicionário com chaves {1, 2, 3}."""
        cleaned = preprocessor.clean_data(dataset)
        periods = preprocessor.split_by_period(cleaned)
        assert set(periods.keys()) == {1, 2, 3}

    def test_split_by_period_sizes(
        self, preprocessor: DataPreprocessor, dataset: pd.DataFrame
    ) -> None:
        """Verifica que cada período tem amostras."""
        cleaned = preprocessor.clean_data(dataset)
        periods = preprocessor.split_by_period(cleaned)
        for p in [1, 2, 3]:
            assert len(periods[p]) > 0
