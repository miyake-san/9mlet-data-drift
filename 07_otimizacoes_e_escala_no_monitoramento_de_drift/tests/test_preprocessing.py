"""
Testes unitários para o módulo data_preprocessing.

Testa DataPreprocessor: carregamento, limpeza, preparação de features,
separação baseline/corrente e amostragem estratificada.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_preprocessing import DataPreprocessor


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def preprocessor() -> DataPreprocessor:
    """Cria instância do DataPreprocessor com seed fixa."""
    return DataPreprocessor(random_state=42)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Cria DataFrame de transações FinBank para testes."""
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame({
        "transaction_id": [f"TXN-{i:05d}" for i in range(n)],
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "month": np.repeat([1, 2, 3, 4, 5, 6, 7, 8], n // 8),
        "amount": rng.normal(300, 100, n).clip(1),
        "num_items": rng.poisson(3, n) + 1,
        "hour_of_day": rng.randint(0, 24, n),
        "day_of_week": rng.randint(0, 7, n),
        "customer_age": rng.normal(35, 10, n).clip(18, 80),
        "account_age_days": rng.poisson(365, n).clip(1),
        "transaction_count_30d": rng.poisson(15, n),
        "avg_amount_30d": rng.normal(250, 80, n).clip(10),
        "channel": rng.choice(["app", "web", "physical", "phone"], n),
        "is_international": rng.binomial(1, 0.1, n),
        "is_fraud": rng.binomial(1, 0.05, n),
    })
    return df


@pytest.fixture
def sample_csv(tmp_path: Path, sample_dataframe: pd.DataFrame) -> str:
    """Salva sample_dataframe em CSV temporário e retorna caminho."""
    filepath = str(tmp_path / "test_transactions.csv")
    sample_dataframe.to_csv(filepath, index=False)
    return filepath


# ======================================================================
# Testes
# ======================================================================


class TestLoadData:
    """Testes para DataPreprocessor.load_data()."""

    def test_load_data_returns_dataframe(
        self, preprocessor: DataPreprocessor, sample_csv: str
    ) -> None:
        """Verifica que load_data retorna um DataFrame."""
        df = preprocessor.load_data(sample_csv)
        assert isinstance(df, pd.DataFrame)

    def test_load_data_shape(
        self, preprocessor: DataPreprocessor, sample_csv: str
    ) -> None:
        """Verifica shape do DataFrame carregado."""
        df = preprocessor.load_data(sample_csv)
        assert df.shape[0] == 200
        assert df.shape[1] == 14

    def test_load_data_file_not_found(
        self, preprocessor: DataPreprocessor
    ) -> None:
        """Verifica exceção para arquivo inexistente."""
        with pytest.raises(FileNotFoundError):
            preprocessor.load_data("nao_existe.csv")


class TestCleanData:
    """Testes para DataPreprocessor.clean_data()."""

    def test_clean_removes_duplicates(
        self, preprocessor: DataPreprocessor, sample_dataframe: pd.DataFrame
    ) -> None:
        """Verifica remoção de duplicatas por transaction_id."""
        df = pd.concat([sample_dataframe, sample_dataframe.iloc[:5]])
        cleaned = preprocessor.clean_data(df)
        assert len(cleaned) == len(sample_dataframe)

    def test_clean_fills_missing_values(
        self, preprocessor: DataPreprocessor, sample_dataframe: pd.DataFrame
    ) -> None:
        """Verifica preenchimento de valores ausentes."""
        df = sample_dataframe.copy()
        df.loc[0, "amount"] = np.nan
        df.loc[1, "channel"] = np.nan
        cleaned = preprocessor.clean_data(df)
        assert cleaned["amount"].isna().sum() == 0
        assert cleaned["channel"].isna().sum() == 0


class TestPrepareFeatures:
    """Testes para DataPreprocessor.prepare_features()."""

    def test_encode_categorical_creates_dummies(
        self, preprocessor: DataPreprocessor, sample_dataframe: pd.DataFrame
    ) -> None:
        """Verifica criação de dummies para channel."""
        prepared = preprocessor.prepare_features(
            sample_dataframe, encode_categorical=True
        )
        assert "channel" not in prepared.columns
        channel_cols = [c for c in prepared.columns if c.startswith("channel_")]
        assert len(channel_cols) > 0

    def test_no_encode_keeps_channel(
        self, preprocessor: DataPreprocessor, sample_dataframe: pd.DataFrame
    ) -> None:
        """Verifica que channel permanece quando encode_categorical=False."""
        prepared = preprocessor.prepare_features(
            sample_dataframe, encode_categorical=False
        )
        assert "channel" in prepared.columns


class TestSplitData:
    """Testes para DataPreprocessor.split_data()."""

    def test_split_baseline_and_current(
        self, preprocessor: DataPreprocessor, sample_dataframe: pd.DataFrame
    ) -> None:
        """Verifica separação baseline vs. corrente."""
        splits = preprocessor.split_data(
            sample_dataframe, baseline_months=(1, 2, 3)
        )
        assert "baseline" in splits
        assert "current" in splits
        baseline_months = set(splits["baseline"]["month"].unique())
        assert baseline_months <= {1, 2, 3}

    def test_split_no_overlap(
        self, preprocessor: DataPreprocessor, sample_dataframe: pd.DataFrame
    ) -> None:
        """Verifica que baseline e current não se sobrepõem."""
        splits = preprocessor.split_data(
            sample_dataframe, baseline_months=(1, 2, 3)
        )
        baseline_ids = set(splits["baseline"]["transaction_id"])
        current_ids = set(splits["current"]["transaction_id"])
        assert len(baseline_ids & current_ids) == 0


class TestSampleData:
    """Testes para DataPreprocessor.sample_data()."""

    def test_sample_reduces_size(
        self, preprocessor: DataPreprocessor, sample_dataframe: pd.DataFrame
    ) -> None:
        """Verifica que amostragem reduz tamanho."""
        sampled = preprocessor.sample_data(sample_dataframe, fraction=0.5)
        assert len(sampled) < len(sample_dataframe)

    def test_sample_invalid_fraction(
        self, preprocessor: DataPreprocessor, sample_dataframe: pd.DataFrame
    ) -> None:
        """Verifica exceção para fração inválida."""
        with pytest.raises(ValueError):
            preprocessor.sample_data(sample_dataframe, fraction=0.0)
