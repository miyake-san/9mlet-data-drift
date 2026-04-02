"""
Testes unitários para o módulo data_preprocessing.

Valida a geração de dados sintéticos, carga, limpeza e divisão,
garantindo que o dataset produzido é consistente com as especificações
da Aula 1 (10.000 instâncias, 5 períodos, 10 features + target + periodo).
"""

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import DataPreprocessor


@pytest.fixture
def preprocessor() -> DataPreprocessor:
    """Fixture: instância do DataPreprocessor com seed fixa."""
    return DataPreprocessor(random_state=42)


@pytest.fixture
def sample_df(preprocessor: DataPreprocessor) -> pd.DataFrame:
    """Fixture: dataset sintético pequeno para testes rápidos."""
    return preprocessor.generate_synthetic_data(n_samples=500, n_periods=5)


# ---------------------------------------------------------------
# Testes de geração de dados sintéticos
# ---------------------------------------------------------------

class TestGenerateSyntheticData:
    """Testes da função generate_synthetic_data."""

    def test_shape_correta(self, preprocessor: DataPreprocessor) -> None:
        """Verifica se o shape retornado está certo."""
        df = preprocessor.generate_synthetic_data(n_samples=1000)
        assert len(df) == 1000
        # 10 features + target + periodo = 12 colunas
        assert df.shape[1] == 12

    def test_colunas_esperadas(self, sample_df: pd.DataFrame) -> None:
        """Verifica presença de todas as colunas."""
        expected = DataPreprocessor.FEATURE_NAMES + ["target", "periodo"]
        assert list(sample_df.columns) == expected

    def test_target_binario(self, sample_df: pd.DataFrame) -> None:
        """Verifica que target contém apenas 0 e 1."""
        assert set(sample_df["target"].unique()).issubset({0, 1})

    def test_periodos_corretos(self, sample_df: pd.DataFrame) -> None:
        """Verifica que existem 5 períodos."""
        assert sorted(sample_df["periodo"].unique()) == [1, 2, 3, 4, 5]

    def test_reproducibilidade_com_seed(self) -> None:
        """Verifica que a mesma seed produz os mesmos dados."""
        p1 = DataPreprocessor(random_state=123)
        p2 = DataPreprocessor(random_state=123)
        df1 = p1.generate_synthetic_data(n_samples=100)
        df2 = p2.generate_synthetic_data(n_samples=100)
        pd.testing.assert_frame_equal(df1, df2)

    def test_drift_injetado(self, preprocessor: DataPreprocessor) -> None:
        """Verifica que a média das features 'driftadas' muda nos períodos com drift."""
        df = preprocessor.generate_synthetic_data(
            n_samples=5000, drift_start_period=3, drift_magnitude=2.0
        )
        mean_p1 = df[df["periodo"] == 1]["idade"].mean()
        mean_p5 = df[df["periodo"] == 5]["idade"].mean()
        # Período 5 deve ter média significativamente maior
        assert mean_p5 > mean_p1 + 1.0


# ---------------------------------------------------------------
# Testes de limpeza e preparação
# ---------------------------------------------------------------

class TestCleanAndPrepare:
    """Testes de limpeza e preparação de features."""

    def test_clean_data_remove_nan(self, preprocessor: DataPreprocessor) -> None:
        """Verifica remoção de NaNs."""
        df = pd.DataFrame({"idade": [1.0, np.nan, 3.0], "target": [0, 1, 0], "periodo": [1, 1, 1]})
        cleaned = preprocessor.clean_data(df)
        assert cleaned.isna().sum().sum() == 0
        assert len(cleaned) == 2

    def test_prepare_features_shape(self, preprocessor: DataPreprocessor, sample_df: pd.DataFrame) -> None:
        """Verifica shape após padronização."""
        X, y = preprocessor.prepare_features(sample_df, fit=True)
        assert X.shape[0] == len(sample_df)
        assert X.shape[1] == len(DataPreprocessor.FEATURE_NAMES)
        assert y.shape[0] == len(sample_df)

    def test_prepare_features_standardized(self, preprocessor: DataPreprocessor, sample_df: pd.DataFrame) -> None:
        """Verifica que features padronizadas têm média ~0 e std ~1."""
        X, _ = preprocessor.prepare_features(sample_df, fit=True)
        assert abs(X.mean()) < 0.5  # tolerância para dataset finito
        assert abs(X.std() - 1.0) < 0.5


# ---------------------------------------------------------------
# Testes de split
# ---------------------------------------------------------------

class TestSplitData:
    """Testes de divisão de dados."""

    def test_split_data_proporcao(self, preprocessor: DataPreprocessor, sample_df: pd.DataFrame) -> None:
        """Verifica proporção treino/teste."""
        X, y = preprocessor.prepare_features(sample_df, fit=True)
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.2)
        assert len(X_train) + len(X_test) == len(X)
        assert abs(len(X_test) / len(X) - 0.2) < 0.05

    def test_split_by_period(self, preprocessor: DataPreprocessor, sample_df: pd.DataFrame) -> None:
        """Verifica divisão por períodos."""
        df_train, df_test = preprocessor.split_by_period(
            sample_df, train_periods=[1, 2], test_periods=[3, 4, 5]
        )
        assert all(df_train["periodo"].isin([1, 2]))
        assert all(df_test["periodo"].isin([3, 4, 5]))
