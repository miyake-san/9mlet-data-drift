"""
Testes unitários para o módulo data_preprocessing.

Testa a geração de dataset sintético, carregamento de dados,
limpeza, preparação de features e separação referência/produção.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ajustar path para importar src/
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
def small_dataset(tmp_path: Path) -> pd.DataFrame:
    """Gera dataset pequeno para testes rápidos."""
    output_path = str(tmp_path / "test_dataset.csv")
    df = DataPreprocessor.generate_dataset(
        n_reference=200,
        n_production=200,
        output_path=output_path,
        random_state=42,
    )
    return df


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Cria DataFrame simples para testes de limpeza."""
    rng = np.random.RandomState(42)
    n = 100
    df = pd.DataFrame({
        "idade": rng.normal(35, 10, n),
        "renda_mensal": rng.lognormal(9.0, 0.8, n),
        "score_credito": rng.normal(600, 150, n),
        "tempo_emprego": rng.exponential(5, n),
        "valor_emprestimo": rng.lognormal(9.5, 0.6, n),
        "taxa_utilizacao_credito": rng.beta(2, 5, n),
        "num_parcelas": rng.choice([12, 24, 36], n),
        "historico_atrasos": rng.poisson(1, n),
        "saldo_conta": rng.lognormal(8.0, 1.0, n),
        "qtd_dependentes": rng.poisson(1, n),
        "is_production": [0] * 50 + [1] * 50,
        "target": rng.choice([0, 1], n),
    })
    # Introduzir alguns missings
    df.loc[0, "idade"] = np.nan
    df.loc[1, "renda_mensal"] = np.nan
    return df


# ======================================================================
# Testes
# ======================================================================


class TestGenerateDataset:
    """Testes para geração de dataset sintético."""

    def test_generate_dataset_shape(self, tmp_path: Path) -> None:
        """Verifica shape do dataset gerado."""
        output = str(tmp_path / "ds.csv")
        df = DataPreprocessor.generate_dataset(
            n_reference=100, n_production=100,
            output_path=output, random_state=42,
        )
        assert df.shape[0] == 200
        assert df.shape[1] == 12  # 10 features + is_production + target

    def test_generate_dataset_columns(self, small_dataset: pd.DataFrame) -> None:
        """Verifica que todas as colunas esperadas existem."""
        expected = {
            "idade", "renda_mensal", "score_credito", "tempo_emprego",
            "valor_emprestimo", "taxa_utilizacao_credito", "num_parcelas",
            "historico_atrasos", "saldo_conta", "qtd_dependentes",
            "is_production", "target",
        }
        assert set(small_dataset.columns) == expected

    def test_generate_dataset_reference_production_split(
        self, small_dataset: pd.DataFrame
    ) -> None:
        """Verifica separação referência/produção."""
        ref_count = (small_dataset["is_production"] == 0).sum()
        prod_count = (small_dataset["is_production"] == 1).sum()
        assert ref_count == 200
        assert prod_count == 200

    def test_generate_dataset_saves_csv(self, tmp_path: Path) -> None:
        """Verifica que CSV é salvo no caminho especificado."""
        output = str(tmp_path / "saved.csv")
        DataPreprocessor.generate_dataset(
            n_reference=50, n_production=50,
            output_path=output, random_state=42,
        )
        assert os.path.exists(output)

    def test_generate_dataset_target_binary(
        self, small_dataset: pd.DataFrame
    ) -> None:
        """Verifica que target é binário (0 ou 1)."""
        assert set(small_dataset["target"].unique()).issubset({0, 1})


class TestLoadAndClean:
    """Testes para carregamento e limpeza de dados."""

    def test_load_data_from_file(
        self, preprocessor: DataPreprocessor, tmp_path: Path
    ) -> None:
        """Verifica carregamento de dados de arquivo CSV."""
        output = str(tmp_path / "test.csv")
        DataPreprocessor.generate_dataset(
            n_reference=50, n_production=50,
            output_path=output, random_state=42,
        )
        df = preprocessor.load_data(output)
        assert df.shape[0] == 100

    def test_load_data_file_not_found(
        self, preprocessor: DataPreprocessor
    ) -> None:
        """Verifica exceção quando arquivo não existe."""
        with pytest.raises(FileNotFoundError):
            preprocessor.load_data("/nonexistent/path.csv")

    def test_clean_data_fills_missings(
        self, preprocessor: DataPreprocessor, sample_dataframe: pd.DataFrame
    ) -> None:
        """Verifica que limpeza preenche valores ausentes."""
        cleaned = preprocessor.clean_data(sample_dataframe)
        assert cleaned.isnull().sum().sum() == 0

    def test_clean_data_preserves_shape(
        self, preprocessor: DataPreprocessor, sample_dataframe: pd.DataFrame
    ) -> None:
        """Verifica que limpeza preserva número de colunas."""
        cleaned = preprocessor.clean_data(sample_dataframe)
        assert cleaned.shape[1] == sample_dataframe.shape[1]


class TestPrepareFeatures:
    """Testes para preparação de features."""

    def test_prepare_features_shapes(
        self, preprocessor: DataPreprocessor, sample_dataframe: pd.DataFrame
    ) -> None:
        """Verifica shapes de X e y."""
        X, y = preprocessor.prepare_features(sample_dataframe)
        assert X.shape[0] == 100
        assert X.shape[1] == 10  # 10 features
        assert len(y) == 100

    def test_prepare_features_no_target_in_X(
        self, preprocessor: DataPreprocessor, sample_dataframe: pd.DataFrame
    ) -> None:
        """Verifica que target não está nas features."""
        X, _ = preprocessor.prepare_features(sample_dataframe)
        assert "target" not in X.columns
        assert "is_production" not in X.columns

    def test_split_reference_production(
        self, preprocessor: DataPreprocessor, sample_dataframe: pd.DataFrame
    ) -> None:
        """Verifica separação por flag is_production."""
        df_ref, df_prod = preprocessor.split_reference_production(sample_dataframe)
        assert len(df_ref) == 50
        assert len(df_prod) == 50
        assert (df_ref["is_production"] == 0).all()
        assert (df_prod["is_production"] == 1).all()
