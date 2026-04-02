"""
Testes unitários para o módulo data_preprocessing.

Valida o DataPreprocessor: carregamento, limpeza, preparação de features
e separação de dados de referência/produção conforme a Aula 5.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import DataPreprocessor
from src.utils import set_seed


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Fixture com DataFrame de teste contendo embeddings sintéticos."""
    set_seed(42)
    n = 100
    embedding_dim = 8
    rng = np.random.RandomState(42)

    data = {
        "sample_id": range(n),
        "period": (["reference"] * 50 + ["production_stable"] * 25
                   + ["production_drift"] * 25),
        "category": rng.choice(["esportes", "politica", "clima"], n),
    }
    for i in range(embedding_dim):
        data[f"emb_{i}"] = rng.randn(n)

    return pd.DataFrame(data)


@pytest.fixture
def csv_path(sample_dataframe: pd.DataFrame, tmp_path: Path) -> str:
    """Fixture que salva DataFrame em CSV temporário e retorna o caminho."""
    filepath = tmp_path / "test_dataset.csv"
    sample_dataframe.to_csv(filepath, index=False)
    return str(filepath)


@pytest.fixture
def preprocessor() -> DataPreprocessor:
    """Fixture com DataPreprocessor configurado para 8 dimensões."""
    return DataPreprocessor(embedding_dim=8)


class TestLoadData:
    """Testes para DataPreprocessor.load_data()."""

    def test_load_data_valid_file(
        self, preprocessor: DataPreprocessor, csv_path: str
    ) -> None:
        """Testa carregamento de arquivo CSV válido."""
        df = preprocessor.load_data(csv_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert preprocessor.embedding_dim == 8

    def test_load_data_file_not_found(
        self, preprocessor: DataPreprocessor
    ) -> None:
        """Testa erro quando arquivo não existe."""
        with pytest.raises(FileNotFoundError):
            preprocessor.load_data("/nao_existe/fake.csv")

    def test_load_data_no_embedding_cols(
        self, tmp_path: Path
    ) -> None:
        """Testa erro quando não há colunas de embedding."""
        filepath = tmp_path / "no_emb.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(filepath, index=False)
        preprocessor = DataPreprocessor(embedding_dim=8)
        with pytest.raises(ValueError, match="Colunas de embedding"):
            preprocessor.load_data(str(filepath))

    def test_load_data_auto_detects_dims(
        self, csv_path: str
    ) -> None:
        """Testa detecção automática de dimensionalidade."""
        preprocessor = DataPreprocessor(embedding_dim=999)  # errado de propósito
        preprocessor.load_data(csv_path)
        assert preprocessor.embedding_dim == 8  # corrigido automaticamente


class TestCleanData:
    """Testes para DataPreprocessor.clean_data()."""

    def test_clean_data_removes_nans(
        self, preprocessor: DataPreprocessor, csv_path: str
    ) -> None:
        """Testa remoção de linhas com NaN nos embeddings."""
        df = preprocessor.load_data(csv_path)
        # Injeta NaN
        df.loc[0, "emb_0"] = np.nan
        df.loc[1, "emb_3"] = np.nan
        preprocessor.data = df

        df_clean = preprocessor.clean_data()
        assert len(df_clean) == 98

    def test_clean_data_removes_zero_embeddings(
        self, preprocessor: DataPreprocessor, csv_path: str
    ) -> None:
        """Testa remoção de embeddings completamente zerados."""
        df = preprocessor.load_data(csv_path)
        for i in range(preprocessor.embedding_dim):
            df.loc[0, f"emb_{i}"] = 0.0
        preprocessor.data = df

        df_clean = preprocessor.clean_data()
        assert len(df_clean) == 99

    def test_clean_data_no_data_raises(
        self, preprocessor: DataPreprocessor
    ) -> None:
        """Testa erro quando nenhum dado foi carregado."""
        with pytest.raises(ValueError, match="Nenhum dado carregado"):
            preprocessor.clean_data()


class TestPrepareFeatures:
    """Testes para DataPreprocessor.prepare_features()."""

    def test_prepare_features_shape(
        self, preprocessor: DataPreprocessor, csv_path: str
    ) -> None:
        """Testa shape correto dos embeddings processados."""
        preprocessor.load_data(csv_path)
        embeddings = preprocessor.prepare_features(normalize=False)
        assert embeddings.shape == (100, 8)
        assert embeddings.dtype == np.float64

    def test_prepare_features_normalization(
        self, preprocessor: DataPreprocessor, csv_path: str
    ) -> None:
        """Testa normalização com StandardScaler."""
        preprocessor.load_data(csv_path)
        embeddings = preprocessor.prepare_features(normalize=True)
        # Após StandardScaler, média ≈ 0 e std ≈ 1
        assert np.abs(embeddings.mean(axis=0)).max() < 0.1
        assert np.abs(embeddings.std(axis=0) - 1.0).max() < 0.2


class TestSplitData:
    """Testes para DataPreprocessor.split_data()."""

    def test_split_data_correct_sizes(
        self, preprocessor: DataPreprocessor, csv_path: str
    ) -> None:
        """Testa tamanhos corretos após split referência/produção."""
        preprocessor.load_data(csv_path)
        emb_ref, emb_prod, df_ref, df_prod = preprocessor.split_data()
        assert len(emb_ref) == 50
        assert len(emb_prod) == 50
        assert emb_ref.shape[1] == 8

    def test_split_data_missing_column(
        self, preprocessor: DataPreprocessor, csv_path: str
    ) -> None:
        """Testa erro quando coluna de período não existe."""
        preprocessor.load_data(csv_path)
        with pytest.raises(ValueError, match="não encontrada"):
            preprocessor.split_data(period_column="inexistente")
