# -*- coding: utf-8 -*-
"""
test_preprocessing.py – Testes unitários para DataPreprocessor.

Cobre geração de dados sintéticos, carregamento, limpeza,
preparação de features e divisão por período.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import DataPreprocessor


@pytest.fixture
def preprocessor() -> DataPreprocessor:
    """Fixture: DataPreprocessor com seed fixa e tamanho reduzido."""
    return DataPreprocessor(n_samples=200, seed=42)


@pytest.fixture
def sample_df(preprocessor: DataPreprocessor) -> pd.DataFrame:
    """Fixture: DataFrame gerado pelo preprocessor."""
    return preprocessor.generate_dataset()


class TestGenerateDataset:
    """Testes para DataPreprocessor.generate_dataset()."""

    def test_generate_dataset_shape(self, preprocessor: DataPreprocessor) -> None:
        """Verifica shape correto: 2 * n_samples linhas, 7 colunas."""
        df = preprocessor.generate_dataset()
        assert df.shape == (400, 7)

    def test_generate_dataset_columns(self, sample_df: pd.DataFrame) -> None:
        """Verifica que todas as colunas esperadas estão presentes."""
        expected = {"idade", "renda", "divida", "score_credito",
                    "tempo_emprego", "num_parcelas", "periodo"}
        assert set(sample_df.columns) == expected

    def test_generate_dataset_periods(self, sample_df: pd.DataFrame) -> None:
        """Verifica que ambos os períodos estão presentes e balanceados."""
        counts = sample_df["periodo"].value_counts()
        assert set(counts.index) == {"referencia", "atual"}
        assert counts["referencia"] == 200
        assert counts["atual"] == 200

    def test_generate_dataset_reproducibility(self) -> None:
        """Verifica reprodutibilidade com mesma seed."""
        dp1 = DataPreprocessor(n_samples=100, seed=99)
        dp2 = DataPreprocessor(n_samples=100, seed=99)
        df1 = dp1.generate_dataset()
        df2 = dp2.generate_dataset()
        pd.testing.assert_frame_equal(df1, df2)

    def test_generate_dataset_value_ranges(self, sample_df: pd.DataFrame) -> None:
        """Verifica intervalos razoáveis para variáveis clipsadas."""
        assert sample_df["idade"].min() >= 18
        assert sample_df["idade"].max() <= 80
        assert sample_df["renda"].min() >= 500
        assert sample_df["score_credito"].min() >= 0
        assert sample_df["score_credito"].max() <= 1000


class TestCleanData:
    """Testes para DataPreprocessor.clean_data()."""

    def test_clean_removes_duplicates(self, preprocessor: DataPreprocessor) -> None:
        """Verifica que duplicatas são removidas."""
        df = preprocessor.generate_dataset()
        # Duplica primeira linha
        df = pd.concat([df, df.iloc[:1]], ignore_index=True)
        cleaned = preprocessor.clean_data(df)
        assert len(cleaned) == len(df) - 1

    def test_clean_removes_nulls(self, preprocessor: DataPreprocessor) -> None:
        """Verifica que linhas com NaN são removidas."""
        df = preprocessor.generate_dataset()
        df.loc[0, "idade"] = np.nan
        cleaned = preprocessor.clean_data(df)
        assert cleaned["idade"].isna().sum() == 0


class TestPrepareFeatures:
    """Testes para DataPreprocessor.prepare_features()."""

    def test_prepare_features_numeric_only(
        self, preprocessor: DataPreprocessor, sample_df: pd.DataFrame
    ) -> None:
        """Verifica que apenas colunas numéricas são retornadas."""
        features = preprocessor.prepare_features(sample_df)
        assert "periodo" not in features.columns
        assert features.shape[1] == 6

    def test_prepare_features_normalized(
        self, preprocessor: DataPreprocessor, sample_df: pd.DataFrame
    ) -> None:
        """Verifica que features normalizadas têm ~média 0 e ~std 1."""
        features = preprocessor.prepare_features(sample_df, normalize=True)
        for col in features.columns:
            assert abs(features[col].mean()) < 0.1
            assert abs(features[col].std() - 1.0) < 0.1


class TestSplitData:
    """Testes para DataPreprocessor.split_data()."""

    def test_split_data_sizes(
        self, preprocessor: DataPreprocessor, sample_df: pd.DataFrame
    ) -> None:
        """Verifica tamanhos corretos após divisão."""
        df_ref, df_cur = preprocessor.split_data(sample_df)
        assert len(df_ref) == 200
        assert len(df_cur) == 200


class TestGenerateAndSave:
    """Testes para DataPreprocessor.generate_and_save()."""

    def test_generate_and_save_creates_file(
        self, preprocessor: DataPreprocessor
    ) -> None:
        """Verifica que o CSV é criado no disco."""
        with tempfile.TemporaryDirectory() as tmp:
            path = preprocessor.generate_and_save(output_dir=tmp)
            assert os.path.isfile(path)
            df = pd.read_csv(path)
            assert df.shape[0] == 400
