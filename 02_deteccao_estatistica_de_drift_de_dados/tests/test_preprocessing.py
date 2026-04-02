"""
Testes unitários para o módulo data_preprocessing.

Valida a geração, carga, limpeza e preparação de dados sintéticos
de crédito utilizados na Aula 2 — Detecção Estatística de Drift.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import DataPreprocessor


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def preprocessor() -> DataPreprocessor:
    return DataPreprocessor(seed=42, n_samples=200)


@pytest.fixture
def dataset(preprocessor: DataPreprocessor) -> pd.DataFrame:
    return preprocessor.generate_dataset()


@pytest.fixture
def ref_prod(preprocessor: DataPreprocessor, dataset: pd.DataFrame):
    ref = dataset[dataset["origem"] == "referencia"].copy()
    prod = dataset[dataset["origem"] == "producao"].copy()
    return ref, prod


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------

class TestDataPreprocessorGeneration:
    """Testes de geração do dataset sintético."""

    def test_generate_dataset_shape(self, preprocessor: DataPreprocessor) -> None:
        """Verifica que o dataset gerado tem o número correto de linhas."""
        df = preprocessor.generate_dataset()
        # n_samples por grupo × 2 (referência + produção)
        assert len(df) == 2 * preprocessor.n_samples

    def test_generate_dataset_columns(self, dataset: pd.DataFrame) -> None:
        """Verifica que todas as colunas esperadas estão presentes."""
        expected_cols = {
            "idade", "renda_mensal", "score_credito", "tempo_emprego",
            "valor_emprestimo", "taxa_utilizacao_credito", "num_parcelas_atraso",
            "tipo_residencia", "categoria_risco", "origem", "inadimplente",
        }
        assert expected_cols.issubset(set(dataset.columns))

    def test_generate_dataset_origem_groups(self, dataset: pd.DataFrame) -> None:
        """Verifica que existem exatamente dois grupos: referencia e producao."""
        assert set(dataset["origem"].unique()) == {"referencia", "producao"}

    def test_generate_dataset_target_binary(self, dataset: pd.DataFrame) -> None:
        """Verifica que o target é binário (0 ou 1)."""
        assert set(dataset["inadimplente"].unique()).issubset({0, 1})

    def test_generate_dataset_age_range(self, dataset: pd.DataFrame) -> None:
        """Verifica que idades estão no range válido (18–80)."""
        assert dataset["idade"].min() >= 18
        assert dataset["idade"].max() <= 80

    def test_generate_dataset_taxa_range(self, dataset: pd.DataFrame) -> None:
        """Verifica que taxa_utilizacao_credito está entre 0 e 1."""
        assert dataset["taxa_utilizacao_credito"].min() >= 0
        assert dataset["taxa_utilizacao_credito"].max() <= 1

    def test_drift_injected_in_age(self, ref_prod) -> None:
        """Verifica que a média de idade mudou entre ref e prod (drift injetado)."""
        ref, prod = ref_prod
        # Conforme REF_PARAMS: média 40; PROD_PARAMS: média 30
        assert ref["idade"].mean() > prod["idade"].mean()


class TestDataPreprocessorLoadClean:
    """Testes de carga e limpeza de dados."""

    def test_load_data_separation(self, preprocessor: DataPreprocessor, tmp_path) -> None:
        """Verifica que load_data separa corretamente referência e produção."""
        df = preprocessor.generate_dataset()
        filepath = tmp_path / "test_dataset.csv"
        df.to_csv(filepath, index=False)

        ref, prod = preprocessor.load_data(str(filepath))
        assert len(ref) > 0
        assert len(prod) > 0
        assert (ref["origem"] == "referencia").all()
        assert (prod["origem"] == "producao").all()

    def test_load_data_file_not_found(self, preprocessor: DataPreprocessor) -> None:
        """Verifica que FileNotFoundError é levantado para arquivo inexistente."""
        with pytest.raises(FileNotFoundError):
            preprocessor.load_data("/caminho/inexistente.csv")

    def test_clean_data_no_nulls(self, preprocessor: DataPreprocessor, dataset: pd.DataFrame) -> None:
        """Verifica que clean_data remove valores nulos."""
        # Injetar nulos artificiais
        df_dirty = dataset.copy()
        df_dirty.loc[0, "idade"] = np.nan
        df_cleaned = preprocessor.clean_data(df_dirty)
        assert df_cleaned.isnull().sum().sum() == 0

    def test_prepare_features_separation(self, preprocessor: DataPreprocessor, ref_prod) -> None:
        """Verifica que prepare_features retorna listas corretas."""
        ref, _ = ref_prod
        _, num_cols, cat_cols = preprocessor.prepare_features(ref)
        assert len(num_cols) >= 5
        assert len(cat_cols) >= 1
        assert "idade" in num_cols
        assert "tipo_residencia" in cat_cols
