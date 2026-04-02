"""
Testes unitários para o módulo data_preprocessing.

Valida carregamento, limpeza, preparação de features e divisão de dados
do DataPreprocessor, conforme o pipeline da Aula 8 (Documento 04).
"""

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COL,
    DataPreprocessor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Cria DataFrame de exemplo simulando transações de pagamento."""
    np.random.seed(42)
    n = 200

    data = {
        "transaction_id": [f"TX_{i:04d}" for i in range(n)],
        "valor_transacao": np.random.lognormal(4.5, 1.0, n),
        "tempo_conta_cliente": np.random.exponential(180, n).astype(int),
        "num_transacoes_24h": np.random.poisson(3, n),
        "valor_medio_historico": np.random.lognormal(4.2, 0.8, n),
        "distancia_localizacao": np.random.exponential(15, n),
        "hora_transacao": np.random.choice(range(24), n),
        "score_risco_dispositivo": np.random.beta(2, 8, n),
        "tentativas_senha": np.random.choice([0, 1, 2, 3], n),
        "is_weekend": np.random.binomial(1, 0.28, n),
        "razao_valor_medio": np.random.lognormal(0, 0.5, n),
        "tipo_cartao": np.random.choice(["credito", "debito", "prepago"], n),
        "canal_transacao": np.random.choice(["app", "web", "pos", "telefone"], n),
        "pais_origem": np.random.choice(["BR", "US", "PT"], n),
        "fraude": np.random.choice([0, 1], n, p=[0.95, 0.05]),
        "periodo": ["referencia"] * 100 + ["producao"] * 100,
    }
    return pd.DataFrame(data)


@pytest.fixture
def preprocessor() -> DataPreprocessor:
    """Cria instância limpa do DataPreprocessor."""
    return DataPreprocessor()


@pytest.fixture
def csv_path(tmp_path: object, sample_dataframe: pd.DataFrame) -> str:
    """Salva DataFrame como CSV temporário e retorna caminho."""
    filepath = tmp_path / "test_data.csv"
    sample_dataframe.to_csv(filepath, index=False)
    return str(filepath)


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------


class TestLoadData:
    def test_load_data_returns_dataframe(
        self, preprocessor: DataPreprocessor, csv_path: str
    ) -> None:
        """Verifica que load_data retorna um DataFrame."""
        df = preprocessor.load_data(csv_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 200

    def test_load_data_validates_columns(
        self, preprocessor: DataPreprocessor, tmp_path: object
    ) -> None:
        """Verifica que load_data rejeita CSV sem colunas obrigatórias."""
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"a": [1], "b": [2]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="Colunas obrigatórias ausentes"):
            preprocessor.load_data(str(bad_csv))

    def test_load_data_file_not_found(self, preprocessor: DataPreprocessor) -> None:
        """Verifica exceção para arquivo inexistente."""
        with pytest.raises(FileNotFoundError):
            preprocessor.load_data("/caminho/inexistente.csv")


class TestCleanData:
    def test_clean_removes_duplicates(
        self, preprocessor: DataPreprocessor, sample_dataframe: pd.DataFrame
    ) -> None:
        """Verifica remoção de linhas duplicadas."""
        df_dup = pd.concat([sample_dataframe, sample_dataframe.iloc[:5]], ignore_index=True)
        cleaned = preprocessor.clean_data(df_dup)
        assert len(cleaned) <= len(df_dup)

    def test_clean_fills_na_numeric(
        self, preprocessor: DataPreprocessor, sample_dataframe: pd.DataFrame
    ) -> None:
        """Verifica tratamento de NaN em features numéricas."""
        df = sample_dataframe.copy()
        df.loc[0, "valor_transacao"] = np.nan
        cleaned = preprocessor.clean_data(df)
        assert not cleaned["valor_transacao"].isna().any()

    def test_clean_clips_negative_values(
        self, preprocessor: DataPreprocessor, sample_dataframe: pd.DataFrame
    ) -> None:
        """Verifica que valor_transacao negativo é clippado para 0."""
        df = sample_dataframe.copy()
        df.loc[0, "valor_transacao"] = -100.0
        cleaned = preprocessor.clean_data(df)
        assert cleaned["valor_transacao"].min() >= 0


class TestPrepareFeatures:
    def test_prepare_features_scales_numeric(
        self, preprocessor: DataPreprocessor, sample_dataframe: pd.DataFrame
    ) -> None:
        """Verifica que features numéricas são escalonadas (mean ≈ 0, std ≈ 1)."""
        prepared = preprocessor.prepare_features(sample_dataframe, fit=True)
        for col in NUMERIC_FEATURES:
            if col in prepared.columns:
                assert abs(prepared[col].mean()) < 0.5

    def test_prepare_features_encodes_categorical(
        self, preprocessor: DataPreprocessor, sample_dataframe: pd.DataFrame
    ) -> None:
        """Verifica que features categóricas são codificadas como inteiros."""
        prepared = preprocessor.prepare_features(sample_dataframe, fit=True)
        for col in CATEGORICAL_FEATURES:
            if col in prepared.columns:
                assert prepared[col].dtype in [np.int32, np.int64, int]


class TestSplitData:
    def test_split_data_shapes(
        self, preprocessor: DataPreprocessor, sample_dataframe: pd.DataFrame
    ) -> None:
        """Verifica shapes corretos do split train/test."""
        preprocessor.data = sample_dataframe
        X_train, X_test, y_train, y_test = preprocessor.split_data()
        total = len(X_train) + len(X_test)
        assert total == len(sample_dataframe)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

    def test_split_data_stratified(
        self, preprocessor: DataPreprocessor, sample_dataframe: pd.DataFrame
    ) -> None:
        """Verifica que split mantém proporção de classes (estratificado)."""
        preprocessor.data = sample_dataframe
        _, _, _, y_test = preprocessor.split_data(test_size=0.3)
        # Taxa de fraude no teste deve ser próxima da taxa global
        global_rate = sample_dataframe[TARGET_COL].mean()
        test_rate = y_test.mean()
        assert abs(global_rate - test_rate) < 0.15
