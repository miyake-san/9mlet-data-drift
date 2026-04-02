"""
Módulo de pré-processamento de dados para o pipeline de monitoramento de fraude.

Implementa carregamento, limpeza, preparação de features e separação de dados
conforme o caso de negócio da Aula 8 (plataforma de pagamentos global).
O pré-processamento separa os dados em períodos de referência e produção,
fundamentais para comparação via testes estatísticos de drift (KS, Wasserstein,
Jensen-Shannon) discutidos na seção 'Saiba Mais' do Documento 04.

Referências:
    Rabanser, S. et al. (NeurIPS 2019). Failing loudly: an empirical
        study of methods for detecting dataset shift.
    Moreno-Torres, J. G. et al. (2012). A unifying view on dataset shift
        in classification. Pattern Recognition, 45(1), 521-530.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Colunas numéricas do dataset de transações
NUMERIC_FEATURES: List[str] = [
    "valor_transacao",
    "tempo_conta_cliente",
    "num_transacoes_24h",
    "valor_medio_historico",
    "distancia_localizacao",
    "hora_transacao",
    "score_risco_dispositivo",
    "tentativas_senha",
    "is_weekend",
    "razao_valor_medio",
]

# Colunas categóricas do dataset
CATEGORICAL_FEATURES: List[str] = [
    "tipo_cartao",
    "canal_transacao",
    "pais_origem",
]

TARGET_COL: str = "fraude"
PERIOD_COL: str = "periodo"


class DataPreprocessor:
    """
    Pré-processador de dados de transações para detecção de fraude e drift.

    Implementa o pipeline completo de preparação de dados conforme discutido
    na Aula 8, separando dados em referência (baseline) e produção (com drift),
    o que permite a aplicação de ferramentas como Evidently AI e NannyML para
    comparação de distribuições P_ref(X) vs P_prod(X).

    Conforme Moreno-Torres et al. (2012), a separação temporal dos dados é
    fundamental para identificar dataset shift em cenários reais.

    Attributes:
        data: DataFrame com os dados carregados.
        scaler: StandardScaler para normalização de features numéricas.
        label_encoders: Dicionário de LabelEncoders para features categóricas.
        is_fitted: Indica se o preprocessador foi ajustado (fit).
    """

    def __init__(self) -> None:
        """Inicializa o DataPreprocessor."""
        self.data: Optional[pd.DataFrame] = None
        self.scaler: StandardScaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.is_fitted: bool = False

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Carrega dados de transações a partir de um arquivo CSV.

        Implementa validação básica de schema, verificando a presença
        de colunas obrigatórias conforme recomendado pelo Great Expectations
        (seção 'Validação de Dados' do Documento 04).

        Args:
            filepath: Caminho para o arquivo CSV.

        Returns:
            DataFrame com os dados carregados.

        Raises:
            FileNotFoundError: Se o arquivo não existir.
            ValueError: Se colunas obrigatórias estiverem ausentes.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

        self.data = pd.read_csv(filepath)

        # Validação de schema mínimo
        required_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL]
        missing = set(required_cols) - set(self.data.columns)
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

        return self.data

    def clean_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Limpa os dados removendo duplicatas, tratando nulos e outliers.

        Implements data quality checks inspired by Great Expectations
        expectations, conforme discutido na seção 'Validação de Dados'
        do Documento 04. Valores negativos em valor_transacao e nulos
        em campos críticos são tratados.

        Args:
            df: DataFrame a limpar. Se None, usa self.data.

        Returns:
            DataFrame limpo.

        Raises:
            ValueError: Se nenhum dado estiver disponível.
        """
        if df is None:
            df = self.data
        if df is None:
            raise ValueError("Nenhum dado disponível. Carregue dados primeiro.")

        df = df.copy()

        # Remover duplicatas por transaction_id (se existir)
        if "transaction_id" in df.columns:
            df = df.drop_duplicates(subset=["transaction_id"])

        # Tratar nulos em features numéricas com mediana
        for col in NUMERIC_FEATURES:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        # Tratar nulos em features categóricas com moda
        for col in CATEGORICAL_FEATURES:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode()[0])

        # Garantir valor_transacao >= 0 (regra do Great Expectations)
        if "valor_transacao" in df.columns:
            df["valor_transacao"] = df["valor_transacao"].clip(lower=0)

        self.data = df
        return df

    def prepare_features(
        self, df: Optional[pd.DataFrame] = None, fit: bool = True
    ) -> pd.DataFrame:
        """
        Prepara features para modelagem: encoding categórico + scaling numérico.

        O scaling é essencial para que testes estatísticos de drift (como KS
        e Wasserstein) operem em escalas comparáveis, conforme discutido
        na seção 'Detecção Estatística de Data Drift' do Documento 04.

        Args:
            df: DataFrame com dados. Se None, usa self.data.
            fit: Se True, ajusta scaler/encoders; se False, aplica transformação.

        Returns:
            DataFrame com features preparadas.

        Raises:
            ValueError: Se nenhum dado estiver disponível ou fit não realizado.
        """
        if df is None:
            df = self.data
        if df is None:
            raise ValueError("Nenhum dado disponível. Carregue dados primeiro.")

        df = df.copy()

        # Encoding de features categóricas
        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col not in self.label_encoders:
                    raise ValueError(f"LabelEncoder não ajustado para '{col}'. Chame com fit=True primeiro.")
                le = self.label_encoders[col]
                # Tratar categorias não vistas — mapear para -1
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in known else -1
                )

        # Scaling de features numéricas
        num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        if fit:
            df[num_cols] = self.scaler.fit_transform(df[num_cols])
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler não ajustado. Chame prepare_features com fit=True primeiro.")
            df[num_cols] = self.scaler.transform(df[num_cols])

        return df

    def split_data(
        self,
        df: Optional[pd.DataFrame] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divide os dados em treino e teste.

        Utiliza estratificação pelo target (fraude) para manter a proporção
        de classes, conforme boas práticas de ML descritas no Documento 04.

        Args:
            df: DataFrame com features preparadas. Se None, usa self.data.
            test_size: Fração dos dados para teste (padrão: 20%).
            random_state: Semente para reproduzibilidade.

        Returns:
            Tupla (X_train, X_test, y_train, y_test).
        """
        if df is None:
            df = self.data
        if df is None:
            raise ValueError("Nenhum dado disponível.")

        feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        available_cols = [c for c in feature_cols if c in df.columns]

        X = df[available_cols]
        y = df[TARGET_COL]

        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    def split_by_period(
        self, df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separa dados por período (referência vs produção).

        Essa separação é o fundamento para detecção de drift, pois permite
        comparar as distribuições P_ref(X) vs P_prod(X) conforme formalizado
        por Rabanser et al. (2019) e Moreno-Torres et al. (2012).

        Args:
            df: DataFrame com coluna 'periodo'. Se None, usa self.data.

        Returns:
            Tupla (df_referencia, df_producao).

        Raises:
            ValueError: Se coluna 'periodo' não existir.
        """
        if df is None:
            df = self.data
        if df is None:
            raise ValueError("Nenhum dado disponível.")
        if PERIOD_COL not in df.columns:
            raise ValueError(f"Coluna '{PERIOD_COL}' não encontrada nos dados.")

        df_ref = df[df[PERIOD_COL] == "referencia"].copy()
        df_prod = df[df[PERIOD_COL] == "producao"].copy()
        return df_ref, df_prod

    def get_feature_columns(self) -> List[str]:
        """Retorna lista de todas as colunas de features disponíveis."""
        return NUMERIC_FEATURES + CATEGORICAL_FEATURES

    def get_numeric_columns(self) -> List[str]:
        """Retorna lista de colunas numéricas."""
        return NUMERIC_FEATURES.copy()

    def get_categorical_columns(self) -> List[str]:
        """Retorna lista de colunas categóricas."""
        return CATEGORICAL_FEATURES.copy()
