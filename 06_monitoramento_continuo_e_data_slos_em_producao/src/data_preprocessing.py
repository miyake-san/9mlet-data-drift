"""
Módulo de pré-processamento de dados para monitoramento de ML.

Implementa geração de dataset sintético de crédito digital (fintech)
e pré-processamento para separação de dados de referência vs produção,
conforme descrito na seção 'Hands On' do material da Aula 6.

Referências:
    Gama, J. et al. (2014). A survey on concept drift adaptation.
    ACM Computing Surveys, 46(4).
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """Pré-processamento e geração de dados de crédito digital.

    Gera dados sintéticos simulando cenário de fintech com drift,
    conforme descrito no caso de crédito digital do DOCUMENTO_AULA_6.md.

    Attributes:
        random_state: Seed para reproduzibilidade.
        data: DataFrame com dados carregados.
    """

    def __init__(self, random_state: int = 42) -> None:
        """Inicializa o preprocessor.

        Args:
            random_state: Seed para reproduzibilidade dos resultados.
        """
        self.random_state = random_state
        self.data: Optional[pd.DataFrame] = None
        self._rng = np.random.RandomState(random_state)

    # ------------------------------------------------------------------
    # Geração de dataset sintético
    # ------------------------------------------------------------------

    @staticmethod
    def generate_dataset(
        n_reference: int = 5000,
        n_production: int = 5000,
        output_path: Optional[str] = None,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Gera dataset sintético de crédito digital com drift.

        Cria dados de referência (treino) e dados de produção com drift
        introduzido, simulando o cenário descrito no material da aula:
        'Imagine que uma fintech acaba de lançar um modelo de ML para
        aprovação instantânea de crédito'.

        O drift é introduzido nas features idade (+5 anos),
        renda_mensal (-15%) e score_credito (-50 pontos), conforme
        discutido na seção 'Data Drift & Concept Drift' do doc 04.

        Args:
            n_reference: Número de amostras de referência.
            n_production: Número de amostras de produção.
            output_path: Caminho para salvar o CSV. Se None, usa path padrão.
            random_state: Seed para reproduzibilidade.

        Returns:
            DataFrame com dados de referência e produção concatenados.
        """
        rng = np.random.RandomState(random_state)

        # --- Dados de REFERÊNCIA (treino) ---
        ref = DataPreprocessor._generate_partition(
            n_samples=n_reference,
            rng=rng,
            is_production=False,
        )

        # --- Dados de PRODUÇÃO (com drift) ---
        prod = DataPreprocessor._generate_partition(
            n_samples=n_production,
            rng=rng,
            is_production=True,
        )

        dataset = pd.concat([ref, prod], ignore_index=True)

        # Salvar CSV
        if output_path is None:
            base_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
            base_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(base_dir / "dataset.csv")

        dataset.to_csv(output_path, index=False)
        print(f"Dataset salvo em: {output_path}")
        print(f"  Referência: {n_reference} amostras | Produção: {n_production} amostras")

        return dataset

    @staticmethod
    def _generate_partition(
        n_samples: int,
        rng: np.random.RandomState,
        is_production: bool = False,
    ) -> pd.DataFrame:
        """Gera uma partição de dados (referência ou produção).

        Implementa drift nas features para dados de produção, conforme
        discutido na seção 'Data Drift & Concept Drift' do doc 04.

        Args:
            n_samples: Número de amostras.
            rng: RandomState para reproduzibilidade.
            is_production: Se True, aplica drift nas features.

        Returns:
            DataFrame com features e target.
        """
        # Parâmetros base
        idade_media = 35.0 + (5.0 if is_production else 0.0)
        renda_fator = 0.85 if is_production else 1.0
        score_offset = -50.0 if is_production else 0.0

        # Features
        idade = rng.normal(idade_media, 10, n_samples).clip(18, 70)
        renda_mensal = (rng.lognormal(9.0, 0.8, n_samples) * renda_fator).clip(1000, 50000)
        score_credito = (rng.normal(600 + score_offset, 150, n_samples)).clip(0, 1000)
        tempo_emprego = rng.exponential(5, n_samples).clip(0, 40)
        valor_emprestimo = rng.lognormal(9.5, 0.6, n_samples).clip(500, 200000)
        taxa_utilizacao = rng.beta(2, 5, n_samples)
        num_parcelas = rng.choice([6, 12, 18, 24, 36, 48, 60], n_samples)
        historico_atrasos = rng.poisson(1.5 if is_production else 0.8, n_samples).clip(0, 20)
        saldo_conta = rng.lognormal(8.0, 1.0, n_samples).clip(0, 500000)
        qtd_dependentes = rng.poisson(1.2, n_samples).clip(0, 8)

        # Target: probabilidade de inadimplência baseada nas features
        # Implementa concept drift: em produção a relação muda
        logit = (
            -2.0
            + 0.02 * (idade - 35)
            - 0.00005 * renda_mensal
            - 0.003 * score_credito
            - 0.05 * tempo_emprego
            + 0.000005 * valor_emprestimo
            + 1.5 * taxa_utilizacao
            + 0.15 * historico_atrasos
            - 0.00001 * saldo_conta
            + 0.1 * qtd_dependentes
        )

        if is_production:
            # Concept drift: mudança na relação features-target
            logit += 0.5

        prob_default = 1 / (1 + np.exp(-logit))
        target = (rng.random(n_samples) < prob_default).astype(int)

        # Introduzir missings (~0.5% referência, ~2% produção)
        missing_rate = 0.02 if is_production else 0.005

        df = pd.DataFrame({
            "idade": idade,
            "renda_mensal": renda_mensal,
            "score_credito": score_credito,
            "tempo_emprego": tempo_emprego,
            "valor_emprestimo": valor_emprestimo,
            "taxa_utilizacao_credito": taxa_utilizacao,
            "num_parcelas": num_parcelas,
            "historico_atrasos": historico_atrasos,
            "saldo_conta": saldo_conta,
            "qtd_dependentes": qtd_dependentes,
            "is_production": int(is_production),
            "target": target,
        })

        # Introduzir valores ausentes aleatórios em features numéricas
        features_with_missing = [
            "idade", "renda_mensal", "score_credito",
            "tempo_emprego", "saldo_conta",
        ]
        for col in features_with_missing:
            mask = rng.random(n_samples) < missing_rate
            df.loc[mask, col] = np.nan

        return df

    # ------------------------------------------------------------------
    # Carregamento e pré-processamento
    # ------------------------------------------------------------------

    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """Carrega dados do CSV.

        Args:
            filepath: Caminho do arquivo CSV. Se None, usa path padrão.

        Returns:
            DataFrame com os dados carregados.

        Raises:
            FileNotFoundError: Se o arquivo não existir.
        """
        if filepath is None:
            filepath = str(
                Path(__file__).resolve().parent.parent / "data" / "raw" / "dataset.csv"
            )

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Arquivo não encontrado: {filepath}. "
                "Execute DataPreprocessor.generate_dataset() primeiro."
            )

        self.data = pd.read_csv(filepath)
        print(f"Dados carregados: {self.data.shape[0]} amostras, {self.data.shape[1]} colunas")
        return self.data

    def clean_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Limpa dados removendo duplicatas e tratando missings.

        Implementa verificação de qualidade conforme Snippet 3 do
        DOCUMENTO_AULA_6.md: 'Verificação de Qualidade dos Dados'.

        Args:
            df: DataFrame a limpar. Se None, usa self.data.

        Returns:
            DataFrame limpo.
        """
        if df is None:
            df = self.data.copy()
        else:
            df = df.copy()

        # Remover duplicatas
        n_before = len(df)
        df = df.drop_duplicates()
        n_removed = n_before - len(df)
        if n_removed > 0:
            print(f"Duplicatas removidas: {n_removed}")

        # Preencher missings com mediana (estratégia simples)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"  {col}: {n_missing} missings preenchidos com mediana ({median_val:.2f})")

        return df

    def prepare_features(
        self, df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepara features e target para modelagem.

        Args:
            df: DataFrame de entrada. Se None, usa self.data.

        Returns:
            Tupla (X, y) com features e target.
        """
        if df is None:
            df = self.data

        feature_cols = [
            "idade", "renda_mensal", "score_credito", "tempo_emprego",
            "valor_emprestimo", "taxa_utilizacao_credito", "num_parcelas",
            "historico_atrasos", "saldo_conta", "qtd_dependentes",
        ]

        X = df[feature_cols].copy()
        y = df["target"].copy()

        return X, y

    def split_data(
        self,
        df: Optional[pd.DataFrame] = None,
        test_size: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Divide dados em treino e teste.

        Args:
            df: DataFrame de entrada. Se None, usa self.data.
            test_size: Fração para teste.

        Returns:
            Tupla (X_train, X_test, y_train, y_test).
        """
        if df is None:
            df = self.data

        X, y = self.prepare_features(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")
        return X_train, X_test, y_train, y_test

    def split_reference_production(
        self, df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Separa dados de referência e produção pela flag is_production.

        Implementa a separação necessária para detecção de drift,
        conforme discutido na seção 'Data Drift & Concept Drift' do doc 04.

        Args:
            df: DataFrame de entrada. Se None, usa self.data.

        Returns:
            Tupla (df_reference, df_production).
        """
        if df is None:
            df = self.data

        df_ref = df[df["is_production"] == 0].copy()
        df_prod = df[df["is_production"] == 1].copy()

        print(f"Referência: {len(df_ref)} | Produção: {len(df_prod)}")
        return df_ref, df_prod
