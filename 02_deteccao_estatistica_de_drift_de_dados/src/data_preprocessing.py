"""
Módulo de pré-processamento e geração de dados sintéticos para detecção de drift.

Gera datasets sintéticos simulando o cenário de fintech de crédito descrito no
Documento da Aula 2, onde a distribuição de clientes muda ao longo do tempo
(ex.: clientes mais jovens, menor renda) causando data drift.

Referências:
    Gama, J. et al. (2014). A Survey on Concept Drift Adaptation.
    ACM Computing Surveys, 46(4), 44.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class DataPreprocessor:
    """Gera e prepara dados sintéticos de crédito com drift injetado.

    Implementa o cenário descrito na seção 'Saiba Mais' do Documento da Aula 2:
    'se antes a maioria dos tomadores de empréstimo tinha entre 30 e 50 anos,
    mas agora grande parte dos novos clientes é de jovens de 20 e poucos anos,
    a distribuição da variável idade se deslocou.'

    Attributes:
        seed: Semente para reprodutibilidade.
        n_samples: Número de amostras por grupo (referência e produção).
    """

    # Parâmetros de distribuição para dados de referência (treinamento)
    REF_PARAMS: Dict[str, Dict[str, float]] = {
        "idade": {"loc": 40.0, "scale": 10.0},
        "renda_mensal": {"loc": 5000.0, "scale": 2000.0},
        "score_credito": {"loc": 650.0, "scale": 100.0},
        "tempo_emprego": {"loc": 8.0, "scale": 4.0},
        "valor_emprestimo": {"loc": 15000.0, "scale": 8000.0},
        "taxa_utilizacao_credito": {"loc": 0.3, "scale": 0.15},
        "num_parcelas_atraso": {"loc": 1.0, "scale": 1.5},
    }

    # Parâmetros de distribuição para dados de produção (com drift)
    # Conforme descrito na seção 'Saiba Mais': deslocamento de média e dispersão
    PROD_PARAMS: Dict[str, Dict[str, float]] = {
        "idade": {"loc": 30.0, "scale": 8.0},
        "renda_mensal": {"loc": 3500.0, "scale": 1500.0},
        "score_credito": {"loc": 550.0, "scale": 120.0},
        "tempo_emprego": {"loc": 4.0, "scale": 3.0},
        "valor_emprestimo": {"loc": 12000.0, "scale": 7000.0},
        "taxa_utilizacao_credito": {"loc": 0.5, "scale": 0.2},
        "num_parcelas_atraso": {"loc": 2.5, "scale": 2.0},
    }

    # Distribuições categóricas: referência vs produção
    REF_CATEGORICALS: Dict[str, Dict[str, List]] = {
        "tipo_residencia": {
            "categories": ["aluguel", "proprio", "financiado"],
            "probs": [0.50, 0.30, 0.20],
        },
        "categoria_risco": {
            "categories": ["Baixo", "Medio", "Alto"],
            "probs": [0.50, 0.30, 0.20],
        },
    }

    PROD_CATEGORICALS: Dict[str, Dict[str, List]] = {
        "tipo_residencia": {
            "categories": ["aluguel", "proprio", "financiado"],
            "probs": [0.30, 0.25, 0.45],
        },
        "categoria_risco": {
            "categories": ["Baixo", "Medio", "Alto"],
            "probs": [0.25, 0.35, 0.40],
        },
    }

    def __init__(self, seed: int = 42, n_samples: int = 5000) -> None:
        """Inicializa o preprocessador.

        Args:
            seed: Semente para numpy.random (reprodutibilidade).
            n_samples: Número de amostras por grupo.
        """
        self.seed = seed
        self.n_samples = n_samples
        self.rng = np.random.RandomState(seed)

    def _generate_numerical(
        self, params: Dict[str, Dict[str, float]], n: int
    ) -> pd.DataFrame:
        """Gera features numéricas a partir de distribuições normais.

        Args:
            params: Dict feature_name -> {loc, scale}.
            n: Número de amostras.

        Returns:
            DataFrame com features numéricas geradas.
        """
        data: Dict[str, np.ndarray] = {}
        for feature, p in params.items():
            values = self.rng.normal(loc=p["loc"], scale=p["scale"], size=n)
            # Aplicar limites físicos para manter dados realistas
            if feature == "idade":
                values = np.clip(values, 18, 80)
            elif feature == "renda_mensal":
                values = np.clip(values, 500, 30000)
            elif feature == "score_credito":
                values = np.clip(values, 0, 1000)
            elif feature == "tempo_emprego":
                values = np.clip(values, 0, 40)
            elif feature == "valor_emprestimo":
                values = np.clip(values, 500, 100000)
            elif feature == "taxa_utilizacao_credito":
                values = np.clip(values, 0.0, 1.0)
            elif feature == "num_parcelas_atraso":
                values = np.clip(np.round(values).astype(int), 0, 20)
            data[feature] = values
        return pd.DataFrame(data)

    def _generate_categorical(
        self, cat_params: Dict[str, Dict[str, List]], n: int
    ) -> pd.DataFrame:
        """Gera features categóricas a partir de distribuições multinomiais.

        Args:
            cat_params: Dict feature_name -> {categories, probs}.
            n: Número de amostras.

        Returns:
            DataFrame com features categóricas geradas.
        """
        data: Dict[str, np.ndarray] = {}
        for feature, p in cat_params.items():
            data[feature] = self.rng.choice(
                p["categories"], size=n, p=p["probs"]
            )
        return pd.DataFrame(data)

    def _generate_target(
        self, df: pd.DataFrame, base_rate: float = 0.2
    ) -> np.ndarray:
        """Gera variável target (inadimplente) com lógica baseada em features.

        A probabilidade de inadimplência depende de score_credito e
        taxa_utilizacao_credito, simulando cenário realista de crédito.

        Args:
            df: DataFrame com features.
            base_rate: Taxa base de inadimplência.

        Returns:
            Array binário (0/1) com target.
        """
        # Probabilidade influenciada por score_credito e taxa_utilizacao
        prob = base_rate * np.ones(len(df))
        if "score_credito" in df.columns:
            # Menor score -> maior probabilidade de inadimplência
            score_norm = (df["score_credito"] - 300) / 700
            prob += (1 - score_norm) * 0.15
        if "taxa_utilizacao_credito" in df.columns:
            # Maior utilização -> maior probabilidade
            prob += df["taxa_utilizacao_credito"] * 0.1
        prob = np.clip(prob, 0.01, 0.95)
        return (self.rng.random(len(df)) < prob).astype(int)

    def generate_dataset(self) -> pd.DataFrame:
        """Gera dataset completo com dados de referência e produção.

        Implementa o cenário da fintech de crédito descrito no Documento da
        Aula 2, com drift injetado nas features de produção conforme a tabela
        de drift especificada em data/README.md.

        Returns:
            DataFrame com 2*n_samples linhas (referência + produção).
        """
        self.rng = np.random.RandomState(self.seed)

        # Gerar dados de referência (sem drift)
        ref_num = self._generate_numerical(self.REF_PARAMS, self.n_samples)
        ref_cat = self._generate_categorical(
            self.REF_CATEGORICALS, self.n_samples
        )
        ref_df = pd.concat([ref_num, ref_cat], axis=1)
        ref_df["origem"] = "referencia"
        ref_df["inadimplente"] = self._generate_target(ref_df, base_rate=0.15)

        # Gerar dados de produção (com drift)
        prod_num = self._generate_numerical(self.PROD_PARAMS, self.n_samples)
        prod_cat = self._generate_categorical(
            self.PROD_CATEGORICALS, self.n_samples
        )
        prod_df = pd.concat([prod_num, prod_cat], axis=1)
        prod_df["origem"] = "producao"
        prod_df["inadimplente"] = self._generate_target(
            prod_df, base_rate=0.25
        )

        # Combinar e retornar
        dataset = pd.concat([ref_df, prod_df], ignore_index=True)
        return dataset

    def generate_and_save(
        self, output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Gera dataset e salva em CSV.

        Args:
            output_path: Caminho para salvar o CSV. Se None, usa padrão.

        Returns:
            DataFrame gerado.
        """
        if output_path is None:
            output_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "data", "raw", "dataset.csv",
            )
        dataset = self.generate_dataset()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dataset.to_csv(output_path, index=False)
        print(f"Dataset salvo em: {output_path} ({len(dataset)} linhas)")
        return dataset

    def load_data(
        self, filepath: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Carrega dataset e separa em referência e produção.

        Conforme o Documento da Aula 2, os dados são separados pela coluna
        'origem' para permitir comparação das distribuições antes e depois
        do drift.

        Args:
            filepath: Caminho do CSV. Se None, usa padrão.

        Returns:
            Tupla (ref_data, prod_data) com DataFrames separados.

        Raises:
            FileNotFoundError: Se o arquivo não existir.
        """
        if filepath is None:
            filepath = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "data", "raw", "dataset.csv",
            )
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Dataset não encontrado em {filepath}. "
                "Execute generate_and_save() primeiro."
            )
        df = pd.read_csv(filepath)
        ref_data = df[df["origem"] == "referencia"].copy()
        prod_data = df[df["origem"] == "producao"].copy()
        return ref_data, prod_data

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpa dados removendo nulos e valores inconsistentes.

        Args:
            df: DataFrame a ser limpo.

        Returns:
            DataFrame limpo.
        """
        df_clean = df.dropna().copy()
        # Remover idades fora do range válido
        if "idade" in df_clean.columns:
            df_clean = df_clean[
                (df_clean["idade"] >= 18) & (df_clean["idade"] <= 80)
            ]
        # Garantir que taxa de utilização está entre 0 e 1
        if "taxa_utilizacao_credito" in df_clean.columns:
            df_clean = df_clean[
                (df_clean["taxa_utilizacao_credito"] >= 0)
                & (df_clean["taxa_utilizacao_credito"] <= 1)
            ]
        return df_clean.reset_index(drop=True)

    def prepare_features(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Separa features numéricas e categóricas para análise de drift.

        Args:
            df: DataFrame com todas as features.

        Returns:
            Tupla (df, numerical_cols, categorical_cols).
        """
        exclude_cols = {"origem", "inadimplente"}
        numerical_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude_cols
        ]
        categorical_cols = [
            c for c in df.select_dtypes(include=["object", "category"]).columns
            if c not in exclude_cols
        ]
        return df, numerical_cols, categorical_cols

    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Divide dados em treino e teste (para avaliação de modelos).

        Args:
            df: DataFrame a dividir.
            test_size: Proporção para teste.
            random_state: Semente para reprodutibilidade.

        Returns:
            Tupla (train_df, test_df).
        """
        from sklearn.model_selection import train_test_split

        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
