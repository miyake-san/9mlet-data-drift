"""
Módulo de pré-processamento de dados para detecção de drift em embeddings.

Implementa carregamento, limpeza e preparação de features de embeddings
conforme discutido na seção 'Saiba Mais' do documento da Aula 5, onde
embeddings são descritos como vetores densos de alta dimensionalidade
que capturam o significado semântico dos dados (Mikolov et al., 2013;
Devlin et al., 2019).

Referências:
    Mikolov, T. et al. (2013). Efficient estimation of word representations
    in vector space. ICLR.
    Devlin, J. et al. (2019). BERT: Pre-training of deep bidirectional
    transformers for language understanding. NAACL-HLT.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """
    Pré-processador de dados de embeddings para detecção de drift.

    Implementa o pipeline de preparação de dados conforme discutido na Aula 5,
    incluindo carregamento de embeddings pré-computados, normalização e separação
    em conjuntos de referência (P_ref) e produção (P_novo).

    A separação em conjuntos é fundamental para os testes de duas amostras
    (Gretton et al., 2012; Massey, 1951) que verificam se P_ref(X) = P_novo(X).

    Attributes:
        data: DataFrame com os dados carregados.
        scaler: StandardScaler para normalização de embeddings.
        embedding_dim: Dimensionalidade dos embeddings.
        embedding_cols: Lista de nomes das colunas de embedding.
        is_fitted: Indica se o scaler foi ajustado aos dados de referência.
    """

    def __init__(self, embedding_dim: int = 64) -> None:
        """
        Inicializa o DataPreprocessor.

        Args:
            embedding_dim: Dimensionalidade esperada dos embeddings (default: 64).
        """
        self.data: Optional[pd.DataFrame] = None
        self.scaler = StandardScaler()
        self.embedding_dim = embedding_dim
        self.embedding_cols: List[str] = [f"emb_{i}" for i in range(embedding_dim)]
        self.is_fitted: bool = False

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Carrega dados de embeddings a partir de um arquivo CSV.

        Conforme discutido na Aula 5, os embeddings representam textos ou
        imagens em espaços vetoriais contínuos, onde proximidade geométrica
        reflete semelhanças semânticas (Mikolov et al., 2013).

        Args:
            filepath: Caminho para o arquivo CSV.

        Returns:
            DataFrame com os dados carregados.

        Raises:
            FileNotFoundError: Se o arquivo não existir.
            ValueError: Se colunas de embedding não forem encontradas.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

        self.data = pd.read_csv(filepath)

        # Detecta colunas de embedding automaticamente
        emb_cols_found = sorted(
            [c for c in self.data.columns if c.startswith("emb_")],
            key=lambda x: int(x.split("_")[1]),
        )
        if emb_cols_found:
            self.embedding_cols = emb_cols_found
            self.embedding_dim = len(emb_cols_found)
        else:
            raise ValueError(
                "Colunas de embedding não encontradas no arquivo. "
                "Esperado formato: emb_0, emb_1, ..., emb_N"
            )

        return self.data

    def clean_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Limpa e valida dados de embeddings.

        Remove valores ausentes, infinitos e amostras com embeddings
        zerados (que indicam falhas na extração de features conforme
        modelos como BERT - Devlin et al., 2019).

        Args:
            df: DataFrame a limpar (se None, usa self.data).

        Returns:
            DataFrame limpo.

        Raises:
            ValueError: Se nenhum dado estiver carregado.
        """
        if df is None:
            df = self.data
        if df is None:
            raise ValueError("Nenhum dado carregado. Chame load_data() primeiro.")

        df_clean = df.copy()
        initial_len = len(df_clean)

        # Remove linhas com NaN nas colunas de embedding
        df_clean = df_clean.dropna(subset=self.embedding_cols)

        # Remove embeddings com valores infinitos
        emb_data = df_clean[self.embedding_cols]
        mask_finite = np.isfinite(emb_data).all(axis=1)
        df_clean = df_clean[mask_finite]

        # Remove embeddings completamente zerados (falha na extração)
        emb_norms = np.linalg.norm(
            df_clean[self.embedding_cols].values, axis=1
        )
        mask_nonzero = emb_norms > 1e-10
        df_clean = df_clean[mask_nonzero]

        removed = initial_len - len(df_clean)
        if removed > 0:
            print(f"Removidas {removed} amostras inválidas ({removed / initial_len * 100:.1f}%)")

        self.data = df_clean.reset_index(drop=True)
        return self.data

    def prepare_features(
        self,
        df: Optional[pd.DataFrame] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Prepara features de embeddings para análise de drift.

        A normalização dos embeddings é importante para garantir que as
        métricas de distância (como MMD com kernel RBF) reflitam diferenças
        semânticas reais e não apenas diferenças de escala, conforme discutido
        no documento da Aula 5 sobre o uso de kernels Gaussianos
        (Gretton et al., 2012).

        Args:
            df: DataFrame com embeddings (se None, usa self.data).
            normalize: Se True, aplica StandardScaler.

        Returns:
            Array numpy (n_samples, embedding_dim) com embeddings processados.

        Raises:
            ValueError: Se nenhum dado estiver carregado.
        """
        if df is None:
            df = self.data
        if df is None:
            raise ValueError("Nenhum dado carregado. Chame load_data() primeiro.")

        embeddings = df[self.embedding_cols].values.astype(np.float64)

        if normalize:
            if not self.is_fitted:
                embeddings = self.scaler.fit_transform(embeddings)
                self.is_fitted = True
            else:
                embeddings = self.scaler.transform(embeddings)

        return embeddings

    def split_data(
        self,
        df: Optional[pd.DataFrame] = None,
        period_column: str = "period",
        reference_label: str = "reference",
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """
        Separa dados em conjuntos de referência e produção.

        Implementa a separação descrita na Aula 5, onde os dados de referência
        correspondem a P_ref(X) e os dados de produção a P_novo(X). A detecção
        de drift consiste em verificar se P_ref(X) ≠ P_novo(X).

        Conforme Gama et al. (2014):
            P_ref(X, y) ≠ P_novo(X, y)

        Args:
            df: DataFrame com embeddings e coluna de período.
            period_column: Nome da coluna que indica o período.
            reference_label: Valor que identifica dados de referência.

        Returns:
            Tupla (emb_ref, emb_prod, df_ref, df_prod).

        Raises:
            ValueError: Se coluna de período não existir ou dados estiverem vazios.
        """
        if df is None:
            df = self.data
        if df is None:
            raise ValueError("Nenhum dado carregado. Chame load_data() primeiro.")

        if period_column not in df.columns:
            raise ValueError(f"Coluna '{period_column}' não encontrada no DataFrame")

        df_ref = df[df[period_column] == reference_label].copy()
        df_prod = df[df[period_column] != reference_label].copy()

        if len(df_ref) == 0:
            raise ValueError(f"Nenhum dado de referência com label '{reference_label}'")
        if len(df_prod) == 0:
            raise ValueError("Nenhum dado de produção encontrado")

        emb_ref = df_ref[self.embedding_cols].values.astype(np.float64)
        emb_prod = df_prod[self.embedding_cols].values.astype(np.float64)

        return emb_ref, emb_prod, df_ref, df_prod
