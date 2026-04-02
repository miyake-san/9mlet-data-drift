"""
Módulo do modelo de detecção de drift em embeddings.

Implementa a classe EmbeddingDriftDetector com três estratégias de detecção:
1. MMD (Maximum Mean Discrepancy) - Gretton et al., 2012
2. Teste de Kolmogorov-Smirnov (KS) - Massey, 1951
3. Classificador Adversário - Lopez-Paz & Oquab, 2017

Conforme discutido na seção 'Saiba Mais' do documento da Aula 5, estes métodos
comparam estatisticamente dois conjuntos de dados (referência e produção) para
verificar se provêm da mesma distribuição, com a hipótese nula H0 de que
P_ref(X) = P_novo(X).

Referências:
    Gretton, A. et al. (2012). A kernel two-sample test. JMLR, 13, 723-773.
    Massey Jr., F.J. (1951). The Kolmogorov-Smirnov Test for Goodness of Fit.
    JASA, 46(253), 68-78.
    Lopez-Paz, D. & Oquab, M. (2017). Revisiting classifier two-sample tests. ICLR.
    Feldhans, R. et al. (2021). Drift Detection in Text Data with Document Embeddings.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import cross_val_score


class EmbeddingDriftDetector:
    """
    Detector de drift em embeddings usando múltiplos métodos estatísticos.

    Implementa testes de duas amostras para comparar distribuições de embeddings
    de referência (P_ref) e produção (P_novo), conforme descrito no documento
    da Aula 5. Suporta três métodos:

    - **MMD**: Discrepância de Máxima Média com kernel RBF (Gretton et al., 2012).
      Mapeia amostras para um espaço de características via kernel e compara médias.
    - **KS**: Teste de Kolmogorov-Smirnov por dimensão com correção de Bonferroni
      (Massey, 1951). Compara CDFs empíricas em cada dimensão do embedding.
    - **Adversarial**: Classificador adversário que tenta distinguir dados de
      referência e produção (Lopez-Paz & Oquab, 2017). Accuracy > 0.5 sugere drift.

    Attributes:
        method: Método de detecção ('mmd', 'ks', 'adversarial').
        gamma: Parâmetro do kernel RBF para MMD (k(u,v) = exp(-γ||u-v||²)).
        threshold: Limiar para decisão de drift.
        n_permutations: Número de permutações para teste de significância do MMD.
        reference_embeddings: Embeddings de referência ajustados via fit().
        is_fitted: Indica se o detector foi ajustado com dados de referência.
    """

    VALID_METHODS = ("mmd", "ks", "adversarial")

    def __init__(
        self,
        method: str = "mmd",
        gamma: float = 1.0,
        threshold: Optional[float] = None,
        n_permutations: int = 100,
    ) -> None:
        """
        Inicializa o EmbeddingDriftDetector.

        Args:
            method: Método de detecção. Opções: 'mmd', 'ks', 'adversarial'.
            gamma: Parâmetro do kernel RBF para MMD (default: 1.0).
            threshold: Limiar para decidir drift. Se None, calculado automaticamente.
            n_permutations: Número de permutações para p-value do MMD (default: 100).

        Raises:
            ValueError: Se método inválido.
        """
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Método '{method}' inválido. Opções: {self.VALID_METHODS}"
            )

        self.method = method
        self.gamma = gamma
        self.threshold = threshold
        self.n_permutations = n_permutations
        self.reference_embeddings: Optional[np.ndarray] = None
        self.is_fitted: bool = False
        self._classifier: Optional[Any] = None

    def fit(self, X_reference: np.ndarray) -> "EmbeddingDriftDetector":
        """
        Ajusta o detector com embeddings de referência.

        Define a distribuição de referência P_ref(X) para comparação futura com
        novos dados. Para o método MMD, calcula o limiar via bootstrapping se
        não fornecido explicitamente.

        Conforme o documento da Aula 5, os dados de referência representam
        uma amostra {x_i}_{i=1}^n de P_ref(X).

        Args:
            X_reference: Array (n_samples, embedding_dim) com embeddings de referência.

        Returns:
            Self (para encadeamento de métodos).
        """
        self.reference_embeddings = np.array(X_reference, dtype=np.float64)

        if self.threshold is None and self.method == "mmd":
            self.threshold = self._compute_threshold_bootstrap()
        elif self.threshold is None:
            self.threshold = 0.05 if self.method == "ks" else 0.6

        self.is_fitted = True
        return self

    def predict(self, X_new: np.ndarray) -> Dict[str, Any]:
        """
        Detecta drift comparando novos dados com a referência.

        Implementa os testes de duas amostras discutidos na seção 'Saiba Mais':
        H0: P_ref(X) = P_novo(X) (sem drift)
        H1: P_ref(X) ≠ P_novo(X) (drift detectado)

        Args:
            X_new: Array (n_samples, embedding_dim) com novos embeddings.

        Returns:
            Dicionário com chaves:
                - score: Valor numérico do drift (MMD² ou D-statistic).
                - p_value: P-value do teste (quando disponível).
                - drift_detected: Boolean indicando se drift foi detectado.
                - method: Método utilizado.

        Raises:
            RuntimeError: Se o detector não foi ajustado (fit não chamado).
        """
        if not self.is_fitted:
            raise RuntimeError("Detector não ajustado. Chame fit() primeiro.")

        X_new = np.array(X_new, dtype=np.float64)

        if self.method == "mmd":
            return self._predict_mmd(X_new)
        elif self.method == "ks":
            return self._predict_ks(X_new)
        else:
            return self._predict_adversarial(X_new)

    def score(self, X_new: np.ndarray) -> float:
        """
        Retorna o score de drift (quanto maior, mais drift).

        Args:
            X_new: Array (n_samples, embedding_dim) com novos embeddings.

        Returns:
            Score numérico de drift.
        """
        result = self.predict(X_new)
        return result["score"]

    # ------------------------------------------------------------------
    # Métodos de cálculo (públicos para uso direto nos notebooks)
    # ------------------------------------------------------------------

    def compute_mmd(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        gamma: Optional[float] = None,
    ) -> float:
        """
        Calcula MMD² (Maximum Mean Discrepancy quadrático) entre dois conjuntos.

        Implementa a fórmula de Gretton et al. (2012) conforme apresentada na
        seção 'Saiba Mais' do documento da Aula 5:

            MMD²(P, Q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]

        onde k é o kernel RBF: k(u,v) = exp(-γ||u-v||²).

        A MMD mede o distanciamento entre distribuições comparando a proximidade
        média de pontos dentro de cada conjunto versus entre conjuntos. Se as
        distribuições forem idênticas, MMD ≈ 0.

        Corresponde ao Snippet 2 da seção Hands On do documento da Aula 5.

        Args:
            X: Embeddings do conjunto 1 (n, d).
            Y: Embeddings do conjunto 2 (m, d).
            gamma: Parâmetro do kernel RBF (se None, usa self.gamma).

        Returns:
            Valor de MMD² (float). Maior → mais discrepância.

        Referência:
            Gretton, A., Borgwardt, K.M., Rasch, M.J., Schölkopf, B. & Smola, A.
            (2012). A kernel two-sample test. JMLR, 13(Mar), 723-773.
        """
        if gamma is None:
            gamma = self.gamma

        # Matrizes de kernel: informação intra-conjunto e inter-conjunto
        KXX = rbf_kernel(X, X, gamma=gamma)
        KYY = rbf_kernel(Y, Y, gamma=gamma)
        KXY = rbf_kernel(X, Y, gamma=gamma)

        # MMD² = média(K_XX) + média(K_YY) - 2*média(K_XY)
        mmd_squared = float(KXX.mean() + KYY.mean() - 2.0 * KXY.mean())
        return mmd_squared

    def compute_ks_test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Teste de Kolmogorov-Smirnov multivariado (por dimensão com Bonferroni).

        Conforme discutido na seção 'Saiba Mais' da Aula 5, o teste KS compara
        as funções de distribuição acumulada empíricas das duas amostras:
            D_{n,m} = sup_x |F_ref(x) - F_novo(x)|

        Para dados multivariados (embeddings), aplica o teste KS a cada dimensão
        separadamente e ajusta a significância com correção de Bonferroni para
        múltiplos testes. Conforme observado no documento, esta simplificação
        pode perder sensibilidade a mudanças sutis na estrutura multivariada.

        Corresponde ao Snippet 3 da seção Hands On do documento da Aula 5.

        Args:
            X: Embeddings do conjunto 1 (n, d).
            Y: Embeddings do conjunto 2 (m, d).

        Returns:
            Dicionário com d_statistics, p_values, adjusted_p_value e max_d.

        Referência:
            Massey Jr., F.J. (1951). The Kolmogorov-Smirnov Test for Goodness
            of Fit. JASA, 46(253), 68-78.
        """
        n_dims = X.shape[1]
        p_values: List[float] = []
        d_stats: List[float] = []

        for d in range(n_dims):
            d_stat, p_val = ks_2samp(X[:, d], Y[:, d])
            p_values.append(float(p_val))
            d_stats.append(float(d_stat))

        # Correção de Bonferroni para múltiplos testes
        adjusted_p = min(min(p_values) * n_dims, 1.0)

        return {
            "d_statistics": d_stats,
            "p_values": p_values,
            "adjusted_p_value": adjusted_p,
            "max_d": max(d_stats),
        }

    def compute_adversarial_score(
        self,
        X_ref: np.ndarray,
        X_new: np.ndarray,
        n_estimators: int = 50,
        max_depth: int = 3,
        cv: int = 3,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Classificador adversário para detecção de drift.

        Implementa a abordagem de Lopez-Paz & Oquab (2017) conforme discutida
        na seção 'Saiba Mais' da Aula 5: treina um classificador para distinguir
        dados de referência de dados de produção. Se o classificador atinge
        accuracy > 0.5 (aleatório), há evidência de drift.

        Args:
            X_ref: Embeddings de referência (n, d).
            X_new: Embeddings novos (m, d).
            n_estimators: Número de estimadores do GBM.
            max_depth: Profundidade máxima das árvores.
            cv: Número de folds para cross-validation.
            seed: Seed para reproduzibilidade.

        Returns:
            Dicionário com accuracy, score e drift_detected.

        Referência:
            Lopez-Paz, D. & Oquab, M. (2017). Revisiting classifier
            two-sample tests. ICLR.
        """
        X_combined = np.vstack([X_ref, X_new])
        y_combined = np.array([0] * len(X_ref) + [1] * len(X_new))

        clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
        )
        scores = cross_val_score(
            clf, X_combined, y_combined, cv=cv, scoring="accuracy"
        )
        accuracy = float(scores.mean())

        return {
            "accuracy": accuracy,
            "accuracy_std": float(scores.std()),
            "score": max(0.0, (accuracy - 0.5) * 2.0),  # Normalizado [0, 1]
            "drift_detected": accuracy > 0.6,
        }

    # ------------------------------------------------------------------
    # Métodos internos de predição
    # ------------------------------------------------------------------

    def _predict_mmd(self, X_new: np.ndarray) -> Dict[str, Any]:
        """Predição com MMD (Gretton et al., 2012)."""
        mmd_val = self.compute_mmd(self.reference_embeddings, X_new)
        p_value = self._permutation_test_mmd(mmd_val, X_new)

        return {
            "score": mmd_val,
            "p_value": p_value,
            "drift_detected": mmd_val > self.threshold,
            "threshold": self.threshold,
            "method": "mmd",
        }

    def _predict_ks(self, X_new: np.ndarray) -> Dict[str, Any]:
        """Predição com teste KS (Massey, 1951)."""
        ks_result = self.compute_ks_test(self.reference_embeddings, X_new)

        return {
            "score": ks_result["max_d"],
            "p_value": ks_result["adjusted_p_value"],
            "drift_detected": ks_result["adjusted_p_value"] < self.threshold,
            "threshold": self.threshold,
            "method": "ks",
            "details": ks_result,
        }

    def _predict_adversarial(self, X_new: np.ndarray) -> Dict[str, Any]:
        """Predição com classificador adversário (Lopez-Paz & Oquab, 2017)."""
        adv_result = self.compute_adversarial_score(
            self.reference_embeddings, X_new
        )

        return {
            "score": adv_result["score"],
            "accuracy": adv_result["accuracy"],
            "drift_detected": adv_result["accuracy"] > self.threshold,
            "threshold": self.threshold,
            "method": "adversarial",
        }

    def _compute_threshold_bootstrap(self) -> float:
        """
        Calcula limiar de MMD via bootstrapping dos dados de referência.

        Divide aleatoriamente os dados de referência em duas metades e calcula
        o MMD entre elas, repetindo n_permutations vezes. O percentil 95
        dos valores resultantes define o limiar sob H0 (sem drift).
        """
        n = len(self.reference_embeddings)
        half = n // 2
        scores = []

        for _ in range(self.n_permutations):
            idx = np.random.permutation(n)
            X1 = self.reference_embeddings[idx[:half]]
            X2 = self.reference_embeddings[idx[half : 2 * half]]
            mmd = self.compute_mmd(X1, X2)
            scores.append(mmd)

        return float(np.percentile(scores, 95))

    def _permutation_test_mmd(
        self,
        observed_mmd: float,
        X_new: np.ndarray,
    ) -> float:
        """
        Teste de permutação para significância estatística do MMD.

        Verifica quantas permutações aleatórias produzem MMD >= ao observado.
        O p-value é (count + 1) / (n_permutations + 1).
        """
        combined = np.vstack([self.reference_embeddings, X_new])
        n_ref = len(self.reference_embeddings)
        count = 0

        for _ in range(self.n_permutations):
            idx = np.random.permutation(len(combined))
            perm_X = combined[idx[:n_ref]]
            perm_Y = combined[idx[n_ref:]]
            perm_mmd = self.compute_mmd(perm_X, perm_Y)
            if perm_mmd >= observed_mmd:
                count += 1

        return (count + 1) / (self.n_permutations + 1)
