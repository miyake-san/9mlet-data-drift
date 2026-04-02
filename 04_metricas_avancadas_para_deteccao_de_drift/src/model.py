# -*- coding: utf-8 -*-
"""
model.py – Detectores de drift: PSI, MMD, Wasserstein e Energy Distance.

Implementa as métricas avançadas de detecção de drift discutidas no
Documento 04 (Seção "Saiba Mais"):
  - PSI (Population Stability Index): métrica univariada clássica.
  - MMD (Maximum Mean Discrepancy): métrica multivariada com kernel RBF.
  - Wasserstein Distance: distância de transporte ótimo.
  - Energy Distance: métrica baseada em distâncias inter-amostras.

Referências:
    Gretton, A. et al. (2012). A Kernel Two-Sample Test. JMLR, 13, 723–773.
    Arjovsky, M.; Chintala, S.; Bottou, L. (2017). Wasserstein GAN. ICML.
    Székely, G. J.; Rizzo, M. L. (2013). Energy statistics. JSPI, 143(8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


# ======================================================================
# Data classes para resultados
# ======================================================================
@dataclass
class DriftResult:
    """Resultado de um teste de drift.

    Attributes:
        metric_name: Nome da métrica utilizada.
        statistic: Valor da estatística de teste.
        threshold: Limiar de detecção (se disponível).
        drift_detected: Se drift foi detectado.
        details: Informações adicionais.
    """

    metric_name: str
    statistic: float
    threshold: Optional[float] = None
    drift_detected: bool = False
    details: Dict[str, object] = field(default_factory=dict)


# ======================================================================
# Detectores individuais
# ======================================================================
class PSICalculator:
    """Calcula o Population Stability Index (PSI).

    Conforme Documento 04, Seção "Saiba Mais" e Snippet 1 (Hands On):
    o PSI compara a distribuição de frequências de uma variável entre
    dois períodos usando a fórmula:

        PSI = Σ (p_i - q_i) * ln(p_i / q_i)

    onde p_i e q_i são as proporções em cada bin nos períodos de
    referência e atual, respectivamente.

    **Limitação**: o PSI analisa cada variável isoladamente e não
    detecta drifts multivariados sutis, como inversão de correlação
    (Documento 04, Seção "Limites de Métricas Simples").

    Args:
        n_bins: Número de bins para discretização. Default: 10.
        eps: Fator de suavização para evitar log(0). Default: 1e-8.
    """

    def __init__(self, n_bins: int = 10, eps: float = 1e-8) -> None:
        self.n_bins = n_bins
        self.eps = eps

    def calculate(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: Optional[np.ndarray] = None,
    ) -> float:
        """Calcula PSI entre duas distribuições univariadas.

        Implementa o cálculo do Snippet 1 do Hands On (Documento 04).

        Args:
            reference: Dados do período de referência (1D).
            current: Dados do período atual (1D).
            bins: Limites de bins customizados. Se None, auto-calculados.

        Returns:
            Valor do PSI.
        """
        reference = np.asarray(reference).ravel()
        current = np.asarray(current).ravel()

        if bins is None:
            # Cria bins cobrindo ambos os conjuntos (Snippet 1, Hands On)
            all_data = np.concatenate([reference, current])
            bins = np.linspace(all_data.min(), all_data.max(), self.n_bins + 1)

        freq_ref, _ = np.histogram(reference, bins=bins)
        freq_cur, _ = np.histogram(current, bins=bins)

        # Proporções com suavização (evita divisão por zero)
        pct_ref = freq_ref / len(reference) + self.eps
        pct_cur = freq_cur / len(current) + self.eps

        # PSI = Σ (p_i - q_i) * ln(p_i / q_i) — Documento 04, Saiba Mais
        psi = float(np.sum((pct_cur - pct_ref) * np.log(pct_cur / pct_ref)))
        return psi

    def calculate_multifeature(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> List[float]:
        """Calcula PSI para cada feature individualmente.

        Conforme Documento 04, Seção "Limites de Métricas Simples":
        o PSI aplicado feature a feature pode falhar em detectar
        drifts multivariados quando as marginais permanecem similares.

        Args:
            reference: Dados de referência (n_samples, n_features).
            current: Dados atuais (n_samples, n_features).

        Returns:
            Lista de valores PSI, um por feature.
        """
        reference = np.asarray(reference)
        current = np.asarray(current)

        if reference.ndim == 1:
            return [self.calculate(reference, current)]

        n_features = reference.shape[1]
        return [
            self.calculate(reference[:, j], current[:, j])
            for j in range(n_features)
        ]


class MMDCalculator:
    """Calcula a Maximum Mean Discrepancy (MMD) com kernel RBF.

    Conforme Documento 04, Seção "Saiba Mais" — MMD e Teste de Duas
    Amostras, e Snippet 2 do Hands On:

        MMD²(P,Q) = E[k(x,x')] + E[k(y,y')] - 2·E[k(x,y)]

    onde k é um kernel gaussiano (RBF) k(x,x') = exp(-γ‖x-x'‖²).
    A MMD é zero se e somente se P = Q (Gretton et al., 2012).

    Referência:
        Gretton, A. et al. (2012). A Kernel Two-Sample Test.
        Journal of Machine Learning Research, 13, 723–773.

    Args:
        gamma: Parâmetro do kernel RBF. Se None, usa mediana heuristic.
    """

    def __init__(self, gamma: Optional[float] = None) -> None:
        self.gamma = gamma

    def _compute_gamma(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Calcula γ via mediana heuristic (median trick).

        Conforme recomendação de Gretton et al. (2012), o parâmetro
        γ do kernel RBF pode ser definido como 1/(2·median²) das
        distâncias inter-pontos.

        Args:
            X: Primeira amostra.
            Y: Segunda amostra.

        Returns:
            Valor de γ estimado.
        """
        combined = np.vstack([X, Y])
        # Amostragem para eficiência em datasets grandes
        if len(combined) > 1000:
            idx = np.random.choice(len(combined), 1000, replace=False)
            combined = combined[idx]
        dists = cdist(combined, combined, metric="sqeuclidean")
        median_dist = np.median(dists[dists > 0])
        return 1.0 / (2.0 * median_dist) if median_dist > 0 else 1.0

    def calculate(
        self, X: np.ndarray, Y: np.ndarray
    ) -> float:
        """Calcula MMD² entre duas amostras multivariadas.

        Implementa a fórmula do Documento 04, Seção "Saiba Mais"
        e Snippet 2 do Hands On.

        Args:
            X: Amostra de referência (n, d).
            Y: Amostra atual (m, d).

        Returns:
            Valor de MMD² (≥ 0).
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        gamma = self.gamma if self.gamma is not None else self._compute_gamma(X, Y)

        # Kernel RBF: k(x,y) = exp(-γ ‖x-y‖²)
        # Conforme Documento 04, Snippet 2 — Hands On
        K_XX = np.exp(-gamma * cdist(X, X, metric="sqeuclidean"))
        K_YY = np.exp(-gamma * cdist(Y, Y, metric="sqeuclidean"))
        K_XY = np.exp(-gamma * cdist(X, Y, metric="sqeuclidean"))

        # MMD² = mean(K_XX) + mean(K_YY) - 2·mean(K_XY)
        mmd_sq = float(K_XX.mean() + K_YY.mean() - 2 * K_XY.mean())
        return max(mmd_sq, 0.0)  # Garante não-negatividade numérica

    def permutation_test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_permutations: int = 100,
        seed: int = 42,
    ) -> DriftResult:
        """Teste de permutação para significância da MMD.

        Conforme Documento 04, Seção "Saiba Mais": para uso prático,
        define-se um limiar de significância estatística (p-valor)
        via permutações (Gretton et al., 2012).

        Args:
            X: Amostra de referência.
            Y: Amostra atual.
            n_permutations: Número de permutações.
            seed: Semente para reprodutibilidade.

        Returns:
            DriftResult com estatística, p-valor e decisão.
        """
        rng = np.random.RandomState(seed)
        observed_mmd = self.calculate(X, Y)

        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        combined = np.vstack([X, Y])
        n = len(X)
        null_mmds = []

        for _ in range(n_permutations):
            perm = rng.permutation(len(combined))
            X_perm = combined[perm[:n]]
            Y_perm = combined[perm[n:]]
            null_mmds.append(self.calculate(X_perm, Y_perm))

        p_value = float(np.mean(np.array(null_mmds) >= observed_mmd))
        return DriftResult(
            metric_name="MMD",
            statistic=observed_mmd,
            threshold=float(np.percentile(null_mmds, 95)),
            drift_detected=p_value < 0.05,
            details={"p_value": p_value, "n_permutations": n_permutations},
        )


class WassersteinCalculator:
    """Calcula a distância de Wasserstein entre distribuições.

    Conforme Documento 04, Seção "Saiba Mais" — Métricas de Divergência:
    a distância de Wasserstein (Earth Mover's Distance) mede o "custo"
    para transformar uma distribuição em outra. É mais estável e sensível
    a mudanças sutis do que divergências baseadas apenas em densidades.

    Referência:
        Arjovsky, M.; Chintala, S.; Bottou, L. (2017). Wasserstein GAN.
        ICML, PMLR v.70, 214–223.
    """

    def calculate(
        self, reference: np.ndarray, current: np.ndarray
    ) -> float:
        """Calcula distância de Wasserstein univariada (W1).

        Args:
            reference: Dados de referência (1D).
            current: Dados atuais (1D).

        Returns:
            Distância de Wasserstein.
        """
        return float(
            wasserstein_distance(
                np.asarray(reference).ravel(),
                np.asarray(current).ravel(),
            )
        )

    def calculate_multifeature(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> List[float]:
        """Calcula Wasserstein por feature (aplicação univariada por dimensão).

        Args:
            reference: Dados de referência (n, d).
            current: Dados atuais (m, d).

        Returns:
            Lista de distâncias Wasserstein, uma por feature.
        """
        reference = np.asarray(reference)
        current = np.asarray(current)

        if reference.ndim == 1:
            return [self.calculate(reference, current)]

        n_features = reference.shape[1]
        return [
            self.calculate(reference[:, j], current[:, j])
            for j in range(n_features)
        ]


class EnergyDistanceCalculator:
    """Calcula a Energy Distance entre duas amostras multivariadas.

    Conforme Documento 04, Seção "Saiba Mais": a Energy Distance
    (Székely e Rizzo, 2013) é uma métrica multivariada que mede
    a distância entre distribuições baseada em distâncias Euclidianas
    inter-amostras.

    Referência:
        Székely, G. J.; Rizzo, M. L. (2013). Energy statistics: A class
        of statistics based on distances. JSPI, 143(8), 1249–1272.
    """

    def calculate(
        self, X: np.ndarray, Y: np.ndarray
    ) -> float:
        """Calcula Energy Distance entre duas amostras.

        A fórmula é:
            E(X,Y) = 2·E‖X-Y‖ - E‖X-X'‖ - E‖Y-Y'‖

        Args:
            X: Amostra de referência (n, d).
            Y: Amostra atual (m, d).

        Returns:
            Energy Distance (≥ 0).
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # Distâncias Euclidianas médias
        d_XY = cdist(X, Y, metric="euclidean").mean()
        d_XX = cdist(X, X, metric="euclidean").mean()
        d_YY = cdist(Y, Y, metric="euclidean").mean()

        energy = float(2 * d_XY - d_XX - d_YY)
        return max(energy, 0.0)


# ======================================================================
# Detector unificado
# ======================================================================
class DriftDetector:
    """Detector unificado de drift com múltiplas métricas.

    Implementa as métricas avançadas discutidas no Documento 04:
    PSI (univariada), MMD, Wasserstein e Energy Distance (multivariadas).
    Permite comparação direta entre métricas, conforme demonstrado
    na seção "Hands On" e nos Vídeos 3 e 4 da aula.

    Args:
        psi_threshold: Limiar para PSI. Default: 0.25.
        mmd_gamma: Parâmetro γ do kernel RBF. None = auto.
    """

    def __init__(
        self,
        psi_threshold: float = 0.25,
        mmd_gamma: Optional[float] = None,
    ) -> None:
        self.psi_threshold = psi_threshold
        self.psi_calc = PSICalculator()
        self.mmd_calc = MMDCalculator(gamma=mmd_gamma)
        self.wasserstein_calc = WassersteinCalculator()
        self.energy_calc = EnergyDistanceCalculator()

    def fit(self, reference: np.ndarray) -> "DriftDetector":
        """Armazena dados de referência.

        Args:
            reference: Dados do período de referência.

        Returns:
            self (para encadeamento).
        """
        self.reference_ = np.asarray(reference, dtype=np.float64)
        if self.reference_.ndim == 1:
            self.reference_ = self.reference_.reshape(-1, 1)
        return self

    def predict(self, current: np.ndarray) -> Dict[str, DriftResult]:
        """Executa detecção de drift com todas as métricas.

        Retorna resultados de PSI (por feature), MMD, Wasserstein
        e Energy Distance, permitindo comparação conforme discutido
        nos Vídeos 3 e 4 da aula (Documento 04).

        Args:
            current: Dados do período atual.

        Returns:
            Dicionário com DriftResult para cada métrica.
        """
        current = np.asarray(current, dtype=np.float64)
        if current.ndim == 1:
            current = current.reshape(-1, 1)

        results: Dict[str, DriftResult] = {}

        # PSI por feature — Documento 04, Snippet 1, Hands On
        psi_values = self.psi_calc.calculate_multifeature(
            self.reference_, current
        )
        max_psi = max(psi_values) if psi_values else 0.0
        results["psi"] = DriftResult(
            metric_name="PSI",
            statistic=max_psi,
            threshold=self.psi_threshold,
            drift_detected=max_psi > self.psi_threshold,
            details={"per_feature": psi_values},
        )

        # MMD — Documento 04, Snippet 2, Hands On
        mmd_val = self.mmd_calc.calculate(self.reference_, current)
        results["mmd"] = DriftResult(
            metric_name="MMD",
            statistic=mmd_val,
            drift_detected=False,  # Needs permutation test for p-value
            details={"note": "Use permutation_test() para p-valor"},
        )

        # Wasserstein — Documento 04, Saiba Mais
        wass_values = self.wasserstein_calc.calculate_multifeature(
            self.reference_, current
        )
        results["wasserstein"] = DriftResult(
            metric_name="Wasserstein",
            statistic=float(np.mean(wass_values)),
            details={"per_feature": wass_values},
        )

        # Energy Distance — Documento 04, Saiba Mais (Székely e Rizzo, 2013)
        energy_val = self.energy_calc.calculate(self.reference_, current)
        results["energy"] = DriftResult(
            metric_name="Energy Distance",
            statistic=energy_val,
            details={},
        )

        return results

    def score(self, current: np.ndarray) -> float:
        """Retorna o valor de MMD como score principal de drift.

        Args:
            current: Dados do período atual.

        Returns:
            Valor de MMD².
        """
        current = np.asarray(current, dtype=np.float64)
        if current.ndim == 1:
            current = current.reshape(-1, 1)
        return self.mmd_calc.calculate(self.reference_, current)
