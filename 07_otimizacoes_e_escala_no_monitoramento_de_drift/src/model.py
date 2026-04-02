"""
Modelo de monitoramento de drift para detecção em escala.

Implementa DriftMonitor com PSI, teste K–S e distância de Wasserstein,
conforme discutido na seção 'Saiba Mais' da Aula 7.

O PSI é baseado em divergência de Kullback–Leibler (Kullback & Leibler, 1951):
    PSI = Σ (q_i - p_i) * ln(q_i / p_i)

O teste K–S compara CDFs empíricas (Kolmogorov, 1933; Smirnov, 1948):
    D_{n,m} = sup_x |F_n(x) - G_m(x)|

Referências:
    Kullback, S. & Leibler, R. A. (1951). On Information and Sufficiency.
    Kolmogorov, A. (1933). Sulla determinazione empirica di una legge.
    Smirnov, N. (1948). Table for Estimating the Goodness of Fit.
    Gama, J. et al. (2014). A Survey on Concept Drift Adaptation. ACM.
    Wasserstein, L. N. (1969). Markov Processes over Denumerable Products.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class DriftResult:
    """
    Resultado de detecção de drift para uma feature.

    Armazena métricas computadas e decisão, conforme a combinação
    de PSI e K–S discutida no Snippet 4 do Hands On.
    """
    feature: str
    psi_value: float
    ks_statistic: float
    ks_pvalue: float
    wasserstein_distance: float
    drift_detected: bool
    severity: str  # "none", "warning", "critical"


class DriftMonitor:
    """
    Monitor de drift com PSI, K–S e Wasserstein.

    Implementa as métricas discutidas na seção 'Saiba Mais' (Quadro 1)
    para detecção de drift univariado em cenários de produção.
    Combina métrica interpretável (PSI) com teste estatístico (K–S)
    e distância geométrica (Wasserstein) para sinal robusto.

    Conforme Sculley et al. (2015), o monitoramento deve traduzir
    métricas em decisões e decisões em ações com rastreabilidade.

    Args:
        n_bins: Número de bins para discretização do PSI.
        psi_threshold: Limiar de PSI para drift crítico (default 0.25).
        psi_warning: Limiar de PSI para warning (default 0.10).
        ks_alpha: Nível de significância para teste K–S (default 0.05).
        eps: Epsilon para estabilidade numérica no PSI.
    """

    def __init__(
        self,
        n_bins: int = 10,
        psi_threshold: float = 0.25,
        psi_warning: float = 0.10,
        ks_alpha: float = 0.05,
        eps: float = 1e-6,
    ) -> None:
        self.n_bins = n_bins
        self.psi_threshold = psi_threshold
        self.psi_warning = psi_warning
        self.ks_alpha = ks_alpha
        self.eps = eps
        self._baseline: Optional[Dict[str, np.ndarray]] = None

    def fit(
        self, baseline_data: Dict[str, np.ndarray]
    ) -> "DriftMonitor":
        """
        Armazena distribuição de referência (baseline).

        Conforme discutido na seção 'Saiba Mais': PSI e K–S
        requerem uma distribuição de referência para comparação
        com a distribuição corrente.

        Args:
            baseline_data: Dicionário {feature_name: array de valores}.

        Returns:
            Self para encadeamento.
        """
        self._baseline = {
            k: np.asarray(v, dtype=float) for k, v in baseline_data.items()
        }
        return self

    def predict(
        self, current_data: Dict[str, np.ndarray]
    ) -> List[DriftResult]:
        """
        Detecta drift comparando dados correntes com baseline.

        Implementa a lógica de combinação PSI + K–S conforme
        Snippet 4 do Hands On: se PSI ≥ limiar OU K–S significativo,
        sinaliza drift (Sculley et al., 2015).

        Args:
            current_data: Dicionário {feature_name: array de valores}.

        Returns:
            Lista de DriftResult para cada feature.

        Raises:
            ValueError: Se baseline não foi definido via fit().
        """
        if self._baseline is None:
            raise ValueError("Chame fit() com dados de baseline antes de predict().")

        results = []
        for feature_name, current_values in current_data.items():
            if feature_name not in self._baseline:
                continue

            ref = self._baseline[feature_name]
            cur = np.asarray(current_values, dtype=float)

            psi_val = self.compute_psi(ref, cur)
            ks_stat, ks_pval = self.compute_ks(ref, cur)
            wass_dist = self.compute_wasserstein(ref, cur)

            # Lógica de decisão conforme Snippet 4 do Hands On
            drift_detected = (psi_val >= self.psi_warning) or (ks_pval < self.ks_alpha)

            if psi_val >= self.psi_threshold:
                severity = "critical"
            elif psi_val >= self.psi_warning:
                severity = "warning"
            else:
                severity = "none"

            results.append(DriftResult(
                feature=feature_name,
                psi_value=psi_val,
                ks_statistic=ks_stat,
                ks_pvalue=ks_pval,
                wasserstein_distance=wass_dist,
                drift_detected=drift_detected,
                severity=severity,
            ))

        return results

    def score(
        self, current_data: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Retorna score agregado de drift (PSI médio e % features com drift).

        Args:
            current_data: Dicionário {feature_name: array de valores}.

        Returns:
            Dicionário com métricas agregadas.
        """
        results = self.predict(current_data)
        if not results:
            return {"mean_psi": 0.0, "drift_fraction": 0.0}

        psi_values = [r.psi_value for r in results]
        drift_count = sum(1 for r in results if r.drift_detected)

        return {
            "mean_psi": float(np.mean(psi_values)),
            "max_psi": float(np.max(psi_values)),
            "drift_fraction": drift_count / len(results),
            "n_features_with_drift": drift_count,
            "n_features_total": len(results),
        }

    def compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> float:
        """
        Calcula Population Stability Index (PSI) entre duas distribuições.

        Implementa a fórmula da seção 'Saiba Mais':
            PSI = Σ (q_i - p_i) * ln(q_i / p_i)

        Baseado em divergência de Kullback–Leibler (Kullback & Leibler, 1951).
        O parâmetro bins controla a granularidade do histograma e eps evita
        instabilidades numéricas quando uma faixa recebe frequência nula.

        Args:
            reference: Array da distribuição de referência.
            current: Array da distribuição corrente.

        Returns:
            Valor do PSI (≥ 0).
        """
        # Definir cortes por quantis da referência (Snippet 2 do Hands On)
        cuts = np.quantile(reference, np.linspace(0, 1, self.n_bins + 1))
        # Garantir bins únicos
        cuts = np.unique(cuts)
        if len(cuts) < 2:
            return 0.0

        p, _ = np.histogram(reference, bins=cuts)
        q, _ = np.histogram(current, bins=cuts)

        # Normalizar para proporções e aplicar epsilon
        p = np.clip(p / p.sum(), self.eps, None)
        q = np.clip(q / q.sum(), self.eps, None)

        return float(np.sum((q - p) * np.log(q / p)))

    def compute_ks(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Realiza teste Kolmogorov–Smirnov de duas amostras.

        Implementa o teste conforme descrito na seção 'Saiba Mais':
            D_{n,m} = sup_x |F_n(x) - G_m(x)|

        O teste fornece critério estatístico formal para rejeitar
        a hipótese de que as amostras vêm da mesma distribuição
        (Kolmogorov, 1933; Smirnov, 1948).

        Args:
            reference: Array da distribuição de referência.
            current: Array da distribuição corrente.

        Returns:
            Tupla (estatística D, p-valor).
        """
        stat, p_value = stats.ks_2samp(reference, current)
        return float(stat), float(p_value)

    def compute_wasserstein(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> float:
        """
        Calcula distância de Wasserstein (Earth Mover's Distance).

        Conforme seção 'Saiba Mais': mede o 'trabalho' para
        transformar uma distribuição em outra, sendo intuitiva
        em variáveis numéricas e robusta a pequenas mudanças
        de forma (Wasserstein, 1969).

        Args:
            reference: Array da distribuição de referência.
            current: Array da distribuição corrente.

        Returns:
            Distância de Wasserstein (≥ 0).
        """
        return float(stats.wasserstein_distance(reference, current))


class IncrementalPSI:
    """
    Cálculo incremental de PSI para janelas deslizantes.

    Mantém histogramas de contagem que podem ser atualizados
    incrementalmente, sem recomputar do zero a cada janela.
    Implementa o conceito discutido na seção 'Saiba Mais':
    agregação incremental com estados resumidos (contagens por bin).

    Referências:
        Gama, J. et al. (2014). A Survey on Concept Drift Adaptation.
    """

    def __init__(
        self,
        bins: np.ndarray,
        eps: float = 1e-6,
    ) -> None:
        """
        Args:
            bins: Array com bordas dos bins (definidas a partir do baseline).
            eps: Epsilon para estabilidade numérica.
        """
        self.bins = bins
        self.eps = eps
        self.n_bins = len(bins) - 1
        self._counts = np.zeros(self.n_bins, dtype=float)
        self._total = 0

    def update(self, values: np.ndarray) -> None:
        """
        Adiciona novos valores ao histograma corrente (incremental).

        Args:
            values: Array de novos valores observados.
        """
        counts, _ = np.histogram(values, bins=self.bins)
        self._counts += counts
        self._total += len(values)

    def remove(self, values: np.ndarray) -> None:
        """
        Remove valores antigos do histograma (para janela deslizante).

        Args:
            values: Array de valores a remover.
        """
        counts, _ = np.histogram(values, bins=self.bins)
        self._counts = np.maximum(self._counts - counts, 0)
        self._total = max(self._total - len(values), 0)

    def compute(self, reference_proportions: np.ndarray) -> float:
        """
        Calcula PSI entre proporções de referência e estado acumulado.

        Args:
            reference_proportions: Proporções do baseline (soma = 1).

        Returns:
            Valor do PSI.
        """
        if self._total == 0:
            return 0.0

        q = np.clip(self._counts / self._total, self.eps, None)
        p = np.clip(reference_proportions, self.eps, None)

        return float(np.sum((q - p) * np.log(q / p)))

    def reset(self) -> None:
        """Reseta contadores para nova janela."""
        self._counts[:] = 0
        self._total = 0
