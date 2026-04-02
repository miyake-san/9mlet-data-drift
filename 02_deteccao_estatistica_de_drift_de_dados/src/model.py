"""
Módulo de detectores de drift estatístico para a Aula 2.

Implementa detectores baseados em métricas de dissimilaridade (KL, JS, PSI)
e testes de hipótese (KS, Qui-quadrado), conforme discutido nas seções
'Saiba Mais — Métricas de dissimilaridade' e 'Testes estatísticos de
hipóteses (duas amostras)' do Documento da Aula 2.

Referências:
    Kullback, S., & Leibler, R. A. (1951). On Information and Sufficiency.
    The Annals of Mathematical Statistics, 22(1), 79–86.

    Lin, J. (1991). Divergence Measures Based on the Shannon Entropy.
    IEEE Transactions on Information Theory, 37(1), 145–151.

    Massey, F. J. (1951). The Kolmogorov-Smirnov Test for Goodness of Fit.
    JASA, 46(253), 68–78.

    Yurdakul, B., & Naranjo, J. (2020). Statistical Properties of the
    Population Stability Index. Journal of Risk Model Validation, 14(4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon


# ======================================================================
# Data classes para resultados
# ======================================================================

@dataclass
class DriftResult:
    """Resultado de detecção de drift para uma feature.

    Attributes:
        feature_name: Nome da feature analisada.
        method: Nome do método utilizado (ks, psi, kl, js, chi2).
        statistic: Valor da estatística calculada.
        p_value: p-valor (quando aplicável; None para métricas sem p-valor).
        drift_detected: True se drift foi detectado acima do limiar.
        threshold: Limiar utilizado para decisão.
    """

    feature_name: str
    method: str
    statistic: float
    p_value: Optional[float]
    drift_detected: bool
    threshold: float


@dataclass
class DriftReport:
    """Relatório completo de detecção de drift para múltiplas features.

    Attributes:
        results: Lista de DriftResult individuais.
        n_features_total: Total de features analisadas.
        n_features_drifted: Quantidade de features com drift detectado.
        drifted_features: Nomes das features com drift.
    """

    results: list[DriftResult]
    n_features_total: int
    n_features_drifted: int
    drifted_features: list[str]


# ======================================================================
# DriftDetector — classe principal
# ======================================================================

class DriftDetector:
    """Detector de drift estatístico multi-método.

    Implementa quatro técnicas de detecção de drift conforme discutido
    no Documento da Aula 2:

    1. **Teste KS** (seção 'Testes estatísticos de hipóteses'):
       compara CDFs empíricas de duas amostras contínuas.
    2. **PSI** (seção 'Saiba Mais — PSI'):
       divide features em bins e calcula divergência ponderada.
    3. **KL Divergence** (seção 'Saiba Mais — KL'):
       quantifica perda de informação entre distribuições.
    4. **JS Divergence** (seção 'Saiba Mais — JS'):
       versão simétrica e limitada da KL.
    5. **Qui-quadrado** (seção 'Testes estatísticos'):
       teste para variáveis categóricas.

    Attributes:
        alpha: Nível de significância para testes de hipótese.
        psi_threshold: Limiar de PSI para alerta (default 0.25).
        n_bins: Número de bins para PSI e discretização de KL/JS.
    """

    # Limiares de PSI conforme Documento da Aula 2:
    # < 0.10: estável | 0.10–0.25: drift moderado | > 0.25: drift severo
    PSI_STABLE = 0.10
    PSI_MODERATE = 0.25

    def __init__(
        self,
        alpha: float = 0.05,
        psi_threshold: float = 0.25,
        n_bins: int = 10,
    ) -> None:
        """Inicializa o detector de drift.

        Args:
            alpha: Nível de significância para KS e Qui-quadrado.
            psi_threshold: Limiar de decisão para PSI (>= indica drift).
            n_bins: Número de bins para métricas baseadas em histograma.
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha deve estar no intervalo (0, 1)")
        if psi_threshold <= 0:
            raise ValueError("psi_threshold deve ser positivo")
        if n_bins < 2:
            raise ValueError("n_bins deve ser >= 2")

        self.alpha = alpha
        self.psi_threshold = psi_threshold
        self.n_bins = n_bins

    # ------------------------------------------------------------------
    # Teste de Kolmogorov-Smirnov
    # ------------------------------------------------------------------

    def ks_test(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "feature",
    ) -> DriftResult:
        """Aplica o teste de Kolmogorov-Smirnov a duas amostras.

        Implementa o teste KS conforme descrito na seção 'Testes estatísticos
        de hipóteses (duas amostras)' e no Snippet 1 do Hands On:

            D = sup_x |F_ref(x) - F_current(x)|

        O teste é não-paramétrico e sensível a diferenças na forma,
        posição e dispersão das distribuições.

        Referência:
            Massey, F. J. (1951). The Kolmogorov-Smirnov Test for
            Goodness of Fit. JASA, 46(253), 68–78.

        Args:
            reference: Dados de referência (treinamento).
            current: Dados atuais (produção).
            feature_name: Nome descritivo da feature.

        Returns:
            DriftResult com estatística D, p-valor e flag de drift.
        """
        stat, p_value = stats.ks_2samp(
            reference.ravel(), current.ravel()
        )
        return DriftResult(
            feature_name=feature_name,
            method="ks",
            statistic=float(stat),
            p_value=float(p_value),
            drift_detected=bool(p_value < self.alpha),
            threshold=self.alpha,
        )

    # ------------------------------------------------------------------
    # Population Stability Index (PSI)
    # ------------------------------------------------------------------

    def psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "feature",
        n_bins: int | None = None,
    ) -> DriftResult:
        """Calcula o Population Stability Index (PSI).

        Implementa a fórmula do PSI conforme a seção 'Saiba Mais — PSI'
        e o Snippet 2 do Hands On do Documento da Aula 2:

            PSI = Σ_j (p_j - q_j) * ln(p_j / q_j)

        onde p_j é a proporção no bin j da referência e q_j é a proporção
        no bin j dos dados atuais.

        Limiares (setor bancário, conforme Documento):
            PSI < 0.10  → estável
            0.10 ≤ PSI < 0.25 → drift moderado
            PSI ≥ 0.25  → drift severo

        Referência:
            Yurdakul, B., & Naranjo, J. (2020). Statistical Properties
            of the Population Stability Index. J. Risk Model Validation.

        Args:
            reference: Dados de referência (treinamento).
            current: Dados atuais (produção).
            feature_name: Nome da feature.
            n_bins: Número de bins (override do default).

        Returns:
            DriftResult com valor do PSI e flag de drift.
        """
        bins = n_bins or self.n_bins
        ref = reference.ravel()
        cur = current.ravel()

        # Criar bins baseados na referência
        breakpoints = np.linspace(
            min(ref.min(), cur.min()),
            max(ref.max(), cur.max()),
            bins + 1,
        )

        ref_counts, _ = np.histogram(ref, bins=breakpoints)
        cur_counts, _ = np.histogram(cur, bins=breakpoints)

        # Converter para proporções (com suavização para evitar log(0))
        eps = 1e-8
        ref_props = (ref_counts + eps) / (ref_counts.sum() + eps * bins)
        cur_props = (cur_counts + eps) / (cur_counts.sum() + eps * bins)

        # Fórmula PSI: Σ (p - q) * ln(p / q)
        psi_value = float(
            np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        )

        return DriftResult(
            feature_name=feature_name,
            method="psi",
            statistic=psi_value,
            p_value=None,
            drift_detected=psi_value >= self.psi_threshold,
            threshold=self.psi_threshold,
        )

    # ------------------------------------------------------------------
    # Divergência Kullback-Leibler (KL)
    # ------------------------------------------------------------------

    def kl_divergence(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "feature",
        n_bins: int | None = None,
    ) -> DriftResult:
        """Calcula a divergência Kullback-Leibler.

        Implementa a KL conforme a seção 'Saiba Mais — Divergência KL':

            D_KL(P || Q) = Σ_i P(i) * log(P(i) / Q(i))

        A KL é não-simétrica, sempre ≥ 0, e não limitada superiormente.
        Usada aqui com discretização em bins para dados contínuos.

        Referência:
            Kullback, S., & Leibler, R. A. (1951). On Information and
            Sufficiency. Annals of Mathematical Statistics, 22(1), 79–86.

        Args:
            reference: Distribuição P (referência).
            current: Distribuição Q (atual).
            feature_name: Nome da feature.
            n_bins: Número de bins.

        Returns:
            DriftResult com valor da KL divergência.
        """
        bins = n_bins or self.n_bins
        ref = reference.ravel()
        cur = current.ravel()

        breakpoints = np.linspace(
            min(ref.min(), cur.min()),
            max(ref.max(), cur.max()),
            bins + 1,
        )

        ref_counts, _ = np.histogram(ref, bins=breakpoints)
        cur_counts, _ = np.histogram(cur, bins=breakpoints)

        # Suavização para evitar log(0)
        eps = 1e-8
        p = (ref_counts + eps) / (ref_counts.sum() + eps * bins)
        q = (cur_counts + eps) / (cur_counts.sum() + eps * bins)

        kl_value = float(np.sum(p * np.log(p / q)))

        return DriftResult(
            feature_name=feature_name,
            method="kl",
            statistic=kl_value,
            p_value=None,
            drift_detected=kl_value > 0.1,  # limiar heurístico
            threshold=0.1,
        )

    # ------------------------------------------------------------------
    # Divergência Jensen-Shannon (JS)
    # ------------------------------------------------------------------

    def js_divergence(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "feature",
        n_bins: int | None = None,
    ) -> DriftResult:
        """Calcula a divergência Jensen-Shannon.

        Implementa a JS conforme a seção 'Saiba Mais — Divergência JS':

            D_JS(P || Q) = 1/2 D_KL(P || M) + 1/2 D_KL(Q || M)

        onde M = (P + Q) / 2. A JS é simétrica e limitada entre 0 e 1
        (usando log base 2), oferecendo escala normalizada e intuitiva.

        Referência:
            Lin, J. (1991). Divergence Measures Based on the Shannon
            Entropy. IEEE Trans. on Information Theory, 37(1), 145–151.

        Args:
            reference: Distribuição P.
            current: Distribuição Q.
            feature_name: Nome da feature.
            n_bins: Número de bins.

        Returns:
            DriftResult com JS divergência (0 = idênticas, 1 = máxima).
        """
        bins = n_bins or self.n_bins
        ref = reference.ravel()
        cur = current.ravel()

        breakpoints = np.linspace(
            min(ref.min(), cur.min()),
            max(ref.max(), cur.max()),
            bins + 1,
        )

        ref_counts, _ = np.histogram(ref, bins=breakpoints)
        cur_counts, _ = np.histogram(cur, bins=breakpoints)

        # Suavização
        eps = 1e-8
        p = (ref_counts + eps) / (ref_counts.sum() + eps * bins)
        q = (cur_counts + eps) / (cur_counts.sum() + eps * bins)

        # scipy.spatial.distance.jensenshannon retorna a DISTÂNCIA (raiz da JS)
        # Elevamos ao quadrado para obter a divergência JS propriamente
        js_dist = jensenshannon(p, q, base=2)
        js_value = float(js_dist ** 2)

        return DriftResult(
            feature_name=feature_name,
            method="js",
            statistic=js_value,
            p_value=None,
            drift_detected=js_value > 0.05,  # limiar heurístico
            threshold=0.05,
        )

    # ------------------------------------------------------------------
    # Teste do Qui-quadrado (variáveis categóricas)
    # ------------------------------------------------------------------

    def chi2_test(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "feature",
    ) -> DriftResult:
        """Aplica o teste Qui-quadrado para variáveis categóricas.

        Conforme discutido na seção 'Testes estatísticos de hipóteses':
        avalia se as frequências observadas em cada categoria diferem
        significativamente entre referência e produção.

        Referência:
            Pearson, K. (1900). On the Criterion That a Given System
            of Deviations [...]. Phil. Mag., 50(302), 157–175.

        Args:
            reference: Valores categóricos da referência.
            current: Valores categóricos da produção.
            feature_name: Nome da feature.

        Returns:
            DriftResult com estatística χ² e p-valor.
        """
        ref = reference.ravel()
        cur = current.ravel()

        # Obter todas as categorias presentes
        all_categories = np.union1d(np.unique(ref), np.unique(cur))

        # Contagem de frequências por categoria
        ref_counts = np.array([np.sum(ref == c) for c in all_categories])
        cur_counts = np.array([np.sum(cur == c) for c in all_categories])

        # Frequências esperadas proporcionais à referência
        expected = ref_counts * (cur_counts.sum() / ref_counts.sum())

        # Evitar divisão por zero em categorias ausentes
        mask = expected > 0
        if mask.sum() < 2:
            return DriftResult(
                feature_name=feature_name,
                method="chi2",
                statistic=0.0,
                p_value=1.0,
                drift_detected=False,
                threshold=self.alpha,
            )

        stat, p_value = stats.chisquare(cur_counts[mask], f_exp=expected[mask])

        return DriftResult(
            feature_name=feature_name,
            method="chi2",
            statistic=float(stat),
            p_value=float(p_value),
            drift_detected=bool(p_value < self.alpha),
            threshold=self.alpha,
        )

    # ------------------------------------------------------------------
    # Detecção completa (todas as features)
    # ------------------------------------------------------------------

    def detect_all(
        self,
        reference_df: "pd.DataFrame",
        current_df: "pd.DataFrame",
        numerical_cols: list[str] | None = None,
        categorical_cols: list[str] | None = None,
        methods: list[str] | None = None,
    ) -> DriftReport:
        """Executa detecção de drift em todas as features.

        Para features numéricas aplica os métodos especificados (default:
        KS e PSI). Para features categóricas aplica Qui-quadrado.

        Args:
            reference_df: DataFrame de referência.
            current_df: DataFrame de produção.
            numerical_cols: Colunas numéricas (auto-detecta se None).
            categorical_cols: Colunas categóricas (auto-detecta se None).
            methods: Lista de métodos para numéricas (default: ["ks", "psi"]).

        Returns:
            DriftReport com todos os resultados.
        """
        import pandas as pd

        if methods is None:
            methods = ["ks", "psi"]

        exclude = {"origem", "inadimplente"}
        if numerical_cols is None:
            numerical_cols = [
                c for c in reference_df.select_dtypes(include=[np.number]).columns
                if c not in exclude
            ]
        if categorical_cols is None:
            categorical_cols = [
                c for c in reference_df.select_dtypes(
                    include=["object", "category"]
                ).columns
                if c not in exclude
            ]

        results: list[DriftResult] = []
        method_map = {
            "ks": self.ks_test,
            "psi": self.psi,
            "kl": self.kl_divergence,
            "js": self.js_divergence,
        }

        # Features numéricas
        for col in numerical_cols:
            ref_vals = reference_df[col].dropna().values
            cur_vals = current_df[col].dropna().values
            for method_name in methods:
                if method_name in method_map:
                    result = method_map[method_name](
                        ref_vals, cur_vals, feature_name=col
                    )
                    results.append(result)

        # Features categóricas
        for col in categorical_cols:
            ref_vals = reference_df[col].dropna().values
            cur_vals = current_df[col].dropna().values
            results.append(self.chi2_test(ref_vals, cur_vals, feature_name=col))

        # Montar relatório
        drifted = list({r.feature_name for r in results if r.drift_detected})

        return DriftReport(
            results=results,
            n_features_total=len(numerical_cols) + len(categorical_cols),
            n_features_drifted=len(drifted),
            drifted_features=sorted(drifted),
        )

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------

    @staticmethod
    def interpret_psi(psi_value: float) -> str:
        """Interpreta valor de PSI conforme limiares do setor bancário.

        Conforme o Documento da Aula 2:
            PSI < 0.10  → estável (mudanças mínimas)
            0.10 ≤ PSI < 0.25 → drift moderado
            PSI ≥ 0.25  → drift severo

        Args:
            psi_value: Valor calculado do PSI.

        Returns:
            String descritiva do nível de drift.
        """
        if psi_value < 0.10:
            return "estável (PSI < 0.10)"
        elif psi_value < 0.25:
            return "drift moderado (0.10 ≤ PSI < 0.25)"
        else:
            return "drift severo (PSI ≥ 0.25)"
