"""
Módulo de monitoramento de modelos ML – Pipeline e SLO Monitor.

Implementa MonitoringPipeline e SLOMonitor para verificação contínua
de modelos de ML em produção, incluindo:
- Verificação de acurácia com threshold de SLO (Snippet 1, doc 04)
- Teste KS para detecção de drift por feature (Snippet 2, doc 04)
- Verificação de qualidade de dados (Snippet 3, doc 04)
- Cálculo de PSI (Population Stability Index)
- Integração dos checks em pipeline unificado

Referências:
    Breck, E. et al. (2017). The ML Test Score: A Rubric for ML
    Production Readiness and Technical Debt Reduction. IEEE BigData.

    Naveed, H. et al. (2025). Monitoring Machine Learning Systems:
    A Multivocal Literature Review. arXiv:2509.14294.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


# ======================================================================
# Data Classes para resultados de monitoramento
# ======================================================================


@dataclass
class SLOCheckResult:
    """Resultado de uma verificação de SLO.

    Attributes:
        name: Nome do check (e.g., 'accuracy_slo', 'drift_idade').
        passed: Se o check passou (True) ou violou o SLO (False).
        metric_value: Valor da métrica calculada.
        threshold: Threshold do SLO.
        message: Mensagem descritiva do resultado.
        timestamp: Timestamp da verificação.
    """

    name: str
    passed: bool
    metric_value: float
    threshold: float
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class MonitoringReport:
    """Relatório completo de monitoramento.

    Agrega resultados de múltiplos checks de SLO, conforme arquitetura
    discutida na seção 'Boas Práticas, Ferramentas e Estado da Arte'
    do DOCUMENTO_AULA_6.md.

    Attributes:
        checks: Lista de resultados de verificação.
        overall_status: Status geral ('HEALTHY', 'WARNING', 'CRITICAL').
        summary: Resumo textual.
    """

    checks: List[SLOCheckResult] = field(default_factory=list)
    overall_status: str = "HEALTHY"
    summary: str = ""

    def add_check(self, check: SLOCheckResult) -> None:
        """Adiciona um check ao relatório e atualiza status geral."""
        self.checks.append(check)
        if not check.passed:
            self.overall_status = "CRITICAL"

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def to_dict(self) -> Dict[str, Any]:
        """Converte relatório para dicionário."""
        return {
            "overall_status": self.overall_status,
            "passed": self.passed_count,
            "failed": self.failed_count,
            "total": len(self.checks),
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "metric_value": round(c.metric_value, 4),
                    "threshold": c.threshold,
                    "message": c.message,
                }
                for c in self.checks
            ],
        }


# ======================================================================
# SLO Monitor
# ======================================================================


class SLOMonitor:
    """Monitor de Service Level Objectives para modelos de ML.

    Implementa os três tipos de verificação de SLO descritos nos
    Snippets 1-3 do DOCUMENTO_AULA_6.md:
    1. Acurácia do modelo (Snippet 1)
    2. Detecção de drift via teste KS (Snippet 2)
    3. Qualidade dos dados – taxa de missings (Snippet 3)

    Também implementa o cálculo de PSI (Population Stability Index)
    conforme fórmula apresentada na seção 'Data Drift & Concept Drift'.

    Attributes:
        accuracy_slo: Threshold mínimo de acurácia.
        drift_p_value_threshold: Limiar de p-valor para teste KS.
        missing_rate_slo: Taxa máxima aceitável de valores ausentes.
        psi_threshold: Limiar de PSI para detecção de drift severo.
    """

    def __init__(
        self,
        accuracy_slo: float = 0.85,
        drift_p_value_threshold: float = 0.05,
        missing_rate_slo: float = 0.01,
        psi_threshold: float = 0.25,
    ) -> None:
        """Inicializa o SLO Monitor.

        Implementa definição de Data SLOs conforme discutido na seção
        'Service Level Indicators (SLI) e Data SLOs em ML' do doc 04.

        Args:
            accuracy_slo: SLO de acurácia mínima (default: 85%).
            drift_p_value_threshold: P-valor mínimo para rejeitar drift.
            missing_rate_slo: Taxa máxima de missings por coluna.
            psi_threshold: PSI acima do qual drift é considerado severo.
        """
        self.accuracy_slo = accuracy_slo
        self.drift_p_value_threshold = drift_p_value_threshold
        self.missing_rate_slo = missing_rate_slo
        self.psi_threshold = psi_threshold

    def check_accuracy(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> SLOCheckResult:
        """Verifica se a acurácia do modelo atende ao SLO.

        Implementa Snippet 1 do DOCUMENTO_AULA_6.md:
        'Monitorando a Acurácia do Modelo'.

        Args:
            y_true: Rótulos verdadeiros.
            y_pred: Previsões do modelo.

        Returns:
            SLOCheckResult com resultado da verificação.
        """
        accuracy = accuracy_score(y_true, y_pred)
        passed = accuracy >= self.accuracy_slo

        message = (
            f"Acurácia {accuracy:.1%} {'≥' if passed else '<'} "
            f"SLO de {self.accuracy_slo:.0%}"
        )
        if not passed:
            message += " — ALERTA: performance abaixo do SLO!"

        logger.info(message)

        return SLOCheckResult(
            name="accuracy_slo",
            passed=passed,
            metric_value=accuracy,
            threshold=self.accuracy_slo,
            message=message,
        )

    def check_drift_ks(
        self,
        reference_data: pd.Series,
        production_data: pd.Series,
        feature_name: str = "feature",
    ) -> SLOCheckResult:
        """Detecta drift via teste de Kolmogorov-Smirnov.

        Implementa Snippet 2 do DOCUMENTO_AULA_6.md:
        'Detecção de Deriva de Dados'.

        Conforme discutido na seção 'Data Drift & Concept Drift':
        'Usando um teste estatístico de Kolmogorov-Smirnov (K-S),
        podemos comparar a distribuição de frequências de uma feature'.

        Args:
            reference_data: Dados de referência (treino).
            production_data: Dados de produção (novos).
            feature_name: Nome da feature para o relatório.

        Returns:
            SLOCheckResult com resultado do teste KS.
        """
        # Remover NaN para o teste
        ref_clean = reference_data.dropna()
        prod_clean = production_data.dropna()

        statistic, p_value = ks_2samp(ref_clean, prod_clean)
        passed = p_value >= self.drift_p_value_threshold

        message = (
            f"Drift KS em '{feature_name}': "
            f"statistic={statistic:.4f}, p-value={p_value:.4f}"
        )
        if not passed:
            message += f" — ALERTA: drift detectado (p < {self.drift_p_value_threshold})"

        logger.info(message)

        return SLOCheckResult(
            name=f"drift_ks_{feature_name}",
            passed=passed,
            metric_value=p_value,
            threshold=self.drift_p_value_threshold,
            message=message,
        )

    def check_missing_rate(
        self, df: pd.DataFrame
    ) -> List[SLOCheckResult]:
        """Verifica taxa de valores ausentes por coluna.

        Implementa Snippet 3 do DOCUMENTO_AULA_6.md:
        'Verificação de Qualidade dos Dados'.

        Conforme discutido na seção de Data SLOs:
        'completude (ausência de lacunas ou dados ausentes)'.

        Args:
            df: DataFrame com dados de produção.

        Returns:
            Lista de SLOCheckResult, um por coluna numérica.
        """
        results = []
        missing_rate = df.isnull().mean()

        for col in df.select_dtypes(include=[np.number]).columns:
            rate = missing_rate[col]
            passed = rate <= self.missing_rate_slo

            message = (
                f"Missing rate em '{col}': {rate:.2%} "
                f"{'≤' if passed else '>'} SLO de {self.missing_rate_slo:.0%}"
            )
            if not passed:
                message += " — ALERTA: qualidade de dados abaixo do SLO!"

            results.append(
                SLOCheckResult(
                    name=f"missing_rate_{col}",
                    passed=passed,
                    metric_value=rate,
                    threshold=self.missing_rate_slo,
                    message=message,
                )
            )

        return results

    @staticmethod
    def calculate_psi(
        reference: np.ndarray,
        production: np.ndarray,
        n_bins: int = 10,
        epsilon: float = 1e-6,
    ) -> float:
        """Calcula o Population Stability Index (PSI).

        Implementa a fórmula do PSI conforme apresentada na seção
        'Data Drift & Concept Drift' do DOCUMENTO_AULA_6.md:

        PSI = Σ (P_i - Q_i) × ln(P_i / Q_i)

        onde P_i é a proporção no bin i da referência e Q_i da produção.

        PSI > 0.25 indica deriva severa.
        PSI 0.10-0.25 indica deriva moderada.
        PSI < 0.10 indica pouca ou nenhuma mudança.

        Args:
            reference: Array com dados de referência.
            production: Array com dados de produção.
            n_bins: Número de bins para discretização.
            epsilon: Valor mínimo para evitar divisão por zero.

        Returns:
            Valor do PSI.
        """
        # Remover NaN
        reference = reference[~np.isnan(reference)]
        production = production[~np.isnan(production)]

        # Criar bins baseados na distribuição de referência
        breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        # Calcular proporções em cada bin
        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        prod_counts = np.histogram(production, bins=breakpoints)[0]

        ref_proportions = ref_counts / len(reference) + epsilon
        prod_proportions = prod_counts / len(production) + epsilon

        # Fórmula do PSI
        psi = np.sum(
            (prod_proportions - ref_proportions)
            * np.log(prod_proportions / ref_proportions)
        )

        return float(psi)

    def check_psi(
        self,
        reference: np.ndarray,
        production: np.ndarray,
        feature_name: str = "feature",
        n_bins: int = 10,
    ) -> SLOCheckResult:
        """Verifica PSI de uma feature contra o threshold.

        Implementa verificação de drift via PSI conforme discutido
        na seção 'Data Drift & Concept Drift' do doc 04.

        Args:
            reference: Dados de referência.
            production: Dados de produção.
            feature_name: Nome da feature.
            n_bins: Número de bins.

        Returns:
            SLOCheckResult com resultado.
        """
        psi_value = self.calculate_psi(reference, production, n_bins)
        passed = psi_value < self.psi_threshold

        if psi_value < 0.10:
            severity = "baixo (sem mudança significativa)"
        elif psi_value < 0.25:
            severity = "moderado (mudança moderada)"
        else:
            severity = "SEVERO (mudança significativa)"

        message = (
            f"PSI de '{feature_name}': {psi_value:.4f} — {severity}"
        )

        return SLOCheckResult(
            name=f"psi_{feature_name}",
            passed=passed,
            metric_value=psi_value,
            threshold=self.psi_threshold,
            message=message,
        )


# ======================================================================
# Monitoring Pipeline
# ======================================================================


class MonitoringPipeline:
    """Pipeline integrado de monitoramento de ML.

    Integra os três tipos de verificação (Snippets 1-3) em um pipeline
    unificado, conforme descrito na seção 'Boas Práticas, Ferramentas
    e Estado da Arte' do DOCUMENTO_AULA_6.md: 'reunir dados de
    referência e métricas em tempo real'.

    Implementa a arquitetura de monitoramento discutida no Vídeo 6.2:
    - Definição de Data SLOs
    - Error Budget
    - Fluxo de resposta a alertas

    Attributes:
        slo_monitor: Instância do SLOMonitor.
        features_to_monitor: Lista de features para monitorar drift.
        alerts: Lista de alertas emitidos.
    """

    def __init__(
        self,
        slo_monitor: Optional[SLOMonitor] = None,
        features_to_monitor: Optional[List[str]] = None,
    ) -> None:
        """Inicializa o pipeline de monitoramento.

        Args:
            slo_monitor: Instância de SLOMonitor. Se None, cria default.
            features_to_monitor: Features para verificação de drift.
        """
        self.slo_monitor = slo_monitor or SLOMonitor()
        self.features_to_monitor = features_to_monitor or [
            "idade", "renda_mensal", "score_credito",
            "tempo_emprego", "valor_emprestimo",
        ]
        self.alerts: List[str] = []

    def run_full_check(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        df_reference: pd.DataFrame,
        df_production: pd.DataFrame,
    ) -> MonitoringReport:
        """Executa pipeline completo de monitoramento.

        Integra os três checks definidos nos Snippets 1-3 do
        DOCUMENTO_AULA_6.md em um pipeline unificado, conforme
        implementação discutida no Vídeo 6.3.

        Args:
            y_true: Rótulos verdadeiros (produção).
            y_pred: Previsões do modelo (produção).
            df_reference: DataFrame de referência (treino).
            df_production: DataFrame de produção.

        Returns:
            MonitoringReport com todos os resultados.
        """
        report = MonitoringReport()
        self.alerts = []

        # --- Check 1: Acurácia (Snippet 1) ---
        accuracy_check = self.slo_monitor.check_accuracy(y_true, y_pred)
        report.add_check(accuracy_check)
        if not accuracy_check.passed:
            self.alerts.append(accuracy_check.message)

        # --- Check 2: Drift por feature via KS (Snippet 2) ---
        for feature in self.features_to_monitor:
            if feature in df_reference.columns and feature in df_production.columns:
                drift_check = self.slo_monitor.check_drift_ks(
                    df_reference[feature],
                    df_production[feature],
                    feature_name=feature,
                )
                report.add_check(drift_check)
                if not drift_check.passed:
                    self.alerts.append(drift_check.message)

                # Check PSI adicional para mesma feature
                psi_check = self.slo_monitor.check_psi(
                    df_reference[feature].dropna().values,
                    df_production[feature].dropna().values,
                    feature_name=feature,
                )
                report.add_check(psi_check)
                if not psi_check.passed:
                    self.alerts.append(psi_check.message)

        # --- Check 3: Qualidade dos dados (Snippet 3) ---
        missing_checks = self.slo_monitor.check_missing_rate(df_production)
        for check in missing_checks:
            report.add_check(check)
            if not check.passed:
                self.alerts.append(check.message)

        # Gerar resumo
        report.summary = (
            f"Monitoramento: {report.passed_count}/{len(report.checks)} checks OK. "
            f"Status: {report.overall_status}. "
            f"Alertas: {len(self.alerts)}."
        )

        logger.info(report.summary)
        return report

    def get_alerts(self) -> List[str]:
        """Retorna lista de alertas do último check.

        Returns:
            Lista de mensagens de alerta.
        """
        return self.alerts.copy()
