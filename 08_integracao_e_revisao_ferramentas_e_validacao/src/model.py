"""
Módulo do modelo de detecção de fraude com monitoramento integrado.

Implementa um classificador de fraude (Random Forest) envolvido por
lógica de monitoramento de data drift, conforme o caso de negócio
da Aula 8 — plataforma de pagamentos global.

A classe FraudDetector combina predição com verificações simples
de drift (testes KS por feature), ilustrando a visão integrada
de modelo + monitoramento discutida no Documento 04.

Referências:
    Rabanser, S. et al. (NeurIPS 2019). Failing loudly: an empirical
        study of methods for detecting dataset shift.
    Gama, J. et al. (2014). A survey on concept drift adaptation.
        ACM Computing Surveys, 46(4).
    Sculley, D. et al. (NIPS 2015). Hidden technical debt in ML systems.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class FraudDetector:
    """
    Detector de fraude com monitoramento de drift integrado.

    Combina um classificador Random Forest com verificações de
    covariate drift via teste Kolmogorov-Smirnov (KS), conforme
    a abordagem integrada discutida no Documento 04 da Aula 8.

    A integração de detecção + monitoramento em uma única classe
    demonstra o conceito de 'observabilidade contínua' do modelo,
    alinhado com as práticas de MLOps de Sculley et al. (2015).

    Attributes:
        model: Classificador RandomForest subjacente.
        reference_distributions: Distribuições de referência por feature.
        drift_threshold: Limiar de p-value para sinalizar drift (KS test).
        is_fitted: Indica se o modelo foi treinado.
        feature_names: Nomes das features usadas no treinamento.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        random_state: int = 42,
        drift_threshold: float = 0.05,
        class_weight: str = "balanced",
    ) -> None:
        """
        Inicializa o FraudDetector.

        Args:
            n_estimators: Número de árvores no Random Forest.
            max_depth: Profundidade máxima das árvores.
            random_state: Semente para reproduzibilidade.
            drift_threshold: Limiar de p-value KS para sinalizar drift.
            class_weight: Estratégia de peso de classes ('balanced').
        """
        self.model: RandomForestClassifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1,
        )
        self.drift_threshold: float = drift_threshold
        self.reference_distributions: Dict[str, np.ndarray] = {}
        self.is_fitted: bool = False
        self.feature_names: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        store_reference: bool = True,
    ) -> "FraudDetector":
        """
        Treina o modelo de fraude e armazena distribuições de referência.

        Conforme discutido no Documento 04 (seção 'Pipeline Integrado'),
        o treinamento deve preservar as distribuições de referência para
        posterior comparação com dados de produção (detecção de drift).

        Args:
            X: Features de treinamento.
            y: Labels (0 = legítima, 1 = fraude).
            store_reference: Se True, armazena P_ref(X) para cada feature.

        Returns:
            Self (para encadeamento).
        """
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        self.is_fitted = True

        if store_reference:
            for col in X.select_dtypes(include=[np.number]).columns:
                self.reference_distributions[col] = X[col].values.copy()

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Gera predições de fraude (0/1).

        Args:
            X: Features para predição.

        Returns:
            Array de predições binárias.

        Raises:
            RuntimeError: Se o modelo não tiver sido treinado.
        """
        self._check_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Gera probabilidades de fraude.

        Probabilidades são essenciais para o NannyML CBPE
        (Confidence-Based Performance Estimation), que as utiliza
        para estimar métricas de performance sem ground truth,
        conforme descrito no Documento 04 (seção 'NannyML CBPE').

        Args:
            X: Features para predição.

        Returns:
            Array de probabilidades (n_samples, 2).
        """
        self._check_fitted()
        return self.model.predict_proba(X)

    def score(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Calcula métricas de performance do modelo.

        Retorna um dicionário com métricas discutidas no Documento 04:
        accuracy, precision, recall, f1 e ROC AUC.

        Args:
            X: Features de teste.
            y: Labels verdadeiros.

        Returns:
            Dicionário com métricas de performance.
        """
        self._check_fitted()
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]

        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, y_proba)),
        }

    def check_drift(
        self,
        X_production: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Verifica covariate drift via teste KS para cada feature numérica.

        Implementa a detecção de drift conforme descrito na seção
        'Detecção Estatística de Data Drift' do Documento 04, usando
        o teste Kolmogorov-Smirnov (KS) que calcula:

            D_{n,m} = sup_x |F_ref(x) - F_prod(x)|

        onde F_ref e F_prod são as funções de distribuição empírica
        dos dados de referência e produção, respectivamente.

        Referência:
            Rabanser, S. et al. (NeurIPS 2019). Failing loudly.

        Args:
            X_production: Dados de produção para verificação.
            features: Lista de features a verificar. Se None, usa todas
                      as features numéricas com distribuição de referência.

        Returns:
            Dicionário {feature: {statistic, p_value, drift_detected}}.

        Raises:
            RuntimeError: Se distribuições de referência não estiverem armazenadas.
        """
        if not self.reference_distributions:
            raise RuntimeError(
                "Distribuições de referência não armazenadas. "
                "Treine com store_reference=True."
            )

        if features is None:
            features = list(self.reference_distributions.keys())

        results: Dict[str, Dict[str, Any]] = {}
        for feat in features:
            if feat not in self.reference_distributions:
                continue
            if feat not in X_production.columns:
                continue

            ref_values = self.reference_distributions[feat]
            prod_values = X_production[feat].values

            ks_stat, p_value = stats.ks_2samp(ref_values, prod_values)
            results[feat] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "drift_detected": p_value < self.drift_threshold,
            }

        return results

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Retorna importância de cada feature (Gini importance).

        Útil para identificar quais features contribuem mais para
        a predição de fraude, informação complementar à detecção
        de drift — se uma feature importante sofre drift, o impacto
        no modelo será maior.

        Returns:
            Dicionário {feature_name: importance}.
        """
        self._check_fitted()
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances.tolist()))

    def _check_fitted(self) -> None:
        """Verifica se o modelo foi treinado."""
        if not self.is_fitted:
            raise RuntimeError(
                "Modelo não treinado. Chame fit() antes de predict/score."
            )
