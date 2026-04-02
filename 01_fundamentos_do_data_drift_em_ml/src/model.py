"""
Módulo de modelos para detecção de drift e classificação adaptativa.

Implementa:
- DriftDetector: detecção de data drift via teste Kolmogorov-Smirnov,
  conforme apresentado na seção 'Hands On — Snippet 1' do material da aula.
- AdaptiveClassifier: classificador com suporte a aprendizado incremental
  (partial_fit), conforme a seção 'Hands On — Snippet 2' e a discussão
  sobre aprendizado online na seção 'Saiba Mais — Adaptação e Mitigação'.

Referências:
    Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A.
    (2014). A Survey on Concept Drift Adaptation. ACM Computing Surveys,
    46(4), 44.

    Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., & Zhang, G. (2018).
    Learning under Concept Drift: A Review. IEEE TKDE, 31(12), 2346–2363.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import ks_2samp
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


# ======================================================================
# DriftDetector
# ======================================================================

@dataclass
class DriftResult:
    """Resultado da detecção de drift para uma feature.

    Attributes:
        feature_name: Nome da feature analisada.
        statistic: Estatística KS (distância máxima entre CDFs).
        p_value: p-value do teste KS.
        drift_detected: True se p_value < alpha.
    """
    feature_name: str
    statistic: float
    p_value: float
    drift_detected: bool


class DriftDetector:
    """Detector de data drift usando o teste Kolmogorov-Smirnov.

    O teste KS compara duas amostras e verifica se provêm da mesma
    distribuição.  Conforme discutido na seção 'Monitoramento e Detecção
    de Drift', é uma abordagem não-paramétrica que não assume forma
    funcional para as distribuições.

    A estatística KS é definida como:
        D = sup_x |F_ref(x) - F_prod(x)|

    onde F_ref e F_prod são as funções de distribuição empírica
    acumulada (CDF) das amostras de referência e produção.

    Attributes:
        alpha: Nível de significância para rejeição de H0.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        """Inicializa o detector.

        Args:
            alpha: Nível de significância (default 0.05 = 95% de confiança).
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha deve estar no intervalo (0, 1)")
        self.alpha = alpha

    def detect_univariate(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "feature",
    ) -> DriftResult:
        """Aplica o teste KS a uma única feature.

        Implementa o Snippet 1 do Hands On: comparação de distribuições
        com ``scipy.stats.ks_2samp``.

        Args:
            reference: Dados de referência (treinamento).
            current: Dados atuais (produção).
            feature_name: Nome descritivo da feature.

        Returns:
            DriftResult com estatística, p-value e flag de drift.
        """
        stat, p_value = ks_2samp(reference.ravel(), current.ravel())
        return DriftResult(
            feature_name=feature_name,
            statistic=float(stat),
            p_value=float(p_value),
            drift_detected=p_value < self.alpha,
        )

    def detect_multivariate(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> list[DriftResult]:
        """Aplica o teste KS feature a feature.

        Conforme discutido no material, o monitoramento das distribuições
        de entrada P(X) é essencial para identificar covariate shift.

        Args:
            reference: Matriz de referência (n_ref, n_features).
            current: Matriz atual (n_cur, n_features).
            feature_names: Nomes das features (opcional).

        Returns:
            Lista de DriftResult, um por feature.
        """
        n_features = reference.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        results: list[DriftResult] = []
        for i in range(n_features):
            result = self.detect_univariate(
                reference[:, i], current[:, i], feature_names[i]
            )
            results.append(result)
        return results

    def summary(self, results: list[DriftResult]) -> dict[str, object]:
        """Gera resumo da detecção de drift.

        Args:
            results: Lista de DriftResult.

        Returns:
            Dicionário com contagem de features com drift e detalhes.
        """
        drifted = [r for r in results if r.drift_detected]
        return {
            "total_features": len(results),
            "features_with_drift": len(drifted),
            "drift_ratio": len(drifted) / max(len(results), 1),
            "drifted_features": [r.feature_name for r in drifted],
            "all_results": results,
        }


# ======================================================================
# AdaptiveClassifier
# ======================================================================

class AdaptiveClassifier:
    """Classificador adaptativo com suporte a aprendizado incremental.

    Encapsula ``SGDClassifier`` com ``partial_fit`` para atualização
    contínua, conforme discutido na seção 'Hands On — Snippet 2' e
    'Saiba Mais — Aprendizado online (incremental)'.

    O aprendizado incremental permite ao modelo "aprender continuamente"
    sem precisar ser reconstruído do zero, adequando-se a eventuais
    mudanças nos padrões dos dados.

    Referência:
        Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., & Zhang, G. (2018).
        Learning under Concept Drift: A Review. IEEE TKDE, 31(12),
        2346–2363.

    Attributes:
        model: Instância de SGDClassifier.
        classes: Classes possíveis do problema.
        history: Histórico de métricas por batch.
    """

    def __init__(
        self,
        loss: str = "log_loss",
        random_state: int = 42,
        classes: list[int] | None = None,
    ) -> None:
        """Inicializa o classificador.

        Args:
            loss: Função de perda do SGDClassifier.
            random_state: Semente para reprodutibilidade.
            classes: Classes possíveis (default [0, 1]).
        """
        self.model = SGDClassifier(
            loss=loss,
            random_state=random_state,
            max_iter=1000,
            tol=1e-3,
        )
        self.classes = np.array(classes if classes is not None else [0, 1])
        self.history: list[dict[str, float]] = []
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaptiveClassifier":
        """Treina o modelo do zero (batch completo).

        Args:
            X: Features (n_samples, n_features).
            y: Labels.

        Returns:
            self
        """
        self.model.fit(X, y)
        self._fitted = True
        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> "AdaptiveClassifier":
        """Atualiza o modelo incrementalmente com um batch de dados.

        Implementa aprendizado contínuo com ``partial_fit`` conforme
        o Snippet 2 do Hands On e a seção 'Saiba Mais — Adaptação
        e Mitigação'.

        Args:
            X: Features do batch (n_samples, n_features).
            y: Labels do batch.

        Returns:
            self
        """
        self.model.partial_fit(X, y, classes=self.classes)
        self._fitted = True

        # Registra acurácia no batch para monitoramento
        y_pred = self.model.predict(X)
        acc = float(accuracy_score(y, y_pred))
        self.history.append({"batch_accuracy": acc, "batch_size": len(y)})
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Gera predições.

        Args:
            X: Features.

        Returns:
            Vetor de predições.

        Raises:
            RuntimeError: Se o modelo não foi treinado.
        """
        if not self._fitted:
            raise RuntimeError("Modelo não treinado. Chame fit() ou partial_fit() primeiro.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidades de cada classe.

        Args:
            X: Features.

        Returns:
            Matriz (n_samples, n_classes) de probabilidades.
        """
        if not self._fitted:
            raise RuntimeError("Modelo não treinado.")
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calcula acurácia.

        Args:
            X: Features.
            y: Labels verdadeiras.

        Returns:
            Acurácia (0-1).
        """
        return float(self.model.score(X, y))
