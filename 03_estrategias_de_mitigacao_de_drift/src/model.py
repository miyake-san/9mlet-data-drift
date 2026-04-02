"""
Modelos batch e online para mitigação de drift em churn telecom.

Implementa as arquiteturas discutidas na seção 'Saiba Mais' do documento
acadêmico da Aula 03:
  - Modelo batch (LogisticRegression / RandomForest) como baseline
  - Modelo online com aprendizado incremental (River)
  - Ensemble adaptativo com ADWINBoostingClassifier (River)

O conceito central é que mitigação de drift pode ser incorporada à
própria arquitetura do modelo (ensembles adaptativos), e não apenas
adicionada como um remendo posterior (Krawczyk et al., 2017).

Referências:
    Gomes, H. M., et al. (2017). Adaptive random forests for evolving
    data stream classification. Machine Learning, 106(9-10), 1469-1495.

    Krawczyk, B., et al. (2017). Ensemble learning for data stream analysis:
    A survey. Information Fusion, 37, 132-156.

    Montiel, J., et al. (2021). River: machine learning for streaming
    data in Python. JMLR, 22(110), 1-8.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from river import drift, ensemble, linear_model, preprocessing


class BatchChurnModel:
    """Modelo batch para predição de churn (baseline).

    Implementa o modelo tradicional que sofre degradação sob drift,
    conforme discutido na seção 'Saiba Mais': um modelo treinado em
    regime estável perde aderência quando o ambiente muda.

    Attributes:
        model_type: Tipo de modelo ('logistic' ou 'random_forest').
        model: Instância do modelo scikit-learn.
    """

    def __init__(
        self,
        model_type: str = "logistic",
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.model_type = model_type
        self.random_state = random_state
        if model_type == "logistic":
            self.model = LogisticRegression(
                random_state=random_state, max_iter=1000, **kwargs
            )
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100, random_state=random_state, **kwargs
            )
        else:
            raise ValueError(f"Tipo de modelo não suportado: {model_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BatchChurnModel":
        """Treina o modelo no regime estável.

        Args:
            X: Features de treino.
            y: Target de treino.

        Returns:
            Self (para encadeamento).
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prediz classes.

        Args:
            X: Features.

        Returns:
            Array de predições (0 ou 1).
        """
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Prediz probabilidades.

        Args:
            X: Features.

        Returns:
            Array de probabilidades (n_samples, 2).
        """
        return self.model.predict_proba(X)

    def score(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Calcula métricas de desempenho.

        Retorna accuracy, F1 e AUC-ROC, métricas discutidas na seção
        'Saiba Mais' para avaliar degradação sob drift.

        Args:
            X: Features.
            y: Target verdadeiro.

        Returns:
            Dicionário com métricas.
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        return {
            "accuracy": accuracy_score(y, y_pred),
            "f1": f1_score(y, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y, y_proba),
        }


class OnlineChurnModel:
    """Modelo online com aprendizado incremental para mitigação de drift.

    Implementa a atualização incremental discutida na seção 'Saiba Mais':
        θ_{t+1} = θ_t - η_t ∇_θ ℓ(f_{θ_t}(x_t), y_t)

    Usa LogisticRegression do River com StandardScaler para normalização
    online. O detector ADWIN sinaliza drift conforme Bifet & Gavaldà (2007).

    Referências:
        Bifet, A., & Gavaldà, R. (2007). Learning from time-changing data
        with adaptive windowing. SIAM SDM.

        Cesa-Bianchi, N., & Lugosi, G. (2006). Prediction, Learning, and
        Games. Cambridge University Press.
    """

    def __init__(self, seed: int = 42) -> None:
        self.model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
        self.drift_detector = drift.ADWIN()
        self.seed = seed
        self.n_samples_seen: int = 0
        self.drift_points: list[int] = []
        self._correct: int = 0
        self._total: int = 0

    def learn_one(self, x: dict, y: int) -> Optional[bool]:
        """Atualiza o modelo com uma observação.

        Implementa o gatilho mínimo de mitigação discutido na seção
        'Hands On' (Snippet 1): o detector ADWIN sinaliza drift e
        abre espaço para medidas como recalibração ou re-treinamento.

        Args:
            x: Dicionário com features de uma observação.
            y: Label verdadeiro (0 ou 1).

        Returns:
            True se drift foi detectado, None caso contrário.
        """
        y_pred = self.model.predict_one(x)
        is_correct = int(y_pred == y) if y_pred is not None else 0

        self.model.learn_one(x, y)
        self.n_samples_seen += 1
        self._total += 1
        self._correct += is_correct

        # Atualiza detector ADWIN com acerto/erro
        self.drift_detector.update(is_correct)
        if self.drift_detector.drift_detected:
            self.drift_points.append(self.n_samples_seen)
            return True
        return None

    def predict_one(self, x: dict) -> Optional[int]:
        """Prediz classe para uma observação.

        Args:
            x: Dicionário com features.

        Returns:
            Classe predita (0 ou 1) ou None se modelo ainda não treinado.
        """
        return self.model.predict_one(x)

    def predict_proba_one(self, x: dict) -> dict[int, float]:
        """Prediz probabilidades para uma observação.

        Args:
            x: Dicionário com features.

        Returns:
            Dicionário {classe: probabilidade}.
        """
        return self.model.predict_proba_one(x)

    @property
    def accuracy(self) -> float:
        """Acurácia acumulada."""
        return self._correct / self._total if self._total > 0 else 0.0


class AdaptiveEnsembleModel:
    """Ensemble adaptativo ADWINBoostingClassifier para absorver drift.

    Implementa o conceito discutido na seção 'Saiba Mais' e no Snippet 3
    da seção 'Hands On': em vez de depender de um único classificador
    que envelhece, o ensemble incorpora monitoramento de mudança e
    substituição gradual dos membros mais fracos.

    Conforme Krawczyk et al. (2017): "a vantagem estatística e operacional
    está na redundância — se um membro perde aderência por causa do drift,
    outros preservam parte do desempenho".

    Referências:
        Krawczyk, B., et al. (2017). Ensemble learning for data stream
        analysis: A survey. Information Fusion, 37, 132-156.
    """

    def __init__(self, n_models: int = 5, seed: int = 42) -> None:
        self.n_models = n_models
        self.seed = seed
        self.model = ensemble.ADWINBoostingClassifier(
            model=preprocessing.StandardScaler()
            | linear_model.LogisticRegression(),
            n_models=n_models,
            seed=seed,
        )
        self.n_samples_seen: int = 0
        self._correct: int = 0
        self._total: int = 0

    def learn_one(self, x: dict, y: int) -> None:
        """Atualiza o ensemble com uma observação.

        Args:
            x: Dicionário com features.
            y: Label verdadeiro.
        """
        y_pred = self.model.predict_one(x)
        is_correct = int(y_pred == y) if y_pred is not None else 0
        self._correct += is_correct
        self._total += 1

        self.model.learn_one(x, y)
        self.n_samples_seen += 1

    def predict_one(self, x: dict) -> Optional[int]:
        """Prediz classe para uma observação.

        Args:
            x: Dicionário com features.

        Returns:
            Classe predita ou None.
        """
        return self.model.predict_one(x)

    def predict_proba_one(self, x: dict) -> dict[int, float]:
        """Prediz probabilidades para uma observação.

        Args:
            x: Dicionário com features.

        Returns:
            Dicionário {classe: probabilidade}.
        """
        return self.model.predict_proba_one(x)

    @property
    def accuracy(self) -> float:
        """Acurácia acumulada."""
        return self._correct / self._total if self._total > 0 else 0.0
