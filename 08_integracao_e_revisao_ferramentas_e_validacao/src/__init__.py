"""
Aula 8: Integração e Revisão (Ferramentas e Validação)

Módulos para construção de um pipeline integrado de monitoramento de modelos
de ML em produção, combinando detecção de data drift (Evidently AI),
estimativa de performance sem ground truth (NannyML CBPE) e validação
de dados (Great Expectations).

Referências:
    Rabanser, S. et al. (NeurIPS 2019). Failing loudly: an empirical
        study of methods for detecting dataset shift.
    Sculley, D. et al. (NIPS 2015). Hidden technical debt in ML systems.
    Müller, R. et al. (2024). Open-source drift detection tools in action.
    Breck, E. et al. (MLSys 2019). Data validation for machine learning.
    Gama, J. et al. (2014). A survey on concept drift adaptation.
"""

from .data_preprocessing import DataPreprocessor
from .model import FraudDetector
from .training import train_model, cross_validate_model
from .evaluation import (
    calculate_metrics,
    calculate_drift_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)
from .utils import save_model, load_model, save_metrics, set_seed

__all__ = [
    "DataPreprocessor",
    "FraudDetector",
    "train_model",
    "cross_validate_model",
    "calculate_metrics",
    "calculate_drift_metrics",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "save_model",
    "load_model",
    "save_metrics",
    "set_seed",
]
