"""
Aula 5: Drift em Embeddings e Testes de Duas Amostras

Módulos para detecção de drift em representações vetoriais (embeddings)
utilizando testes estatísticos de duas amostras, classificadores adversários
e técnicas de visualização de alta dimensionalidade.

Referências:
    Gretton, A. et al. (2012). A kernel two-sample test. JMLR, 13, 723-773.
    Devlin, J. et al. (2019). BERT: Pre-training of deep bidirectional transformers. NAACL.
    Lopez-Paz, D. & Oquab, M. (2017). Revisiting classifier two-sample tests. ICLR.
    Feldhans, R. et al. (2021). Drift Detection in Text Data with Document Embeddings.
"""

from .data_preprocessing import DataPreprocessor
from .model import EmbeddingDriftDetector
from .training import train_adversarial_classifier, compute_reference_statistics
from .evaluation import calculate_drift_metrics, plot_embedding_distributions
from .utils import save_model, load_model, set_seed, generate_synthetic_dataset

__all__ = [
    "DataPreprocessor",
    "EmbeddingDriftDetector",
    "train_adversarial_classifier",
    "compute_reference_statistics",
    "calculate_drift_metrics",
    "plot_embedding_distributions",
    "save_model",
    "load_model",
    "set_seed",
    "generate_synthetic_dataset",
]
