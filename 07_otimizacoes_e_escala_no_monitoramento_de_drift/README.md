# Aula 7 — Otimizações e Escala no Monitoramento de Drift

## 🎯 Objetivos de Aprendizagem

Ao final desta aula o aluno será capaz de:

1. Explicar por que *data drift* e *concept drift* degradam modelos em produção e como isso se intensifica em cenários de Big Data.
2. Projetar pipelines de *streaming* (message broker + processamento em janelas) para monitoramento contínuo com baixa latência.
3. Implementar métricas interpretáveis de drift (PSI) e testes estatísticos (K–S) de forma incremental e em janelas deslizantes.
4. Comparar arquiteturas Lambda e Kappa e justificar qual se adequa melhor a diferentes cenários de monitoramento.
5. Aplicar estratégias de amostragem e janelas deslizantes para reduzir custo computacional sem comprometer qualidade estatística.
6. Projetar triggers automáticos de retraining baseados em drift detectado.

## 📖 Teoria-Chave

### PSI — Population Stability Index

O PSI mede divergência entre distribuição de referência e distribuição corrente, baseado em divergência de Kullback–Leibler:

$$PSI = \sum_{i=1}^{B} (q_i - p_i)\,\ln\left(\frac{q_i}{p_i}\right)$$

Onde $p_i$ e $q_i$ são proporções por faixa na referência e na janela atual, respectivamente (Kullback & Leibler, 1951).

### Teste Kolmogorov–Smirnov (K–S)

Compara CDFs empíricas; a estatística é a maior diferença absoluta:

$$D_{n,m} = \sup_x |F_n(x) - G_m(x)|$$

Fornece p-valor para decisão estatística formal (Kolmogorov, 1933; Smirnov, 1948).

### Arquitetura Lambda vs. Kappa

| Característica | Lambda | Kappa |
|---|---|---|
| Camadas | Batch + Streaming | Streaming unificado |
| Latência | Variável | Baixa e consistente |
| Complexidade | Maior (lógica duplicada) | Menor (replay) |

## 🛠️ Tecnologias Utilizadas

| Biblioteca | Versão Mínima | Uso |
|---|---|---|
| numpy | ≥1.24.0 | Computação numérica |
| pandas | ≥2.0.0 | Manipulação de dados |
| scipy | ≥1.10.0 | Teste K–S e estatísticas |
| scikit-learn | ≥1.3.0 | Modelos e métricas |
| matplotlib | ≥3.7.0 | Visualizações |
| seaborn | ≥0.12.0 | Gráficos estatísticos |
| pytest | ≥7.4.0 | Testes unitários |

## 📂 Estrutura de Arquivos

```
07_otimizacoes_e_escala_no_monitoramento_de_drift/
├── README.md                          # Este arquivo
├── requirements.txt                   # Dependências
├── scripts/
│   └── generate_dataset.py            # Geração do dataset sintético
├── notebooks/
│   ├── 01_exploracao.ipynb            # EDA e análise de drift em escala
│   ├── 02_treinamento.ipynb           # Janelas deslizantes e amostragem
│   └── 03_avaliacao.ipynb             # Avaliação e auto-retraining
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py          # Ingestão e pré-processamento
│   ├── model.py                       # DriftMonitor com PSI/KS
│   ├── training.py                    # Pipeline de janelas e treino
│   ├── evaluation.py                  # Métricas e visualizações
│   └── utils.py                       # Utilitários gerais
├── data/
│   ├── raw/
│   │   └── finbank_transactions.csv   # Dataset sintético FinBank
│   ├── processed/
│   └── README.md                      # Documentação dos dados
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_evaluation.py
└── outputs/
    ├── models/
    ├── figures/
    └── logs/
```

## 🚀 Como Executar

### Instalação de dependências

```bash
pip install -r requirements.txt
```

### Geração do dataset

```bash
python scripts/generate_dataset.py
```

### Execução dos notebooks (sequencial)

```bash
jupyter notebook notebooks/01_exploracao.ipynb
jupyter notebook notebooks/02_treinamento.ipynb
jupyter notebook notebooks/03_avaliacao.ipynb
```

### Execução via scripts Python

```python
from src.data_preprocessing import DataPreprocessor
from src.model import DriftMonitor
from src.training import train_sliding_window_pipeline
from src.evaluation import evaluate_drift_detection

# Carregar e preparar dados
preprocessor = DataPreprocessor()
df = preprocessor.load_data("data/raw/finbank_transactions.csv")

# Monitorar drift
monitor = DriftMonitor(n_bins=10, psi_threshold=0.25, ks_alpha=0.05)
```

## 📊 Datasets

Dataset sintético **FinBank Transactions** simulando transações financeiras com drift temporal.

- **Instâncias:** 10.000 transações
- **Features:** 8 numéricas + 2 categóricas
- **Drift injetado:** gradual (meses 4-6) e abrupto (meses 7-8)

Detalhes em [data/README.md](data/README.md).

## 🧪 Testes

```bash
# Executar todos os testes
pytest tests/ -v

# Executar testes específicos
pytest tests/test_preprocessing.py -v
pytest tests/test_model.py -v
pytest tests/test_evaluation.py -v
```

## 📚 Referências

- Sculley, D. et al. (2015). Hidden Technical Debt in Machine Learning Systems. *NeurIPS 2015*.
- Gama, J. et al. (2014). A Survey on Concept Drift Adaptation. *ACM Computing Surveys*, 46(4).
- Kreps, J. et al. (2011). Kafka: A Distributed Messaging System for Log Processing. *NetDB 2011*.
- Carbone, P. et al. (2015). Apache Flink™: Stream and Batch Processing in a Single Engine. *IEEE TCDE Bulletin*, 36(4).
- Kolmogorov, A. (1933). Sulla determinazione empirica di una legge di distribuzione.
- Smirnov, N. (1948). Table for Estimating the Goodness of Fit of Empirical Distributions.
- Kullback, S. & Leibler, R. A. (1951). On Information and Sufficiency. *Ann. Math. Stat.*, 22(1).
- Marz, N. & Warren, J. (2015). *Big Data: Principles and Best Practices of Scalable Realtime Data Systems*. Manning.
- Kreps, J. (2014). *Questioning the Lambda Architecture*. O'Reilly Radar.
- Wasserstein, L. N. (1969). Markov Processes over Denumerable Products of Spaces. *Problems of Information Transmission*, 5(3).

## 🎥 Vídeos Relacionados

| Vídeo | Título | Duração |
|-------|--------|---------|
| 1 | Otimizações e Escala no Monitoramento de Drift | 15 min |
| 2 | Arquiteturas Lambda e Kappa para Streaming de Drift | 18 min |
| 3 | Monitoramento de Drift com Amostragem e Janelas Deslizantes | 20 min |
| 4 | Escala e Auto-Retraining em Produção | 18 min |

## 🔗 Links Úteis

- [Apache Kafka — Documentação](https://kafka.apache.org/documentation/)
- [Apache Flink — Documentação](https://nightlies.apache.org/flink/flink-docs-stable/)
- [SciPy — ks_2samp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html)
- [Evidently AI — Data Drift](https://docs.evidentlyai.com/)
- [Scikit-learn — Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
