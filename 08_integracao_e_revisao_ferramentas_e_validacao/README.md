# Aula 8 — Integração e Revisão (Ferramentas e Validação)

## 🎯 Objetivos de Aprendizagem

Ao final desta aula o aluno será capaz de:

1. Integrar conceitos de **data drift**, **concept drift** e **prior drift** vistos nas Aulas 1–7 em uma visão unificada de monitoramento de modelos ML.
2. Utilizar **Evidently AI** para gerar relatórios automatizados de data drift comparando distribuições de referência e produção.
3. Aplicar **NannyML (CBPE)** para estimar performance de modelos de classificação **sem ground truth** disponível.
4. Definir **Great Expectations** para validação de qualidade de dados de entrada antes de alimentar o modelo.
5. Comparar ferramentas **open-source** (Evidently, NannyML, GX) com **comerciais** (WhyLabs, Arize, Fiddler, SageMaker Monitor).
6. Construir um **pipeline end-to-end** de detecção de drift, validação de dados e re-treino automático.
7. Aplicar boas práticas de **MLOps** e **governança de modelos** (auditoria, documentação, checklist de produção).

## 📖 Teoria-Chave

### Tipos de Drift (Recap)

| Tipo | Definição Formal | Exemplo (Fraude) |
|------|------------------|------------------|
| **Covariate Drift** | $P_t(X) \neq P_{t+\Delta}(X)$ | Aumento no `valor_transacao` pós-promoção |
| **Prior Drift** | $P_t(Y) \neq P_{t+\Delta}(Y)$ | Proporção de fraude caiu de 5% → 1% |
| **Concept Drift** | $P_t(Y \mid X) \neq P_{t+\Delta}(Y \mid X)$ | Fraudadores adotam novo canal (telefone) |

### Detecção Estatística

- **Kolmogorov-Smirnov (KS)**: $D_{n,m} = \sup_x |F_{\text{ref}}(x) - F_{\text{prod}}(x)|$
- **Jensen-Shannon Divergence**: Versão simétrica da KL divergence.
- **Wasserstein Distance**: Custo de transformar distribuição $P$ em $Q$.
- **Maximum Mean Discrepancy (MMD)**: Testes multivariados em espaço de kernel.

### NannyML — CBPE

O **Confidence-Based Performance Estimation** utiliza probabilidades de predição do modelo para estimar métricas (como AUC) sem rótulos verdadeiros, segmentando dados em *chunks* temporais.

### Great Expectations

Framework de validação de dados com regras declarativas ("expectativas") que capturam problemas de schema, valores fora de faixa e anomalias antes de alimentar o modelo.

> Referências completas: Rabanser et al. (NeurIPS 2019); Sculley et al. (NIPS 2015); Müller et al. (2024); Gama et al. (2014); Polyzotis et al. (2019).

## 🛠️ Tecnologias Utilizadas

| Biblioteca | Versão Mínima | Uso |
|---|---|---|
| numpy | ≥1.24.0 | Computação numérica |
| pandas | ≥2.0.0 | Manipulação de dados |
| scikit-learn | ≥1.3.0 | Modelo Random Forest, métricas, validação cruzada |
| scipy | ≥1.11.0 | Testes estatísticos (KS) |
| evidently | ≥0.4.0 | Relatórios de data drift e data quality |
| nannyml | ≥0.10.0 | Estimativa de performance sem ground truth (CBPE) |
| great-expectations | ≥0.18.0 | Validação de dados de entrada |
| matplotlib | ≥3.7.0 | Visualizações |
| seaborn | ≥0.12.0 | Gráficos estatísticos |
| joblib | ≥1.3.0 | Serialização de modelos |
| pytest | ≥7.4.0 | Testes unitários |

## 📂 Estrutura de Arquivos

```
08_integracao_e_revisao_ferramentas_e_validacao/
├── README.md                           # Este documento
├── requirements.txt                    # Dependências da aula
├── notebooks/
│   ├── 01_exploracao.ipynb             # EDA + Evidently drift report
│   ├── 02_treinamento.ipynb            # Modelo de fraude + NannyML CBPE
│   └── 03_avaliacao.ipynb              # Pipeline integrado + Great Expectations
├── src/
│   ├── __init__.py                     # Exports do pacote
│   ├── data_preprocessing.py           # DataPreprocessor (carga, limpeza, features)
│   ├── model.py                        # FraudDetector (RF + drift check KS)
│   ├── training.py                     # Treinamento, cross-validation, re-treino
│   ├── evaluation.py                   # Métricas, drift metrics, visualizações
│   └── utils.py                        # Serialização, logging, helpers
├── scripts/
│   └── generate_dataset.py             # Gerador de dataset sintético
├── data/
│   ├── raw/
│   │   └── dataset.csv                 # Dataset de transações (ref + prod)
│   ├── processed/                      # Dados processados
│   └── README.md                       # Documentação do dataset
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py           # Testes do DataPreprocessor
│   ├── test_model.py                   # Testes do FraudDetector
│   └── test_evaluation.py              # Testes de métricas e drift
└── outputs/
    ├── models/                         # Modelos salvos (.pkl)
    ├── figures/                        # Gráficos gerados
    └── logs/                           # Logs de execução
```

## 🚀 Como Executar

### 1. Instalar Dependências

```bash
# Com uv (recomendado)
uv pip install -r requirements.txt

# Ou com pip
pip install -r requirements.txt
```

### 2. Gerar Dataset

```bash
cd 08_integracao_e_revisao_ferramentas_e_validacao
python scripts/generate_dataset.py
```

### 3. Executar Notebooks (sequencialmente)

```bash
jupyter notebook notebooks/01_exploracao.ipynb
jupyter notebook notebooks/02_treinamento.ipynb
jupyter notebook notebooks/03_avaliacao.ipynb
```

### 4. Executar via Scripts Python

```bash
# Treinar modelo e avaliar
python -c "
from src import DataPreprocessor, train_model, calculate_metrics
import pandas as pd

df = pd.read_csv('data/raw/dataset.csv')
prep = DataPreprocessor()
prep.load_data('data/raw/dataset.csv')
ref = df[df['periodo'] == 'referencia']
X_train, X_test, y_train, y_test = prep.split_data(prep.clean_data(ref))
model = train_model(X_train, y_train)
metrics = model.score(X_test, y_test)
print(metrics)
"
```

## 📊 Datasets

Dataset sintético de **10.000 transações de pagamento** com drift simulado. Detalhes completos em [data/README.md](data/README.md).

| Período | Registros | % Fraude | Drift |
|---------|-----------|----------|-------|
| Referência | 5.000 | ~5% | Baseline |
| Produção | 5.000 | ~8% | Covariate + Prior + Concept |

## 🧪 Testes

```bash
# Executar todos os testes
pytest tests/ -v

# Testes específicos
pytest tests/test_preprocessing.py -v
pytest tests/test_model.py -v
pytest tests/test_evaluation.py -v

# Com cobertura
pytest tests/ -v --cov=src --cov-report=term-missing
```

## 📚 Referências

- **Rabanser, S. et al.** (NeurIPS 2019). *Failing loudly: an empirical study of methods for detecting dataset shift.*
- **Sculley, D. et al.** (NIPS 2015). *Hidden technical debt in machine learning systems.*
- **Müller, R. et al.** (2024). *Open-source drift detection tools in action.* arXiv:2404.18673.
- **Gama, J. et al.** (2014). *A survey on concept drift adaptation.* ACM Computing Surveys, 46(4).
- **Breck, E. et al.** (MLSys 2019). *Data validation for machine learning.*
- **Polyzotis, N. et al.** (2019). *Data lifecycle challenges in production ML.* ACM SIGMOD.
- **Moreno-Torres, J. G. et al.** (2012). *A unifying view on dataset shift.* Pattern Recognition, 45(1).
- **Hinder, F. et al.** (2024). *One or two things we know about concept drift.* Frontiers in AI.

## 🎥 Vídeos Relacionados

| Vídeo | Título | Duração |
|-------|--------|---------|
| 8.1 | Integração e Revisão: Ferramentas e Validação | 15 min |
| 8.2 | Ecossistema de Ferramentas para Data Drift | 18 min |
| 8.3 | Pipeline Integrado End-to-End de Detecção e Mitigação | 20 min |
| 8.4 | Validação, Governança e Tendências Futuras | 18 min |

## 🔗 Links Úteis

- [Evidently AI — Documentação](https://docs.evidentlyai.com/)
- [NannyML — Documentação](https://nannyml.readthedocs.io/)
- [Great Expectations — Documentação](https://docs.greatexpectations.io/)
- [WhyLabs — Plataforma](https://whylabs.ai/docs)
- [Arize AI — Plataforma](https://docs.arize.com/)
- [MLOps — Google ML Best Practices](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
