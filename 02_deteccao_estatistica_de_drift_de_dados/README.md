# Aula 02 – Detecção Estatística de Drift de Dados

## 🎯 Objetivos de Aprendizagem

Ao final desta aula, o aluno será capaz de:

1. **Compreender** o conceito de *data drift* (deriva de covariáveis) e seu impacto na degradação de modelos de Machine Learning em produção.
2. **Aplicar** métricas de dissimilaridade entre distribuições — Divergência Kullback-Leibler (KL), Divergência Jensen-Shannon (JS) e Population Stability Index (PSI) — para quantificar mudanças nos dados.
3. **Implementar** testes de hipótese de duas amostras (Kolmogorov-Smirnov e Qui-quadrado) para avaliar a significância estatística do drift detectado.
4. **Construir** pipelines automatizados de monitoramento de drift usando ferramentas open-source (Evidently, NannyML).
5. **Interpretar** resultados de detecção de drift e conectá-los a ações corretivas em contextos de MLOps.

## 📖 Teoria-Chave

### Data Drift (Deriva de Covariáveis)

> *"Drift de dados significa que a distribuição estatística dos dados de entrada mudou ao longo do tempo. [...] Como os modelos de ML supõem que os dados futuros seguem a mesma distribuição dos dados de treinamento, quando essa suposição é violada o desempenho do modelo tende a degradar."*
> — Documento da Aula 2

### Métricas de Dissimilaridade

| Métrica | Fórmula | Características |
|---------|---------|-----------------|
| **KL Divergence** | $D_{KL}(P \| Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}$ | Não simétrica, não limitada superiormente |
| **JS Divergence** | $D_{JS}(P \| Q) = \frac{1}{2}D_{KL}(P \| M) + \frac{1}{2}D_{KL}(Q \| M)$ | Simétrica, limitada entre 0 e 1 |
| **PSI** | $PSI = \sum_j (p_j - q_j) \ln \frac{p_j}{q_j}$ | Limiares: <0,1 estável; 0,1–0,25 moderado; >0,25 severo |

### Testes de Hipótese

- **Kolmogorov-Smirnov (KS):** Teste não paramétrico para distribuições contínuas. Compara CDFs.
- **Qui-quadrado (χ²):** Teste para variáveis categóricas. Compara frequências observadas vs. esperadas.

## 🛠️ Tecnologias Utilizadas

| Biblioteca | Versão Mínima | Uso |
|-----------|--------------|-----|
| `numpy` | ≥1.24.0 | Cálculos numéricos, PSI |
| `pandas` | ≥2.0.0 | Manipulação de dados |
| `scipy` | ≥1.11.0 | Teste KS, Qui-quadrado, entropia |
| `scikit-learn` | ≥1.3.0 | Pré-processamento, binning |
| `matplotlib` | ≥3.7.0 | Visualizações |
| `seaborn` | ≥0.12.0 | Visualizações estatísticas |
| `evidently` | ≥0.4.0 | Pipeline de monitoramento de drift |
| `nannyml` | ≥0.10.0 | Detecção de drift em produção |
| `pytest` | ≥7.4.0 | Testes unitários |

## 📂 Estrutura de Arquivos

```
02_deteccao_estatistica_de_drift_de_dados/
├── README.md                          # Este arquivo
├── requirements.txt                   # Dependências específicas da aula
├── notebooks/
│   ├── 01_exploracao.ipynb            # EDA e visualização de distribuições
│   ├── 02_treinamento.ipynb           # Implementação de KS, KL, JS e PSI
│   └── 03_avaliacao.ipynb             # Pipeline automatizado com Evidently/NannyML
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py          # Geração e preparo de dados sintéticos
│   ├── model.py                       # Detectores de drift (KS, PSI, KL, JS)
│   ├── training.py                    # Execução de detecção em lotes
│   ├── evaluation.py                  # Avaliação e visualização de resultados
│   └── utils.py                       # Utilitários (I/O, configuração)
├── data/
│   ├── raw/
│   │   └── dataset.csv                # Dataset sintético de crédito
│   ├── processed/                     # Dados processados
│   └── README.md                      # Documentação do dataset
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py          # Testes do pré-processamento
│   ├── test_model.py                  # Testes dos detectores de drift
│   └── test_evaluation.py             # Testes das métricas de avaliação
└── outputs/
    ├── models/                        # Resultados salvos
    ├── figures/                        # Gráficos gerados
    └── logs/                          # Logs de execução
```

## 🚀 Como Executar

### 1. Instalação de Dependências

```bash
cd 02_deteccao_estatistica_de_drift_de_dados
pip install -r requirements.txt
```

### 2. Gerar Dataset Sintético

```bash
python -c "from src.data_preprocessing import DataPreprocessor; dp = DataPreprocessor(); dp.generate_and_save()"
```

### 3. Executar Notebooks (sequencialmente)

```bash
jupyter notebook notebooks/01_exploracao.ipynb
jupyter notebook notebooks/02_treinamento.ipynb
jupyter notebook notebooks/03_avaliacao.ipynb
```

### 4. Execução via Scripts Python

```python
from src.data_preprocessing import DataPreprocessor
from src.model import DriftDetector
from src.evaluation import DriftEvaluator

# Carregar dados
dp = DataPreprocessor()
ref_data, prod_data = dp.load_data("data/raw/dataset.csv")

# Detectar drift
detector = DriftDetector()
results = detector.detect_all(ref_data, prod_data)

# Avaliar resultados
evaluator = DriftEvaluator()
evaluator.generate_report(results)
```

## 📊 Datasets

Dataset sintético simulando dados de clientes de uma fintech de crédito, com distribuições de referência (treinamento) e produção (com drift). Detalhes completos em [`data/README.md`](data/README.md).

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

1. Kullback, S., & Leibler, R. A. (1951). *On Information and Sufficiency*. The Annals of Mathematical Statistics, 22(1), 79–86.
2. Lin, J. (1991). *Divergence Measures Based on the Shannon Entropy*. IEEE Trans. on Information Theory, 37(1), 145–151.
3. Massey, F. J. (1951). *The Kolmogorov-Smirnov Test for Goodness of Fit*. JASA, 46(253), 68–78.
4. Yurdakul, B., & Naranjo, J. (2020). *Statistical Properties of the Population Stability Index*. Journal of Risk Model Validation, 14(4), 89–100.
5. Rabanser, S., Günnemann, S., & Lipton, Z. C. (2019). *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift*. NeurIPS 2019.
6. Gama, J. et al. (2014). *A Survey on Concept Drift Adaptation*. ACM Computing Surveys, 46(4), 44.
7. Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NIPS 2015.

## 🎥 Vídeos Relacionados

| Vídeo | Título | Duração |
|-------|--------|---------|
| 1 | Detecção Estatística de Drift de Dados | 15 min |
| 2 | Métricas de Dissimilaridade: KL, JS e PSI | 18 min |
| 3 | Detectando Drift com KS e PSI em Python | 20 min |
| 4 | Pipeline Automatizado de Monitoramento de Drift | 18 min |

## 🔗 Links Úteis

- [SciPy – KS Test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html)
- [Evidently AI – Data Drift](https://docs.evidentlyai.com/presets/data-drift)
- [NannyML – Documentation](https://nannyml.readthedocs.io/)
- [Kullback-Leibler Divergence Explained](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)
- [AWS SageMaker Model Monitor](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html)
