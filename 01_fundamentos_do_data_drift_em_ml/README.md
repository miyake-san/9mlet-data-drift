# Aula 1 – Fundamentos do Data Drift em ML

## 🎯 Objetivos de Aprendizagem

Ao final desta aula, o aluno será capaz de:

1. **Definir** Data Drift, Concept Drift e Label Drift, entendendo suas diferenças formais via distribuição conjunta $P(X, Y)$.
2. **Identificar** ambientes não-estacionários e a quebra da hipótese IID em cenários reais.
3. **Classificar** padrões de drift (abrupto, gradual e sazonal/recorrente) e suas causas.
4. **Compreender** as consequências do drift para modelos em produção (queda de acurácia, impacto de negócio, riscos de compliance).
5. **Implementar** detecção de drift usando o teste Kolmogorov-Smirnov (`scipy.stats.ks_2samp`).
6. **Aplicar** aprendizado incremental com `partial_fit` (SGDClassifier) para adaptação contínua.
7. **Projetar** um ciclo básico de MLOps para monitoramento e atualização de modelos.

## 📖 Teoria-Chave

### Data Drift vs. Concept Drift

- **Data Drift (Feature Drift / Covariate Shift):** $P_{\text{treino}}(X) \neq P_{\text{produção}}(X)$, enquanto $P(Y|X)$ permanece constante.
- **Concept Drift:** $P_{\text{treino}}(Y|X) \neq P_{\text{produção}}(Y|X)$ — o significado dos atributos para prever $Y$ muda.
- **Label Drift (Prior Probability Shift):** $P_{\text{treino}}(Y) \neq P_{\text{produção}}(Y)$ — a frequência das classes muda.

### Padrões de Drift

| Padrão | Descrição | Exemplo |
|--------|-----------|---------|
| **Abrupto** | Mudança repentina na distribuição | Lockdown em pandemia |
| **Gradual** | Mudança lenta e contínua | Evolução de preferências musicais |
| **Sazonal/Recorrente** | Padrões cíclicos previsíveis | Vendas no Natal vs. Janeiro |

### Detecção Estatística

O **teste Kolmogorov-Smirnov (KS)** compara duas amostras e retorna:
- **Estatística KS**: distância máxima entre as CDFs empíricas.
- **p-value**: probabilidade de as distribuições serem iguais. Se $p < 0.05$, rejeitamos $H_0$ (distribuições iguais).

### Aprendizado Incremental

Uso de `partial_fit()` para atualização contínua sem re-treino completo, especialmente útil com SGDClassifier para cenários de drift gradual.

## 🛠️ Tecnologias Utilizadas

| Biblioteca | Versão | Uso |
|------------|--------|-----|
| `numpy` | ≥1.24.0 | Operações numéricas e geração de dados sintéticos |
| `pandas` | ≥2.0.0 | Manipulação de DataFrames |
| `scikit-learn` | ≥1.3.0 | SGDClassifier, métricas, pré-processamento |
| `scipy` | ≥1.11.0 | Teste Kolmogorov-Smirnov (`ks_2samp`) |
| `matplotlib` | ≥3.7.0 | Visualizações e gráficos |
| `seaborn` | ≥0.12.0 | Gráficos estatísticos |
| `jupyter` | ≥1.0.0 | Notebooks interativos |
| `pytest` | ≥7.4.0 | Testes unitários |

## 📂 Estrutura de Arquivos

```
01_fundamentos_do_data_drift_em_ml/
├── README.md                          # Este arquivo
├── requirements.txt                   # Dependências da aula
├── notebooks/
│   ├── 01_exploracao.ipynb            # EDA e visualização de drift
│   ├── 02_treinamento.ipynb           # Treinamento e detecção de drift
│   └── 03_avaliacao.ipynb             # Avaliação e aprendizado incremental
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py          # Geração e pré-processamento de dados
│   ├── model.py                       # DriftDetector e classificador adaptativo
│   ├── training.py                    # Lógica de treinamento e validação cruzada
│   ├── evaluation.py                  # Métricas e visualizações de avaliação
│   └── utils.py                       # Utilitários (save/load, configuração)
├── scripts/
│   └── generate_dataset.py            # Script para gerar dataset sintético
├── data/
│   ├── raw/                           # Dados brutos gerados
│   │   └── dataset.csv
│   ├── processed/                     # Dados processados
│   └── README.md                      # Documentação dos dados
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py          # Testes do pré-processamento
│   ├── test_model.py                  # Testes do modelo
│   └── test_evaluation.py             # Testes de avaliação
└── outputs/
    ├── models/                        # Modelos salvos (.joblib)
    ├── figures/                       # Gráficos gerados (.png)
    └── logs/                          # Logs de execução
```

## 🚀 Como Executar

### 1. Instalar dependências (via uv)

```bash
cd mlet_data-drift
uv sync               # instala dependências do pyproject.toml
uv sync --extra dev   # inclui pytest, ruff, taskipy
```

### 2. Gerar dataset sintético

```bash
uv run python 01_fundamentos_do_data_drift_em_ml/scripts/generate_dataset.py
```

### 3. Executar notebooks (sequencialmente)

```bash
uv run jupyter lab notebooks/01_exploracao.ipynb
uv run jupyter lab notebooks/02_treinamento.ipynb
uv run jupyter lab notebooks/03_avaliacao.ipynb
```

### 4. Executar via scripts Python

```bash
uv run python -m src.data_preprocessing
uv run python -m src.training
uv run python -m src.evaluation
```

## 📊 Datasets

O dataset sintético simula o cenário motivador da loja online descrito no documento da aula:
- **10.000 instâncias** divididas em 5 períodos temporais
- **10 features** simulando atributos de clientes e comportamento de compra
- **Drift injetado** nos períodos 3–5 (mudança gradual na distribuição)
- Detalhes completos em [`data/README.md`](data/README.md)

## 🧪 Testes

```bash
# Executar todos os testes
uv run pytest tests/ -v

# Executar testes específicos
uv run pytest tests/test_preprocessing.py -v
uv run pytest tests/test_model.py -v
uv run pytest tests/test_evaluation.py -v
```

## 📚 Referências

1. **Widmer, G. & Kubat, M. (1996)**. *Learning in the presence of concept drift and hidden contexts*. Machine Learning, 23(1), 69–101.
2. **Gama, J. et al. (2014)**. *A Survey on Concept Drift Adaptation*. ACM Computing Surveys, 46(4), 44.
3. **Lu, J. et al. (2018)**. *Learning under Concept Drift: A Review*. IEEE TKDE, 31(12), 2346–2363.
4. **Sculley, D. et al. (2015)**. *Hidden Technical Debt in Machine Learning Systems*. NeurIPS 28.
5. **Moreno-Torres, J.G. et al. (2012)**. *A unifying view on dataset shift in classification*. Pattern Recognition, 45(1), 521–530.
6. **Pang, G. et al. (2023)**. *AI Aging: Quantifying Temporal Degradation of ML Models*. Nature Communications, 14(1), 2165.
7. **Huyen, C. (2022)**. *Data Distribution Shifts and Monitoring*. Stanford CS 329S.

## 🎥 Vídeos Relacionados

| Vídeo | Título | Duração | Tópicos |
|-------|--------|---------|---------|
| 1 | Fundamentos do Data Drift em ML | 15 min | Data Drift, Concept Drift, hipótese IID, P(X,Y) |
| 2 | Padrões e Consequências do Data Drift | 18 min | Drift abrupto/gradual/sazonal, causas, consequências |
| 3 | Detectando Drift na Prática com Python | 20 min | Teste KS, histogramas, simulação, p-values |
| 4 | Aprendizado Incremental e Resposta ao Drift | 18 min | partial_fit, SGDClassifier, gatilhos, MLOps |

## 🔗 Links Úteis

- [scipy.stats.ks_2samp — Documentação](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html)
- [SGDClassifier — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
- [River — Online ML Framework](https://riverml.xyz/)
- [Evidently AI — Monitoramento de Modelos](https://www.evidentlyai.com/)
- [Chip Huyen — Data Distribution Shifts](https://huyenchip.com/ml-monitors)
