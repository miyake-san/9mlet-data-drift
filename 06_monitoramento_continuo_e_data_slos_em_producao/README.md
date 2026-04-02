# Aula 6 — Monitoramento Contínuo e Data SLOs em Produção

## 🎯 Objetivos de Aprendizagem

Ao final desta aula, o estudante será capaz de:

1. **Compreender** por que modelos de ML em produção necessitam de monitoramento contínuo e quais riscos existem sem supervisão ativa.
2. **Definir** Service Level Indicators (SLIs) e Data SLOs para sistemas de ML, incluindo métricas de desempenho, drift e qualidade de dados.
3. **Implementar** um pipeline de monitoramento em Python que verifica acurácia, detecta drift via teste Kolmogorov-Smirnov e valida qualidade de dados.
4. **Calcular** métricas avançadas como PSI (Population Stability Index) e Brier Score para quantificar drift e calibração.
5. **Conhecer** ferramentas de MLOps (Evidently, Great Expectations, Prometheus/Grafana) para monitoramento em escala.
6. **Analisar** casos reais (Zillow, LinkedIn, Uber) para entender impactos de falhas e benefícios do monitoramento proativo.

## 📖 Teoria-Chave

### Data Drift e Concept Drift

- **Data Drift**: mudança nas distribuições das features de entrada ao longo do tempo.
- **Concept Drift**: mudança na relação entre features e target (Gama et al., 2014).
- Métricas: **PSI** (Population Stability Index), teste **Kolmogorov-Smirnov**, distância de Wasserstein.

### SLIs e Data SLOs

- **SLI** (Service Level Indicator): grandeza que mede qualidade (acurácia, latência, taxa de erro).
- **SLO** (Service Level Objective): meta aceitável para o SLI (e.g., acurácia ≥ 85%).
- **Data SLOs**: SLOs aplicados à qualidade dos dados — completude, acurácia, consistência, validade, atualidade.
- **Error Budget**: margem de tolerância entre SLO e estado atual.

### Brier Score (Calibração)

$$
BS = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2
$$

### Population Stability Index (PSI)

$$
PSI = \sum_{i=1}^{n} (P_i - Q_i) \ln\!\left(\frac{P_i}{Q_i}\right)
$$

- PSI < 0.10 → sem mudança | 0.10–0.25 → moderada | > 0.25 → severa.

## 🛠️ Tecnologias Utilizadas

| Biblioteca | Versão | Uso |
|---|---|---|
| Python | ≥ 3.10 | Linguagem base |
| NumPy | ≥ 1.24.0 | Operações numéricas |
| Pandas | ≥ 2.0.0 | Manipulação de dados |
| scikit-learn | ≥ 1.3.0 | Modelagem e métricas |
| SciPy | ≥ 1.11.0 | Testes estatísticos (KS) |
| Matplotlib | ≥ 3.7.0 | Visualizações |
| Seaborn | ≥ 0.12.0 | Gráficos estatísticos |
| Jupyter | ≥ 1.0.0 | Notebooks interativos |
| pytest | ≥ 7.4.0 | Testes unitários |

## 📂 Estrutura de Arquivos

```
06_monitoramento_continuo_e_data_slos_em_producao/
├── README.md                           # Este arquivo
├── requirements.txt                    # Dependências
├── generate_dataset.py                 # Script para gerar dataset sintético
├── notebooks/
│   ├── 01_exploracao.ipynb             # EDA e análise de drift nos dados
│   ├── 02_treinamento.ipynb            # Treinamento do modelo e SLOs
│   └── 03_avaliacao.ipynb              # Pipeline de monitoramento e avaliação
├── src/
│   ├── __init__.py                     # Exports do pacote
│   ├── data_preprocessing.py           # Geração e pré-processamento de dados
│   ├── model.py                        # SLOMonitor e MonitoringPipeline
│   ├── training.py                     # Treinamento do modelo de crédito
│   ├── evaluation.py                   # Métricas de avaliação e Brier Score
│   └── utils.py                        # Utilitários (save/load, config, logging)
├── data/
│   ├── raw/
│   │   └── dataset.csv                 # Dataset sintético de crédito digital
│   ├── processed/                      # Dados processados
│   └── README.md                       # Documentação dos dados
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py           # Testes do DataPreprocessor
│   ├── test_model.py                   # Testes do SLOMonitor e Pipeline
│   └── test_evaluation.py             # Testes das métricas de avaliação
└── outputs/
    ├── models/                         # Modelos treinados (.joblib)
    ├── figures/                        # Gráficos gerados
    └── logs/                           # Logs de monitoramento
```

## 🚀 Como Executar

### 1. Instalar Dependências

```bash
cd 06_monitoramento_continuo_e_data_slos_em_producao
pip install -r requirements.txt
```

### 2. Gerar Dataset

```bash
python generate_dataset.py
```

### 3. Executar Notebooks (sequencialmente)

```bash
jupyter notebook notebooks/01_exploracao.ipynb
jupyter notebook notebooks/02_treinamento.ipynb
jupyter notebook notebooks/03_avaliacao.ipynb
```

### 4. Executar via Scripts Python

```bash
python -m src.training        # Treina modelo
python -m src.evaluation      # Avalia modelo e gera relatórios
```

## 📊 Datasets

O dataset sintético simula um cenário de **fintech de crédito digital** com drift, conforme descrito no documento da aula. Veja detalhes completos em [data/README.md](data/README.md).

- **10.000 instâncias** (5.000 referência + 5.000 produção)
- **10 features** numéricas + flag de produção + target
- Drift introduzido em `idade`, `renda_mensal` e `score_credito`

## 🧪 Testes

```bash
# Executar todos os testes
pytest tests/ -v

# Executar com cobertura
pytest tests/ -v --tb=short
```

## 📚 Referências

- **Breck, E. et al. (2017).** The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction. IEEE BigData.
- **Gama, J. et al. (2014).** A survey on concept drift adaptation. ACM Computing Surveys, 46(4).
- **Naveed, H. et al. (2025).** Monitoring Machine Learning Systems: A Multivocal Literature Review. arXiv:2509.14294.
- **Ovadia, Y. et al. (2019).** Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift. NeurIPS 2019.
- **Sculley, D. et al. (2015).** Hidden Technical Debt in Machine Learning Systems. NIPS 2015.
- **Xu, H. et al. (2023).** Deep Isolation Forest for Anomaly Detection. IEEE TKDE.

## 🎥 Vídeos Relacionados

| Vídeo | Título | Tópicos |
|-------|--------|---------|
| 6.1 | Monitoramento Contínuo e Data SLOs em Produção | Por que monitorar, conceito de SLOs/SLIs, caso fintech |
| 6.2 | Arquitetura de Monitoramento e Definição de SLOs | Definição de SLOs, Error Budget, dashboards Grafana/Prometheus |
| 6.3 | Pipeline de Monitoramento com SLOs em Python | Acurácia + KS + qualidade em pipeline integrado |
| 6.4 | Ferramentas de MLOps para Monitoramento Contínuo | Evidently, Great Expectations, AWS/Azure, CI/CD |

## 🔗 Links Úteis

- [Evidently AI — Documentação](https://docs.evidentlyai.com/)
- [Great Expectations — Docs](https://docs.greatexpectations.io/)
- [Prometheus — Monitoring](https://prometheus.io/docs/)
- [Grafana — Dashboards](https://grafana.com/docs/)
- [scikit-learn — Métricas](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [SciPy — ks_2samp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html)
