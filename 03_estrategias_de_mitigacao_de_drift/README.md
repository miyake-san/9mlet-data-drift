# Aula 03 — Estratégias de Mitigação de Drift

## 🎯 Objetivos de Aprendizagem

Ao final desta aula, o aluno será capaz de:

1. **Distinguir** estratégias reativas e proativas de mitigação de drift em sistemas de ML em produção.
2. **Implementar** re-treinamento com janela deslizante e validação segmentada por subgrupos críticos.
3. **Aplicar** aprendizado incremental com ADWIN (River) para resposta contínua a mudanças de distribuição.
4. **Construir** ensembles adaptativos (ADWINBoostingClassifier) que absorvem drift sem interromper a operação.
5. **Projetar** políticas de resposta em camadas — emergencial, adaptativa e estrutural — proporcionais à velocidade da mudança e ao custo do erro.

## 📖 Teoria-Chave

### Ciclo de Controle Adaptativo

Mitigar drift é uma decisão de arquitetura, não um detalhe operacional. A mitigação deve ser entendida como um **ciclo de controle adaptativo**: monitorar, diagnosticar, intervir e revalidar (Moreno-Torres et al., 2012; Quionero-Candela et al., 2009).

### Camadas de Resposta

| Camada | Função |
|--------|--------|
| **Observabilidade** | Produzir sinais confiáveis de mudança |
| **Emergencial** | Ajuste de limiar, isolamento de anomalias, regras de negócio |
| **Adaptativa** | Aprendizado incremental, ensembles adaptativos |
| **Estrutural** | Re-treinamento, revisão de features, nova validação |

### Re-treinamento com Reponderação (Covariate Shift)

$$\hat{R}_{t}(f) \approx \frac{1}{n}\sum_{i=1}^{n}\frac{p_{t}(x_i)}{p_{s}(x_i)}\,\ell\bigl(f(x_i), y_i\bigr)$$

### Atualização Incremental (Online Learning)

$$\theta_{t+1} = \theta_t - \eta_t\,\nabla_{\theta}\,\ell\bigl(f_{\theta_t}(x_t), y_t\bigr)$$

### Limiar Ótimo Sensível a Custo

$$t^{\star} = \arg\min_t \left[C_{FN}\,\pi\,\bigl(1-\mathrm{TPR}(t)\bigr) + C_{FP}\,\bigl(1-\pi\bigr)\,\mathrm{FPR}(t)\right]$$

## 🛠️ Tecnologias Utilizadas

| Biblioteca | Versão | Uso |
|------------|--------|-----|
| Python | ≥3.10 | Linguagem principal |
| numpy | ≥1.26.0 | Operações numéricas |
| pandas | ≥2.2.0 | Manipulação de dados |
| scikit-learn | ≥1.5.0 | Modelos batch e métricas |
| river | ≥0.23.0 | Aprendizado incremental e detecção de drift |
| evidently | ≥0.7.0 | Monitoramento e relatórios de drift |
| matplotlib | ≥3.9.0 | Visualizações |
| seaborn | ≥0.13.0 | Visualizações estatísticas |
| pytest | ≥8.3.0 | Testes unitários |

## 📂 Estrutura de Arquivos

```
03_estrategias_de_mitigacao_de_drift/
├── README.md                          # Este arquivo
├── requirements.txt                   # Dependências específicas da aula
├── notebooks/
│   ├── 01_exploracao.ipynb            # EDA do dataset de churn + simulação de drift
│   ├── 02_treinamento.ipynb           # Re-treinamento, champion/challenger, online learning
│   └── 03_avaliacao.ipynb             # Ensembles adaptativos, métricas e comparação
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py          # Geração e pré-processamento do dataset de churn
│   ├── model.py                       # Modelos batch e online (incluindo ensembles adaptativos)
│   ├── training.py                    # Re-treinamento com janela deslizante e online learning
│   ├── evaluation.py                  # Métricas, curvas ROC e comparação de estratégias
│   └── utils.py                       # Utilitários: serialização, logging e configuração
├── data/
│   ├── raw/
│   │   └── dataset.csv                # Dataset sintético de churn telecom com drift
│   ├── processed/
│   └── README.md                      # Descrição detalhada do dataset
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py          # Testes do DataPreprocessor
│   ├── test_model.py                  # Testes dos modelos batch e online
│   └── test_evaluation.py             # Testes das funções de avaliação
└── outputs/
    ├── models/                        # Modelos salvos (.pkl)
    ├── figures/                        # Gráficos gerados
    └── logs/                          # Logs de execução
```

## 🚀 Como Executar

### 1. Instalação de dependências

```bash
cd 03_estrategias_de_mitigacao_de_drift
pip install -r requirements.txt
```

### 2. Gerar dataset sintético

```bash
python -c "from src.data_preprocessing import DataPreprocessor; DataPreprocessor().generate_and_save()"
```

### 3. Executar notebooks sequencialmente

```bash
jupyter lab notebooks/01_exploracao.ipynb
# Em seguida:
jupyter lab notebooks/02_treinamento.ipynb
jupyter lab notebooks/03_avaliacao.ipynb
```

### 4. Executar via scripts Python

```bash
python -m src.training
python -m src.evaluation
```

## 📊 Datasets

Dataset sintético de **churn em telecomunicações** com drift simulado, representando o cenário discutido no documento acadêmico: uma oferta agressiva do concorrente altera hábitos de consumo e desloca a taxa de cancelamento.

Detalhes completos em [`data/README.md`](data/README.md).

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

1. Bifet, A., & Gavaldà, R. (2007). *Learning from time-changing data with adaptive windowing*. SIAM SDM.
2. Cesa-Bianchi, N., & Lugosi, G. (2006). *Prediction, Learning, and Games*. Cambridge University Press.
3. Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Computing Surveys, 46*(4).
4. Gomes, H. M., et al. (2017). Adaptive random forests for evolving data stream classification. *Machine Learning, 106*(9-10), 1469-1495.
5. Krawczyk, B., et al. (2017). Ensemble learning for data stream analysis: A survey. *Information Fusion, 37*, 132-156.
6. Lu, J., et al. (2019). Learning under concept drift: A review. *IEEE TKDE, 31*(12), 2346-2363.
7. Montiel, J., et al. (2021). River: machine learning for streaming data in Python. *JMLR, 22*(110), 1-8.
8. Moreno-Torres, J. G., et al. (2012). A unifying view on dataset shift in classification. *Pattern Recognition, 45*(1).
9. Provost, F., & Fawcett, T. (2013). *Data Science for Business*. O'Reilly Media.
10. Sugiyama, M., & Kawanabe, M. (2012). *Machine Learning in Non-Stationary Environments*. MIT Press.

## 🎥 Vídeos Relacionados

| Vídeo | Título | Conteúdo |
|-------|--------|----------|
| 1 | Estratégias de Mitigação de Data Drift | Caso churn; reativas vs proativas; panorama de estratégias |
| 2 | Re-treinamento, Validação e Deploy Seguro | Janela deslizante; champion/challenger; canary release |
| 3 | Online Learning e Detecção Adaptativa com River | ADWIN; gatilho mínimo; monitoramento com Evidently |
| 4 | Ensembles Adaptativos para Absorver Drift | ADWINBoostingClassifier; camadas de resposta; comparação |

## 🔗 Links Úteis

- [River — Machine Learning for Streaming Data](https://riverml.xyz/)
- [River JMLR Paper](https://www.jmlr.org/papers/v22/20-1380.html)
- [Evidently AI — ML Monitoring](https://docs.evidentlyai.com/)
- [Concept Drift Detection Notebook (River)](https://github.com/online-ml/river/blob/main/docs/introduction/getting-started/concept-drift-detection.ipynb)
- [ADWINBoostingClassifier (River)](https://github.com/online-ml/river/blob/main/river/ensemble/boosting.py)
