# 🐳 Stack de Monitoramento — Aula 6 (Prometheus + Grafana + Drift Exporter)

Complemento prático da **Aula 6 — Monitoramento Contínuo e Data SLOs em
Produção**. Esta stack sobe, com um único `docker compose up`, um ambiente
completo de observabilidade para o modelo de **crédito digital** estudado no
notebook [`notebooks/03_avaliacao.ipynb`](../notebooks/03_avaliacao.ipynb),
permitindo visualizar **antes e depois** da chegada de drift em tempo real.

## 🧱 Componentes

| Serviço          | Imagem                  | Porta host | Função                                                                                  |
| ---------------- | ----------------------- | ---------- | --------------------------------------------------------------------------------------- |
| `drift-exporter` | build local (Python 3.11) | `8000`   | App Python que treina o modelo, simula batches pré/pós-drift e expõe `/metrics`         |
| `prometheus`     | `prom/prometheus:v2.54.1` | `9090`   | Coleta as métricas do exporter a cada 5s (TSDB local)                                   |
| `grafana`        | `grafana/grafana:11.2.0`  | `3000`   | Dashboard pré-provisionado ("Aula 6 — ML Monitoring (Pre/Pos Drift)")                  |

A rede `monitoring` (bridge) é criada automaticamente pelo Compose e conecta
os três serviços por DNS interno (`drift-exporter`, `prometheus`, `grafana`).

## 📁 Estrutura

```
monitoring_stack/
├── docker-compose.yml              # Orquestração dos 3 serviços
├── README.md                       # Este arquivo
├── app/
│   ├── Dockerfile                  # Imagem do exporter (reaproveita src/ da aula)
│   ├── requirements.txt            # Dependências (inclui prometheus-client)
│   └── drift_exporter.py           # App Python que publica as métricas
├── prometheus/
│   └── prometheus.yml              # Config de scrape (5s)
└── grafana/
    ├── provisioning/
    │   ├── datasources/datasource.yml   # Datasource Prometheus auto-configurada
    │   └── dashboards/dashboards.yml    # Loader de dashboards
    └── dashboards/
        └── ml_monitoring_dashboard.json # Dashboard pronto da aula
```

## 🧠 Como o exporter funciona (resumo)

O [`drift_exporter.py`](./app/drift_exporter.py) **reaproveita exatamente o
código de `src/`** (mesma fonte usada pelo notebook):

1. Gera/lê o dataset sintético com `DataPreprocessor.generate_dataset()`.
2. Treina um modelo de regressão logística com `CreditModelTrainer` usando
   **apenas a partição de referência** (`is_production == 0`).
3. Em loop, alterna duas fases de simulação para mostrar o efeito do drift:
   - **`pre_drift`** — amostras retiradas da partição de referência → o
     modelo opera dentro do SLO (acurácia alta, KS p-valor alto, PSI baixo).
   - **`pos_drift`** — amostras retiradas da partição de produção (com drift
     em `idade`, `renda_mensal`, `score_credito`) → o `MonitoringPipeline`
     detecta queda de acurácia, p-valor KS abaixo do threshold e PSI > 0.25.
4. A cada iteração executa `MonitoringPipeline.run_full_check(...)`
   (os 3 Snippets da aula) e publica todas as métricas via `prometheus_client`.

### Métricas expostas (`/metrics`)

| Métrica                          | Tipo    | Labels      | Origem                                |
| -------------------------------- | ------- | ----------- | ------------------------------------- |
| `ml_phase`                       | Gauge   | —           | 0 = pre_drift, 1 = pos_drift          |
| `ml_overall_status`              | Gauge   | —           | 1 = HEALTHY, 0 = CRITICAL             |
| `ml_model_accuracy`              | Gauge   | —           | Snippet 1 — `accuracy_score`           |
| `ml_model_brier_score`           | Gauge   | —           | `brier_score_loss`                    |
| `ml_drift_ks_pvalue`             | Gauge   | `feature`   | Snippet 2 — `ks_2samp`                |
| `ml_drift_psi`                   | Gauge   | `feature`   | `SLOMonitor.calculate_psi`            |
| `ml_data_missing_rate`           | Gauge   | `feature`   | Snippet 3 — taxa de missings          |
| `ml_slo_check_status`            | Gauge   | `check`     | Cada `SLOCheckResult.passed` (1/0)    |
| `ml_slo_checks_passed/failed`    | Gauge   | —           | Contagem por batch                    |
| `ml_slo_alerts_total`            | Counter | `check`     | Total acumulado de violações          |
| `ml_batches_processed_total`     | Counter | `phase`     | Lotes processados                     |
| `ml_slo_*_threshold`             | Gauge   | —           | Thresholds dos SLOs (overlay)         |

## ⚙️ Variáveis de ambiente do exporter

Configuráveis no `docker-compose.yml` (seção `drift-exporter.environment`):

| Variável                  | Default | Descrição                                                         |
| ------------------------- | ------- | ----------------------------------------------------------------- |
| `SCRAPE_INTERVAL_SECONDS` | `10`    | Segundos entre cada batch / atualização de métricas               |
| `ITERATIONS_PER_PHASE`    | `12`    | Quantas iterações ficam em cada fase antes de alternar pre↔pos    |
| `BATCH_SIZE`              | `500`   | Tamanho do batch de produção amostrado a cada iteração            |
| `EXPORTER_PORT`           | `8000`  | Porta HTTP onde o `/metrics` é publicado                          |
| `ACCURACY_SLO`            | `0.85`  | SLO mínimo de acurácia                                            |
| `DRIFT_P_VALUE_THRESHOLD` | `0.05`  | p-valor mínimo do teste KS                                        |
| `MISSING_RATE_SLO`        | `0.01`  | Taxa máxima de missings por coluna                                |
| `PSI_THRESHOLD`           | `0.25`  | PSI acima do qual o check é considerado violado                   |

Com os defaults, **cada ciclo completo (pre → pos → pre)** leva
`2 × 12 × 10s ≈ 4 minutos`, suficiente para a demo em sala.

---

## 🚀 Passo a passo

> **Pré-requisitos**: Docker Desktop (Windows/Mac) ou Docker Engine + Docker
> Compose v2 (Linux). Recomenda-se ≥ 4 GB de RAM livres.

### Passo 1 — Entrar na pasta da stack

```powershell
cd 06_monitoramento_continuo_e_data_slos_em_producao\monitoring_stack
```

> ⚠️ **Importante**: execute o Compose **a partir desta pasta**. O Dockerfile
> usa `context: ..` (a pasta da Aula 6) para conseguir copiar o pacote
> `src/` original — o mesmo importado pelos notebooks. Isso garante que o
> exporter use exatamente a mesma lógica de `SLOMonitor` e `MonitoringPipeline`
> apresentada na aula.

### Passo 2 — Subir os serviços

```powershell
docker compose up -d --build
```

O que acontece:

1. **Build da imagem `aula6/drift-exporter`** (~1 a 2 min na primeira vez): 
   Python 3.11-slim + dependências de ML (numpy, pandas, scikit-learn, scipy)
   + `prometheus_client`.
2. **Download** das imagens oficiais `prom/prometheus:v2.54.1` e
   `grafana/grafana:11.2.0`.
3. **Start dos três containers** na rede `monitoring`. O Compose aguarda o
   healthcheck do exporter (`wget /metrics`) antes de marcar o Prometheus
   como dependente saudável.

Acompanhe os logs do exporter para ver o treino inicial e o início do loop:

```powershell
docker compose logs -f drift-exporter
```

Deve aparecer algo como:

```
... drift_exporter :: Subindo HTTP server Prometheus em :8000/metrics
... drift_exporter :: Dataset não encontrado em /app/data/raw/dataset.csv — gerando...
... drift_exporter :: Modelo de referência treinado: logistic
... drift_exporter :: iter=0 phase=pre_drift status=HEALTHY  passed=16 failed=0
... drift_exporter :: iter=12 phase=pos_drift status=CRITICAL passed=8  failed=8
```

### Passo 3 — Validar que o Prometheus está coletando

Abra <http://localhost:9090/targets> e confirme que o target
**`drift-exporter`** aparece com estado **UP** (cor verde).

Faça uma query de teste em <http://localhost:9090/graph>, por exemplo:

```promql
ml_model_accuracy
```

Você deve ver uma série temporal já se desenhando.

### Passo 4 — Abrir o Grafana

Acesse <http://localhost:3000>:

- **Usuário/senha**: `admin` / `admin` (não pedirá troca; também há acesso
  anônimo como Viewer, por isso o dashboard abre direto).
- A **datasource Prometheus** já está provisionada (UID `prometheus`,
  URL interna `http://prometheus:9090`).
- O **dashboard "Aula 6 — ML Monitoring (Pre/Pos Drift)"** é carregado
  automaticamente como dashboard padrão (pasta *Aula 6*).

### Passo 5 — Acompanhar a transição pré → pós drift

No dashboard você verá, em tempo real:

| Painel                                         | O que observar                                                                                 |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Fase Atual da Simulação**                    | Alterna entre **PRE-DRIFT** (verde) e **POS-DRIFT** (vermelho) a cada `ITERATIONS_PER_PHASE`   |
| **Status Geral**                               | Vai de HEALTHY para CRITICAL assim que entra na fase pos-drift                                 |
| **Acurácia do Modelo vs SLO**                  | Linha de acurácia cai abaixo da linha tracejada do SLO durante a fase pos-drift                |
| **Brier Score**                                | Aumenta na fase pos-drift (calibração se deteriora — Ovadia et al., 2019)                      |
| **PSI por Feature**                            | `idade`, `renda_mensal` e `score_credito` cruzam o threshold 0.25 → drift severo               |
| **p-valor KS por Feature**                     | p-valores despencam abaixo de 0.05 nas features afetadas                                       |
| **Status dos Checks de SLO (heatmap)**         | Mostra exatamente quais checks (accuracy, drift_ks_<f>, psi_<f>, missing_rate_<f>) violam      |
| **Taxa de Missings por Feature vs SLO**        | Sobe na fase pos-drift (~2% vs ~0.5% na referência), violando o SLO de 1%                      |
| **Alertas por tipo de check (acumulado)**      | Counter cresce; ajuda a priorizar quais SLOs falham mais                                       |

### Passo 6 — Explorar e experimentar

Sugestões de manipulação ao vivo na aula:

- **Forçar mais drift mais rápido**: edite `docker-compose.yml`,
  `ITERATIONS_PER_PHASE: "4"` e rode `docker compose up -d` (sobe só o que
  mudou).
- **Apertar o SLO de acurácia** para mostrar violação mesmo em pre-drift:
  `ACCURACY_SLO: "0.92"`.
- **Recarregar config do Prometheus** sem reiniciar:
  `curl -X POST http://localhost:9090/-/reload`.
- **Inspecionar métricas raw**: <http://localhost:8000/metrics>.

### Passo 7 — Parar e limpar

```powershell
# Para os containers mantendo volumes (histórico do Prometheus preservado)
docker compose down

# Limpeza completa (apaga TSDB do Prometheus e dados do Grafana)
docker compose down -v
```

---

## 🧪 Validação local sem Docker (opcional)

Para depurar o exporter sem precisar buildar a imagem:

```powershell
cd 06_monitoramento_continuo_e_data_slos_em_producao
pip install -r requirements.txt
pip install prometheus-client
python monitoring_stack\app\drift_exporter.py
```

Em seguida, abra <http://localhost:8000/metrics>.

## 🩺 Troubleshooting

| Sintoma                                                 | Causa provável / solução                                                                                                |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `drift-exporter` reiniciando em loop                    | Veja `docker compose logs drift-exporter`. Em geral falta de RAM no Docker Desktop ou erro de import (path do `src/`).  |
| Prometheus target DOWN                                  | Confirme com `docker compose ps` que `drift-exporter` está `healthy`. A primeira leitura demora ~30s (treino do modelo). |
| Dashboard vazio no Grafana                              | Aguarde 30–60s após o start; o intervalo de scrape é 5s, mas o exporter espera o primeiro batch antes de publicar.      |
| Porta 3000/9090/8000 já em uso                          | Altere a porta em `docker-compose.yml`, ex. `"3001:3000"` para o Grafana.                                               |
| Mudou `src/model.py` mas o exporter não reflete         | Faça `docker compose build drift-exporter && docker compose up -d drift-exporter` para reconstruir a imagem.            |

## 🔗 Referências

- [Prometheus — Best practices on instrumentation](https://prometheus.io/docs/practices/instrumentation/)
- [`prometheus_client` (Python)](https://github.com/prometheus/client_python)
- [Grafana — Provisioning datasources & dashboards](https://grafana.com/docs/grafana/latest/administration/provisioning/)
- Notebook da aula: [`notebooks/03_avaliacao.ipynb`](../notebooks/03_avaliacao.ipynb)
- Código reutilizado: [`src/model.py`](../src/model.py), [`src/data_preprocessing.py`](../src/data_preprocessing.py), [`src/training.py`](../src/training.py)
