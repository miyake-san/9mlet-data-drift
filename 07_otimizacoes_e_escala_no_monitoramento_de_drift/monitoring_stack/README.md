# 🐳 Stack de Monitoramento — Aula 7 (Streaming Kappa + Drift em Tempo Real)

Complemento prático da **Aula 7 — Otimizações e Escala no Monitoramento de
Drift**. Inspirada na stack da Aula 6, esta versão evolui o cenário para uma
arquitetura de **streaming (Kappa)** com a *tech stack* da fintech fictícia
**FinBank**: um **producer** gera transações de forma contínua e aleatória,
publica em um **broker (Redis Streams)**, e um **consumer** avalia drift em
**janelas deslizantes** expondo métricas para **Prometheus + Grafana**.

Com um único `docker compose up`, sobe um ambiente completo de observabilidade
que mostra, em tempo real, a transição entre períodos **sem drift** e períodos
**com drift** (gradual e abrupto).

## 🧱 Componentes

| Serviço          | Imagem                     | Porta host | Função                                                                          |
| ---------------- | -------------------------- | ---------- | ------------------------------------------------------------------------------- |
| `redis`          | `redis:7.4-alpine`         | `6379`     | Message broker (Redis Streams) — camada de ingestão                             |
| `producer`       | build local (Python 3.11)  | —          | Gera transações FinBank alternando regimes **sem/com drift** e publica no stream |
| `drift-consumer` | build local (Python 3.11)  | `8000`     | Consome o stream em janela deslizante, calcula PSI/KS/Wasserstein e expõe `/metrics` |
| `prometheus`     | `prom/prometheus:v2.54.1`  | `9090`     | Coleta as métricas do consumer a cada 5s (TSDB local)                           |
| `grafana`        | `grafana/grafana:11.2.0`   | `3000`     | Dashboard pré-provisionado ("Aula 7 — FinBank Drift Streaming (Kappa)")          |

A rede `monitoring` (bridge) conecta todos os serviços por DNS interno
(`redis`, `producer`, `drift-consumer`, `prometheus`, `grafana`).

## 🏗️ Arquitetura (Kappa)

```
┌──────────┐   XADD    ┌───────────────┐  XREADGROUP  ┌─────────────────┐  /metrics  ┌────────────┐   ┌─────────┐
│ producer │ ───────▶ │ redis (stream) │ ───────────▶ │ drift-consumer  │ ─────────▶ │ prometheus │ ─▶│ grafana │
│ (regimes)│           │ finbank:txns   │              │ janela + PSI/KS │            │   (TSDB)   │   │ (dash)  │
└──────────┘           └───────────────┘              └─────────────────┘            └────────────┘   └─────────┘
   sem/com drift          broker / fila                 speed + serving layer
```

Um único caminho de streaming unificado (sem camada batch separada) processa o
fluxo em janelas e produz sinais de drift de baixa latência — a essência da
arquitetura **Kappa** discutida na Aula 7 (Carbone et al., 2015).

## 📁 Estrutura

```
monitoring_stack/
├── docker-compose.yml                   # Orquestração dos 5 serviços
├── README.md                            # Este arquivo
├── app/
│   ├── Dockerfile                       # Imagem compartilhada (producer + consumer)
│   ├── requirements.txt                 # Dependências (redis, prometheus-client, scipy...)
│   ├── transaction_generator.py         # Gerador FinBank por regime (sem/com drift)
│   ├── producer.py                      # Publica transações aleatórias no stream
│   └── drift_consumer.py                # Consome, calcula drift e expõe /metrics
├── prometheus/
│   └── prometheus.yml                   # Config de scrape (5s)
└── grafana/
    ├── provisioning/
    │   ├── datasources/datasource.yml   # Datasource Prometheus auto-configurada
    │   └── dashboards/dashboards.yml    # Loader de dashboards
    └── dashboards/
        └── drift_streaming_dashboard.json # Dashboard pronto da aula
```

## 🎲 Como o producer gera dados (sem/com drift)

O [`producer.py`](./app/producer.py) usa o
[`transaction_generator.py`](./app/transaction_generator.py) para gerar
transações de forma **estocástica e com persistência** entre três regimes que
reproduzem o caso FinBank de `scripts/generate_dataset.py`:

| Regime     | Código | Significado     | O que muda                                                              |
| ---------- | :----: | --------------- | ---------------------------------------------------------------------- |
| `baseline` |   0    | **sem drift**   | Distribuição estável (meses 1–3 do dataset original)                   |
| `gradual`  |   1    | drift gradual   | Deslocamento progressivo em `amount`, `customer_age`, `channel`, etc.  |
| `abrupt`   |   2    | drift abrupto   | Mudança forte em múltiplas features + transações noturnas + mais fraude |

A cada lote, com probabilidade `REGIME_SWITCH_PROB` o producer **sorteia** um
novo regime (pesos: 50% baseline, 30% gradual, 20% abrupto); no regime gradual,
a intensidade do drift (`drift_factor`) também é aleatória em `[0.3, 1.0]`. Isso
simula janelas saudáveis intercaladas com surtos de drift — o cenário realista
que exige **monitoramento contínuo** (Gama et al., 2014).

O regime corrente é propagado em cada mensagem e exposto como
`finbank_producer_regime`, servindo de **ground truth** para comparar, no
dashboard, o drift **real** (injetado) com o drift **detectado** (PSI/KS).

## 🧠 Como o consumer detecta drift

O [`drift_consumer.py`](./app/drift_consumer.py) **reaproveita o `DriftMonitor`
de `src/model.py`** (mesma classe usada nos notebooks):

1. No start, gera um **baseline estável** (`BASELINE_SIZE` amostras do regime
   `baseline`) e ajusta o `DriftMonitor` via `fit()`.
2. Consome o stream com `XREADGROUP` (consumer group) e mantém uma **janela
   deslizante** (`deque` de `WINDOW_SIZE` transações).
3. A cada `SCRAPE_INTERVAL_SECONDS`, compara a janela corrente com o baseline,
   calculando **PSI**, teste **K–S** e distância de **Wasserstein** por feature.
4. Publica todas as métricas via `prometheus_client` em `/metrics`.

### Métricas expostas (`/metrics`)

| Métrica                              | Tipo    | Labels     | Origem                                       |
| ------------------------------------ | ------- | ---------- | -------------------------------------------- |
| `finbank_producer_regime`            | Gauge   | —          | 0 = baseline, 1 = gradual, 2 = abrupto       |
| `finbank_producer_regime_active`     | Gauge   | `regime`   | Indicador booleano por regime                |
| `finbank_drift_psi`                  | Gauge   | `feature`  | `DriftMonitor.compute_psi`                   |
| `finbank_drift_ks_pvalue`            | Gauge   | `feature`  | `DriftMonitor.compute_ks` (p-valor)          |
| `finbank_drift_ks_statistic`         | Gauge   | `feature`  | `DriftMonitor.compute_ks` (estatística D)    |
| `finbank_drift_wasserstein`          | Gauge   | `feature`  | `DriftMonitor.compute_wasserstein`           |
| `finbank_drift_detected`             | Gauge   | `feature`  | 1 = drift detectado, 0 = ok                  |
| `finbank_mean_psi` / `finbank_max_psi` | Gauge | —          | Agregados de PSI (`DriftMonitor.score`)      |
| `finbank_drift_fraction`             | Gauge   | —          | Fração de features com drift                 |
| `finbank_features_with_drift`        | Gauge   | —          | Nº de features com drift                     |
| `finbank_window_size`                | Gauge   | —          | Transações na janela deslizante              |
| `finbank_window_fraud_rate`          | Gauge   | —          | Taxa de fraude observada na janela           |
| `finbank_transactions_consumed_total`| Counter | —          | Total de transações consumidas               |
| `finbank_messages_consumed_total`    | Counter | —          | Total de lotes consumidos                    |
| `finbank_drift_alerts_total`         | Counter | `feature`  | Total acumulado de alertas de drift          |
| `finbank_psi_threshold` / `_warning` | Gauge   | —          | Thresholds de PSI (overlay)                  |
| `finbank_ks_alpha`                   | Gauge   | —          | Significância do teste K–S (overlay)         |

## ⚙️ Variáveis de ambiente

### `producer` (seção `producer.environment`)

| Variável                   | Default               | Descrição                                              |
| -------------------------- | --------------------- | ------------------------------------------------------ |
| `REDIS_HOST` / `REDIS_PORT`| `redis` / `6379`      | Endereço do broker                                     |
| `STREAM_KEY`               | `finbank:transactions`| Nome do Redis Stream                                   |
| `STREAM_MAXLEN`            | `50000`               | Tamanho máximo aproximado do stream                    |
| `PRODUCE_INTERVAL_SECONDS` | `2`                   | Intervalo entre lotes publicados                       |
| `BATCH_SIZE`               | `200`                 | Transações por lote                                    |
| `REGIME_SWITCH_PROB`       | `0.15`                | Probabilidade de troca de regime por lote              |
| `SEED`                     | `2025`                | Semente do gerador                                     |

### `drift-consumer` (seção `drift-consumer.environment`)

| Variável                   | Default               | Descrição                                              |
| -------------------------- | --------------------- | ------------------------------------------------------ |
| `WINDOW_SIZE`              | `2000`                | Tamanho da janela deslizante                           |
| `BASELINE_SIZE`            | `5000`                | Tamanho da amostra de referência                       |
| `SCRAPE_INTERVAL_SECONDS`  | `5`                   | Intervalo entre avaliações de drift                    |
| `EXPORTER_PORT`            | `8000`                | Porta do endpoint `/metrics`                           |
| `PSI_THRESHOLD`            | `0.25`                | Limiar de PSI para drift crítico                       |
| `PSI_WARNING`              | `0.10`                | Limiar de PSI para warning                             |
| `KS_ALPHA`                 | `0.05`                | Significância do teste K–S                             |

## 🚀 Como executar

A partir da pasta `07_otimizacoes_e_escala_no_monitoramento_de_drift`:

```bash
cd monitoring_stack
docker compose up --build
```

Em seguida, acesse:

| Serviço     | URL                                   | Credenciais |
| ----------- | ------------------------------------- | ----------- |
| Grafana     | http://localhost:3000                 | admin/admin |
| Prometheus  | http://localhost:9090                 | —           |
| Métricas    | http://localhost:8000/metrics         | —           |

O dashboard **"Aula 7 — FinBank Drift Streaming (Kappa)"** já abre como home.
Observe o painel **"Regime do Producer"** (drift real injetado) e compare com
o **PSI por feature** e os **alertas de drift** (drift detectado).

Para encerrar:

```bash
docker compose down -v
```

## 📚 Referências

- Carbone, P. et al. (2015). *Apache Flink: Stream and Batch Processing.*
- Gama, J. et al. (2014). *A Survey on Concept Drift Adaptation.* ACM.
- Kullback, S. & Leibler, R. A. (1951). *On Information and Sufficiency.*
- Kolmogorov, A. (1933); Smirnov, N. (1948). *Teste K–S.*
- Sculley, D. et al. (2015). *Hidden Technical Debt in ML Systems.* NeurIPS.
