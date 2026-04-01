# Diagrama de Arquitetura de Sistema

Este diagrama resume a arquitetura end-to-end recomendada para os experimentos e exemplos da disciplina. Ele mostra como dados de varias naturezas entram no sistema, sao validados, alimentam modelos e geram sinais de observabilidade para resposta operacional.

```mermaid
flowchart LR
    subgraph Sources["Fontes de dados"]
        Trx["Transacoes e eventos"]
        Txt["Textos e embeddings"]
        Sens["Sensores e telemetria"]
    end

    subgraph Platform["Plataforma de dados e ML"]
        Broker["Broker e ingestao"]
        Lake[("Data lake e storage")]
        Feature["Feature engineering"]
        Registry["Model registry e baseline"]
        Model["Modelos de deteccao e predicao"]
        Monitor["Monitoramento de drift"]
        Perf["Estimativa de performance"]
        Quality["Validacao de dados"]
        Orch["Orquestracao de pipelines"]
    end

    subgraph Operations["Operacoes"]
        Dash["Dashboards e relatorios"]
        Alert["Alertas"]
        Retrain["Re-treino e rollback"]
    end

    Trx --> Broker
    Txt --> Broker
    Sens --> Broker
    Broker --> Lake
    Lake --> Quality
    Quality --> Feature
    Feature --> Model
    Feature --> Registry
    Registry --> Model
    Model --> Monitor
    Model --> Perf
    Monitor --> Dash
    Perf --> Dash
    Monitor --> Alert
    Perf --> Alert
    Alert --> Retrain
    Retrain --> Registry
    Orch --> Broker
    Orch --> Quality
    Orch --> Monitor
    Orch --> Retrain
```

## Leituras do diagrama

- A observabilidade do modelo nao e um componente lateral; ela faz parte do fluxo central do sistema.
- Baseline e versionamento sao tratados como artefatos de primeira classe.
- Orquestracao, alertas e re-treino fecham o loop de aprendizagem continua.