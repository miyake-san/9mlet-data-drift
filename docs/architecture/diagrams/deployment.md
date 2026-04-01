# Diagrama de Deployment

Este diagrama mostra uma topologia de referencia que combina laboratorio local, automacao em nuvem e camadas de governanca. A proposta e suportar tanto ensino quanto reproducao em cenarios proximos de producao.

```mermaid
flowchart LR
    subgraph OnPrem["On-premise e laboratorio"]
        Dev["Notebook e experimentacao"]
        LocalStore[("Storage local")]
        LocalBroker["Broker local"]
        LocalReport["Relatorios HTML"]
    end

    subgraph Cloud["Cloud e producao"]
        Obj[("Object storage")]
        Stream["Kafka, Event Hub ou PubSub"]
        Jobs["Pipelines orquestrados"]
        Serve["Servico de inferencia"]
        Monitor["Evidently, NannyML e GX"]
        Obs["Dashboards e alertas"]
    end

    subgraph Governance["Governanca"]
        Git["GitHub"]
        ADR["ADRs e documentacao"]
        CI["CI e qualidade"]
    end

    Dev --> LocalStore
    Dev --> LocalBroker
    LocalStore --> Obj
    LocalBroker --> Stream
    Obj --> Jobs
    Stream --> Jobs
    Jobs --> Serve
    Serve --> Monitor
    Jobs --> Monitor
    Monitor --> Obs
    Git --> CI
    CI --> Jobs
    ADR --> Git
    LocalReport --> Git
```

## Interpretacao

- O laboratorio local acelera a iteracao didatica e a exploracao.
- A nuvem concentra orquestracao, servico de inferencia e monitoramento continuo.
- Git, ADRs e CI garantem rastreabilidade e repetibilidade ao longo do ciclo de vida.