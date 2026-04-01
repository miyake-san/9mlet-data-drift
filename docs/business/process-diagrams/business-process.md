# Processo End-to-End de ML em Producao

O diagrama abaixo resume o processo de negocio e engenharia recomendado para construir, operar e manter modelos de ML sujeitos a drift.

```mermaid
flowchart TD
    Start([Demanda de negocio]) --> Problem["Definir problema e KPI"]
    Problem --> Collect["Coletar dados"]
    Collect --> Quality["Validar schema e qualidade"]
    Quality --> Explore["EDA e baseline"]
    Explore --> Features["Features e embeddings"]
    Features --> Train["Treinar modelos"]
    Train --> Evaluate{"Metricas e risco aceitos?"}
    Evaluate -- "Nao" --> Tune["Ajustar features e hiperparametros"]
    Tune --> Train
    Evaluate -- "Sim" --> Deploy["Publicar em producao"]
    Deploy --> Monitor["Monitorar drift e performance"]
    Monitor --> Check{"SLOs violados?"}
    Check -- "Nao" --> Monitor
    Check -- "Sim" --> Decide["Triagem e decisao"]
    Decide --> Retrain["Re-treinar ou recalibrar"]
    Retrain --> Deploy
```

## Observacoes

- O processo explicita que treinamento sem monitoramento deixa o ciclo incompleto.
- Validacao de dados e observabilidade devem entrar antes e depois da inferencia.
- A resposta a drift precisa estar acoplada a KPIs, SLOs e responsabilidade operacional.