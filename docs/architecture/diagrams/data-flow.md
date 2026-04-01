# Diagrama de Fluxo de Dados

O fluxo abaixo destaca a trajetoria dos dados desde a construcao do baseline ate a deteccao de drift, a estimativa de performance e a tomada de acao em producao.

```mermaid
flowchart TD
    A["Dados historicos"] --> B["Curadoria e baseline"]
    B --> C["Treinamento e calibracao"]
    C --> D["Modelo em producao"]
    E["Dados recentes"] --> F["Validacao de schema e qualidade"]
    F --> G["Chunking batch ou streaming"]
    G --> H["Features e embeddings"]
    H --> D
    B --> I["Conjunto de referencia"]
    H --> J["Testes univariados\nKS, PSI e Wasserstein"]
    H --> K["Testes multivariados\nMMD e classifier drift"]
    D --> L["Predicoes e scores"]
    L --> M["Estimativa de performance\nCBPE e DLE"]
    I --> J
    I --> K
    J --> N{"Thresholds e SLOs\nviolados?"}
    K --> N
    M --> N
    N -- "Nao" --> O["Relatorio periodico"]
    N -- "Sim" --> P["Alerta e triagem"]
    P --> Q["Mitigacao\nrecalibracao, regras e retraining"]
    Q --> C
```

## Ponto de atencao

Este fluxo representa um loop operacional. O objetivo nao e apenas detectar uma mudanca, mas conectar o sinal estatistico a uma resposta concreta e verificavel.