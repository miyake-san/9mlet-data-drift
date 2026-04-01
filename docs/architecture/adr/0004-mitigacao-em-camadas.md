# ADR-0004: Mitigacao em camadas com re-treino e aprendizado online

## Status

Aceito

## Contexto

Nem todo sinal de drift deve levar imediatamente a re-treino completo. Em alguns cenarios, o custo computacional e operacional e alto, os rotulos demoram a chegar ou o sinal e temporario. Em outros, medidas paliativas so postergam o problema e aumentam o risco.

## Decisao

Modelar a resposta a drift em camadas. A primeira camada contempla ajustes taticos, como recalibracao de threshold, regras temporarias e triagem manual. A segunda contempla re-treino programado com dados recentes. A terceira introduz aprendizado online ou ensembles adaptativos para contextos de mudanca frequente.

## Consequencias

### Positivas

- Evita respostas unicas para problemas com perfis de risco distintos.
- Ensina a combinar mitigacao reativa e adaptacao estrutural ao longo do tempo.

### Negativas

- Aumenta a necessidade de criterios claros para selecionar a resposta correta.
- Pode gerar complexidade adicional de operacao, versionamento e validacao comparativa entre modelos.

## Alternativas Consideradas

1. Apenas re-treino periodico - simples de governar, mas lento para responder a mudancas rapidas.
2. Apenas aprendizado online - adaptativo, mas nem sempre adequado para todos os modelos, dados e requisitos regulatorios.

## Referencias

- Gama et al. Survey on concept drift adaptation.
- River documentation and examples.