# ADR-0003: Monitoramento continuo orientado a Data SLOs

## Status

Aceito

## Contexto

Detectar drift sem definir qual acao tomar produz dashboards ricos e pouca resposta operacional. Em ambientes de negocio, sinais estatisticos precisam ser convertidos em prioridade, urgencia e responsabilidade. A disciplina tambem precisa mostrar que observabilidade sem governanca tende a falhar na pratica.

## Decisao

Adotar monitoramento continuo orientado a Data SLOs, combinando Evidently para visao ampla de drift, NannyML para estimativa de performance sem rotulo e Great Expectations para qualidade de dados antes da inferencia. Alertas devem ser associados a thresholds, janelas temporais e trilhas de resposta previamente definidas.

## Consequencias

### Positivas

- Transforma metricas tecnicas em objetivos operacionais acionaveis.
- Reduz o tempo entre a mudanca nos dados e a decisao de intervir no pipeline.

### Negativas

- Requer desenho disciplinado de SLIs, SLOs e runbooks para evitar ruido operacional.
- Pode gerar sobrecarga de manutencao se thresholds e segmentos nao forem revisados periodicamente.

## Alternativas Consideradas

1. Analise ad hoc em notebooks - adequada para exploracao inicial, mas fraca para producao e governanca.
2. Somente dashboards sem SLO - melhora visibilidade, mas nao fecha o ciclo de tomada de decisao.

## Referencias

- Sculley et al. Hidden technical debt in machine learning systems.
- Muller et al. Open-source drift detection tools in action.
- Evidently, NannyML and Great Expectations documentation.