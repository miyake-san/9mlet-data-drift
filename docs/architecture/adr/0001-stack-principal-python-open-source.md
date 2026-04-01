# ADR-0001: Stack principal em Python e open source

## Status

Aceito

## Contexto

A disciplina cobre desde testes estatisticos classicos para dados tabulares ate embeddings, aprendizado em fluxo, monitoramento continuo e validacao de dados em producao. Isso exige uma stack capaz de unir prototipacao rapida, maturidade academica, ecossistema de bibliotecas e baixo atrito para reproducao em ambiente educacional.

## Decisao

Adotar Python como linguagem principal e priorizar um ecossistema open source composto por NumPy, Pandas, SciPy, scikit-learn, River, PyTorch, Transformers, Evidently, NannyML e Great Expectations. JupyterLab sera o ambiente padrao para exploracao e ensino, enquanto Ruff e Taskipy serao usados para qualidade e automacao basicas do repositorio.

## Consequencias

### Positivas

- Reduz o custo de entrada para alunos e instrutores ao concentrar a maior parte do trabalho em uma linguagem dominante em dados e ML.
- Facilita a conexao entre teoria, experimento e operacao em producao usando bibliotecas amplamente adotadas.

### Negativas

- Alguns componentes de streaming e baixa latencia podem exigir integracao com outras tecnologias fora do nucleo Python.
- A gestao de dependencias pode se tornar pesada em aulas com embeddings e bibliotecas de NLP.

## Alternativas Consideradas

1. Stack poliglota com Python, Java e Scala - aumentaria a fidelidade para streaming industrial, mas elevaria demais a complexidade pedagogica.
2. Stack centrada em plataforma comercial gerenciada - reduziria esforco operacional, mas introduziria lock-in e menor transparencia para fins didaticos.

## Referencias

- Pedregosa et al. Scikit-learn: Machine Learning in Python.
- Lopez Paz and Oquab. Revisiting classifier two-sample tests.
- River documentation and JMLR paper.