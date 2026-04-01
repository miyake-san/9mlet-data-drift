# Guia de Contribuicao

Este repositorio foi estruturado para servir como base academica e tecnica para a disciplina DATA DRIFT. O objetivo deste guia e manter a consistencia editorial, a qualidade do codigo e a rastreabilidade das decisoes conforme novos exemplos, notebooks e estudos de caso forem adicionados.

## Como reportar issues

- Use issues para registrar bugs, lacunas de conteudo, melhorias de documentacao ou sugestoes de exemplos praticos.
- Dê preferencia a titulos objetivos, como `Adicionar exemplo de MMD para aula 4` ou `Corrigir thresholds no business case de fraude`.
- Descreva contexto, impacto esperado, passos para reproduzir e, quando aplicavel, trechos de codigo ou capturas de tela.
- Se o issue estiver ligado a uma aula especifica, referencie a pasta correspondente no titulo ou na descricao.

## Como submeter pull requests

- Crie um branch curto e descritivo, por exemplo: `feat/aula-05-embeddings` ou `docs/adr-monitoramento`.
- Mantenha cada PR focado em um unico objetivo: conteudo de aula, documentacao, diagramas, business case ou configuracao.
- Atualize a documentacao relacionada sempre que uma mudanca afetar arquitetura, stack, fluxo de estudo ou exemplos praticos.
- Em PRs com codigo, inclua uma breve secao `Como validar` com comandos e artefatos esperados.

## Padroes de codigo

- Siga PEP 8 para estilo geral de Python.
- Use type hints em funcoes publicas e em funcoes utilitarias reutilizaveis.
- Prefira nomes descritivos para variaveis, funcoes e notebooks.
- Inclua docstrings curtas em funcoes nao triviais, especialmente quando houver logica estatistica, thresholds ou calculos de drift.
- Use `ruff` para lint e formatacao automatizada.

## Como executar testes e validacoes

Instale as dependencias e execute:

```bash
task lint
task format
task test
```

Se voce estiver adicionando ou editando diagramas Mermaid:

- valide a sintaxe antes de publicar;
- confirme se os fluxos permanecem legiveis em Markdown;
- evite diagramas excessivamente densos quando um conjunto de diagramas menores for mais claro.

## Como adicionar uma nova aula

1. Crie ou atualize a pasta de aula no padrao `NN_nome_da_aula`.
2. Adicione README, notebook, script, dataset sintetico e artefatos de apoio conforme o escopo da aula.
3. Atualize o README principal com objetivo, teoria-chave, tecnologias e link para a pasta.
4. Se a nova aula alterar a arquitetura ou a stack, registre a decisao em um ADR novo ou atualize o ADR correspondente.
5. Se houver novo caso de negocio ou nova fase do pipeline, atualize os diagramas em `docs/architecture/` e `docs/business/process-diagrams/`.

## Boas praticas editoriais

- Mantenha a linguagem em nivel de mestrado, com foco em clareza e rigor tecnico.
- Quando possivel, conecte teoria, implementacao e impacto de negocio.
- Prefira exemplos reproduziveis, datasets sinteticos e referencias publicas.
- Evite adicionar dependencias sem justificar o ganho pedagogico ou operacional.