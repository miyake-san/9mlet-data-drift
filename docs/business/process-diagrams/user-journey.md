# Jornada do Estudante e do Cientista de Dados

Este diagrama representa a experiencia esperada de quem usa o repositorio para estudar, gerar codigo e evoluir os exemplos ao longo do curso.

```mermaid
flowchart TD
    A([Inicio do estudo]) --> B["Ler README e objetivos"]
    B --> C["Revisar arquitetura e ADRs"]
    C --> D["Escolher uma aula"]
    D --> E["Abrir pasta da aula"]
    E --> F["Executar notebook ou script"]
    F --> G["Comparar resultados com business cases"]
    G --> H["Registrar aprendizados e duvidas"]
    H --> I["Avancar para monitoramento e escala"]
    I --> J["Contribuir com exemplos, issues ou PRs"]
```

## Observacoes

- A jornada foi desenhada para partir de fundamentos e evoluir ate operacao em escala.
- ADRs e business cases funcionam como ponte entre teoria, arquitetura e implementacao.
- O repositorio foi preparado para receber conteudo pratico por aula sem perder a coerencia global.