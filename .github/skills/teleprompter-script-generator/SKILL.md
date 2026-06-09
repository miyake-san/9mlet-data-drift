---
name: teleprompter-script-generator
description: 'Generate teleprompter-ready plain text scripts from notebooks, markdown, class documents, and presentation materials. Use when creating scripts for livestreams, video recordings, or meetings with audience/style adaptation (executive, teaching, casual), fixed duration targets (1, 3, 5, 10, 15, 20, 25 minutes), optional chunking (2-3 parts), and preservation of tool/app names in original form.'
argument-hint: 'Provide source files, audience/style, target duration, channel (live/video/meeting), and language preferences.'
user-invocable: true
---

# Teleprompter Script Generator

## What This Skill Produces
This skill creates teleprompter scripts in plain text from mixed learning/content artifacts:
- Notebook files (`.ipynb`)
- Markdown files (`.md`)
- Class/support documents
- Presentation content

Output is optimized for spoken delivery in:
- Live sessions
- Video recording
- Meetings

## Rules (Always Enforced)
1. Output must be plain text style (no emojis, no decorative symbols).
2. Keep product/tool/app names untranslated exactly as they appear in source material.
3. Adapt language and tone to audience profile:
- executive
- teaching
- casual
- custom audience profile (if provided)
4. Respect target duration options:
- 1 minute
- 3 minutes
- 5 minutes
- 10 minutes
- 15 minutes
- 20 minutes
- 25 minutes
5. If source volume is high, split script into 2 or 3 coherent parts.

## Input Contract
Collect or infer the following before writing:
1. Source files/locations to parse.
2. Delivery channel (`live`, `video`, `meeting`).
3. Audience profile (`executive`, `teaching`, `casual`, or custom).
4. Target duration (one of 1, 3, 5, 10, 15, 20, 25).
5. Preferred language and linguistic register.
6. Chunking preference (`auto`, `2 parts`, `3 parts`, `single`).

If any of items 2-4 are missing, request them explicitly before drafting.

## Workflow
1. Ingest and classify source material.
- Read notebooks, markdown, and supporting class/presentation artifacts.
- Extract: core message, supporting points, examples, calls to action, transitions.

2. Build a speaking outline.
- Organize into: opening, development blocks, closing.
- Remove repetition and merge overlapping points.
- Keep conceptual flow logical for oral delivery.

3. Calibrate for duration.
- Use a speaking pace target of ~150 words/minute as baseline.
- Compute rough word budget:
  - 1 min: ~130-170 words
  - 3 min: ~390-510 words
  - 5 min: ~650-850 words
  - 10 min: ~1300-1700 words
  - 15 min: ~1950-2550 words
  - 20 min: ~2600-3400 words
  - 25 min: ~3250-4250 words
- Prioritize clarity and pacing over strict word count when trade-offs occur.

4. Apply audience/style transformation.
- executive: concise, decision-oriented, impact-first.
- teaching: explanatory, scaffolded, concept-before-detail.
- casual: natural, friendly, direct.
- custom: follow explicit user style instructions.

5. Preserve canonical names.
- Do not translate names of tools, apps, frameworks, APIs, products, libraries, or services.
- Preserve official terms and branded wording where possible.

6. Decide chunking.
- `single`: one continuous script.
- `2 parts` or `3 parts`: divide by narrative milestones, not arbitrary length.
- `auto`: split only when content density makes single-part delivery hard to follow.

7. Write the final teleprompter text.
- Plain prose, easy to read aloud.
- Short/medium sentences with natural breathing rhythm.
- Strong transitions between sections.
- No markdown headings in final script unless user asks for section labels.

8. Quality check before delivery.
- Confirm tone matches target audience.
- Confirm duration target is plausibly met.
- Confirm untranslated names are preserved.
- Confirm no emojis or decorative formatting.
- Confirm chunk count complies with requested mode.

## Completion Criteria
The output is complete only when:
1. It is teleprompter-ready plain text.
2. It matches one of the allowed duration targets.
3. It reflects the requested audience/style.
4. It preserves tool/app names without translation.
5. It is split into valid parts when required.

## Example Invocation Prompts
- Create a 10-minute teaching teleprompter script from these notebook and markdown files for a recorded lesson.
- Generate a 3-minute executive script for a meeting update based on this class summary document.
- Build a 20-minute casual live script from all lesson materials; split automatically if needed.
- Produce a 5-minute teleprompter script in Portuguese from this presentation, preserving all tool names in English.
