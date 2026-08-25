<!-- active-model-version: v31 -->
# The input spec described a collection design the model never used

## Status

- Date: 2026-08-25
- Status: **corrected**; `presidential_issue_engine/IMPORT_SPEC.md` rewritten
- No model, prediction or frozen artifact changed

## What it said

`IMPORT_SPEC.md` opened with a section numbered **1. BIGKinds 뉴스 메타데이터**,
instructing a reader to export news metadata from `bigkinds.or.kr` and stating
that the importer `bigkinds_metadata.metadata_to_salience` consumes it. Its flow
diagram fed four instruments into the salience layer:

```
연합 제목(크롤, 최근)        ┐
BIGKinds 메타(당신, 과거포함) ├─► salience (instrument별 provenance)
                             ┘
```

and it closed by saying every salience row is tagged
`yonhap_title_count / bigkinds_meta / datalab_search`.

## What is true

The active model reads exactly one salience file, `data/issue_salience_assembly.csv`,
and every one of its 1001 rows carries `instrument = assembly_speech`.

| planned instrument | rows in the active model |
| --- | ---: |
| `bigkinds_meta` | **0** |
| `bigkinds_count` | **0** |
| `yonhap_title_count` | **0** |
| `datalab_search` | **0** |
| `assembly_speech` | 1001 |

The V31 input manifest (41 entries) contains no BigKinds-derived file, and the
wheel's runtime bundle contains no `news_collector` module — only an unused
mapping YAML. `docs/PUBLIC_DATA_SOURCES.json` registers no BigKinds source
family, so the rights register and the input spec disagreed with each other.

## Why it matters more than a typo

`IMPORT_SPEC.md` **ships in the source distribution**. A reviewer reading the
distributed package would conclude that this model builds issue salience from
commercial news metadata, when it builds it from National Assembly proceedings.
That is a claim about provenance, and provenance is what the project asks to be
judged on.

It also pointed at a source whose terms were never assessed, because the source
was never used — the register has no entry to assess.

## How it happened

The document is an early planning note, and it says so if read closely: section
2 ends with *"이 임포터는 형식 확정되면 BIGKinds 것과 같은 패턴으로 바로 만든다"*
— future tense, a plan to build. The project then settled on Assembly
proceedings alone, and the note was never revised. Nothing checked it, because
nothing connects a prose document to the instruments the data actually carries.

## What changed

`IMPORT_SPEC.md` now describes the one input the active model consumes, and
carries an explicit table of the importers that exist in the tree but are not on
the active path, with their zero row counts. The importers themselves are kept:
they are the record of the original design, and deleting them would delete that
record.

## What would have caught it

A check that every `instrument` value named in prose appears in the shipped
salience data, or that every importer named as active produces a file the input
manifest lists. Neither exists. This is left as a note rather than built, since
the prose surface is small and now correct.
