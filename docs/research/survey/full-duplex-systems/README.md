# Full-duplex systems survey source

This directory contains the LaTeX source and compiled form of
**Engineering Full-Duplex Voice Agents Around External Reasoning Models: A
Systems Survey and Reference Architecture**.

The paper is intentionally narrower than the two existing dedicated surveys:

- [From Turn-Taking to Synchronous Dialogue](../../papers/full-duplex/2509.14515v1.pdf)
  organizes full-duplex spoken language models by engineered versus learned
  synchronization.
- [A Survey of Full-Duplex Spoken Dialogue Systems](../../papers/full-duplex/2606.19453v1.pdf)
  contributes an L0--L3 hierarchy, interaction ontology, and state machine.

This report focuses on the implementation boundary around an external
reasoning/tool model: AEC, target-speaker evidence, reversible interruption,
endpoint anticipation, heard-prefix consistency, and safe tool execution.

## Files

- `full-duplex-systems-survey.tex` — LaTeX source
- `references.bib` — BibTeX primary-source bibliography
- [Compiled survey PDF](full-duplex-systems-survey.pdf)

## Build

The checked PDF is built with Tectonic, a reproducible XeTeX/LaTeX engine that
downloads standard TeX Live bundle packages on demand:

```bash
tectonic full-duplex-systems-survey.tex
```
