# arXiv Submission Metadata

Copy/paste the fields below into the arXiv submission form.
Generated against `paper/main.tex` (2026-05-15): revised abstract (protocol definitions, training + 8B regime contrast, forward-pass diagnostic); stock `[preprint]{neurips_2026}` footer, Introduction on page~1, no `float` package.

Source bundle: `paper/arxiv_submission.zip` (contains `main.tex`, `neurips_2026.sty`, `main.bbl`, `references.bib`, `appendix_roadmap_tables.tex`, `figures/`).

---

## Title

No Free Swap: Protocol-Dependent Layer Redundancy in Transformers

## Authors

Gabriel Garcia (Independent Researcher)

Affiliation: Independent Researcher
Email: gpgabriel25@gmail.com

## Abstract  (1,183 characters; arXiv limit is 1920)

When researchers ask whether two transformer layers are "equivalent" for compression, they often conflate distinct tests. Replacement asks whether one layer's map can substitute for another's in place; interchange asks whether two layers approximately commute when their positions are swapped. Both are output-grounded swap-KL probes, but they need not agree: on pretrained transformers the protocol gap can change which layers look safe to prune by several-fold under the same evaluator, especially when replacement distances are high.

We measure both protocols across checkpoints and architectures. On a Pythia training trajectory (410M and 1.4B), the replacement-interchange gap grows from initialization to convergence. Under one matched WikiText-2 contract at 8B scale, Qwen3-8B enters a divergent regime: interchange-guided removal is several-fold safer than replacement-guided at the same layer budgets, while Llama-3.1-8B ties the two protocols for pruning cost even though interchange KL is lower, showing metric gaps need not map one-to-one to removal. Before layer removal or merging, score both swap-KLs on the target checkpoint; the diagnostic requires only unlabeled forward passes.

## Comments

40 pages, 8 figures, 24 tables. Code is available at https://github.com/Gpgabriel25/ProtocolGapDiagnostic

## arXiv Subject Classes (Primary / Secondary)

Primary: cs.LG (Machine Learning)

Secondary: cs.CL (Computation and Language)

## ACM-class (ACM Computing Classification System, 1998)

I.2.6; I.2.7

(I.2.6 [Artificial Intelligence]: Learning; I.2.7 [Artificial Intelligence]: Natural Language Processing.)

## MSC-class (optional, leave blank)

(none)

## DOI

(none yet)

## Journal Reference

(none)

## License

We recommend: CC BY 4.0 (arXiv.org perpetual, non-exclusive license is required either way).

---

## Notes for submitter

- Upload `paper/arxiv_submission.zip`. arXiv will run TeX Live and rebuild from `main.tex`.
- `main.tex` uses `\usepackage[preprint]{neurips_2026}`; this de-anonymizes the manuscript and is the correct mode for arXiv.
- The bundled `main.bbl` removes any need for arXiv to run BibTeX, but `references.bib` is included for completeness.
- Do not add the `float` package to this upload: it breaks the NeurIPS preprint noticebox on page~1.
- No font, .cls, .sty, or graphics conversion is needed; the tree compiles cleanly with `pdflatex main.tex` (run twice for cross-references). Include `appendix_roadmap_tables.tex` next to `main.tex` (it is `\input` from the appendix).
