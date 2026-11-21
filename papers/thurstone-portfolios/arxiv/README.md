# arXiv package

Build with `python3 papers/thurstone-portfolios/arxiv/make_arxiv.py`. It writes
`thurstone_portfolios_arxiv.zip` and a local `arxiv_preview.pdf`.

## What is in the zip

- `main.tex`: the paper flattened into one file (shared preamble and sections inlined).
- `main.bbl`: the compiled bibliography. arXiv uses this and does not need to run BibTeX.
- `refs.bib`: the cited references only, kept for completeness.
- `figures/`: the two vector figures. Figure 1 is drawn in TikZ inside `main.tex`.

## Differences from the site PDF

The package uses natbib with the `plainnat` style where the repository build uses
biblatex. A biblatex `.bbl` only works with the biblatex version that made it, and
arXiv's version differs from Tectonic's; a natbib `.bbl` is plain LaTeX and has no
such dependency. Citations therefore read "(Black and Litterman, 1992)" with a
comma, and the reference list is formatted by `plainnat`. The text is otherwise
identical.

## Submission form

- **Title:** Portfolio-Implied Contests, with an Application to Tail Risk
- **Author:** Peter Cotton
- **Abstract:** paste `abstract.txt`.
- **Suggested categories:** q-fin.PM (primary), q-fin.MF, stat.ME.
- **Processor:** pdfLaTeX. No special TeX Live version is needed.

## Before uploading

Rerun the script after any edit to the paper, and check `arxiv_preview.pdf`.
