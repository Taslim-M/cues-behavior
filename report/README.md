# User Axis — ICLR technical report

Compile on Overleaf (pdfLaTeX). Two ways:

**A. As-is (self-contained).** Upload the whole `report/` folder (main.tex,
`iclr2024_conference.sty`, `figures/`) to a blank Overleaf project. Set the
compiler to **pdfLaTeX** and compile `main.tex`. The bundled
`iclr2024_conference.sty` is a faithful reconstruction of the ICLR template, so
it builds with no extra files.

**B. Official ICLR style.** Create a project from Overleaf's **“ICLR 2024”**
template (Menu → New Project → Templates), then drop in this `main.tex` and the
`figures/` folder and **overwrite** the template's `iclr2024_conference.sty` — or
just delete the bundled one so Overleaf uses theirs. `main.tex` is written to work
with either style unchanged.

To reveal author names for a camera-ready, uncomment `\iclrfinalcopy` and fill in
the `\author{...}` block.

## Files
- `main.tex` — the report (body + 8 appendices, all detailed numbers in the appendix)
- `iclr2024_conference.sty` — bundled style (swap for official if desired)
- `figures/` — 10 PNGs (5 per readout: variance curve, PC1-vs-tags, cosine-per-layer,
  persona 3-D scatter, elicitation-arm agreement)

All numbers trace to `results/useraxis/llama-3.3-70b/` (`axis_validation.json`,
`interpretation.json`, `steering.json`, `stage_d.jsonl`).
