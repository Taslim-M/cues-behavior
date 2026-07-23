# User Axis — ICLR technical reports

Two companion reports, both compiled with **pdfLaTeX** and sharing the bundled
`iclr2024_conference.sty` and `figures/`:

- **`main.tex` → `main.pdf`** (13 pp) — *The User Axis*: recovering, interpreting,
  and steering the unsupervised User Axis (Stages A–E + steering). This is the
  primary report.
- **`link_report.tex` → `link_report.pdf`** (7 pp) — *Does the User Axis Predict
  Assistant-Axis Drift?*: the Stage-F de-circularization test of the User→Assistant
  drift link (a negative result). Self-contained; reuses the same `.sty` and
  `figures/`.

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

To reveal author names for a camera-ready, uncomment `\iclrfinalcopy` and fill in
the `\author{...}` block (both reports).

## Files
- `main.tex` — primary report: body + appendices incl. **The user personas**
  appendix (archetype table + persona-space figure) and the note on why the
  held-out-tag → PC regression is needed. All detailed numbers in the appendices.
- `link_report.tex` — companion report: the Stage-F link test + full tier tables.
- `iclr2024_conference.sty` — bundled style (swap for official if desired).
- `figures/` — User-Axis figures (`resp_*`, `lastuser_*`, `persona_space_labeled`)
  and link figures (`link_*`).
- `user-axis-interactive-summary.html` — interactive one-page overview of both.

All numbers trace to `results/useraxis/llama-3.3-70b/` (`axis_validation.json`,
`interpretation.json`, `steering.json`, `stage_d.jsonl`, and `analysis/*.json`).
