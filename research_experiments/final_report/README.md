# Final report (Overleaf project)

`main.tex` is the entry point; it inputs the `sec_*.tex` section files and the `figures/` folder.
- **Overleaf:** New Project → Upload Project → choose the zip. The compiler is pdfLaTeX (the default).
- **Locally:** `latexmk -pdf main.tex`.
- **Before submitting:** fill in the author and supervisor names on line 23 of `main.tex`.
- **Figures:** made by `python -m research_experiments.final_report.make_figures` from committed aggregate results only.
  The four PNGs reused from the phase folders are aggregate plots.
- **Contents:** aggregate metrics only. No patient names, identifiers, per-ECG predictions or ECG images.
