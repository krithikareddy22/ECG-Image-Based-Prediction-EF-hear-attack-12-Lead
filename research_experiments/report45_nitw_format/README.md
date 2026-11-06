# Report 45 in the NIT Warangal template format (LaTeX, PDF and Word)

Report 45 (Phase I progress report: four-class LVEF from 12-lead ECG printout images) rebuilt in the font and layout of
the NITW Phase I template / NITW thesis template. Same text, tables, figures and references as report 45; aggregates only.

| Path | What it is |
|---|---|
| `Report45_NITW_template.pdf` | Compiled report (58 pages) |
| `latex/` | Overleaf project: zip this folder and upload it (Overleaf → New Project → Upload Project), compiler pdfLaTeX |
| `word/Report45_NITW_template.docx` | Editable Word version (same layout); install the fonts in `word/LatinModern_fonts/` first |
| `confusion_matrices/` | Best confusion matrices: 50s (IMG-C3, locked test, 0.553), 70s (triage, locked test, 0.747 on answered) |
| `tools/latex/` | Scripts that converted report 45's PDF into the LaTeX sources (text, tables, figures, references) |
| `tools/word/` | Scripts that build the Word file from the LaTeX sources and calibrate its spacing; confusion-matrix plots |

The tool scripts record how the files were made; they use the working paths of the session that produced them.

---

## LaTeX project

Upload the `latex/` folder (zipped) to Overleaf (New Project → Upload Project). Compiler: **pdfLaTeX** (Overleaf default).
Main file: `main.tex`. Locally: `latexmk -pdf main.tex`.

| File / folder | Contents |
|---|---|
| `main.tex` | Document skeleton: front matter, the 7 chapters, references |
| `nitw-report.sty` | All layout settings of the template (fonts, margins, spacing, headings, headers, captions, contents, references) |
| `frontmatter.tex` | Title page, approval, certificate, declaration, acknowledgements, abstract, contents, lists, abbreviations |
| `body/ch1.tex` … `body/ch7.tex` | Chapter text |
| `tables/tab_X_Y.tex` | The 17 tables |
| `figures/` | Figures (vector PDF crops of report 45; Figure 3.1 redrawn in TikZ: `fig3_1_system.tex`) and the NITW logo |
| `equations.tex` | Equations 3.1–3.4 |
| `references.tex` | The 39 references (IEEE style) |

## Fill in before submission

All personal details are macros at the top of `frontmatter.tex` (change once, every page updates):

- `\StudentB`, `\RollB`, `\StudentC`, `\RollC`: students 2 and 3 (placeholders, as in the template)
- `\Guide`, `\GuideDesig`: supervisor's name with title, and designation
- Student 1, programme and department are taken from your template (Sai Chaithr, 23ECB0A20, Electronics and Communication Engineering). Change `\StudentA`, `\RollA`, `\Programme`, `\Dept` and `\DeptUpper` if needed.
- `\MonthYear` (November 2026). Also fill the `DD/MM/YYYY` date on the approval page.
- Print the Certificate page on the departmental letter-head, and sign the approval, certificate and declaration pages.

## Formatting guidelines (measured from the template)

**Page and font**
- A4, `report` class, 12 pt, Latin Modern (`lmodern`, T1 encoding), pdfLaTeX. No microtype (the template has neither font expansion nor margin protrusion).
- Margins: left 1.25 in, right 1 in, top 1 in, bottom 1 in (binding side wider). Header height 15 pt, header gap 25 pt, footer gap 30 pt.
- Body text: justified, 1.4 line spacing (20.3 pt from line to line), first-line indent 1.5 em, 4 pt between paragraphs. No single line of a paragraph is left alone at the top or bottom of a page.

**Chapters and headings**
- Every chapter starts at the top of a new page. "Chapter N" (`\Huge` bold) sits 50 pt below the top margin, the chapter title 30 pt below it, and the text starts 40 pt lower. There are no blank pages between chapters (one-sided).
- Sections: `\Large` bold, numbered 1.1. Subsections: `\large` bold, numbered 1.1.1.
- Contents, List of Figures, List of Tables and References use the chapter style without a number. Acknowledgements, Abstract and List of Abbreviations use a smaller `\LARGE` bold heading.
- Never put `\newpage`, `\clearpage` or `\vspace` before a `\chapter`: it already starts a new page at the right height.

**Headers, footers and page numbers**
- Text pages: "Chapter N. Title" on the left of the header, page number on the right, a thin (0.4 pt) rule under the header and above the footer.
- Chapter-opening pages: no header, page number centred at the bottom.
- Front matter is numbered i, ii, iii, …. The title page has no number. The title, approval, certificate, declaration and acknowledgements pages have no header. The abstract shows only the page number at top right. Chapter 1 starts at page 1.

**Front-matter order** (as in the template)
Title page → Approval of the Project Evaluation Board → Certificate → Declaration → Acknowledgements → Abstract with Index Terms → Contents → List of Figures → List of Tables → List of Abbreviations (with Notation) → Chapters 1–7 → References.

**Figures and tables**
- Captions: `\small`, single-spaced, bold label followed by a space ("**Table 3.1** Preprocessing …"), no colon. A caption that fits on one line is centred; a longer one is justified. Table captions above the table, figure captions below the figure.
- Tables: booktabs rules only (top, middle, bottom; no vertical lines), `\small`, single-spaced inside (each row 1.12 × the line height), bold header row, left-aligned columns. Very wide tables (8–9 columns) or tables taller than a page use `\footnotesize` with narrower column gaps (Tables 3.2, 4.5, 5.2). A table longer than a page continues with "Table x.y (continued)" and its header repeated (Table 2.1).
- Figures: centred, vector PDF where possible, at most 80 % of the text height, placed with `[htbp]`.
- Equations are numbered by chapter: (3.1), (3.2), ….

**Contents and lists**
- Contents, List of Figures and List of Tables: 1.2 line spacing, 4 pt between entries, 1 em extra before each chapter entry. Chapter and front-matter entries are bold. All entries are blue links.
- List of Abbreviations: 1.2 line spacing, indented 18 pt, abbreviation column 2.6 cm wide.
- Bulleted and numbered lists in the text: 2.2 em indent, 2 pt between items, 4 pt above and below.

**References**
- IEEE numeric style, cited as [n], with ranges compressed ([14–16]). Citations and links are blue.
- Reference list in 12 pt, 1.1 line spacing, 9 pt between entries, hanging indent after the [n] label.

## Editing tips

- Add a chapter: create `body/ch8.tex` starting with `\chapter{Title}\label{ch:8}` and add `\input{body/ch8}` in `main.tex`.
- Add a figure: `\begin{figure}[htbp]\centering\includegraphics[width=0.8\linewidth]{figures/name}\caption{…}\label{fig:x.y}\end{figure}`.
- Add a table: copy any file in `tables/` and keep the `\caption` above the `tabular`.
- Refer with `Figure~\ref{fig:x.y}`, `Table~\ref{tab:x.y}`, `Section~\ref{sec:x.y}` and `\cite{refN}`.
- Overleaf compiles twice automatically, so the contents, lists and cross-references are always up to date.

## Word version (`word/Report45_NITW_template.docx`)

The same report as an editable Word file, built from these LaTeX sources with the same layout:

- **Fonts:** install the Latin Modern fonts first (folder `word/LatinModern_fonts`: select all `.otf` files → right-click → Install). Without them Word substitutes Times New Roman, and line and page breaks shift.
- **Contents and lists:** Word may ask "This document contains fields that may refer to other files. Update the fields?" on opening; choose **Yes**. Page numbers in the Contents, List of Figures and List of Tables are live fields (also: right-click a list → Update Field → Update entire table, or press Ctrl+A, then F9).
- **Styles:** body text "Body Text NITW" (12 pt, exact 20.3 pt line spacing, 1.5 em first-line indent, 4 pt after); headings "Heading 1–3"; captions "Caption". Each chapter is its own section (new page, own header "Chapter N. Title", page number centred on the chapter's first page).
- **Equations** are native Word equations (editable with the equation editor). Citations [n] and Figure/Table/Section numbers are internal links.
