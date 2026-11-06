// Build the Word version of report 45 in the NIT Warangal template format from doc.json.
// node build.js  ->  report45.docx (equations are inserted as native Word equations by post.py)
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, BorderStyle,
  AlignmentType, TabStopType, LeaderType, LineRuleType, PageNumber, NumberFormat, SectionType, Header, Footer,
  Bookmark, InternalHyperlink, ExternalHyperlink, PageReference, TableLayoutType, VerticalAlignTable, LevelFormat, Tab,
} = require("docx");

const D = JSON.parse(fs.readFileSync("doc.json", "utf8"));
const C = JSON.parse(fs.readFileSync("cal.json", "utf8"));
const M = D.front.macros;
const TW = 8666;                         // text width in twips (A4, left 1.25 in, right 1 in)
const pt = (v) => Math.round(v * 20);    // points -> twips
const BODY = "LM Roman 12", SMALLF = "LM Roman 10", BLUE = "0000FF";
const LINE = pt(20.3);                   // body baseline-to-baseline (12 pt x 1.4)
const NONE = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const NOBORDERS = { top: NONE, bottom: NONE, left: NONE, right: NONE };
const RULE = (sz) => ({ style: BorderStyle.SINGLE, size: sz, color: "000000" });

// ---------------------------------------------------------------- runs
function runs(rs, o = {}) {
  const out = [];
  for (const r of rs) {
    const p = {
      text: r.t, bold: !!(r.b || o.bold), italics: !!(r.i || o.italics), font: o.font, size: o.size,
      superScript: !!r.sup, subScript: !!r.sub,
    };
    if (r.link) out.push(new InternalHyperlink({ anchor: r.link, children: [new TextRun({ ...p, color: BLUE })] }));
    else if (r.url) out.push(new ExternalHyperlink({ link: r.url, children: [new TextRun({ ...p, color: BLUE })] }));
    else out.push(new TextRun(p));
  }
  return out;
}
const T = (text, o = {}) => new TextRun({ text, ...o });
const plain = (rs) => rs.map((r) => r.t).join("");
const anchorOf = (kind, num) => `${kind}_${String(num).replace(/\./g, "_")}`;

// ---------------------------------------------------------------- headers / footers
const emptyHF = () => ({ header: new Header({ children: [new Paragraph({})] }), footer: new Footer({ children: [new Paragraph({})] }) });
function fancyHeader(left) {
  return new Header({ children: [new Paragraph({
    tabStops: [{ type: TabStopType.RIGHT, position: TW }],
    spacing: { line: pt(C.headerLine), lineRule: LineRuleType.EXACT, before: 0, after: 0 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 3, space: 2, color: "000000" } },
    children: [T(left), new TextRun({ children: [new Tab(), PageNumber.CURRENT] })],
  })] });
}
const ruleFooter = () => new Footer({ children: [new Paragraph({
  spacing: { line: pt(12), lineRule: LineRuleType.EXACT, before: 0, after: 0 },
  border: { top: { style: BorderStyle.SINGLE, size: 3, space: 1, color: "000000" } }, children: [],
})] });
const plainFooter = () => new Footer({ children: [new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 0, after: 0 }, children: [new TextRun({ children: [PageNumber.CURRENT] })],
})] });
const blankHeader = () => new Header({ children: [new Paragraph({})] });
// empty paragraph of exact height: positions the first line of a page (space-before is dropped at a page top)
const spacer = (h, o = {}) => new Paragraph({ ...o, spacing: { before: 0, after: 0, line: Math.max(20, pt(h)), lineRule: LineRuleType.EXACT }, children: [] });

const PAGE = (numbers) => ({
  page: {
    size: { width: 11906, height: 16838 },
    margin: { top: 1440, bottom: 1440, left: 1800, right: 1440, header: pt(C.headerDist), footer: pt(C.footerDist) },
    pageNumbers: numbers,
  },
});

// ---------------------------------------------------------------- headings
function chapterHead(num, title) {
  return [
    spacer(C.chapBefore),
    new Paragraph({ style: "ChapterLabel", spacing: { before: 0, after: 0, line: pt(C.titleLine), lineRule: LineRuleType.EXACT }, children: [T(`Chapter ${num}`)] }),
    new Paragraph({ style: "Heading1", spacing: { before: pt(C.chapGap), after: 0, line: pt(C.titleLine), lineRule: LineRuleType.EXACT },
      children: [new Bookmark({ id: anchorOf("ch", num), children: [T(title)] })] }),
  ];
}
function starHead(title, id, extra = 0) { // Contents, List of Figures/Tables, References (chapter style, no number)
  return [spacer(C.starBefore + extra), new Paragraph({ style: "Heading1", spacing: { before: 0, after: 0, line: pt(36), lineRule: LineRuleType.EXACT },
    children: [new Bookmark({ id, children: [T(title)] })] })];
}
function frontHead(title, id, before, o = {}) { // Acknowledgements, Abstract, List of Abbreviations (\LARGE)
  return [spacer(before, o), new Paragraph({ style: "FrontHead", spacing: { before: 0, after: 0, line: pt(35), lineRule: LineRuleType.EXACT },
    children: [new Bookmark({ id, children: [T(title)] })] })];
}

// ---------------------------------------------------------------- body blocks
const ENUM_INSTANCES = { n: 0 };
function bodyPara(rs, opts = {}) {
  return new Paragraph({
    style: "Body", indent: opts.indent === false ? { firstLine: 0 } : undefined,
    spacing: opts.before !== undefined ? { before: pt(opts.before), after: 80, line: LINE, lineRule: LineRuleType.EXACT } : undefined,
    children: runs(rs),
  });
}
function equationTable(num, indentLeft = 0) {
  const w = TW - indentLeft;
  const side = 1000;
  const cell = (children, width, align) => new TableCell({
    width: { size: width, type: WidthType.DXA }, borders: NOBORDERS, verticalAlign: VerticalAlignTable.CENTER,
    margins: { left: 0, right: 0, top: 0, bottom: 0 },
    children: [new Paragraph({ alignment: align, spacing: { before: 100, after: 100 }, children })],
  });
  return new Table({
    width: { size: w, type: WidthType.DXA }, columnWidths: [side, w - 2 * side, side], layout: TableLayoutType.FIXED,
    indent: { size: indentLeft, type: WidthType.DXA }, borders: { ...NOBORDERS, insideHorizontal: NONE, insideVertical: NONE },
    rows: [new TableRow({ cantSplit: true, children: [
      cell([], side, AlignmentType.LEFT),
      cell([T(`@@EQ${num}@@`)], w - 2 * side, AlignmentType.CENTER),
      cell([new Bookmark({ id: anchorOf("eq", num), children: [T(`(${num})`)] })], side, AlignmentType.RIGHT),
    ] })],
  });
}
function listBlock(b, prevBefore) {
  const out = [];
  let ref;
  if (b.kind === "number") { ENUM_INSTANCES.n += 1; ref = { reference: "enum", level: 0, instance: ENUM_INSTANCES.n }; }
  else ref = { reference: "bullet", level: 0 };
  b.items.forEach((it, k) => {
    out.push(new Paragraph({
      style: "ListItem", numbering: ref,
      spacing: { before: k === 0 ? pt(prevBefore) : pt(3), after: 80, line: LINE, lineRule: LineRuleType.EXACT },
      children: runs(it.runs),
    }));
    if (it.eq) out.push(equationTable(it.eq, 516));
  });
  return out;
}
function captionPara(kind, num, rs, single, before, after) {
  return new Paragraph({
    style: "Caption", alignment: single ? AlignmentType.CENTER : AlignmentType.JUSTIFIED, keepNext: kind === "Table",
    spacing: { before: pt(before), after: pt(after), line: pt(13.6), lineRule: LineRuleType.EXACT },
    children: [new Bookmark({ id: anchorOf(kind === "Table" ? "tab" : "fig", num), children: [T(`${kind} ${num}`, { bold: true })] }), T(" "), ...runs(rs)],
  });
}
function dataTable(b) {
  const size = b.small ? 22 : 20, line = b.small ? pt(13.6) : pt(12), pad = b.small ? 120 : 80;
  const widths = b.fracs.map((f) => Math.round(f * TW));
  widths[widths.length - 1] += TW - widths.reduce((a, c) => a + c, 0);
  const n = widths.length;
  const rows = b.rows.map((r, k) => {
    const next = b.rows[k + 1];
    const last = k === b.rows.length - 1;
    const ruleBelow = last || (next && next.rule === "mid");
    let col = 0;
    const cells = r.cells.map((c) => {
      const span = c.span, w = widths.slice(col, col + span).reduce((a, x) => a + x, 0);
      const startCol = col + 1; col += span;
      const top = r.rule === "top" ? RULE(8) : r.rule === "mid" ? RULE(5)
        : (r.cmid_below || []).some(([a, z]) => startCol >= a && startCol <= z) ? RULE(4) : NONE;
      return new TableCell({
        width: { size: w, type: WidthType.DXA }, columnSpan: span > 1 ? span : undefined,
        borders: { top, bottom: last ? RULE(8) : NONE, left: NONE, right: NONE },
        margins: { left: pad, right: pad, top: r.rule ? 60 : 16, bottom: ruleBelow ? 40 : 16 },
        children: [new Paragraph({ alignment: AlignmentType.LEFT, spacing: { before: 0, after: 0, line, lineRule: LineRuleType.EXACT },
          children: runs(c.runs, { font: SMALLF, size }) })],
      });
    });
    while (col < n) {                                   // pad short rows
      cells.push(new TableCell({ width: { size: widths[col], type: WidthType.DXA }, borders: { top: NONE, bottom: last ? RULE(8) : NONE, left: NONE, right: NONE },
        children: [new Paragraph({ children: [] })] }));
      col += 1;
    }
    return new TableRow({ tableHeader: r.header, cantSplit: true, children: cells });
  });
  return new Table({ width: { size: TW, type: WidthType.DXA }, columnWidths: widths, layout: TableLayoutType.FIXED,
    borders: { ...NOBORDERS, insideHorizontal: NONE, insideVertical: NONE }, rows });
}
function figureBlocks(b) {
  const maxW = TW / 20 * 96 / 72, maxH = 0.80 * 697.9 * 96 / 72;
  let w = b.frac * maxW, h = w * b.px[1] / b.px[0];
  if (h > maxH) { w *= maxH / h; h = maxH; }
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 240, after: 0 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(`media/${b.png}`), transformation: { width: Math.round(w), height: Math.round(h) } })] }),
    captionPara("Figure", b.num, b.caption, b.cap_single, 8, 14),
  ];
}

// ---------------------------------------------------------------- chapters
const chapters = [];
let cur = null, prev = null;
for (const b of D.blocks) {
  if (b.type === "chapter") { cur = { num: b.num, title: b.title, children: chapterHead(b.num, b.title) }; chapters.push(cur); prev = "chapter"; continue; }
  const ch = cur.children;
  if (b.type === "section") {
    const sub = b.level === 2;
    const before = prev === "chapter" ? C.secAfterTitle : sub ? C.subBefore : C.secBefore;
    ch.push(new Paragraph({
      style: sub ? "Heading3" : "Heading2",
      spacing: { before: pt(before), after: 0, line: pt(sub ? 25.2 : 30.8), lineRule: LineRuleType.EXACT },
      tabStops: [{ type: TabStopType.LEFT, position: sub ? 1060 : 880 }], indent: { left: sub ? 1060 : 880, hanging: sub ? 1060 : 880 },
      children: [new Bookmark({ id: anchorOf("sec", b.num), children: [T(b.num), new TextRun({ children: [new Tab()] }), ...runs(b.title_runs)] })],
    }));
    prev = sub ? "sub" : "section";
  } else if (b.type === "para") {
    const before = prev === "chapter" ? C.textAfterTitle : prev === "section" ? C.textAfterSec : prev === "sub" ? C.textAfterSub
      : prev === "list" ? 7 : prev === "float" ? 0 : undefined;
    ch.push(bodyPara(b.runs, { indent: b.indent, before }));
    prev = "para";
  } else if (b.type === "list") {
    const before = prev === "chapter" ? C.textAfterTitle : prev === "section" ? C.textAfterSec : prev === "sub" ? C.textAfterSub : 7;
    ch.push(...listBlock(b, before));
    prev = "list";
  } else if (b.type === "equation") {
    ch.push(equationTable(b.num));
    prev = "eq";
  } else if (b.type === "table") {
    ch.push(captionPara("Table", b.num, b.caption, b.cap_single, 12, 6));
    ch.push(dataTable(b));
    ch.push(new Paragraph({ spacing: { before: 0, after: 0, line: pt(10), lineRule: LineRuleType.EXACT }, children: [] }));
    prev = "float";
  } else if (b.type === "figure") {
    ch.push(...figureBlocks(b));
    prev = "float";
  }
}

// ---------------------------------------------------------------- contents and lists (titles link to their target; page numbers are live PAGEREF fields)
function tocEntry(level, anchor, num, title) {
  const link = (children) => new InternalHyperlink({ anchor, children });
  if (level === 0) {
    return new Paragraph({ style: "TocBase", spacing: { before: 235, after: 80, line: pt(17.4), lineRule: LineRuleType.EXACT },
      tabStops: [{ type: TabStopType.LEFT, position: 352 }, { type: TabStopType.RIGHT, position: TW }], indent: { left: num ? 352 : 0, hanging: num ? 352 : 0 },
      children: [link([...(num ? [T(num, { bold: true, color: BLUE }), new TextRun({ children: [new Tab()] })] : []), T(title, { bold: true, color: BLUE })]),
        new TextRun({ children: [new Tab()], bold: true }), new TextRun({ bold: true, children: [new PageReference(anchor)] })] });
  }
  const left = level === 1 ? 890 : 1644, hang = level === 1 ? 538 : 752;
  return new Paragraph({ style: "TocBase", spacing: { before: 0, after: 80, line: pt(17.4), lineRule: LineRuleType.EXACT },
    tabStops: [{ type: TabStopType.LEFT, position: left }, { type: TabStopType.RIGHT, position: TW, leader: LeaderType.DOT }],
    indent: { left, hanging: hang, right: 500 },
    children: [link([T(num, { color: BLUE }), new TextRun({ children: [new Tab()] }), T(title, { color: BLUE })]),
      new TextRun({ children: [new Tab()] }), new PageReference(anchor)] });
}
const toc = [...starHead("Contents", "fm_toc"), new Paragraph({ spacing: { before: 0, after: pt(C.starAfter - 12) }, children: [] })];
for (const [t, id] of [["Certificate of Approval", "fm_approval"], ["Certificate", "fm_cert"], ["Declaration", "fm_decl"], ["Acknowledgements", "fm_ack"],
  ["Abstract", "fm_abs"], ["List of Figures", "fm_lof"], ["List of Tables", "fm_lot"], ["List of Abbreviations", "fm_abbr"]]) toc.push(tocEntry(0, id, "", t));
for (const b of D.blocks) {
  if (b.type === "chapter") toc.push(tocEntry(0, anchorOf("ch", b.num), b.num, b.title));
  if (b.type === "section") toc.push(tocEntry(b.level, anchorOf("sec", b.num), b.num, b.title_runs.map((r) => r.t).join("")));
}
toc.push(tocEntry(0, "fm_refs", "", "References"));
function listOf(kind) {
  const out = [...starHead(kind === "figure" ? "List of Figures" : "List of Tables", kind === "figure" ? "fm_lof" : "fm_lot"),
    new Paragraph({ spacing: { before: 0, after: pt(C.starAfter - 12) }, children: [] })];
  let lastCh = null;
  for (const b of D.blocks) {
    if (b.type !== kind) continue;
    const chn = b.num.split(".")[0];
    const p = tocEntry(1, anchorOf(kind === "figure" ? "fig" : "tab", b.num), b.num, plain(b.caption));
    if (lastCh !== null && chn !== lastCh) out.push(new Paragraph({ spacing: { before: 0, after: 0, line: 200, lineRule: LineRuleType.EXACT }, children: [] }));
    lastCh = chn;
    out.push(p);
  }
  return out;
}

// ---------------------------------------------------------------- front matter
const centred = (children, o = {}) => new Paragraph({ alignment: AlignmentType.CENTER, ...o, children });
const exact = (before, line) => ({ before: pt(before), after: 0, line: pt(line), lineRule: LineRuleType.EXACT });
const BI = { bold: true, italics: true, font: SMALLF };
const titlePage = [
  ...D.front.title_lines.map((t, k) => centred([T(t, { bold: true, size: 35 })], { spacing: exact(0, k ? 24.2 : C.tp_title) })),
  centred([T("Phase I Project Progress Report submitted to the", BI)], { spacing: exact(C.tp_A, 18.97) }),
  centred([T("National Institute of Technology Warangal", BI)], { spacing: exact(0, 18.97) }),
  centred([T("in partial fulfilment of the requirements for the award of the degree", BI)], { spacing: exact(0, 18.97) }),
  centred([T("of", BI)], { spacing: exact(C.tp_B, 15.95) }),
  centred([T("Bachelor of Technology", { bold: true, size: 29 })], { spacing: exact(C.tp_C, 19.8) }),
  centred([T(`in ${M.Programme}`, { bold: true, size: 29 })], { spacing: exact(4, 19.8) }),
  centred([T("by", BI)], { spacing: exact(C.tp_E, 15.95) }),
  ...[[M.StudentA, M.RollA], [M.StudentB, M.RollB], [M.StudentC, M.RollC]].map(([n, r], k) =>
    centred([T(n, { bold: true, size: 29 }), T(` (Roll No. ${r})`)], { spacing: exact(k ? 3.5 : C.tp_F, 19.8) })),
  centred([T("Under the guidance of")], { spacing: exact(C.tp_H, 15.95) }),
  centred([T(M.Guide, { bold: true, size: 29 })], { spacing: exact(C.tp_I, 19.8) }),
  centred([T(`${M.GuideDesig}, Department of ${M.Dept}`, { size: 22, font: SMALLF })], { spacing: exact(0, 15) }),
  centred([new ImageRun({ type: "png", data: fs.readFileSync("/tmp/claude-0/r45tex/figures/nitw_logo.png"), transformation: { width: 87, height: 102 } })],
    { spacing: { before: pt(C.tp_K), after: 0 } }),
  centred([T(`DEPARTMENT OF ${M.DeptUpper}`, { bold: true })], { spacing: exact(C.tp_bottom, 15.95) }),
  centred([T("NATIONAL INSTITUTE OF TECHNOLOGY WARANGAL", { bold: true })], { spacing: exact(3, 15.95) }),
  centred([T(M.MonthYear, { bold: true })], { spacing: exact(3, 15.95) }),
];
const tabs3 = [{ type: TabStopType.CENTER, position: TW / 2 }, { type: TabStopType.RIGHT, position: TW }];
const approval = [
  centred([new Bookmark({ id: "fm_approval", children: [T("APPROVAL OF THE PROJECT EVALUATION BOARD", { bold: true, size: 29 })] })], { pageBreakBefore: true, spacing: exact(0, C.ap_head) }),
  new Paragraph({ alignment: AlignmentType.RIGHT, spacing: exact(C.ap_date, 20.3), children: [T("DD/MM/YYYY ")] }),
  new Paragraph({ style: "Body", spacing: exact(C.ap_text, 20.3), children: runs(D.front.approval) }),
  new Paragraph({ tabStops: tabs3, spacing: exact(C.ap_B, 20.3), children: [T("(Member of the Board)"), new TextRun({ children: [new Tab()] }), T("(Member of the Board)"), new TextRun({ children: [new Tab()] }), T("(Member of the Board)")] }),
  new Paragraph({ tabStops: tabs3, spacing: exact(C.ap_C, 20.3), children: [T("(Supervisor)"), new TextRun({ children: [new Tab(), new Tab()] }), T("(Co-Supervisor)")] }),
  new Paragraph({ tabStops: tabs3, spacing: exact(C.ap_D, 20.3), children: [T("(Head of the Department)"), new TextRun({ children: [new Tab(), new Tab()] }), T("(Chairperson, Evaluation Board)")] }),
];
const sigCell = (children, w, top) => new TableCell({ width: { size: w, type: WidthType.DXA }, borders: { top: top ? RULE(3) : NONE, bottom: NONE, left: NONE, right: NONE },
  margins: { left: 0, right: 0, top: 60, bottom: 0 }, children });
const cline = (text) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 0, line: pt(20.3), lineRule: LineRuleType.EXACT }, children: [T(text)] });
const certificate = [
  centred([T("(The certificate is to be printed on the Departmental Letter-Head)")], { pageBreakBefore: true, spacing: exact(0, C.ce_note) }),
  centred([new Bookmark({ id: "fm_cert", children: [T("CERTIFICATE", { bold: true, size: 29, underline: {} })] })], { spacing: exact(C.ce_head, 19.8) }),
  new Paragraph({ style: "Body", indent: { firstLine: 0 }, spacing: exact(C.ce_text, 20.3), children: runs(D.front.certificate) }),
  new Paragraph({ spacing: exact(C.ce_sig, 1), children: [] }),
  new Table({ width: { size: 3402, type: WidthType.DXA }, columnWidths: [3402], layout: TableLayoutType.FIXED, borders: { ...NOBORDERS, insideHorizontal: NONE, insideVertical: NONE },
    rows: [new TableRow({ children: [sigCell([cline("Supervisor"), cline(M.Guide)], 3402, true)] })] }),
  new Paragraph({ spacing: exact(C.ce_date, 20.3), children: [T("Date:")] }),
];
const declaration = [
  centred([new Bookmark({ id: "fm_decl", children: [T("DECLARATION", { bold: true, size: 29 })] })], { pageBreakBefore: true, spacing: exact(0, C.de_head) }),
  new Paragraph({ spacing: exact(C.de_we, 20.3), children: [T("We certify that")] }),
  ...D.front.declaration.map((rs, k) => new Paragraph({ style: "Body", indent: { left: 468, hanging: 298 }, numbering: { reference: "alpha", level: 0 },
    spacing: { before: pt(k ? C.de_item : C.de_first), after: 0, line: pt(20.3), lineRule: LineRuleType.EXACT }, children: runs(rs) })),
  new Paragraph({ spacing: exact(C.de_sig, 1), children: [] }),
  new Table({ width: { size: 3 * 2381 + 2 * 702, type: WidthType.DXA }, columnWidths: [2381, 702, 2381, 702, 2381], layout: TableLayoutType.FIXED,
    borders: { ...NOBORDERS, insideHorizontal: NONE, insideVertical: NONE },
    rows: [new TableRow({ children: [
      sigCell([cline(M.StudentA), cline(`(${M.RollA})`)], 2381, true), sigCell([new Paragraph({})], 702, false),
      sigCell([cline(M.StudentB), cline(`(${M.RollB})`)], 2381, true), sigCell([new Paragraph({})], 702, false),
      sigCell([cline(M.StudentC), cline(`(${M.RollC})`)], 2381, true)] })] }),
  centred([T("Signatures of the Students")], { spacing: exact(C.de_cap, 20.3) }),
];
const ack = [
  ...frontHead("Acknowledgements", "fm_ack", C.frontBefore, { pageBreakBefore: true }),
  bodyPara(D.front.ack, { before: C.frontAfter }),
  new Paragraph({ alignment: AlignmentType.RIGHT, spacing: { before: pt(24), after: 0 }, children: [T(`${M.StudentA}, ${M.StudentB} and ${M.StudentC}`, { bold: true })] }),
];
const abstract = [...frontHead("Abstract", "fm_abs", C.frontBefore),
  ...D.front.abstract.map((rs, k) => bodyPara(rs, { before: k === 0 ? C.frontAfter : undefined }))];
const abbrTable = (rows, keep = false) => new Table({
  width: { size: TW - 360, type: WidthType.DXA }, columnWidths: [1714, TW - 360 - 1714], layout: TableLayoutType.FIXED,
  indent: { size: 360, type: WidthType.DXA }, borders: { ...NOBORDERS, insideHorizontal: NONE, insideVertical: NONE },
  rows: rows.map(([a, b]) => new TableRow({ cantSplit: true, children: [a, b].map((rs, k) => new TableCell({
    width: { size: k ? TW - 360 - 1714 : 1714, type: WidthType.DXA }, borders: NOBORDERS, margins: { left: 0, right: k ? 0 : 240, top: 20, bottom: 20 },
    children: [new Paragraph({ keepNext: keep, spacing: { before: 0, after: 0, line: pt(17.4), lineRule: LineRuleType.EXACT }, children: runs(rs) })] })) })),
});
const abbreviations = [...frontHead("List of Abbreviations", "fm_abbr", C.frontBefore - 5),
  new Paragraph({ spacing: { before: 0, after: 0, line: pt(C.frontAfter - 10 + C.abbrExtra), lineRule: LineRuleType.EXACT }, children: [] }),
  abbrTable(D.front.abbreviations),
  new Paragraph({ keepNext: true, spacing: { before: pt(14), after: pt(6), line: pt(21.6), lineRule: LineRuleType.EXACT }, children: [T("Notation", { bold: true, size: 29 })] }),
  abbrTable(D.front.notation, true)];        // Notation block kept on one page with its heading

// ---------------------------------------------------------------- references
const refs = [...starHead("References", "fm_refs", -3), new Paragraph({ spacing: { before: 0, after: pt(C.starAfter - 20) }, children: [] }),
  ...D.refs.map((r) => new Paragraph({
    alignment: AlignmentType.JUSTIFIED, indent: { left: 482, hanging: 482 },
    tabStops: [{ type: TabStopType.RIGHT, position: 364 }, { type: TabStopType.LEFT, position: 482 }],
    spacing: { before: 0, after: pt(9), line: pt(15.95), lineRule: LineRuleType.EXACT },
    children: [new TextRun({ children: [new Tab()] }), new Bookmark({ id: `ref_${r.n}`, children: [T(`[${r.n}]`)] }), new TextRun({ children: [new Tab()] }), ...runs(r.runs)],
  }))];

// ---------------------------------------------------------------- sections
const sections = [];
const hfNone = emptyHF();
sections.push({ properties: { ...PAGE({ start: 1, formatType: NumberFormat.LOWER_ROMAN }) },
  headers: { default: hfNone.header }, footers: { default: hfNone.footer },
  children: [...titlePage, ...approval, ...certificate, ...declaration, ...ack] });
sections.push({ properties: { type: SectionType.NEXT_PAGE, ...PAGE({ formatType: NumberFormat.LOWER_ROMAN }) },
  headers: { default: fancyHeader("") }, footers: { default: ruleFooter() }, children: abstract });
const starSection = (name, children) => ({ properties: { type: SectionType.NEXT_PAGE, titlePage: true, ...PAGE({ formatType: NumberFormat.LOWER_ROMAN }) },
  headers: { default: fancyHeader(name), first: blankHeader() }, footers: { default: ruleFooter(), first: plainFooter() }, children });
sections.push(starSection("Contents", toc));
sections.push(starSection("List of Figures", listOf("figure")));
sections.push(starSection("List of Tables", listOf("table")));
sections.push({ properties: { type: SectionType.NEXT_PAGE, ...PAGE({ formatType: NumberFormat.LOWER_ROMAN }) },
  headers: { default: fancyHeader("List of Abbreviations") }, footers: { default: ruleFooter() }, children: abbreviations });
chapters.forEach((c, k) => sections.push({
  properties: { type: SectionType.NEXT_PAGE, titlePage: true, ...PAGE(k === 0 ? { start: 1, formatType: NumberFormat.DECIMAL } : { formatType: NumberFormat.DECIMAL }) },
  headers: { default: fancyHeader(`Chapter ${c.num}. ${c.title}`), first: blankHeader() }, footers: { default: ruleFooter(), first: plainFooter() },
  children: c.children,
}));
sections.push({ properties: { type: SectionType.NEXT_PAGE, titlePage: true, ...PAGE({ formatType: NumberFormat.DECIMAL }) },
  headers: { default: fancyHeader("References"), first: blankHeader() }, footers: { default: ruleFooter(), first: plainFooter() }, children: refs });

// ---------------------------------------------------------------- document
const doc = new Document({
  creator: M.StudentA, title: M.ReportTitle, description: "Phase I Project Progress Report",
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: BODY, size: 24 }, paragraph: { spacing: { before: 0, after: 0 } } } },
    paragraphStyles: [
      { id: "Body", name: "Body Text NITW", basedOn: "Normal", run: { font: BODY, size: 24 },
        paragraph: { alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 359 }, spacing: { before: 0, after: 80, line: LINE, lineRule: LineRuleType.EXACT }, widowControl: true } },
      { id: "ListItem", name: "List Item NITW", basedOn: "Body", run: { font: BODY, size: 24 }, paragraph: { indent: { left: 516, hanging: 300 } } },
      { id: "ChapterLabel", name: "Chapter Label", basedOn: "Normal", next: "Heading1", run: { font: BODY, size: 50, bold: true }, paragraph: { keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Body", run: { font: BODY, size: 50, bold: true }, paragraph: { keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Body", run: { font: BODY, size: 35, bold: true }, paragraph: { keepNext: true, keepLines: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Body", run: { font: BODY, size: 29, bold: true }, paragraph: { keepNext: true, keepLines: true, outlineLevel: 2 } },
      { id: "FrontHead", name: "Front Matter Heading", basedOn: "Normal", next: "Body", run: { font: BODY, size: 41, bold: true }, paragraph: { keepNext: true } },
      { id: "Caption", name: "Caption", basedOn: "Normal", run: { font: SMALLF, size: 22 }, paragraph: {} },
      { id: "TocBase", name: "TOC Entry NITW", basedOn: "Normal", run: { font: BODY, size: 24 }, paragraph: {} },
    ],
  },
  numbering: { config: [
    { reference: "bullet", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 516, hanging: 300 } }, run: { font: BODY } } }] },
    { reference: "enum", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 516, hanging: 300 } } } }] },
    { reference: "alpha", levels: [{ level: 0, format: LevelFormat.LOWER_LETTER, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 468, hanging: 298 } } } }] },
  ] },
  sections,
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync("report45_raw.docx", buf); console.log("written report45_raw.docx", buf.length); });
