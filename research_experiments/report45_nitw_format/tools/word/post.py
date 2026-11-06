"""Post-process the docx written by build.js: native Word equations (OMML from pandoc) in place of the
@@EQx.y@@ placeholders, and a serif fallback entry for the Latin Modern fonts in the font table.
python3 -I post.py report45_raw.docx report45.docx
"""
import re
import shutil
import sys
import zipfile

src, dst = sys.argv[1], sys.argv[2]
import json, os
PAGECACHE = json.load(open("pagecache.json")) if os.path.exists("pagecache.json") else {}
eqdoc = open("/tmp/claude-0/r45docx/eqs_unz/word/document.xml", encoding="utf8").read()
paras = re.findall(r"<w:p>.*?</w:p>|<w:p .*?</w:p>", eqdoc, re.S)
omml, label = {}, None
for p in paras:
    t = "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", p))
    m = re.match(r"EQ(\d\.\d)$", t.strip())
    if m:
        label = m.group(1); continue
    mp = re.search(r"<m:oMathPara>.*</m:oMathPara>", p, re.S)
    if mp and label:
        omml[label] = mp.group(0); label = None
assert len(omml) == 4, omml.keys()

zin = zipfile.ZipFile(src)
tmp = dst + ".tmp"
zout = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    if item.filename.endswith("/"):
        continue
    data = zin.read(item.filename)
    if item.filename == "word/document.xml":
        x = data.decode("utf8")
        if 'xmlns:m="' not in x[:3000]:
            x = x.replace("<w:document ", '<w:document xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" ', 1)
        for num, om in omml.items():
            pat = re.compile(r"<w:r>(?:(?!<w:r>).)*?@@EQ" + re.escape(num) + r"@@</w:t></w:r>", re.S)
            x, n = pat.subn(lambda m: om, x)
            assert n == 1, (num, n)
        assert "@@EQ" not in x
        # PAGEREF fields: docx-js nests them in a run and gives no result; write the standard field form
        # (begin / instruction / separate / cached result / end) carrying the surrounding run formatting
        cache = PAGECACHE

        def field(rpr, bm):
            r = f"<w:rPr>{rpr}</w:rPr>" if rpr else ""
            res = cache.get(bm, "")
            return (f'<w:r>{r}<w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r>'
                    f'<w:r>{r}<w:instrText xml:space="preserve"> PAGEREF {bm} \\h </w:instrText></w:r>'
                    f'<w:r>{r}<w:fldChar w:fldCharType="separate"/></w:r><w:r>{r}<w:t>{res}</w:t></w:r>'
                    f'<w:r>{r}<w:fldChar w:fldCharType="end"/></w:r>')
        FLD = r'<w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/><w:instrText xml:space="preserve">PAGEREF (\w+)</w:instrText><w:fldChar w:fldCharType="end"/></w:r>'
        x, n1 = re.subn(r"<w:r><w:rPr>((?:(?!</w:rPr>).)*)</w:rPr>" + FLD + r"</w:r>", lambda m: field(m.group(1), m.group(2)), x)
        x, n2 = re.subn(FLD, lambda m: field("", m.group(1)), x)
        print("page references:", n1 + n2, "| cached:", sum(1 for k in re.findall(r"PAGEREF (\w+)", x) if k in cache))
        # pandoc OMML -> schema order
        x = x.replace('<m:nor /><m:sty m:val="p" />', "<m:nor />")
        # one upright run per function name (pandoc writes \\mathrm{ReLU} letter by letter)
        UP = r'<m:r><m:rPr><m:sty m:val="p" /></m:rPr><m:t>([A-Za-z0-9]+)</m:t></m:r>'
        prevx = None
        while prevx != x:
            prevx = x
            x = re.sub(UP + UP, lambda m: f'<m:r><m:rPr><m:sty m:val="p" /></m:rPr><m:t>{m.group(1)}{m.group(2)}</m:t></m:r>', x)
        x = re.sub(r'<m:dPr>(<m:begChr [^>]*/>)(<m:endChr [^>]*/>)(<m:sepChr [^>]*/>)', r"<m:dPr>\1\3\2", x)
        x = re.sub(r'<m:mcPr>(<m:mcJc [^>]*/>)(<m:count [^>]*/>)</m:mcPr>', r"<m:mcPr>\2\1</m:mcPr>", x)
        # unique bookmark ids (start/end pairs nest, never overlap)
        stack, counter = [], [0]

        def bm(m):
            if m.group(1) == "Start":
                counter[0] += 1; stack.append(counter[0]); i = counter[0]
            else:
                i = stack.pop()
            return m.group(0).replace(m.group(2), f'w:id="{i}"')
        x = re.sub(r"<w:bookmark(Start|End)\b[^>]*?(w:id=\"\d+\")", bm, x)
        assert not stack
        data = x.encode("utf8")
    if item.filename == "word/fontTable.xml":
        x = data.decode("utf8")
        extra = "".join(
            f'<w:font w:name="{f}"><w:altName w:val="Times New Roman"/><w:panose1 w:val="00000500000000000000"/>'
            f'<w:charset w:val="00"/><w:family w:val="roman"/><w:pitch w:val="variable"/></w:font>'
            for f in ("LM Roman 12", "LM Roman 10") if f'w:name="{f}"' not in x)
        x = x.replace("</w:fonts>", extra + "</w:fonts>")
        data = x.encode("utf8")
    zout.writestr(item, data)
zout.close()
shutil.move(tmp, dst)
print("equations inserted:", sorted(omml))
