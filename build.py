"""Assemble the manuscript markdown, convert with pandoc (native Word equations) and enforce the
black-and-white journal format with python-docx."""
import copy
import os
import re
import subprocess
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Inches, Cm, RGBColor

P = 'paper'
FIGS = {
    '01': ('figs/fig01_system.png', 6.3, 'System diagram of GATC: the strategic layer runs once per signal cycle, the tactical layer checks the stop-line zone of the green arm every second, and the graph attention estimator is trained offline in simulation.'),
    '02': ('figs/fig02_corridor.png', 6.3, 'Three-junction corridor with left-hand traffic, 60 m camera fields of view on all 12 arms, link lengths and storage, S2 boundary flows and the location of the S5 lane blockage.'),
    '03': ('figs/fig04_mix.png', 5.6, 'Vehicle composition of the base traffic mix and the design PCU weight w of each of the seven vehicle groups.'),
    '04': ('figs/fig05_demand.png', 5.8, 'Network inflow over time in the demand scenarios; shaded bands mark the warm-up period and the S5 incident window.'),
    '05': ('figs/fig03_graph.png', 6.2, 'Junction graph used by the estimator: 12 arm nodes, 3 junction nodes and four edge sets, with directed edge counts in brackets.'),
    '06': ('figs/fig06_webster.png', 5.4, "Simulator verification: average control delay at an isolated fixed-time junction against Webster's delay formula (cars only, 96 s cycle, 19 s greens)."),
    '07': ('figs/fig07_training.png', 5.6, 'Training and held-out loss of the four estimators.'),
    '08': ('figs/fig08_scatter.png', 3.9, 'Predicted against observed next-cycle arm demand on held-out episodes for the calibrated camera snapshot and the GAT estimator (2,500 random arm snapshots each).'),
    '09': ('figs/fig09_bars.png', 6.3, 'Average waiting time by scenario and controller: means of 20 seeds with bootstrap 95% confidence intervals. MP-C and FT-W* are omitted for scale (see Table 8).'),
    '10': ('figs/fig10_heat_change.png', 5.8, 'Relative change in average waiting time of the proposed controller modes against each baseline; blue cells favour the proposed controller and colours saturate at ±60%.'),
    '11': ('figs/fig11_dqn_learning.png', 5.8, 'Learning curves of the deep Q-networks: greedy checkpoints of the selected DQN-C configuration, of a camera configuration trained with the DQN-O settings and of DQN-O (exact state), evaluated on the tuning seeds (mean of S1-S5), with the tuned controllers on the same seeds; open circles mark the frozen checkpoints (logarithmic axis).'),
    '12': ('figs/fig11_series_s5.png', 6.0, 'Queued demand summed over all arms during the S5 lane blockage (seed 1, 60 s moving average).'),
    '13': ('figs/fig12_cdf_s3.png', 5.6, 'Distribution of per-vehicle waiting time in the oversaturated scenario S3 (all measured vehicles of 20 seeds).'),
    '14': ('figs/fig13_ablation.png', 6.2, 'Ablation: change in average waiting time relative to the full controller mode, averaged over S1-S5, with bootstrap 95% intervals.'),
    '15': ('figs/fig14_sweep_GATC-S.png', 4.4, 'Sensitivity of GATC-S to detector recall and false positives in the peak scenario S2 (10 seeds per cell).'),
    '16': ('figs/fig15_attention.png', 4.9, 'Mean second-layer attention coefficients of the GAT estimator during S4 (seed 1), averaged over heads and decisions; empty cells are not graph neighbours.'),
}
TABLE_ORDER = ['lit', 'notation', 'classes', 'profiles', 'hyper', 'tuning', 'estimator', 'main', 'stats', 'secondary',
               'ablation', 'perception', 'cost']
SECTIONS = ['01_intro.md', '02_litreview.md', '03_methods.md', '04_results.md', '05_discussion.md', '06_conclusion.md']


def load_table(key):
    t = open(f'{P}/tables/{key}.md').read().strip()
    cap, body = t.split('\n\n', 1)
    cap = cap.replace('CAPTION: ', '')
    return cap, body


def assemble(cite):
    front = open(f'{P}/00_front.md').read()
    title = re.search(r'^TITLE: (.*)$', front, re.M).group(1)
    kw = re.search(r'^KEYWORDS: (.*)$', front, re.M).group(1)
    abstract = front.split('# Abstract')[1].split('KEYWORDS:')[0].strip()
    out = [f'{{{{TITLE}}}}{title}', '', '# Abstract', '', abstract, '', f'{{{{KW}}}}Keywords: {kw}', '']
    sec = 0
    tnum = {}
    fnum = 0
    for fn in SECTIONS:
        txt = open(f'{P}/{fn}').read()
        lines = []
        sub = 0
        for line in txt.split('\n'):
            if line.startswith('# '):
                sec += 1
                sub = 0
                lines.append(f'# {sec}. {line[2:].strip()}')
            elif line.startswith('## '):
                sub += 1
                lines.append(f'## {sec}.{sub} {line[3:].strip()}')
            elif line.startswith('TABLE:'):
                key = line.split(':', 1)[1].strip()
                cap, body = load_table(key)
                n = TABLE_ORDER.index(key) + 1
                tnum[key] = n
                assert cap.startswith(f'Table {n}.'), (key, cap[:20], n)
                lines.append(f'{{{{TCAP}}}}{cap}')
                lines.append('')
                lines.append(body)
            elif line.startswith('FIGURE:'):
                key = line.split(':', 1)[1].strip()
                fnum += 1
                assert int(key) == fnum, (key, fnum)
                path, w, cap = FIGS[key]
                lines.append(f'![]({path}){{width={w}in}}')
                lines.append('')
                lines.append(f'{{{{FCAP}}}}Figure {fnum}. {cap}')
            elif line.startswith('ALGORITHM:'):
                lines.append('{{ALGO1}}')
            else:
                lines.append(line)
        out.append('\n'.join(lines))
        out.append('')
    for fn in ['07_declarations.md']:
        out.append(open(f'{P}/{fn}').read())
        out.append('')
    refs = open(f'{P}/refs.txt').read().split('\n')
    out.append('# References')
    out.append('')
    for i, r in enumerate(refs, 1):
        out.append(f'{{{{REF}}}}\\[{i}\\] {r}')
        out.append('')
    md = '\n'.join(out)
    if cite:
        md = re.sub(r'\[@(\d+)\]', r'[\1]', md)
    else:
        md = re.sub(r'\s*\[@\d+\](,\s*\[@\d+\])*', '', md)
    # check that every figure/table number mentioned in text exists
    return md, tnum, fnum


def make_reference_docx(path):
    subprocess.run(['pandoc', '-o', path, '--print-default-data-file', 'reference.docx'], check=True)
    d = Document(path)
    for st in d.styles:
        try:
            f = st.font
        except Exception:
            continue
        if f is None:
            continue
        f.name = 'Times New Roman'
        f.color.rgb = RGBColor(0, 0, 0)
        rpr = st.element.get_or_add_rPr()
        rfonts = rpr.find(qn('w:rFonts'))
        if rfonts is None:
            rfonts = OxmlElement('w:rFonts')
            rpr.append(rfonts)
        for att in ('w:ascii', 'w:hAnsi', 'w:cs', 'w:eastAsia'):
            rfonts.set(qn(att), 'Times New Roman')
        for att in ('w:asciiTheme', 'w:hAnsiTheme', 'w:cstheme', 'w:eastAsiaTheme'):
            if rfonts.get(qn(att)) is not None:
                del rfonts.attrib[qn(att)]
        col = rpr.find(qn('w:color'))
        if col is not None:
            for att in ('w:themeColor', 'w:themeShade', 'w:themeTint'):
                if col.get(qn(att)) is not None:
                    del col.attrib[qn(att)]
            col.set(qn('w:val'), '000000')
        u = rpr.find(qn('w:u'))
        if u is not None:
            rpr.remove(u)
    d.save(path)


def set_cell_borders(table):
    tbl = table._tbl
    tblPr = tbl.tblPr
    for old in tblPr.findall(qn('w:tblBorders')):
        tblPr.remove(old)
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        el = OxmlElement(f'w:{edge}')
        el.set(qn('w:val'), 'single')
        el.set(qn('w:sz'), '6')
        el.set(qn('w:space'), '0')
        el.set(qn('w:color'), '000000')
        borders.append(el)
    tblPr.append(borders)
    # remove table style look (banding) and shading
    for old in tblPr.findall(qn('w:tblLook')):
        tblPr.remove(old)
    for old in tblPr.findall(qn('w:shd')):
        tblPr.remove(old)
    w = tblPr.find(qn('w:tblW'))
    if w is None:
        w = OxmlElement('w:tblW')
        tblPr.append(w)
    w.set(qn('w:type'), 'pct')
    w.set(qn('w:w'), '5000')
    for row in table.rows:
        for cell in row.cells:
            tcPr = cell._tc.get_or_add_tcPr()
            for old in tcPr.findall(qn('w:tcBorders')):
                tcPr.remove(old)
            for old in tcPr.findall(qn('w:shd')):
                tcPr.remove(old)
            tb = OxmlElement('w:tcBorders')
            for edge in ('top', 'left', 'bottom', 'right'):
                el = OxmlElement(f'w:{edge}')
                el.set(qn('w:val'), 'single')
                el.set(qn('w:sz'), '6')
                el.set(qn('w:space'), '0')
                el.set(qn('w:color'), '000000')
                tb.append(el)
            tcPr.append(tb)
            mar = OxmlElement('w:tcMar')
            for edge, v in (('top', 25), ('bottom', 25), ('left', 70), ('right', 70)):
                el = OxmlElement(f'w:{edge}')
                el.set(qn('w:w'), str(v))
                el.set(qn('w:type'), 'dxa')
                mar.append(el)
            for old in tcPr.findall(qn('w:tcMar')):
                tcPr.remove(old)
            tcPr.append(mar)


def clean_run(r, size=None, bold=None, italic=None):
    rpr = r._r.get_or_add_rPr()
    for tag in ('w:color', 'w:u', 'w:highlight', 'w:shd'):
        for el in rpr.findall(qn(tag)):
            rpr.remove(el)
    r.font.name = 'Times New Roman'
    rfonts = rpr.find(qn('w:rFonts'))
    if rfonts is not None:
        for att in ('w:ascii', 'w:hAnsi', 'w:cs', 'w:eastAsia'):
            rfonts.set(qn(att), 'Times New Roman')
    if size is not None:
        r.font.size = Pt(size)
    if bold is not None:
        r.font.bold = bold
    if italic is not None:
        r.font.italic = italic


def para_fmt(p, align=None, before=0, after=6, line=1.15, keep=False, left=None, first=None):
    pf = p.paragraph_format
    if align is not None:
        p.alignment = align
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line
    pf.keep_with_next = keep
    if left is not None:
        pf.left_indent = left
    if first is not None:
        pf.first_line_indent = first


def strip_marker(p, marker):
    for r in p.runs:
        if marker in r.text:
            r.text = r.text.replace(marker, '')
            return True
    return False


def bold_label(p, pattern):
    """split the first run so that the label (e.g. 'Table 3.') is bold."""
    text = p.text
    m = re.match(pattern, text)
    if not m:
        return
    label = m.group(0)
    first = p.runs[0]
    if first.text.startswith(label):
        rest = first.text[len(label):]
        first.text = label
        first.font.bold = True
        if rest:
            new = copy.deepcopy(first._r)
            first._r.addnext(new)
            from docx.text.run import Run
            nr = Run(new, p)
            nr.text = rest
            nr.font.bold = False


def add_algorithm(doc, anchor_p):
    lines = open(f'{P}/algorithm.txt').read().split('\n')
    title = lines[0].replace('ALGORITHM_TITLE: ', '')
    tbl = doc.add_table(rows=2, cols=1)
    anchor_p._p.addnext(tbl._tbl)
    hdr = tbl.rows[0].cells[0].paragraphs[0]
    m = re.match(r'(Algorithm 1\.)(.*)', title)
    r = hdr.add_run(m.group(1))
    clean_run(r, 10, True)
    r = hdr.add_run(m.group(2))
    clean_run(r, 10, True)
    para_fmt(hdr, WD_ALIGN_PARAGRAPH.LEFT, 2, 2, 1.0)
    cell = tbl.rows[1].cells[0]
    first = True
    for ln in lines[1:]:
        p = cell.paragraphs[0] if first else cell.add_paragraph()
        first = False
        m = re.match(r'^(\d+):\s*(.*)$', ln)
        indent = 0
        if m:
            num, rest = m.group(1), m.group(2)
            raw = ln.split(':', 1)[1]
            indent = (len(raw) - len(raw.lstrip(' '))) // 4
            head = f'{num}:'
            rr = p.add_run(head + '\t')
            clean_run(rr, 9.5)
            body = rest
        else:
            body = ln
        # render _{...} as subscript
        parts = re.split(r'(_\{[^}]*\})', body)
        for part in parts:
            if part.startswith('_{'):
                rr = p.add_run(part[2:-1])
                clean_run(rr, 9.5)
                rr.font.subscript = True
            elif part:
                if part.startswith('Input:') or part.startswith('Output:'):
                    lab = part.split(':', 1)[0] + ':'
                    rr = p.add_run(lab)
                    clean_run(rr, 9.5, True)
                    part = part[len(lab):]
                rr = p.add_run(part)
                clean_run(rr, 9.5)
        para_fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 0, 1, 1.0, left=Cm(0.9 + 0.5 * indent) if m else None,
                 first=Cm(-0.9) if m else None)
        p.paragraph_format.tab_stops.add_tab_stop(Cm(0.9 + 0.5 * indent))
    set_cell_borders(tbl)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for row in tbl.rows:
        trPr = row._tr.get_or_add_trPr()
        trPr.append(OxmlElement('w:cantSplit'))
    for p in tbl.rows[0].cells[0].paragraphs:
        p.paragraph_format.keep_with_next = True
    # blank spacer after the table
    spacer = OxmlElement('w:p')
    tbl._tbl.addnext(spacer)
    anchor_p._p.getparent().remove(anchor_p._p)


def postprocess(path, out):
    doc = Document(path)
    # every style (including unused syntax-highlighting styles) in black, without theme colours or underline
    for col in doc.styles.element.iter(qn('w:color')):
        col.set(qn('w:val'), '000000')
        for att in ('w:themeColor', 'w:themeShade', 'w:themeTint'):
            if col.get(qn(att)) is not None:
                del col.attrib[qn(att)]
    for u in list(doc.styles.element.iter(qn('w:u'))):
        u.getparent().remove(u)
    for s in doc.sections:
        s.page_height = Cm(29.7)
        s.page_width = Cm(21.0)
        s.left_margin = s.right_margin = Cm(2.3)
        s.top_margin = s.bottom_margin = Cm(2.3)
        s.header_distance = Cm(1.25)
        s.footer_distance = Cm(1.25)
        sp = s._sectPr
        for tag in ('w:headerReference', 'w:footerReference', 'w:pgNumType'):
            for el in sp.findall(qn(tag)):
                sp.remove(el)
    algo_anchor = None
    in_refs = False
    body_paras = list(doc.paragraphs)
    for p in body_paras:
        txt = p.text
        sname = p.style.name if p.style is not None else ''
        for r in p.runs:
            clean_run(r)
        if '{{ALGO1}}' in txt:
            algo_anchor = p
            continue
        if txt.startswith('{{TITLE}}'):
            strip_marker(p, '{{TITLE}}')
            for r in p.runs:
                clean_run(r, 16, True)
            para_fmt(p, WD_ALIGN_PARAGRAPH.CENTER, 0, 14, 1.1)
            continue
        if txt.startswith('{{KW}}'):
            strip_marker(p, '{{KW}}')
            bold_label(p, r'Keywords:')
            for r in p.runs:
                clean_run(r, 10.5)
            para_fmt(p, WD_ALIGN_PARAGRAPH.JUSTIFY, 4, 10, 1.15)
            continue
        if txt.startswith('{{TCAP}}'):
            strip_marker(p, '{{TCAP}}')
            for r in p.runs:
                clean_run(r, 9.5)
            bold_label(p, r'Table \d+\.')
            para_fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 8, 3, 1.05, keep=True)
            continue
        if txt.startswith('{{FCAP}}'):
            strip_marker(p, '{{FCAP}}')
            for r in p.runs:
                clean_run(r, 9.5)
            bold_label(p, r'Figure \d+\.')
            para_fmt(p, WD_ALIGN_PARAGRAPH.CENTER, 2, 10, 1.05)
            continue
        if txt.startswith('{{REF}}'):
            strip_marker(p, '{{REF}}')
            for r in p.runs:
                clean_run(r, 9.5)
            para_fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 0, 3, 1.05, left=Cm(0.75), first=Cm(-0.75))
            continue
        if sname.startswith('Heading 1'):
            for r in p.runs:
                clean_run(r, 12.5, True, False)
            para_fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 14, 4, 1.1, keep=True)
            continue
        if sname.startswith('Heading 2'):
            for r in p.runs:
                clean_run(r, 11, True, False)
            para_fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 10, 3, 1.1, keep=True)
            continue
        has_img = bool(p._p.findall('.//' + qn('w:drawing')))
        has_math = bool(p._p.findall('.//' + qn('m:oMathPara')))
        if has_img:
            para_fmt(p, WD_ALIGN_PARAGRAPH.CENTER, 6, 2, 1.0, keep=True)
            continue
        if has_math:
            para_fmt(p, WD_ALIGN_PARAGRAPH.CENTER, 3, 3, 1.0)
            continue
        for r in p.runs:
            clean_run(r, 11)
        if sname in ('Compact',) or p._p.getparent().tag == qn('w:tc'):
            continue
        para_fmt(p, WD_ALIGN_PARAGRAPH.JUSTIFY, 0, 6, 1.15)
    # tables
    for t in doc.tables:
        set_cell_borders(t)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, row in enumerate(t.rows):
            # repeat header row, keep rows intact
            trPr = row._tr.get_or_add_trPr()
            if i == 0:
                el = OxmlElement('w:tblHeader')
                trPr.append(el)
            cs = OxmlElement('w:cantSplit')
            trPr.append(cs)
            for cell in row.cells:
                for p in cell.paragraphs:
                    for r in p.runs:
                        clean_run(r, 8.5, True if i == 0 else None)
                        if i > 0 and r.font.bold and not r.text.strip().startswith(('GATC', 'Table')):
                            pass
                    para_fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 0, 0, 1.0)
    if algo_anchor is not None:
        add_algorithm(doc, algo_anchor)
    # remove any hyperlinks' styling
    for rs in doc.element.body.iter(qn('w:rStyle')):
        if rs.get(qn('w:val')) in ('Hyperlink',):
            rs.getparent().remove(rs)
    doc.save(out)


def build(cite, out):
    md, tnum, fnum = assemble(cite)
    mdpath = out.replace('.docx', '.md')
    open(mdpath, 'w').write(md)
    ref = 'build_reference.docx'
    if not os.path.exists(ref):
        make_reference_docx(ref)
    tmp = out.replace('.docx', '_raw.docx')
    subprocess.run(['pandoc', mdpath, '-f', 'markdown+tex_math_dollars-auto_identifiers', '-t', 'docx',
                    '--reference-doc', ref, '--wrap=none', '-o', tmp], check=True)
    postprocess(tmp, out)
    os.remove(tmp)
    return md


if __name__ == '__main__':
    os.makedirs('out', exist_ok=True)
    build(False, 'out/manuscript.docx')
    build(True, 'out/manuscript_with_citations.docx')
    print('built')
