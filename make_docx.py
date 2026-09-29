"""
Generate theory.docx from theory.md.

Reads the module's Markdown write-up and renders it into a Word document with a
simple, readable style.  Supported Markdown (enough for theory.md):

    # title          -> document title
    ## heading       -> section heading
    - item / 1. item -> bullet
    indented block   -> shaded monospaced formula block
    **bold** / `code`-> inline emphasis
    ---             -> paragraph spacer

The math is written in plain ASCII, so no equation renderer is needed.
Requires: pip install python-docx
Run:      python3 make_docx.py
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).parent
SRC = ROOT / "theory.md"
OUT = ROOT / "theory.docx"

doc = Document()

normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(11)


# ----------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------

def shade(paragraph, fill):
    """Light grey background for a paragraph (formula block)."""
    p_pr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    p_pr.append(shd)


def add_code_runs(paragraph, text, bold=False):
    """Add runs to a paragraph, honouring inline `code` backticks."""
    for j, seg in enumerate(text.split("`")):
        if seg == "":
            continue
        run = paragraph.add_run(seg)
        run.bold = bold
        if j % 2 == 1:  # inside backticks -> monospaced
            run.font.name = "Courier New"
            run.font.size = Pt(9.5)


def add_inline(paragraph, text):
    """Add runs honouring **bold** and `code` markers."""
    for i, seg in enumerate(text.split("**")):
        if seg == "":
            continue
        add_code_runs(paragraph, seg, bold=(i % 2 == 1))


def title(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(18)
    run.font.color.rgb = RGBColor(0x1A, 0x2B, 0x4A)
    p.paragraph_format.space_after = Pt(4)


def subtitle(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(10.5)
    run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    p.paragraph_format.space_after = Pt(12)


def heading(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(13)
    run.font.color.rgb = RGBColor(0x1A, 0x2B, 0x4A)
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)


def body(text):
    p = doc.add_paragraph()
    add_inline(p, text)
    p.paragraph_format.space_after = Pt(6)


def bullet(text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.space_after = Pt(3)
    add_inline(p, "-  " + text)


def formula(text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.2)
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(8)
    for j, line in enumerate(text.split("\n")):
        run = p.add_run(line)
        run.font.name = "Courier New"
        run.font.size = Pt(9.5)
        if j < len(text.split("\n")) - 1:
            run.add_break()
    shade(p, "F2F3F7")


def spacer():
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)


# ----------------------------------------------------------------------
# Markdown -> blocks
# ----------------------------------------------------------------------

ORDERED = re.compile(r"^\d+\.\s")


def parse(path: Path) -> list[tuple[str, str]]:
    lines = path.read_text().splitlines()
    blocks: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.strip() == "":
            i += 1
            continue
        if line.strip() == "---":
            blocks.append(("hr", ""))
            i += 1
            continue
        if line.startswith("# "):
            blocks.append(("title", line[2:].strip()))
            i += 1
            continue
        if line.startswith("## "):
            blocks.append(("heading", line[3:].strip()))
            i += 1
            continue
        if line.startswith("    ") or line.startswith("\t"):
            code = []
            while i < len(lines) and (lines[i].startswith("    ") or lines[i].startswith("\t") or lines[i].strip() == ""):
                code.append(lines[i][4:] if lines[i].startswith("    ") else lines[i].lstrip())
                i += 1
            while code and code[-1].strip() == "":
                code.pop()
            blocks.append(("code", "\n".join(code)))
            continue
        if line.startswith("- "):
            blocks.append(("bullet", line[2:].strip()))
            i += 1
            continue
        if ORDERED.match(line):
            blocks.append(("bullet", ORDERED.sub("", line).strip()))
            i += 1
            continue

        para = [line]
        i += 1
        while (i < len(lines) and lines[i].strip() != ""
               and not lines[i].startswith(("#", "---", "- ", "    ", "\t"))
               and not ORDERED.match(lines[i])):
            para.append(lines[i])
            i += 1
        blocks.append(("para", " ".join(para)))
    return blocks


# ----------------------------------------------------------------------
# Render
# ----------------------------------------------------------------------

for kind, text in parse(SRC):
    if kind == "title":
        title(text)
    elif kind == "heading":
        heading(text)
    elif kind == "para":
        # The line right after the title is the module subtitle.
        body(text)
    elif kind == "bullet":
        bullet(text)
    elif kind == "code":
        formula(text)
    elif kind == "hr":
        spacer()

doc.save(OUT)
print("Wrote", OUT)
