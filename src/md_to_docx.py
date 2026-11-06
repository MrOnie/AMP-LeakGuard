"""
md_to_docx.py
=============
Converts MANUSCRITO_BORRADOR_COMPLETO.md to a beautifully styled Microsoft Word (.docx) document:
- Professional typography (Calibri/Arial, 1-inch margins).
- Formatted Headings (Title, H1, H2, H3).
- Real Word Tables with shaded headers, borders, and centered text.
- Embedded high-resolution figures (300 DPI) where referenced.
- Formatted bullet points and paragraphs.
"""

import re
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn


def set_cell_background(cell, color_hex="F2F2F2"):
    """Sets background shading for a table cell."""
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets cell padding in dxa (1 pt = 20 dxa)."""
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def add_formatted_text(paragraph, text):
    """Parses basic markdown inline styles (**bold**, *italic*, `code`)."""
    # Regex for tokens: bold (**...**), italic (*...*), code (`...`)
    tokens = re.split(r'(\*\*.*?\*\*|\*.*?\*|`.*?`)', text)
    for token in tokens:
        if not token:
            continue
        if token.startswith("**") and token.endswith("**") and len(token) >= 4:
            run = paragraph.add_run(token[2:-2])
            run.bold = True
        elif token.startswith("*") and token.endswith("*") and len(token) >= 2:
            run = paragraph.add_run(token[1:-1])
            run.italic = True
        elif token.startswith("`") and token.endswith("`") and len(token) >= 2:
            run = paragraph.add_run(token[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(160, 50, 50)
        else:
            paragraph.add_run(token)


def convert_markdown_to_docx(md_path: Path, docx_path: Path):
    print(f"[*] Reading markdown from {md_path}...")
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    doc = Document()

    # Set 1-inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Base Normal Style
    style_normal = doc.styles["Normal"]
    style_normal.font.name = "Arial"
    style_normal.font.size = Pt(10.5)
    style_normal.font.color.rgb = RGBColor(30, 30, 30)
    style_normal.paragraph_format.line_spacing = 1.15
    style_normal.paragraph_format.space_after = Pt(6)

    in_table = False
    table_lines = []
    in_code_block = False
    code_block_lines = []

    def flush_table():
        nonlocal in_table, table_lines
        if not table_lines:
            return
        
        # Parse rows
        rows_data = []
        for t_line in table_lines:
            if re.match(r'^\s*\|[-:\s|]+\|\s*$', t_line):
                continue  # skip separator line
            cells = [c.strip() for c in t_line.strip().strip('|').split('|')]
            if any(cells):
                rows_data.append(cells)

        if len(rows_data) >= 2:
            num_cols = max(len(r) for r in rows_data)
            table = doc.add_table(rows=len(rows_data), cols=num_cols)
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            table.autofit = True

            for r_idx, row in enumerate(rows_data):
                for c_idx, cell_text in enumerate(row):
                    if c_idx < num_cols:
                        cell = table.cell(r_idx, c_idx)
                        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                        p = cell.paragraphs[0]
                        p.paragraph_format.space_after = Pt(2)
                        p.paragraph_format.space_before = Pt(2)
                        p.paragraph_format.line_spacing = 1.0
                        set_cell_margins(cell, top=80, bottom=80, left=120, right=120)

                        if r_idx == 0:  # Header row
                            set_cell_background(cell, "2B5C8F")
                            add_formatted_text(p, f"**{cell_text}**")
                            p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
                            p.runs[0].font.size = Pt(9.5)
                        else:  # Data rows
                            if r_idx % 2 == 1:
                                set_cell_background(cell, "F8F9FA")
                            else:
                                set_cell_background(cell, "FFFFFF")
                            add_formatted_text(p, cell_text)
                            if p.runs:
                                p.runs[0].font.size = Pt(9.0)

            # Space after table
            p_space = doc.add_paragraph()
            p_space.paragraph_format.space_after = Pt(6)

        in_table = False
        table_lines = []

    def flush_code_block():
        nonlocal in_code_block, code_block_lines
        if not code_block_lines:
            return
        block_text = "".join(code_block_lines)
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.right_indent = Inches(0.25)
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(6)
        run = p.add_run(block_text)
        run.font.name = "Consolas"
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor(40, 40, 40)
        in_code_block = False
        code_block_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Handle Code Blocks
        if stripped.startswith("```"):
            if in_code_block:
                flush_code_block()
            else:
                if in_table:
                    flush_table()
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            code_block_lines.append(line)
            i += 1
            continue

        # Handle Tables
        if stripped.startswith("|") and stripped.endswith("|"):
            in_table = True
            table_lines.append(stripped)
            i += 1
            continue
        elif in_table:
            flush_table()

        # Handle Empty Lines
        if not stripped:
            i += 1
            continue

        # Handle Headings
        if stripped.startswith("# "):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(12)
            run = p.add_run(stripped[2:])
            run.font.name = "Arial"
            run.font.size = Pt(18)
            run.bold = True
            run.font.color.rgb = RGBColor(20, 50, 90)

        elif stripped.startswith("## "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run(stripped[3:])
            run.font.name = "Arial"
            run.font.size = Pt(14)
            run.bold = True
            run.font.color.rgb = RGBColor(30, 70, 120)

        elif stripped.startswith("### "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(3)
            run = p.add_run(stripped[4:])
            run.font.name = "Arial"
            run.font.size = Pt(12)
            run.bold = True
            run.font.color.rgb = RGBColor(50, 80, 130)

        elif stripped.startswith("#### "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(stripped[5:])
            run.font.name = "Arial"
            run.font.size = Pt(11)
            run.bold = True
            run.italic = True
            run.font.color.rgb = RGBColor(60, 60, 60)

        # Handle Horizontal Rules
        elif stripped in ("---", "***", "___"):
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(6)

        # Handle Bullet points
        elif stripped.startswith("* ") or stripped.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.15
            add_formatted_text(p, stripped[2:])

        # Handle Numbered lists
        elif re.match(r'^\d+\.\s+', stripped):
            p = doc.add_paragraph(style="List Number")
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.15
            clean_text = re.sub(r'^\d+\.\s+', '', stripped)
            add_formatted_text(p, clean_text)

        # Handle Standalone Equations ($$ ... $$)
        elif stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            run = p.add_run(stripped[2:-2].strip())
            run.font.name = "Cambria Math"
            run.font.size = Pt(10.5)
            run.italic = True

        # Check for image inclusions
        elif "reports/figures/" in stripped and (".png" in stripped or ".jpg" in stripped):
            # Extract image path
            match = re.search(r'reports/figures/[a-zA-Z0-9_\.]+\.(?:png|jpg)', stripped)
            if match:
                img_path = Path(match.group(0))
                if img_path.exists():
                    p_img = doc.add_paragraph()
                    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p_img.paragraph_format.space_before = Pt(8)
                    p_img.paragraph_format.space_after = Pt(4)
                    doc.add_picture(str(img_path), width=Inches(6.0))
                    # Add caption if available
                    p_cap = doc.add_paragraph()
                    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p_cap.paragraph_format.space_after = Pt(8)
                    cap_run = p_cap.add_run(f"Figure: {img_path.name}")
                    cap_run.font.size = Pt(9.0)
                    cap_run.italic = True
            else:
                p = doc.add_paragraph()
                add_formatted_text(p, stripped)

        # Standard Paragraphs
        else:
            p = doc.add_paragraph()
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_after = Pt(6)
            add_formatted_text(p, stripped)

        i += 1

    if in_table:
        flush_table()
    if in_code_block:
        flush_code_block()

    docx_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(docx_path)
    print(f"[+] Successfully converted and saved Microsoft Word document to {docx_path}")


if __name__ == "__main__":
    md_file = Path("MANUSCRITO_BORRADOR_COMPLETO.md")
    docx_file = Path("MANUSCRITO_BORRADOR_COMPLETO.docx")
    convert_markdown_to_docx(md_file, docx_file)
