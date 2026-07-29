"""Convert docs/pipeline_overview.md into a single self-contained HTML file
with all images and GIF embedded as base64. The output is portable: copy the
single .html file anywhere, open in a browser, print to PDF, or import into
Word via "Open with Word".
"""
from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path

import markdown

DOCS = Path(__file__).resolve().parent
MD_PATH = DOCS / "pipeline_overview.md"
HTML_PATH = DOCS / "pipeline_overview.html"


def embed_image(html: str, base_dir: Path) -> str:
    """Replace <img src="figures/..."> with data: URLs."""
    pattern = re.compile(r'<img([^>]*?)\ssrc="([^"]+)"([^>]*)>')

    def repl(match: re.Match) -> str:
        before, src, after = match.group(1), match.group(2), match.group(3)
        if src.startswith("data:") or src.startswith("http"):
            return match.group(0)
        src_path = (base_dir / src).resolve()
        if not src_path.exists():
            return match.group(0)
        mime, _ = mimetypes.guess_type(str(src_path))
        if mime is None:
            mime = "application/octet-stream"
        data = src_path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f'<img{before} src="data:{mime};base64,{b64}"{after}>'

    return pattern.sub(repl, html)


def strip_local_anchors(html: str) -> str:
    """Drop <a href="..."> wrappers when the href is a local relative path.

    These links target repo files outside the HTML (e.g. ../Tools/x.py,
    figures/file.gif). They work from the docs/ folder but break the moment the
    HTML is moved or shared. Stripping the anchor keeps the readable text
    (often already styled as <code>) without misleading the reader with a dead
    link. http(s):// and #fragment links are kept intact.
    """
    pattern = re.compile(r'<a\s+([^>]*?)href="([^"]+)"([^>]*)>(.*?)</a>',
                         flags=re.DOTALL | re.IGNORECASE)

    def repl(match: re.Match) -> str:
        href = match.group(2).strip()
        inner = match.group(4)
        if href.startswith(("http://", "https://", "mailto:", "#", "data:")):
            return match.group(0)
        # Anything else (relative path, ../something, figures/x.png, etc.) is a
        # broken-when-shared link. Keep the visible text only.
        return inner

    return pattern.sub(repl, html)


def main() -> None:
    md_text = MD_PATH.read_text(encoding="utf-8")
    md = markdown.Markdown(extensions=["extra", "toc", "sane_lists", "tables"])
    body = md.convert(md_text)
    body = embed_image(body, DOCS)
    body = strip_local_anchors(body)

    style = """
    <style>
        body { font-family: Calibri, "Segoe UI", Arial, sans-serif;
               max-width: 1080px; margin: 40px auto; padding: 0 24px;
               color: #1f1f1f; line-height: 1.55; }
        h1 { border-bottom: 2px solid #999; padding-bottom: 8px; }
        h2 { border-bottom: 1px solid #ccc; padding-bottom: 6px; margin-top: 36px; }
        h3 { margin-top: 28px; }
        code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-size: 0.94em; }
        pre { background: #f4f4f4; padding: 12px 16px; border-radius: 4px;
              overflow-x: auto; font-size: 0.92em; }
        pre code { background: transparent; padding: 0; }
        table { border-collapse: collapse; margin: 12px 0; }
        th, td { border: 1px solid #bbb; padding: 6px 10px; text-align: left; }
        th { background: #eaeaea; }
        img { max-width: 100%; height: auto; margin: 12px 0;
              border: 1px solid #ddd; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
        blockquote { border-left: 4px solid #d0a000; background: #fff8e0;
                     margin: 12px 0; padding: 8px 14px; }
        a { color: #1a73e8; }
    </style>
    """

    footer = (
        '<hr>\n'
        '<p style="color:#777; font-size:0.85em;">'
        "Filenames shown as code are repository paths in the original markdown "
        "doc. Anchors to local files were removed in this self-contained HTML "
        "build so links don't dead-end when the file is shared. Figures and "
        "GIFs are embedded inline (base64)."
        '</p>'
    )
    html = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<title>FilaSeg — Pipeline Overview</title>\n'
        + style
        + '</head>\n<body>\n'
        + body
        + footer
        + '\n</body>\n</html>\n'
    )

    HTML_PATH.write_text(html, encoding="utf-8")
    size_mb = HTML_PATH.stat().st_size / 1024 / 1024
    print(f"wrote {HTML_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
