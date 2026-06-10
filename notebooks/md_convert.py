#!/usr/bin/env python3
"""Barebones, one-way converter for the ``notebooks/*.md`` cell-markdown format.

The ``.md`` notebooks use HTML-comment markers to delimit cells::

    <!-- md -->      a markdown cell (plain text until the next marker)
    <!-- code -->    a code cell (a ```python fenced block)
    <!-- output -->  a saved output block (dropped — notebooks regenerate it on run)

This converts one of those files into either:

    --py      a runnable script: just the code cells, joined with ``# %%`` cell
              markers. IPython magics (``!pip ...``, ``%matplotlib ...``) are
              rewritten to ``get_ipython()...`` so the file still *parses* under a
              plain ``python`` run and only executes the magic under IPython/Colab.
    --ipynb   a standard Jupyter notebook (nbformat 4); markdown + code cells,
              outputs dropped.

Stdlib only, one direction. For round-tripping, output capture, image hosting, or
Colab upload, use the separate ``colab-utils`` tooling instead.

    python md_convert.py "03 - Analyze Rollout Nuclei.md" --py
    python md_convert.py "03 - Analyze Rollout Nuclei.md" --ipynb
    python md_convert.py in.md --py -o /tmp/out.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_MARKER = re.compile(r"^\s*<!--\s*(md|code|output)\s*-->\s*$")
_BANG = re.compile(r"^(\s*)!(.*)$")                       # !shell command
_LINE_MAGIC = re.compile(r"^(\s*)%(\w+)([ \t].*)?$")      # %magic args
_CELL_MAGIC = re.compile(r"^(\s*)%%(\w+)([ \t].*)?$")     # %%magic (whole cell)


def parse_cells(text: str) -> list[tuple[str, str]]:
    """Split cell-markdown into ``(kind, content)`` cells, kind in md/code/output."""
    cells: list[tuple[str, str]] = []
    kind: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        m = _MARKER.match(line)
        if m:
            if kind is not None:
                cells.append((kind, "\n".join(buf).strip("\n")))
            kind, buf = m.group(1), []
        elif kind is not None:
            buf.append(line)
    if kind is not None:
        cells.append((kind, "\n".join(buf).strip("\n")))
    return cells


def strip_code_fence(content: str) -> str:
    """Drop the leading ```python (or ```) fence and its closing ``` from a code cell."""
    lines = content.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and lines[-1].lstrip().startswith("```"):
            lines = lines[:-1]
    return "\n".join(lines).strip("\n")


def _translate_magics(code: str) -> str:
    """Rewrite IPython magics to get_ipython() calls (nbconvert-style) so the script
    parses under plain ``python``; the call is a no-op import-time and only runs the
    magic when executed inside IPython/Colab."""
    out = []
    for line in code.splitlines():
        if (m := _CELL_MAGIC.match(line)):
            indent, name, arg = m.group(1), m.group(2), (m.group(3) or "").strip()
            out.append(f"{indent}get_ipython().run_cell_magic({name!r}, {arg!r}, '')")
        elif (m := _BANG.match(line)):
            out.append(f"{m.group(1)}get_ipython().system({m.group(2).strip()!r})")
        elif (m := _LINE_MAGIC.match(line)):
            indent, name, arg = m.group(1), m.group(2), (m.group(3) or "").strip()
            out.append(f"{indent}get_ipython().run_line_magic({name!r}, {arg!r})")
        else:
            out.append(line)
    return "\n".join(out)


def to_py(cells: list[tuple[str, str]]) -> str:
    """Code cells only, joined with ``# %%`` markers, magics translated."""
    blocks = []
    for kind, content in cells:
        if kind != "code":
            continue
        code = _translate_magics(strip_code_fence(content))
        if code.strip():
            blocks.append("# %%\n" + code)
    return "\n\n".join(blocks) + "\n"


def _source_lines(s: str) -> list[str]:
    """nbformat source: a list of lines, each with a trailing newline except the last."""
    if not s:
        return []
    lines = s.splitlines()
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def to_ipynb(cells: list[tuple[str, str]]) -> dict:
    """Markdown + code cells as an nbformat-4 notebook; output cells dropped."""
    nb_cells = []
    for kind, content in cells:
        if kind == "output":
            continue
        if kind == "code":
            nb_cells.append({"cell_type": "code", "metadata": {},
                             "execution_count": None, "outputs": [],
                             "source": _source_lines(strip_code_fence(content))})
        else:
            nb_cells.append({"cell_type": "markdown", "metadata": {},
                             "source": _source_lines(content)})
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="path to a cell-markdown .md notebook")
    fmt = ap.add_mutually_exclusive_group(required=True)
    fmt.add_argument("--py", action="store_true", help="emit a runnable .py script")
    fmt.add_argument("--ipynb", action="store_true", help="emit a Jupyter .ipynb notebook")
    ap.add_argument("-o", "--output", help="output path (default: input with new suffix)")
    a = ap.parse_args()

    src = Path(a.input)
    cells = parse_cells(src.read_text(encoding="utf-8"))
    if a.py:
        out = Path(a.output) if a.output else src.with_suffix(".py")
        out.write_text(to_py(cells), encoding="utf-8")
    else:
        out = Path(a.output) if a.output else src.with_suffix(".ipynb")
        out.write_text(json.dumps(to_ipynb(cells), indent=1, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    n_code = sum(k == "code" for k, _ in cells)
    n_md = sum(k == "md" for k, _ in cells)
    print(f"wrote {out}  ({n_code} code, {n_md} markdown cells)")


if __name__ == "__main__":
    main()
