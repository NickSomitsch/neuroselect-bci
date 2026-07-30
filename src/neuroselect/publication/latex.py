"""Deterministic LaTeX assembly for the verified NeuroSelect manuscript."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from neuroselect.publication.display import (
    PublicationDisplayInventory,
    PublicationDisplayItem,
)
from neuroselect.publication.manuscript import (
    ManuscriptReference,
    ManuscriptSpec,
)

_CITATION_PATTERN = re.compile(r"\[@([a-z0-9-]+(?:;\s*@[a-z0-9-]+)*)\]")
_INLINE_PATTERN = re.compile(
    r"(\[@[a-z0-9-]+(?:;\s*@[a-z0-9-]+)*\]|\*\*.+?\*\*|\\\(.+?\\\)|https?://[^\s]+)"
)
_MARKER_PATTERN = re.compile(r"^\{\{(figure|table):([a-z0-9-]+)\}\}$")
_NUMBERED_HEADING_PATTERN = re.compile(r"^\d+(?:\.\d+)*\.?\s*")


@dataclass(frozen=True)
class LatexManuscript:
    """A self-contained LaTeX source set before compilation."""

    source: str
    bibliography: str
    figures: dict[str, bytes]


def _latex_escape(text: str) -> str:
    normalized = (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "--")
        .replace("\u2014", "---")
        .replace("\u2212", "-")
    )
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\u00b1": r"\ensuremath{\pm}",
        "\u00d7": r"\ensuremath{\times}",
        "\u2264": r"\ensuremath{\leq}",
        "\u2265": r"\ensuremath{\geq}",
    }
    return "".join(replacements.get(character, character) for character in normalized)


def _citation_command(token: str) -> str:
    match = _CITATION_PATTERN.fullmatch(token)
    if match is None:
        raise ValueError(f"invalid manuscript citation token: {token}")
    keys = [item.strip().removeprefix("@") for item in match.group(1).split(";")]
    return rf"\cite{{{','.join(keys)}}}"


def _inline_latex(text: str) -> str:
    chunks: list[str] = []
    for chunk in _INLINE_PATTERN.split(text):
        if not chunk:
            continue
        if _CITATION_PATTERN.fullmatch(chunk):
            chunks.append(_citation_command(chunk))
        elif chunk.startswith("**") and chunk.endswith("**"):
            chunks.append(r"\textbf{" + _latex_escape(chunk[2:-2]) + "}")
        elif chunk.startswith(r"\(") and chunk.endswith(r"\)"):
            chunks.append("$" + chunk[2:-2] + "$")
        elif chunk.startswith(("https://", "http://")):
            trailing = ""
            if chunk[-1:] in ".,;:":
                chunk, trailing = chunk[:-1], chunk[-1]
            chunks.append(r"\url{" + chunk + "}" + _latex_escape(trailing))
        else:
            chunks.append(_latex_escape(chunk))
    return "".join(chunks)


def _iter_blocks(source: str) -> list[str]:
    blocks: list[str] = []
    paragraph_lines: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        is_structural = (
            not stripped
            or stripped.startswith("#")
            or stripped.startswith("{{")
            or stripped.startswith("- ")
        )
        if is_structural and paragraph_lines:
            blocks.append(" ".join(paragraph_lines))
            paragraph_lines = []
        if not stripped:
            continue
        if stripped.startswith(("#", "{{", "- ")):
            blocks.append(stripped)
        else:
            paragraph_lines.append(stripped)
    if paragraph_lines:
        blocks.append(" ".join(paragraph_lines))
    return blocks


def _table_label(item_id: str) -> str:
    match = re.match(r"table-([0-9]+[a-z]?)", item_id)
    if match is None:
        raise ValueError(f"invalid table ID: {item_id}")
    return f"Table {match.group(1)}"


def _figure_label(item_id: str) -> str:
    match = re.match(r"figure-([0-9]+)", item_id)
    if match is None:
        raise ValueError(f"invalid figure ID: {item_id}")
    return f"Figure {match.group(1)}"


def _table_latex(display_path: Path, item: PublicationDisplayItem) -> str:
    csv_path = display_path / next(file for file in item.files if file.endswith(".csv"))
    with csv_path.open(encoding="utf-8", newline="") as source_file:
        rows = list(csv.reader(source_file))
    headers = rows[0]
    body = rows[1:]
    columns = "@{}" + "l" * len(headers) + "@{}"
    landscape = len(headers) >= 7
    lines: list[str] = []
    if landscape:
        lines.extend((r"\clearpage", r"\begin{landscape}"))
    lines.extend(
        (
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{" + _latex_escape(f"{item.title}. {item.caption}") + "}",
            rf"\label{{tab:{item.item_id}}}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3.5pt}",
            r"\renewcommand{\arraystretch}{1.15}",
            r"\resizebox{\linewidth}{!}{%",
            rf"\begin{{tabular}}{{{columns}}}",
            r"\toprule",
            " & ".join(r"\textbf{" + _latex_escape(header) + "}" for header in headers) + r" \\",
            r"\midrule",
        )
    )
    for row in body:
        lines.append(" & ".join(_latex_escape(value) for value in row) + r" \\")
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\vspace{2pt}",
            r"\begin{minipage}{0.98\linewidth}",
            r"\footnotesize\textit{Evidence role: "
            + _latex_escape(", ".join(item.evidence_roles))
            + r". Source: checksum-verified publication display.}",
            r"\end{minipage}",
            r"\end{table}",
        )
    )
    if landscape:
        lines.extend((r"\end{landscape}", r"\clearpage"))
    return "\n".join(lines)


def _figure_latex(item: PublicationDisplayItem) -> str:
    source_name = Path(next(file for file in item.files if file.endswith(".pdf"))).name
    return "\n".join(
        (
            r"\begin{figure}[p]",
            r"\centering",
            rf"\includegraphics[width=0.98\linewidth]{{figures/{source_name}}}",
            r"\caption{" + _latex_escape(f"{item.title}. {item.caption}") + "}",
            rf"\label{{fig:{item.item_id}}}",
            r"\end{figure}",
            r"\clearpage",
        )
    )


def render_bibliography(references: tuple[ManuscriptReference, ...]) -> str:
    """Emit portable BibTeX entries while preserving the verified citation text."""

    entries = []
    for reference in references:
        if reference.bibtex_fields:
            fields = dict(reference.bibtex_fields)
            fields.setdefault("url", reference.persistent_url)
            field_lines = []
            for key, value in fields.items():
                rendered = value if key in {"url", "doi"} else _latex_escape(value)
                if key == "title":
                    field_lines.append("  title = {{" + rendered + "}},")
                else:
                    field_lines.append(f"  {key} = {{{rendered}}},")
            entries.append(
                "\n".join(
                    (
                        f"@{reference.bibtex_type}{{{reference.reference_id},",
                        *field_lines,
                        "}",
                    )
                )
            )
            continue
        note = _latex_escape(reference.formatted)
        entries.append(
            "\n".join(
                (
                    f"@misc{{{reference.reference_id},",
                    f"  note = {{{{{note} Available at: "
                    + r"\url{"
                    + reference.persistent_url
                    + r"}}}},",
                    "}",
                )
            )
        )
    return "\n\n".join(entries) + "\n"


def render_latex_manuscript(
    spec: ManuscriptSpec,
    manuscript_source: str,
    ordered_references: tuple[ManuscriptReference, ...],
    display_inventory: PublicationDisplayInventory,
) -> LatexManuscript:
    """Render verified Markdown, display items, and references as journal-neutral LaTeX."""

    item_index = {item.item_id: item for item in display_inventory.items}
    figures: dict[str, bytes] = {}
    preamble = rf"""\documentclass[11pt]{{article}}
\usepackage[letterpaper,margin=1in]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{lmodern}}
\usepackage{{microtype}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{float}}
\usepackage{{pdflscape}}
\usepackage[font=small,labelfont=bf]{{caption}}
\usepackage{{xurl}}
\usepackage[hidelinks]{{hyperref}}
\graphicspath{{{{figures/}}}}
\setlength{{\parindent}}{{1em}}
\setlength{{\parskip}}{{0.25em}}
\setlength{{\emergencystretch}}{{3em}}
\title{{{_latex_escape(spec.title)}}}
\author{{{_latex_escape(spec.author)}}}
\date{{}}

\begin{{document}}
\maketitle
"""
    output = [preamble.rstrip()]
    in_abstract = False
    in_list = False
    for block in _iter_blocks(manuscript_source):
        if in_list and not block.startswith("- "):
            output.append(r"\end{itemize}")
            in_list = False
        if block == "{{pagebreak}}":
            output.append(r"\clearpage")
            continue
        if block == "{{references}}":
            output.extend((r"\bibliographystyle{unsrt}", r"\bibliography{references}"))
            continue
        marker = _MARKER_PATTERN.fullmatch(block)
        if marker:
            kind, item_id = marker.groups()
            item = item_index[item_id]
            if kind == "table":
                output.append(_table_latex(spec.display.path, item))
            else:
                output.append(_figure_latex(item))
                source_path = spec.display.path / next(
                    file for file in item.files if file.endswith(".pdf")
                )
                figures[f"figures/{source_path.name}"] = source_path.read_bytes()
            continue
        if block.startswith("#"):
            level = len(block) - len(block.lstrip("#"))
            title = block[level:].strip()
            if title == "Abstract":
                output.append(r"\begin{abstract}")
                in_abstract = True
                continue
            if in_abstract:
                output.append(r"\end{abstract}")
                in_abstract = False
            if title == "References":
                continue
            clean_title = _NUMBERED_HEADING_PATTERN.sub("", title)
            if level == 1 and title == "Declarations":
                output.extend(
                    (
                        r"\section*{Declarations}",
                        r"\addcontentsline{toc}{section}{Declarations}",
                    )
                )
            elif level == 1:
                output.append(r"\section{" + _latex_escape(clean_title) + "}")
            elif level == 2:
                output.append(r"\subsection{" + _latex_escape(clean_title) + "}")
            else:
                output.append(r"\subsubsection{" + _latex_escape(clean_title) + "}")
            continue
        if block.startswith("- "):
            if not in_list:
                output.append(r"\begin{itemize}")
                in_list = True
            output.append(r"\item " + _inline_latex(block[2:]))
        else:
            output.append(_inline_latex(block) + "\n")
    if in_list:
        output.append(r"\end{itemize}")
    if in_abstract:
        output.append(r"\end{abstract}")
    output.append(r"\end{document}")
    return LatexManuscript(
        source="\n\n".join(output) + "\n",
        bibliography=render_bibliography(ordered_references),
        figures=figures,
    )
