import datetime
import html
import io
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from onehot.report.assets import CSS, JS

# Report categories: directory name -> heading on the index page
CATEGORIES = {
    "experiments": "Experiments",
    "usecases": "Use cases",
}

# Order of the cards on the index page (matches the README)
ORDER = [
    "w_state_scaling",
    "leader_election",
    "fair_dice",
    "entanglement_robustness",
    "grover_one_hot",
    "single_excitation_walk",
    "one_hot_encoding",
    "graph_coloring_qaoa",
    "tsp_qaoa",
]


class HTML(str):
    """
    Class that marks a table cell as trusted HTML, rendered without escaping
    """


@dataclass
class Stat:
    """
    Class that describes one stat tile: a big value with a label and a hint

    Parameters
    ----------
    label: str
        Short label above the value
    value: str
        Formatted value
    hint: str = ""
        Small text under the value (trusted HTML)
    tone: str | None = None
        "good", "bad", "warn" or None; a tone adds a status icon so colour
        never carries the meaning alone
    """

    label: str
    value: str
    hint: str = ""
    tone: str | None = None


@dataclass
class Block:
    """
    Class that holds one rendered piece of a section

    Parameters
    ----------
    kind: str
        Kind of block ("text", "figure", "table", ...)
    html: str
        Rendered HTML
    """

    kind: str
    html: str


@dataclass
class Section:
    """
    Class that collects the blocks of one titled section of a report

    Prose passed to text, callout and intro is trusted HTML written by the
    report scripts, so it may contain <code> or KaTeX \\( ... \\).

    Parameters
    ----------
    title: str
        Section heading, also listed in the table of contents
    intro: str = ""
        Paragraph under the heading
    blocks: list[Block]
        Blocks in display order
    """

    title: str
    intro: str = ""
    blocks: list[Block] = field(default_factory=list)

    @property
    def anchor(self) -> str:
        """
        Function that derives the HTML id of the section from its title

        Returns
        -------
        anchor: str
            Lower-case id made of letters, digits and hyphens
        """
        anchor = re.sub(r"[^a-z0-9]+", "-", self.title.lower()).strip("-") or "section"

        return anchor

    def text(self, body: str) -> "Section":
        """
        Function that adds a paragraph

        Parameters
        ----------
        body: str
            Paragraph content (trusted HTML)

        Returns
        -------
        self: Section
            The section, for chaining
        """
        self.blocks.append(Block("text", f"<p>{body}</p>"))

        return self

    def callout(self, body: str, tone: str = "info", title: str | None = None) -> "Section":
        """
        Function that adds a highlighted box for a takeaway or a warning

        Parameters
        ----------
        body: str
            Callout content (trusted HTML)
        tone: str = "info"
            "info", "good", "warn" or "bad"
        title: str | None = None
            Bold lead-in before the body

        Returns
        -------
        self: Section
            The section, for chaining
        """
        # Pair the tone colour with an icon so the meaning does not rely on colour
        icon = {"info": "i", "good": "✓", "warn": "!", "bad": "✗"}.get(tone, "i")
        head = f"<strong>{html.escape(title)}</strong> " if title else ""
        self.blocks.append(Block(
            "callout",
            f'<div class="callout callout-{tone}"><span class="callout-icon" aria-hidden="true">{icon}</span>'
            f"<div>{head}{body}</div></div>",
        ))

        return self

    def stats(self, stats: list[Stat]) -> "Section":
        """
        Function that adds a row of stat tiles

        Parameters
        ----------
        stats: list[Stat]
            Tiles in display order

        Returns
        -------
        self: Section
            The section, for chaining
        """
        self.blocks.append(Block("stats", stats_html(stats)))

        return self

    def figure(self, fig, caption: str = "", height: int | None = None) -> "Section":
        """
        Function that adds an interactive Plotly chart

        Parameters
        ----------
        fig: plotly.graph_objects.Figure
            Chart, ideally styled with charts.style
        caption: str = ""
            Text under the chart explaining how to read it (trusted HTML)
        height: int | None = None
            Override of the figure height in pixels

        Returns
        -------
        self: Section
            The section, for chaining
        """
        # Embed only the chart div; plotly.js is loaded once in the page head
        if height is not None:
            fig.update_layout(height=height)
        div = fig.to_html(
            full_html=False, include_plotlyjs=False,
            config={"displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
        )
        self.blocks.append(Block("figure", figure_html(div, caption)))

        return self

    def circuit(self, qc, caption: str = "", fold: int = -1, scale: float = 0.8) -> "Section":
        """
        Function that adds a circuit diagram as inline SVG

        Parameters
        ----------
        qc: QuantumCircuit
            Circuit to draw
        caption: str = ""
            Text under the diagram (trusted HTML)
        fold: int = -1
            Qiskit fold width; -1 keeps the circuit on one row (scrolls sideways)
        scale: float = 0.8
            Drawing scale

        Returns
        -------
        self: Section
            The section, for chaining
        """
        self.blocks.append(Block("circuit", figure_html(circuit_svg(qc, fold, scale), caption, "circuit")))

        return self

    def circuits(self, items: list[tuple[Any, str]], fold: int = -1, scale: float = 0.7) -> "Section":
        """
        Function that adds several labelled circuit diagrams, one under another

        Parameters
        ----------
        items: list[tuple[QuantumCircuit, str]]
            (circuit, label) pairs
        fold: int = -1
            Qiskit fold width; -1 keeps each circuit on one row
        scale: float = 0.7
            Drawing scale

        Returns
        -------
        self: Section
            The section, for chaining
        """
        cells = "".join(
            f'<div class="circuit-cell"><div class="circuit-label">{html.escape(label)}</div>'
            f'<div class="circuit">{circuit_svg(qc, fold, scale)}</div></div>'
            for qc, label in items
        )
        self.blocks.append(Block("circuits", f'<div class="circuit-grid">{cells}</div>'))

        return self

    def table(
        self, columns: list[str], rows: list[list[Any]], caption: str = "",
        numeric: list[int] | None = None, max_height: int | None = 420,
    ) -> "Section":
        """
        Function that adds a table whose columns sort when their header is clicked

        Parameters
        ----------
        columns: list[str]
            Column headers
        rows: list[list[Any]]
            Cell values; strings are escaped unless wrapped in HTML
        caption: str = ""
            Text under the table (trusted HTML)
        numeric: list[int] | None = None
            Indices of right-aligned, numerically sorted columns
            (guessed from the first row when None)
        max_height: int | None = 420
            Height in pixels after which the table scrolls (no limit when None)

        Returns
        -------
        self: Section
            The section, for chaining
        """
        # Header cells carry the column index the sorting script reads
        numeric_cols = set(numeric if numeric is not None else guess_numeric(rows))
        head = "".join(
            f'<th class="{"num" if i in numeric_cols else ""}" data-col="{i}" tabindex="0">'
            f'{html.escape(c)}<span class="sort" aria-hidden="true"></span></th>'
            for i, c in enumerate(columns)
        )

        # Body cells carry a raw sort key next to their formatted text
        body = "".join(
            "<tr>" + "".join(cell_html(v, i in numeric_cols) for i, v in enumerate(row)) + "</tr>" for row in rows
        )

        style = f' style="max-height:{max_height}px"' if max_height else ""
        cap = f"<figcaption>{caption}</figcaption>" if caption else ""
        self.blocks.append(Block(
            "table",
            f'<figure class="block"><div class="table-wrap"{style}><table class="sortable">'
            f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{cap}</figure>",
        ))

        return self

    def code(self, source: str, caption: str = "") -> "Section":
        """
        Function that adds a block of code or a shell command

        Parameters
        ----------
        source: str
            Code shown verbatim
        caption: str = ""
            Text under the code (trusted HTML)

        Returns
        -------
        self: Section
            The section, for chaining
        """
        cap = f"<figcaption>{caption}</figcaption>" if caption else ""
        self.blocks.append(Block(
            "code", f'<figure class="block"><pre class="code"><code>{html.escape(source)}</code></pre>{cap}</figure>'
        ))

        return self

    def html(self, raw: str) -> "Section":
        """
        Function that adds raw HTML

        Parameters
        ----------
        raw: str
            Trusted HTML inserted as is

        Returns
        -------
        self: Section
            The section, for chaining
        """
        self.blocks.append(Block("html", raw))

        return self

    def render(self) -> str:
        """
        Function that renders the section

        Returns
        -------
        rendered: str
            HTML of the section
        """
        intro = f'<p class="section-intro">{self.intro}</p>' if self.intro else ""
        inner = "\n".join(b.html for b in self.blocks)
        rendered = (
            f'<section id="{self.anchor}" class="section">'
            f"<h2>{html.escape(self.title)}</h2>{intro}{inner}</section>"
        )

        return rendered


class Page:
    """
    Class that builds one report page and registers it in the index

    Page.save writes <out_root>/<category>/<slug>/index.html and a meta.json
    summary next to it, then rebuilds <out_root>/index.html.

    Parameters
    ----------
    title: str
        Page title, e.g. "Leader election"
    category: str
        "experiments" or "usecases"
    slug: str
        Directory name under the category, e.g. "leader_election"
    summary: str
        One sentence for the subtitle and the index card
    out_root: Path
        Root output directory (outputs/ by default in the scripts)
    tags: list[str] | None = None
        Short keywords shown on the page and the index card
    """

    def __init__(
        self, title: str, *, category: str, slug: str, summary: str, out_root: Path,
        tags: list[str] | None = None,
    ) -> None:
        if category not in CATEGORIES:
            raise ValueError(f"category must be one of {list(CATEGORIES)}")
        self.title = title
        self.category = category
        self.slug = slug
        self.summary = summary
        self.out_root = Path(out_root)
        self.tags = tags or []
        self.leads: list[str] = []
        self.hero: list[Stat] = []
        self.sections: list[Section] = []
        self.command: str | None = None
        self.params: dict[str, Any] = {}

    def lead(self, body: str) -> "Page":
        """
        Function that adds an intro paragraph under the title

        Parameters
        ----------
        body: str
            Paragraph content (trusted HTML)

        Returns
        -------
        self: Page
            The page, for chaining
        """
        self.leads.append(body)

        return self

    def headline(self, stats: list[Stat]) -> "Page":
        """
        Function that sets the key results shown under the intro and on the index card

        Parameters
        ----------
        stats: list[Stat]
            Tiles in display order; the first three also go on the index card

        Returns
        -------
        self: Page
            The page, for chaining
        """
        self.hero = stats

        return self

    def section(self, title: str, intro: str = "") -> Section:
        """
        Function that appends a new section

        Parameters
        ----------
        title: str
            Section heading
        intro: str = ""
            Paragraph under the heading (trusted HTML)

        Returns
        -------
        sec: Section
            The new section, to fill with blocks
        """
        sec = Section(title, intro)
        self.sections.append(sec)

        return sec

    def reproduce(self, params: dict[str, Any], command: str | None = None) -> "Page":
        """
        Function that records the parameters and the command that produced the page

        Parameters
        ----------
        params: dict[str, Any]
            Parameter name -> value, shown as a table
        command: str | None = None
            Command line (default: the current one, prefixed with "uv run python")

        Returns
        -------
        self: Page
            The page, for chaining
        """
        self.params = params
        self.command = command or "uv run python " + " ".join(relative_argv())

        return self

    @property
    def directory(self) -> Path:
        """
        Function that returns the directory the page is written to

        Returns
        -------
        directory: Path
            <out_root>/<category>/<slug>
        """
        directory = self.out_root / self.category / self.slug

        return directory

    def save(self) -> Path:
        """
        Function that writes the page and its summary, then rebuilds the index

        Returns
        -------
        path: Path
            Path of the written index.html
        """
        # Write the page itself
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "index.html"
        path.write_text(self.render(), encoding="utf-8")

        # Write the summary the index page is built from
        meta = {
            "title": self.title, "category": self.category, "slug": self.slug, "summary": self.summary,
            "tags": self.tags, "stats": [asdict(s) for s in self.hero[:3]], "updated": now(),
            "command": self.command,
        }
        (self.directory / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        # Refresh the overview so it lists this page
        build_index(self.out_root)

        return path

    def render(self) -> str:
        """
        Function that renders the whole page

        Returns
        -------
        document: str
            Complete HTML document
        """
        # Append the reproduction section when parameters were recorded
        sections = list(self.sections)
        if self.params or self.command:
            sections.append(reproduce_section(self.params, self.command))

        # Render the header pieces, the table of contents and the sections
        toc = "".join(f'<li><a href="#{s.anchor}">{html.escape(s.title)}</a></li>' for s in sections)
        leads = "".join(f'<p class="lead">{p}</p>' for p in self.leads)
        hero = stats_html(self.hero, hero=True) if self.hero else ""
        tags = "".join(f'<span class="tag">{html.escape(t)}</span>' for t in self.tags)
        body = f"""
<header class="page-header">
  <nav class="crumbs"><a href="../../index.html">OneHot</a><span>/</span>
    <a href="../../index.html#{self.category}">{CATEGORIES[self.category]}</a></nav>
  <h1>{html.escape(self.title)}</h1>
  <p class="subtitle">{html.escape(self.summary)}</p>
  <div class="tags">{tags}</div>
</header>
<div class="layout">
  <aside class="toc"><div class="toc-title">On this page</div><ol>{toc}</ol></aside>
  <main>
    {leads}
    {hero}
    {"".join(s.render() for s in sections)}
  </main>
</div>
<footer>Generated {html.escape(now())} · {html.escape(versions())}</footer>"""
        document = document_html(f"{self.title} · OneHot", body)

        return document


def build_index(out_root: Path) -> Path:
    """
    Function that rebuilds the overview page from every report's meta.json

    Parameters
    ----------
    out_root: Path
        Root output directory

    Returns
    -------
    path: Path
        Path of <out_root>/index.html
    """
    out_root = Path(out_root)

    # Collect the summaries of every generated report, grouped by category
    metas: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    for meta_path in sorted(out_root.glob("*/*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("category") in metas:
            metas[meta["category"]].append(meta)

    # Show known reports in reading order (simple to advanced), unknown ones last
    for category in metas:
        metas[category].sort(key=lambda m: (ORDER.index(m["slug"]) if m["slug"] in ORDER else len(ORDER), m["slug"]))

    # Render one card per report with its headline numbers
    groups = []
    for category, label in CATEGORIES.items():
        cards = []
        for m in metas[category]:
            stats = "".join(
                f'<div class="mini-stat"><b>{tone_icon(s.get("tone"))}{html.escape(s["value"])}</b>'
                f'<span>{html.escape(s["label"])}</span></div>'
                for s in m.get("stats", [])
            )
            tags = "".join(f'<span class="tag">{html.escape(t)}</span>' for t in m.get("tags", []))
            cards.append(
                f'<a class="card" href="{category}/{m["slug"]}/index.html">'
                f'<h3>{html.escape(m["title"])}</h3><p>{html.escape(m["summary"])}</p>'
                f'<div class="mini-stats">{stats}</div><div class="card-foot"><div class="tags">{tags}</div>'
                f'<time>{html.escape(m.get("updated", ""))}</time></div></a>'
            )
        if cards:
            content = f'<div class="cards">{"".join(cards)}</div>'
        else:
            content = f'<p class="empty">No {label.lower()} have been generated yet.</p>'
        groups.append(f'<section id="{category}" class="group"><h2>{label}</h2>{content}</section>')

    # Write the overview page
    body = f"""
<header class="page-header index-header">
  <h1>OneHot</h1>
  <p class="subtitle">Experiments and use cases for the W state
    \\(|W_n\\rangle = \\tfrac{{1}}{{\\sqrt{{n}}}}\\left(|10\\cdots0\\rangle + |01\\cdots0\\rangle + \\cdots + |0\\cdots01\\rangle\\right)\\),
    the equal superposition of every one-hot basis state.</p>
</header>
<main class="index-main">{"".join(groups)}</main>
<footer>Generated {html.escape(now())} · {html.escape(versions())}</footer>"""
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "index.html"
    path.write_text(document_html("OneHot reports", body), encoding="utf-8")

    return path


def now() -> str:
    """
    Function that returns the current local time for "Generated" stamps

    Returns
    -------
    stamp: str
        Time formatted as YYYY-MM-DD HH:MM
    """
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    return stamp


def versions() -> str:
    """
    Function that lists the versions of the quantum packages loaded in this run

    Returns
    -------
    text: str
        e.g. "onehot 0.1.0 · qiskit 2.6.2 · qiskit-aer 0.18.2"
    """
    parts = []
    for name in ("onehot", "qiskit", "qiskit_aer", "qiskit_optimization"):
        module = sys.modules.get(name)
        if module is not None and hasattr(module, "__version__"):
            parts.append(f"{name.replace('_', '-')} {module.__version__}")
    text = " · ".join(parts)

    return text


def relative_argv() -> list[str]:
    """
    Function that returns the command line with the script path relative to the cwd

    Returns
    -------
    argv: list[str]
        Script path and arguments
    """
    argv = list(sys.argv)
    try:
        argv[0] = str(Path(argv[0]).resolve().relative_to(Path.cwd()))
    except ValueError:
        pass

    return argv


def tone_icon(tone: str | None) -> str:
    """
    Function that renders the status icon of a tone

    Parameters
    ----------
    tone: str | None
        "good", "bad", "warn" or None

    Returns
    -------
    icon_html: str
        Icon HTML, or an empty string when there is no tone
    """
    icon = {"good": "✓", "bad": "✗", "warn": "!"}.get(tone or "")
    icon_html = f'<span class="tone tone-{tone}" aria-label="{tone}">{icon}</span>' if icon else ""

    return icon_html


def stats_html(stats: list[Stat], hero: bool = False) -> str:
    """
    Function that renders a row of stat tiles

    Parameters
    ----------
    stats: list[Stat]
        Tiles in display order
    hero: bool = False
        Use the larger headline style

    Returns
    -------
    rendered: str
        HTML of the tile row
    """
    tiles = "".join(
        f'<div class="stat"><div class="stat-label">{html.escape(s.label)}</div>'
        f'<div class="stat-value">{tone_icon(s.tone)}{html.escape(s.value)}</div>'
        + (f'<div class="stat-hint">{s.hint}</div>' if s.hint else "")
        + "</div>"
        for s in stats
    )
    rendered = f'<div class="stats{" hero" if hero else ""}">{tiles}</div>'

    return rendered


def figure_html(inner: str, caption: str, cls: str = "chart") -> str:
    """
    Function that wraps a chart or diagram in a captioned figure

    Parameters
    ----------
    inner: str
        HTML of the chart or diagram
    caption: str
        Caption (trusted HTML); omitted when empty
    cls: str = "chart"
        CSS class of the frame ("chart" or "circuit")

    Returns
    -------
    rendered: str
        HTML of the figure
    """
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    rendered = f'<figure class="block"><div class="{cls}">{inner}</div>{cap}</figure>'

    return rendered


def circuit_svg(qc, fold: int, scale: float) -> str:
    """
    Function that draws a circuit with Qiskit's matplotlib drawer as inline SVG

    Parameters
    ----------
    qc: QuantumCircuit
        Circuit to draw
    fold: int
        Qiskit fold width (-1 for a single row)
    scale: float
        Drawing scale

    Returns
    -------
    svg: str
        <svg> element sized in CSS pixels
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Draw the circuit and serialise the figure to SVG
    fig = qc.draw("mpl", fold=fold, scale=scale, style="clifford")
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight", transparent=False)
    plt.close(fig)
    svg = buffer.getvalue()
    svg = svg[svg.find("<svg"):]

    # Replace the fixed point size with a pixel width taken from the viewBox
    svg = re.sub(r'<svg([^>]*?) width="[^"]*"', r"<svg\1", svg, count=1)
    svg = re.sub(r'<svg([^>]*?) height="[^"]*"', r"<svg\1", svg, count=1)
    match = re.search(r'viewBox="[\d.\-]+ [\d.\-]+ ([\d.]+) ([\d.]+)"', svg)
    width = f'style="width:{float(match.group(1)):.0f}px;max-width:none"' if match else ""
    svg = svg.replace("<svg", f"<svg {width}", 1)

    return svg


def guess_numeric(rows: list[list[Any]]) -> list[int]:
    """
    Function that guesses which columns are numeric from the first row

    Parameters
    ----------
    rows: list[list[Any]]
        Table rows

    Returns
    -------
    columns: list[int]
        Indices of int or float cells in the first row
    """
    columns = []
    if rows:
        columns = [i for i, v in enumerate(rows[0]) if isinstance(v, (int, float)) and not isinstance(v, bool)]

    return columns


def cell_html(value: Any, numeric: bool) -> str:
    """
    Function that renders one table cell with its sort key

    Parameters
    ----------
    value: Any
        Cell value; HTML is inserted as is, other strings are escaped
    numeric: bool
        Right-align the cell

    Returns
    -------
    rendered: str
        HTML of the <td>
    """
    # Format the value and keep a raw key for sorting
    if isinstance(value, HTML):
        text, key = str(value), re.sub(r"<[^>]+>", "", value)
    elif isinstance(value, bool):
        text = key = "✓" if value else "✗"
    elif isinstance(value, float):
        text, key = format_number(value), repr(value)
    else:
        text, key = html.escape(str(value)), str(value)

    cls = ' class="num"' if numeric else ""
    rendered = f'<td{cls} data-sort="{html.escape(key)}">{text}</td>'

    return rendered


def format_number(x: float) -> str:
    """
    Function that formats a float compactly for a table

    Parameters
    ----------
    x: float
        Value to format

    Returns
    -------
    text: str
        Thousands separators above 1,000, up to 3 decimals above 1, up to 4 below
    """
    if x == 0:
        text = "0"
    elif abs(x) >= 1000:
        text = f"{x:,.0f}"
    elif abs(x) >= 1:
        text = f"{x:.3f}".rstrip("0").rstrip(".")
    else:
        text = f"{x:.4f}".rstrip("0").rstrip(".")

    return text


def reproduce_section(params: dict[str, Any], command: str | None) -> Section:
    """
    Function that builds the closing section with the command and parameters

    Parameters
    ----------
    params: dict[str, Any]
        Parameter name -> value
    command: str | None
        Command line that produced the page

    Returns
    -------
    sec: Section
        The "Reproduce" section
    """
    sec = Section("Reproduce")
    if command:
        sec.code(command, "Run from the repository root.")
    if params:
        sec.table(["Parameter", "Value"], [[k, str(v)] for k, v in params.items()], max_height=None, numeric=[])

    return sec


def plotly_js_url() -> str:
    """
    Function that returns the CDN URL of the plotly.js build matching the installed plotly

    Returns
    -------
    url: str
        URL of plotly.min.js
    """
    from plotly.offline import get_plotlyjs_version

    url = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"

    return url


def document_html(title: str, body: str) -> str:
    """
    Function that wraps a page body in a complete HTML document

    Parameters
    ----------
    title: str
        Browser tab title
    body: str
        HTML inside <body>

    Returns
    -------
    document: str
        Complete HTML document with styles, plotly.js and KaTeX
    """
    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<script>try{{var t=localStorage.getItem("onehot-theme");if(t)document.documentElement.dataset.theme=t;}}catch(e){{}}</script>
<script src="{plotly_js_url()}" charset="utf-8"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<style>{CSS}</style>
</head>
<body>
<button class="theme-toggle" type="button" aria-label="Toggle dark mode" title="Toggle dark mode">◐</button>
{body}
<script>{JS}</script>
</body>
</html>"""

    return document
