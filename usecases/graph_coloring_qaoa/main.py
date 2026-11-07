"""
Use case: colour a graph with QAOA, starting from one W state per vertex

Each vertex v gets one qubit per colour c, and x_{v,c} = 1 means "v has
colour c", so a valid colouring is one-hot inside every vertex. The cost is
the number of edges whose two ends share a colour. Three QAOA settings are
compared at depths p = 1..P:

(a) uniform start (H on every qubit) + X mixer + one-hot penalty,
(b) one W state per vertex + X mixer + one-hot penalty,
(c) one W state per vertex + XY ring mixer, which never leaves the one-hot
    subspace, so no penalty is needed.

The results are written to outputs/usecases/graph_coloring_qaoa/index.html.

Examples
--------
uv run python usecases/graph_coloring_qaoa/main.py
uv run python usecases/graph_coloring_qaoa/main.py --graph cycle4 --depth 4
uv run python usecases/graph_coloring_qaoa/main.py --graph triangle --colors 3 --restarts 6
"""

import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from qiskit import QuantumCircuit, transpile

from onehot import w_state
from onehot.report import HTML, Page, Stat, charts

# Make the repository root importable so that usecases.shared resolves when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from usecases.shared.common import base_parser, sample  # noqa: E402
from usecases.shared.qaoa import Qubo, Run, Variant, cost_scale, optimize, qaoa_circuit  # noqa: E402


GRAPHS = {
    "triangle": ([(0, 1), (1, 2), (0, 2)], [(0.0, 1.0), (-0.87, -0.5), (0.87, -0.5)]),
    "paw": ([(0, 1), (1, 2), (0, 2), (2, 3)], [(-0.9, 0.9), (-0.9, -0.9), (0.2, 0.0), (1.4, 0.0)]),
    "cycle4": ([(0, 1), (1, 2), (2, 3), (3, 0)], [(-1.0, 1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)]),
}
COLOR_SWATCHES = [charts.SERIES[4], charts.SERIES[5], charts.SERIES[6], charts.SERIES[7]]


def pct(value: float) -> HTML:
    """
    Function that formats a probability as a percentage table cell

    Parameters
    ----------
    value: float
        Probability between 0 and 1

    Returns
    -------
    cell: HTML
        Cell text such as "7.5%" (sorts numerically)
    """
    cell = HTML(f"{value:.1%}")

    return cell


def qubit(v: int, c: int, colors: int) -> int:
    """
    Function that returns the qubit of variable x_{v,c}

    Parameters
    ----------
    v: int
        Vertex
    c: int
        Colour
    colors: int
        Number of colours K

    Returns
    -------
    index: int
        Qubit index v * K + c (each vertex owns K neighbouring qubits)
    """
    index = v * colors + c

    return index


def objective_qubo(edges: list[tuple[int, int]], vertices: int, colors: int) -> Qubo:
    """
    Function that builds the number of monochromatic edges as a QUBO

    Parameters
    ----------
    edges: list[tuple[int, int]]
        Edges of the graph
    vertices: int
        Number of vertices
    colors: int
        Number of colours K

    Returns
    -------
    qubo: Qubo
        Sum over edges (u, v) and colours c of x_{u,c} x_{v,c}
    """
    # An edge costs 1 for every colour that both of its ends have
    qubo = Qubo(vertices * colors)
    for u, v in edges:
        for c in range(colors):
            qubo.add_quadratic(qubit(u, c, colors), qubit(v, c, colors), 1.0)

    return qubo


def initial_state(kind: str, vertices: int, colors: int) -> QuantumCircuit:
    """
    Function that prepares the initial state of a QAOA setting

    Parameters
    ----------
    kind: str
        "uniform" (H on every qubit) or "w" (one W state per vertex)
    vertices: int
        Number of vertices
    colors: int
        Number of colours K

    Returns
    -------
    qc: QuantumCircuit
        Initial-state circuit on vertices * K qubits
    """
    qc = QuantumCircuit(vertices * colors)

    # Uniform superposition over all 2^(V K) bitstrings, valid or not
    if kind == "uniform":
        qc.h(range(qc.num_qubits))

    # Uniform superposition over the K^V valid colourings only
    else:
        for v in range(vertices):
            qc.compose(w_state(colors), qubits=[qubit(v, c, colors) for c in range(colors)], inplace=True)

    return qc


def decode(index: int, vertices: int, colors: int) -> list[int | None]:
    """
    Function that reads the colour of every vertex from a basis index

    Parameters
    ----------
    index: int
        Basis-state index (qubit i = bit i)
    vertices: int
        Number of vertices
    colors: int
        Number of colours K

    Returns
    -------
    coloring: list[int | None]
        Colour of each vertex, or None when the vertex is not one-hot
    """
    coloring: list[int | None] = []
    for v in range(vertices):
        # Collect the colours switched on for vertex v
        on = [c for c in range(colors) if index >> qubit(v, c, colors) & 1]
        coloring.append(on[0] if len(on) == 1 else None)

    return coloring


def masks(edges: list[tuple[int, int]], vertices: int, colors: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Function that marks the feasible and the properly coloured basis states

    Parameters
    ----------
    edges: list[tuple[int, int]]
        Edges of the graph
    vertices: int
        Number of vertices
    colors: int
        Number of colours K

    Returns
    -------
    feasible: np.ndarray
        True where every vertex is one-hot
    proper: np.ndarray
        True where the state is feasible and no edge is monochromatic
    """
    # A vertex is one-hot when exactly one of its K bits is set
    index = np.arange(2 ** (vertices * colors))
    feasible = np.ones(index.shape, dtype=bool)
    for v in range(vertices):
        ones = sum((index >> qubit(v, c, colors)) & 1 for c in range(colors))
        feasible &= ones == 1

    # Proper colourings are the feasible states with zero monochromatic edges
    conflicts = objective_qubo(edges, vertices, colors).values()
    proper = feasible & (conflicts == 0)

    return feasible, proper


def graph_figure(edges: list[tuple[int, int]], positions: list[tuple[float, float]],
                 coloring: list[int | None]) -> go.Figure:
    """
    Function that draws the graph with its vertices filled by colour

    Parameters
    ----------
    edges: list[tuple[int, int]]
        Edges of the graph
    positions: list[tuple[float, float]]
        Coordinates of every vertex
    coloring: list[int | None]
        Colour of each vertex (vertices that are not one-hot are left out)

    Returns
    -------
    fig: go.Figure
        Styled figure
    """
    # Apply the shared look first so the marker and line sizes below are kept
    fig = charts.style(go.Figure(), height=300, legend=True)

    # Edges: grey when the ends differ, red and thick when they clash
    for u, v in edges:
        clash = coloring[u] is not None and coloring[u] == coloring[v]
        fig.add_trace(go.Scatter(
            x=[positions[u][0], positions[v][0]], y=[positions[u][1], positions[v][1]], mode="lines",
            line=dict(color=charts.CRITICAL if clash else charts.AXIS, width=4 if clash else 2),
            hoverinfo="skip", showlegend=False,
        ))

    # Vertices: one trace per colour so the legend names the colours
    for c in sorted({c for c in coloring if c is not None}):
        members = [v for v, col in enumerate(coloring) if col == c]
        fig.add_trace(go.Scatter(
            x=[positions[v][0] for v in members], y=[positions[v][1] for v in members],
            mode="markers+text", name=f"colour {c}", text=[f"v{v}" for v in members],
            textposition="middle center", textfont=dict(color="#ffffff", size=12),
            marker=dict(size=38, color=COLOR_SWATCHES[c % len(COLOR_SWATCHES)], line=dict(width=2, color="#fcfcfb")),
            hovertemplate="%{text}<extra>colour " + str(c) + "</extra>",
        ))

    # Hide the axes and keep the aspect ratio square
    fig.update_xaxes(visible=False, range=[-1.8, 1.8])
    fig.update_yaxes(visible=False, range=[-1.5, 1.5], scaleanchor="x")

    return fig


def main() -> None:
    """
    Function that runs the comparison and writes the report

    Returns
    -------
    None
    """
    # Read the command-line arguments
    parser = base_parser(__doc__)
    parser.add_argument("--graph", choices=list(GRAPHS), default="paw", help="graph to colour (default: paw)")
    parser.add_argument("--colors", type=int, default=3, help="number of colours K (default: 3)")
    parser.add_argument("--depth", type=int, default=3, help="largest QAOA depth p (default: 3)")
    parser.add_argument("--restarts", type=int, default=4, help="random starts per depth (default: 4)")
    parser.add_argument("--penalty", type=float, default=1.0,
                        help="weight of the one-hot penalty in settings (a) and (b) (default: 1.0)")
    args = parser.parse_args()
    started = time.time()

    edges, positions = GRAPHS[args.graph]
    vertices, colors = len(positions), args.colors
    n = vertices * colors
    groups = [[qubit(v, c, colors) for c in range(colors)] for v in range(vertices)]

    # Objective, penalised cost and the reference masks
    objective = objective_qubo(edges, vertices, colors)
    penalised = objective_qubo(edges, vertices, colors)
    for group in groups:
        penalised.add_one_hot_penalty(group, args.penalty)
    feasible, proper = masks(edges, vertices, colors)
    conflicts = objective.values()
    scale = cost_scale(conflicts, feasible)

    # The three settings to compare, each with a fixed colour
    variants = [
        Variant("H + X mixer + penalty", initial_state("uniform", vertices, colors), penalised, "x",
                color=charts.SERIES[1]),
        Variant("W + X mixer + penalty", initial_state("w", vertices, colors), penalised, "x",
                color=charts.SERIES[0]),
        Variant("W + XY mixer", initial_state("w", vertices, colors), objective, "xy", groups,
                color=charts.SERIES[2]),
    ]
    colors_by_name = {v.name: v.color for v in variants}

    # Optimise every setting at every depth, warm-starting from the previous depth
    runs: dict[tuple[str, int], Run] = {}
    for variant in variants:
        previous = None
        for p in range(1, args.depth + 1):
            run = optimize(variant, p, scale, args.restarts, args.seed + p, warm_start=previous)
            runs[(variant.name, p)] = run
            previous = run.params
            print(f"{variant.name:<24} p={p}  P(proper)={run.probabilities[proper].sum():.3f}  "
                  f"P(feasible)={run.probabilities[feasible].sum():.3f}")

    # Sample the deepest circuits, ideally and with gate noise
    depths = list(range(1, args.depth + 1))
    sampled, cx_counts = {}, {}
    for variant in variants:
        run = runs[(variant.name, args.depth)]
        circuit = qaoa_circuit(variant, args.depth, scale, run.params)
        compiled = transpile(circuit, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1,
                             seed_transpiler=args.seed)
        cx_counts[variant.name] = compiled.count_ops().get("cx", 0)
        rates = {}
        for label, noise in (("ideal", 0.0), ("noisy", args.noise)):
            if label == "noisy" and noise <= 0:
                continue
            counts = sample(circuit, args.shots, args.seed, noise)
            total = sum(counts.values())
            indices = {int(bits.replace(" ", ""), 2): c for bits, c in counts.items()}
            rates[label] = (sum(c for k, c in indices.items() if proper[k]) / total,
                            sum(c for k, c in indices.items() if feasible[k]) / total)
        sampled[variant.name] = rates

    # Reference numbers: random guessing over all states and over valid colourings
    proper_count = int(proper.sum())
    guess_all = proper_count / 2**n
    guess_valid = proper_count / colors**vertices
    best = variants[2]
    best_run = runs[(best.name, args.depth)]
    base_run = runs[(variants[0].name, args.depth)]
    top = int(np.argmax(best_run.probabilities))
    coloring = decode(top, vertices, colors)

    page = Page(
        "Graph colouring with QAOA", category="usecases", slug="graph_coloring_qaoa", out_root=args.out,
        summary="Colour a graph with QAOA using one-hot variables; one W state per vertex plus an XY mixer "
        "keeps every amplitude on valid colourings.",
        tags=["QAOA", "one-hot constraint", "XY mixer", "optimisation"],
    )
    page.lead(
        rf"Colour the <b>{args.graph}</b> graph ({vertices} vertices, {len(edges)} edges) with {colors} colours. "
        r"Each vertex \(v\) owns one qubit per colour, and \(x_{v,c}=1\) means “\(v\) has colour \(c\)”, so a "
        r"valid colouring is <em>one-hot inside every vertex</em>: exactly the structure of a W state. The cost "
        r"\(C=\sum_{(u,v)\in E}\sum_c x_{u,c}x_{v,c}\) counts edges whose ends share a colour."
    )
    page.headline([
        Stat(f"{v.name}", f"{runs[(v.name, args.depth)].probabilities[proper].sum():.1%}",
             f"P(proper colouring) at p = {args.depth}",
             "good" if runs[(v.name, args.depth)].probabilities[proper].sum() > guess_valid else "warn")
        for v in variants
    ] + [Stat("Random valid colouring", f"{guess_valid:.1%}", f"{proper_count} of {colors ** vertices} "
                                                              "one-hot assignments are proper")])

    # Problem and the best answer
    sec = page.section(
        "Problem and answer",
        f"{n} qubits: vertex v uses qubits {colors}v … {colors}v+{colors - 1}. Of the 2^{n} = {2 ** n:,} bitstrings, "
        f"only {colors ** vertices} are one-hot per vertex and {proper_count} of those are proper colourings.",
    )
    sec.figure(graph_figure(edges, positions, coloring),
               f"Most likely outcome of “{best.name}” at p = {args.depth}; it has probability "
               f"{best_run.probabilities[top]:.1%}. Colour labels are interchangeable, so the probability is shared "
               "among the proper colourings that permute them.")
    if None not in coloring and all(coloring[u] != coloring[v] for u, v in edges):
        sec.callout("The most likely outcome is a proper colouring: no edge joins two vertices of the same colour.",
                    "good")

    # Initial states and one layer
    sec = page.section(
        "Circuits",
        r"Settings (b) and (c) start from \(|W_K\rangle^{\otimes V}\): every amplitude already sits on a valid "
        r"assignment. The uniform start of (a) puts only "
        r"\((K/2^K)^V\) = " f"{(colors / 2 ** colors) ** vertices:.2%} of the probability there.",
    )
    sec.circuits([(variants[1].init, "Initial state of (b) and (c): one W state per vertex"),
                  (qaoa_circuit(Variant(best.name, QuantumCircuit(n), best.qubo, "xy", groups), 1, scale),
                   "One layer of (c): cost rotations, then XY rotations around "
                                                 "each vertex's ring of colour qubits")])
    sec.text("The XY gate exchanges amplitude between |01⟩ and |10⟩ of two colour qubits of the same vertex, so it "
             "changes which colour the vertex has but never how many. The X mixer of (a) and (b) flips single bits "
             "and leaks probability out of the valid subspace, which the penalty then has to push back.")

    # Results versus depth
    sec = page.section("Results versus depth", "Exact probabilities of the optimised state at each depth.")
    panels = {
        "P(proper colouring)": {v.name: [runs[(v.name, p)].probabilities[proper].sum() for p in depths]
                                for v in variants},
        "P(valid one-hot)": {v.name: [runs[(v.name, p)].probabilities[feasible].sum() for p in depths]
                             for v in variants},
    }
    fig = charts.small_multiples(depths, panels, colors=colors_by_name, x_title="QAOA depth p",
                                 y_title="probability", hover_format=".1%", height=400)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
    fig.update_xaxes(dtick=1)
    fig.add_hline(y=guess_valid, line=dict(color=charts.MUTED, width=1, dash="dot"), row=1, col=1,
                  annotation_text="random valid colouring", annotation_position="top left",
                  annotation_font=dict(color=charts.MUTED, size=11))
    sec.figure(fig, "Left: probability of measuring a proper colouring. Right: probability of measuring a valid "
                    "(one-hot) assignment at all; both W settings sit at 100 % there, so their lines coincide. "
                    "The dotted line is guessing uniformly among valid assignments.")
    stuck = max(abs(runs[(variants[1].name, p)].probabilities[proper].sum() - guess_valid) for p in depths) < 0.01
    if stuck:
        sec.callout(
            f"“{variants[1].name}” stays at the random-guess level ({guess_valid:.1%}) at every depth. Starting "
            "inside the valid subspace, any X rotation only leaks probability into penalised states, so the "
            "optimiser switches the mixer off (β ≈ 0) and the state never changes. A W start needs a mixer that "
            "respects the constraint.", "warn", "Why (b) does not improve.")
    gain = best_run.probabilities[proper].sum() / max(base_run.probabilities[proper].sum(), 1e-12)
    sec.callout(
        f"At p = {args.depth}, “{best.name}” measures a proper colouring with probability "
        f"{best_run.probabilities[proper].sum():.1%}, {gain:.1f}× the uniform-start setting "
        f"({base_run.probabilities[proper].sum():.1%}), and its valid-assignment probability stays at "
        f"{min(runs[(best.name, p)].probabilities[feasible].sum() for p in depths):.1%} at every depth.",
        "info", "Takeaway.",
    )

    # Optimisation traces
    sec = page.section("Optimisation", f"Best expected cost found by COBYLA at p = {args.depth} "
                                       f"(best of {args.restarts} random starts plus a warm start).")
    fig = go.Figure()
    for v in variants:
        history = runs[(v.name, args.depth)].history
        fig.add_trace(go.Scatter(x=list(range(1, len(history) + 1)), y=history, name=v.name, mode="lines",
                                 line=dict(color=v.color), hovertemplate="%{y:.3f}<extra>" + v.name + "</extra>"))
    charts.style(fig, height=360, x_title="cost evaluation", y_title="expected cost ⟨C⟩", hovermode="x unified",
                 legend=True)
    sec.figure(fig, "The penalised settings minimise conflicts plus penalty; (c) minimises conflicts alone. "
                    "Lower is better; 0 means every outcome is a proper colouring.")

    # Sampling the final circuits
    sec = page.section("Sampling the final circuits",
                       f"The p = {args.depth} circuits are compiled to gates and sampled ({args.shots} shots), "
                       f"ideally and with depolarizing noise (CX error {args.noise:g}).")
    body = []
    for v in variants:
        rates = sampled[v.name]
        exact = runs[(v.name, args.depth)].probabilities[proper].sum()
        row = [v.name, cx_counts[v.name], pct(exact), pct(rates["ideal"][0])]
        row += [pct(rates["noisy"][0]), pct(rates["noisy"][1])] if "noisy" in rates else [HTML("—"), HTML("—")]
        body.append(row)
    sec.table(["setting", "CX gates", "P(proper) exact", "P(proper) sampled", "P(proper) noisy", "P(valid) noisy"],
              body, numeric=[1, 2, 3, 4, 5], max_height=None)
    if args.noise > 0:
        noisy_best = max(variants, key=lambda v: sampled[v.name]["noisy"][0])
        sec.callout(
            f"With noise, “{noisy_best.name}” still gives the most proper colourings "
            f"({sampled[noisy_best.name]['noisy'][0]:.1%}), but every setting loses much of its advantage: its "
            f"depth-{args.depth} circuit compiles to {cx_counts[noisy_best.name]} CX gates on {n} qubits, and each "
            "error can break the one-hot structure that the ideal XY mixer preserves.", "warn", "Noise.")

    # Most likely outcomes of the best setting
    order = np.argsort(best_run.probabilities)[::-1][:10]
    body = []
    for k in order:
        assign = decode(int(k), vertices, colors)
        clashes = int(conflicts[k])
        body.append([format(int(k), f"0{n}b"), " ".join(f"v{v}={c}" for v, c in enumerate(assign)),
                     clashes, bool(proper[k]), pct(best_run.probabilities[k])])
    page.section("Most likely outcomes", f"Top 10 bitstrings of “{best.name}” at p = {args.depth} "
                                         "(qubit 0 on the right).").table(
        ["bitstring", "colouring", "clashing edges", "proper", "probability"], body, numeric=[2, 4])

    page.reproduce({"graph": args.graph, "colours": colors, "depth": args.depth, "restarts": args.restarts,
                    "penalty": args.penalty, "shots": args.shots, "CX error": args.noise, "seed": args.seed,
                    "runtime": f"{time.time() - started:.0f} s"})
    path = page.save()
    print(f"\nReport: {path}")


if __name__ == "__main__":
    main()
