"""
Use case: solve a small travelling salesman problem with QAOA and W-state starts

x_{i,t} = 1 means "city i is visited at time t", so a tour is a permutation
matrix: one-hot in every row (each city once) and every column (one city per
time). As in the original MasterResearch experiment, city 0 is fixed at time 0
and every other city starts in a W state over times 1..N-1, which leaves
(N-1)^2 qubits. Three QAOA settings are compared at depths p = 1..P:

(a) uniform start (H on every qubit) + X mixer + row and column penalties,
(b) W state per city + X mixer + row and column penalties (the research setting),
(c) W state per city + XY ring mixer inside each city row + column penalty only.

The same W-start problem is also solved through qiskit-optimization
(QuadraticProgram + QAOA + MinimumEigenOptimizer), the modern counterpart of
the original code. The results are written to
outputs/usecases/tsp_qaoa/index.html.

Examples
--------
uv run python usecases/tsp_qaoa.py
uv run python usecases/tsp_qaoa.py --cities 4 --depth 4 --restarts 6
uv run python usecases/tsp_qaoa.py --input data/tsp.txt
"""

import itertools
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import StatevectorSampler
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.minimum_eigensolvers import QAOA
from qiskit_optimization.optimizers import COBYLA

from _common import base_parser, sample
from _qaoa import Qubo, Run, Variant, cost_scale, optimize, qaoa_circuit
from onehot import w_state
from onehot.report import HTML, Page, Stat, charts

QISKIT_OPTIMIZATION_SNIPPET = """\
qp = QuadraticProgram("TSP")                      # x_i_t for cities/times 1..N-1
...                                               # tour length + one-hot constraints
qaoa = QAOA(StatevectorSampler(), COBYLA(maxiter=200), reps=p,
            initial_state=w_rows)                 # W state per city row
result = MinimumEigenOptimizer(qaoa).solve(qp)    # constraints become penalties
"""


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


def random_cities(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Function that places n cities at random and returns their distances

    Parameters
    ----------
    n: int
        Number of cities
    seed: int
        Seed of the random positions

    Returns
    -------
    distances: np.ndarray
        Euclidean distance matrix of shape (n, n)
    positions: np.ndarray
        Coordinates of shape (n, 2)
    """
    # Scatter the cities over a 10 x 10 square
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, 10, size=(n, 2))

    # Pairwise Euclidean distances, rounded for readable tables
    distances = np.round(np.linalg.norm(positions[:, None] - positions[None, :], axis=-1), 2)

    return distances, positions


def read_tsp_file(path: Path) -> np.ndarray:
    """
    Function that reads a directed TSP in the original format

    The first line is "V E"; each of the next E lines is "s t d", a directed
    edge from city s to city t of length d. A missing edge is made very long
    (ten times the longest edge) so that tours avoid it.

    Parameters
    ----------
    path: Path
        Text file in the format above

    Returns
    -------
    distances: np.ndarray
        Distance matrix of shape (V, V); distances[i, j] is the edge i -> j
    """
    # Read the header and the edge list
    lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
    v, e = int(lines[0][0]), int(lines[0][1])
    distances = np.full((v, v), np.inf)
    np.fill_diagonal(distances, 0.0)
    for s, t, d in lines[1:1 + e]:
        distances[int(s), int(t)] = float(d)

    # Replace missing edges with a long detour
    longest = distances[np.isfinite(distances)].max()
    distances[~np.isfinite(distances)] = 10 * longest

    return distances


def layout_from_distances(distances: np.ndarray) -> np.ndarray:
    """
    Function that finds 2-D positions whose distances resemble the matrix (classical MDS)

    Parameters
    ----------
    distances: np.ndarray
        Distance matrix (made symmetric before embedding)

    Returns
    -------
    positions: np.ndarray
        Coordinates of shape (n, 2)
    """
    # Double-centre the squared distances and keep the two largest eigenvectors
    sym = (distances + distances.T) / 2
    n = len(sym)
    centre = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centre @ (sym**2) @ centre
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:2]
    positions = vectors[:, order] * np.sqrt(np.maximum(values[order], 0))

    return positions


def var(i: int, t: int, n: int) -> int:
    """
    Function that returns the qubit of x_{i,t} in the reduced encoding

    Parameters
    ----------
    i: int
        City (1..N-1)
    t: int
        Time (1..N-1)
    n: int
        Number of cities N

    Returns
    -------
    index: int
        Qubit (i - 1)(N - 1) + (t - 1); each city row owns N - 1 neighbouring qubits
    """
    index = (i - 1) * (n - 1) + (t - 1)

    return index


def tour_qubo(distances: np.ndarray) -> Qubo:
    """
    Function that builds the tour length as a QUBO with city 0 fixed at time 0

    Parameters
    ----------
    distances: np.ndarray
        Distance matrix; distances[i, j] is the leg i -> j

    Returns
    -------
    qubo: Qubo
        Length of the closed tour 0 -> ... -> 0 on (N-1)^2 variables
    """
    n = len(distances)
    qubo = Qubo((n - 1) ** 2)

    # First leg from city 0 and last leg back to city 0 are linear terms
    for j in range(1, n):
        qubo.add_linear(var(j, 1, n), distances[0, j])
        qubo.add_linear(var(j, n - 1, n), distances[j, 0])

    # Legs between consecutive times link the two cities visited then
    for t in range(1, n - 1):
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    qubo.add_quadratic(var(i, t, n), var(j, t + 1, n), distances[i, j])

    return qubo


def rows_and_columns(n: int) -> tuple[list[list[int]], list[list[int]]]:
    """
    Function that lists the qubits of every city row and every time column

    Parameters
    ----------
    n: int
        Number of cities N

    Returns
    -------
    rows: list[list[int]]
        For each city 1..N-1, its qubits over times 1..N-1
    columns: list[list[int]]
        For each time 1..N-1, its qubits over cities 1..N-1
    """
    rows = [[var(i, t, n) for t in range(1, n)] for i in range(1, n)]
    columns = [[var(i, t, n) for i in range(1, n)] for t in range(1, n)]

    return rows, columns


def decode(index: int, n: int) -> list[int] | None:
    """
    Function that turns a basis index into a tour

    Parameters
    ----------
    index: int
        Basis-state index (qubit q = bit q)
    n: int
        Number of cities N

    Returns
    -------
    tour: list[int] | None
        Cities in visiting order starting with 0, or None when the bits are not a permutation
    """
    # Read which city is visited at each time 1..N-1
    tour: list[int] | None = [0]
    for t in range(1, n):
        cities = [i for i in range(1, n) if index >> var(i, t, n) & 1]
        if len(cities) != 1:
            tour = None
            break
        tour.append(cities[0])

    # A valid tour visits every city exactly once
    if tour is not None and len(set(tour)) != n:
        tour = None

    return tour


def tour_length(tour: list[int], distances: np.ndarray) -> float:
    """
    Function that returns the length of a closed tour

    Parameters
    ----------
    tour: list[int]
        Cities in visiting order, starting with 0
    distances: np.ndarray
        Distance matrix

    Returns
    -------
    length: float
        Sum of every leg, including the way back to the start
    """
    length = float(sum(distances[tour[k], tour[(k + 1) % len(tour)]] for k in range(len(tour))))

    return length


def tour_text(tour: list[int] | None) -> str:
    """
    Function that formats a tour as "0 → 2 → 1 → 0"

    Parameters
    ----------
    tour: list[int] | None
        Tour, or None for an invalid assignment

    Returns
    -------
    text: str
        Readable tour
    """
    text = "not a valid tour" if tour is None else " → ".join(str(c) for c in tour + [tour[0]])

    return text


def initial_state(kind: str, n: int) -> QuantumCircuit:
    """
    Function that prepares the initial state of a QAOA setting

    Parameters
    ----------
    kind: str
        "uniform" (H on every qubit) or "w" (one W state per city row)
    n: int
        Number of cities N

    Returns
    -------
    qc: QuantumCircuit
        Initial-state circuit on (N-1)^2 qubits
    """
    qc = QuantumCircuit((n - 1) ** 2)

    # Every bitstring, valid tour or not, with the same amplitude
    if kind == "uniform":
        qc.h(range(qc.num_qubits))

    # Each city visited exactly once, at a uniformly random time (like create_init_state)
    else:
        for row in rows_and_columns(n)[0]:
            qc.compose(w_state(n - 1), qubits=row, inplace=True)

    return qc


def tour_map(positions: np.ndarray, panels: list[tuple[str, list[int] | None, str]]) -> go.Figure:
    """
    Function that draws the cities and one tour per panel with arrows

    Parameters
    ----------
    positions: np.ndarray
        Coordinates of every city
    panels: list[tuple[str, list[int] | None, str]]
        (title, tour or None, line colour) for each panel

    Returns
    -------
    fig: go.Figure
        Styled figure with one panel per tour
    """
    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=[p[0] for p in panels], horizontal_spacing=0.06)
    charts.style(fig, height=380, legend=False)
    pad = 0.12 * float(np.ptp(positions)) + 0.5

    for col, (_, tour, color) in enumerate(panels, start=1):
        axis = "" if col == 1 else str(col)

        # One arrow per leg, so the direction of travel is visible
        if tour is not None:
            for k in range(len(tour)):
                a, b = positions[tour[k]], positions[tour[(k + 1) % len(tour)]]
                fig.add_annotation(x=b[0], y=b[1], ax=a[0], ay=a[1], xref=f"x{axis}", yref=f"y{axis}",
                                   axref=f"x{axis}", ayref=f"y{axis}", showarrow=True, arrowhead=3,
                                   arrowsize=1.2, arrowwidth=2, arrowcolor=color, standoff=14, startstandoff=14,
                                   text="")

        # Cities as labelled dots; the start city is drawn darker
        fig.add_trace(go.Scatter(
            x=positions[:, 0], y=positions[:, 1], mode="markers+text", text=[str(i) for i in range(len(positions))],
            textposition="middle center", textfont=dict(color="#ffffff", size=12),
            marker=dict(size=26, color=[charts.INK] + [charts.MUTED] * (len(positions) - 1),
                        line=dict(width=2, color="#fcfcfb")),
            hovertemplate="city %{text}<extra></extra>", showlegend=False,
        ), row=1, col=col)

        # Hide the axes and keep distances undistorted
        fig.update_xaxes(visible=False, range=[positions[:, 0].min() - pad, positions[:, 0].max() + pad],
                         row=1, col=col)
        fig.update_yaxes(visible=False, range=[positions[:, 1].min() - pad, positions[:, 1].max() + pad],
                         scaleanchor=f"x{axis}", row=1, col=col)

    fig.update_annotations(selector=dict(text=""), font=dict(size=1))
    fig.update_annotations(selector=lambda a: a.text, font=dict(size=13, color=charts.INK))

    return fig


def summarize(probs: np.ndarray, feasible: np.ndarray, optimal: np.ndarray, lengths: np.ndarray,
              optimum: float) -> dict[str, float]:
    """
    Function that summarises a distribution over basis states

    Parameters
    ----------
    probs: np.ndarray
        Probability of every basis state
    feasible: np.ndarray
        True where the state is a valid tour
    optimal: np.ndarray
        True where the state is an optimal tour
    lengths: np.ndarray
        Tour length of every basis state
    optimum: float
        Optimal tour length

    Returns
    -------
    summary: dict[str, float]
        "optimal": probability of an optimal tour,
        "valid": probability of any valid tour,
        "conditional": probability that a valid outcome is optimal (invalid outcomes discarded),
        "ratio": expected length of the valid outcomes divided by the optimal length
    """
    # Probabilities of the optimal and of all valid tours
    p_opt, p_valid = float(probs[optimal].sum()), float(probs[feasible].sum())

    # Keep only the valid outcomes, as one would after measuring
    conditional = p_opt / p_valid if p_valid > 0 else float("nan")
    ratio = float(probs[feasible] @ lengths[feasible] / p_valid / optimum) if p_valid > 0 else float("nan")
    summary = {"optimal": p_opt, "valid": p_valid, "conditional": conditional, "ratio": ratio}

    return summary


def solve_with_qiskit_optimization(distances: np.ndarray, reps: int, seed: int) -> dict:
    """
    Function that solves the W-start problem with qiskit-optimization's QAOA

    Parameters
    ----------
    distances: np.ndarray
        Distance matrix
    reps: int
        Number of QAOA layers
    seed: int
        Seed of the sampler

    Returns
    -------
    outcome: dict
        Tour, objective value, status, probability of the optimal tour and runtime
    """
    n = len(distances)
    objective = tour_qubo(distances)
    rows, columns = rows_and_columns(n)
    names = [f"x_{i}_{t}" for i in range(1, n) for t in range(1, n)]

    # Model: binary x_i_t in qubit order, tour length, and both one-hot constraint families
    qp = QuadraticProgram("TSP")
    for name in names:
        qp.binary_var(name)
    qp.minimize(linear={names[k]: v for k, v in objective.linear.items()},
                quadratic={(names[i], names[j]): v for (i, j), v in objective.quadratic.items()})
    for k, row in enumerate(rows, start=1):
        qp.linear_constraint({names[q]: 1 for q in row}, "==", 1, name=f"city_{k}")
    for k, column in enumerate(columns, start=1):
        qp.linear_constraint({names[q]: 1 for q in column}, "==", 1, name=f"time_{k}")

    # QAOA with the W-state start; MinimumEigenOptimizer turns constraints into penalties
    started = time.time()
    qaoa = QAOA(StatevectorSampler(seed=seed), COBYLA(maxiter=200), reps=reps,
                initial_state=initial_state("w", n))
    result = MinimumEigenOptimizer(qaoa).solve(qp)
    elapsed = time.time() - started

    # Read the best tour and how often the optimal tour was sampled
    index = sum(int(round(v)) << k for k, v in enumerate(result.x))
    optimum = min(tour_length([0, *perm], distances) for perm in itertools.permutations(range(1, n)))
    p_opt = sum(s.probability for s in result.samples
                if (t := decode(sum(int(round(v)) << k for k, v in enumerate(s.x)), n)) is not None
                and abs(tour_length(t, distances) - optimum) < 1e-9)
    outcome = {"tour": decode(index, n), "fval": float(result.fval), "status": result.status.name,
               "p_optimal": float(p_opt), "seconds": elapsed, "qp": qp}

    return outcome


def main() -> None:
    """
    Function that runs the comparison and writes the report

    Returns
    -------
    None
    """
    # Read the command-line arguments
    parser = base_parser(__doc__)
    parser.add_argument("--cities", type=int, default=5, help="number of random cities N >= 3 (default: 5)")
    parser.add_argument("--input", type=Path, default=None, help="TSP file in the original 'V E' / 's t d' format")
    parser.add_argument("--depth", type=int, default=3, help="largest QAOA depth p (default: 3)")
    parser.add_argument("--restarts", type=int, default=3, help="random starts per depth (default: 3)")
    parser.add_argument("--penalty-scale", type=float, default=1.5,
                        help="penalty weight as a multiple of the longest leg (default: 1.5)")
    args = parser.parse_args()
    started = time.time()

    # Load or generate the cities
    if args.input is not None:
        distances = read_tsp_file(args.input)
        positions = layout_from_distances(distances)
        source = f"file {args.input.name}"
    else:
        distances, positions = random_cities(args.cities, args.seed)
        source = f"{args.cities} random cities (seed {args.seed})"
    n = len(distances)
    if n < 3:
        raise SystemExit("need at least 3 cities")
    num_qubits = (n - 1) ** 2
    rows, columns = rows_and_columns(n)

    # Brute-force the optimal tours for reference
    tours = [[0, *perm] for perm in itertools.permutations(range(1, n))]
    optimum = min(tour_length(t, distances) for t in tours)
    optimal_tours = [t for t in tours if abs(tour_length(t, distances) - optimum) < 1e-9]

    # Objective, penalised costs and masks over all basis states
    weight = args.penalty_scale * float(distances.max())
    objective = tour_qubo(distances)
    both = tour_qubo(distances)
    for group in rows + columns:
        both.add_one_hot_penalty(group, weight)
    columns_only = tour_qubo(distances)
    for group in columns:
        columns_only.add_one_hot_penalty(group, weight)
    lengths = objective.values()
    feasible = np.array([decode(k, n) is not None for k in range(2**num_qubits)])
    optimal = feasible & (np.abs(lengths - optimum) < 1e-9)
    scale = cost_scale(lengths, feasible)

    def metrics(probs: np.ndarray) -> dict[str, float]:
        return summarize(probs, feasible, optimal, lengths, optimum)

    # The three settings to compare, each with a fixed colour
    variants = [
        Variant("H + X mixer + penalty", initial_state("uniform", n), both, "x", color=charts.SERIES[1]),
        Variant("W + X mixer + penalty", initial_state("w", n), both, "x", color=charts.SERIES[0]),
        Variant("W + XY mixer", initial_state("w", n), columns_only, "xy", rows, color=charts.SERIES[2]),
    ]
    colors = {v.name: v.color for v in variants}

    # Optimise every setting at every depth, warm-starting from the previous depth
    depths = list(range(1, args.depth + 1))
    runs: dict[tuple[str, int], Run] = {}
    for variant in variants:
        previous = None
        for p in depths:
            run = optimize(variant, p, scale, args.restarts, args.seed + p, warm_start=previous)
            runs[(variant.name, p)] = run
            previous = run.params
            print(f"{variant.name:<24} p={p}  P(opt)={run.probabilities[optimal].sum():.3f}  "
                  f"P(valid)={run.probabilities[feasible].sum():.3f}")

    # Sample the deepest circuits, ideally and with gate noise
    sampled, cx_counts = {}, {}
    for variant in variants:
        circuit = qaoa_circuit(variant, args.depth, scale, runs[(variant.name, args.depth)].params)
        compiled = transpile(circuit, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1,
                             seed_transpiler=args.seed)
        cx_counts[variant.name] = compiled.count_ops().get("cx", 0)
        rates = {}
        for label, noise in (("ideal", 0.0), ("noisy", args.noise)):
            if label == "noisy" and noise <= 0:
                continue
            counts = sample(circuit, args.shots, args.seed, noise)
            probs = np.zeros(2**num_qubits)
            for bits, c in counts.items():
                probs[int(bits.replace(" ", ""), 2)] += c / args.shots
            rates[label] = metrics(probs)
        sampled[variant.name] = rates

    # The same W-start problem through qiskit-optimization
    modern = solve_with_qiskit_optimization(distances, min(args.depth, 2), args.seed)
    print(f"qiskit-optimization: {tour_text(modern['tour'])}  fval={modern['fval']:.2f}  "
          f"P(opt)={modern['p_optimal']:.3f}  {modern['seconds']:.1f}s")

    # Reference numbers and the best setting
    guess = len(optimal_tours) / len(tours)
    final = {v.name: metrics(runs[(v.name, args.depth)].probabilities) for v in variants}
    best = max(variants, key=lambda v: final[v.name]["conditional"])
    best_run = runs[(best.name, args.depth)]
    top = int(np.argmax(np.where(feasible, best_run.probabilities, -1.0)))

    page = Page(
        "Travelling salesman with QAOA", category="usecases", slug="tsp_qaoa", out_root=args.out,
        summary="Solve a small TSP with QAOA using one-hot city/time variables, comparing a uniform start, "
        "the W-state start of the original research and a W start with an XY mixer.",
        tags=["QAOA", "TSP", "one-hot constraint", "qiskit-optimization"],
    )
    page.lead(
        rf"Find the shortest closed tour through {n} cities ({source}). The variable \(x_{{i,t}}=1\) means "
        r"“city \(i\) is visited at time \(t\)”, so a tour is a permutation matrix: one-hot in every row and "
        r"every column. Following the original experiment, city 0 is fixed at time 0 and every other city starts "
        rf"in a W state \(|W_{{{n - 1}}}\rangle\) over times \(1,\dots,{n - 1}\), leaving \({n - 1}^2 = {num_qubits}\) "
        r"qubits. The cost is the tour length \(\sum_t\sum_{i,j} d_{ij}\,x_{i,t}\,x_{j,t+1}\)."
    )
    page.headline([
        Stat(v.name, f"{final[v.name]['conditional']:.1%}",
             f"of valid outcomes are optimal at p = {args.depth} ({final[v.name]['valid']:.1%} of outcomes valid)",
             "good" if final[v.name]["conditional"] > guess + 0.01 else "warn")
        for v in variants
    ] + [Stat("Random valid tour", f"{guess:.1%}", f"{len(optimal_tours)} of {len(tours)} tours are optimal "
                                                    f"(length {optimum:.2f})")])

    # Map of the optimal tour and the best setting's most likely outcome
    sec = page.section("Cities and tours", "Arrows show the direction of travel; city 0 (dark) is the fixed start.")
    sec.figure(tour_map(positions, [
        (f"Optimal tour · length {optimum:.2f}", optimal_tours[0], charts.INK),
        (f"Most likely valid tour of “{best.name}”, p = {args.depth}", decode(top, n), best.color),
    ]), f"The most likely valid tour of “{best.name}” has probability {best_run.probabilities[top]:.1%} and length "
        f"{tour_length(decode(top, n), distances):.2f}. The distances below are directed: d[i, j] is the leg "
        "from i to j.")
    sec.table(["from \\ to"] + [str(j) for j in range(n)],
              [[str(i)] + [float(distances[i, j]) for j in range(n)] for i in range(n)],
              numeric=list(range(1, n + 1)), max_height=None)

    # Circuits
    sec = page.section(
        "Circuits",
        rf"The W start puts all probability on assignments where every city is visited exactly once: "
        rf"\({n - 1}^{{{n - 1}}}\) = {(n - 1) ** (n - 1)} states, {len(tours)} of which are valid tours. "
        r"The uniform start spreads it over all " + f"{2 ** num_qubits:,} bitstrings.",
    )
    sec.circuits([(variants[1].init, "Initial state of (b) and (c): one W state per city row"),
                  (qaoa_circuit(Variant(best.name, QuantumCircuit(num_qubits), variants[2].qubo, "xy", rows), 1,
                                scale), "One layer of (c): tour-length and column-penalty rotations, then XY "
                                        "rotations inside each city row")])
    sec.text("The XY rotations move a city between times but never remove it or duplicate it, so the row "
             "constraint holds throughout and only the column constraint needs a penalty. The X mixer of (a) and (b) "
             "flips single bits, so both families of constraints must be enforced by penalties.")

    # Results versus depth
    sec = page.section("Results versus depth", "Exact probabilities of the optimised state at each depth.")
    series = {key: {v.name: [metrics(runs[(v.name, p)].probabilities)[key] for p in depths] for v in variants}
              for key in ("optimal", "valid", "conditional", "ratio")}
    panels = {"P(optimal | valid)": series["conditional"], "P(valid tour)": series["valid"],
              "P(optimal tour)": series["optimal"]}
    fig = charts.small_multiples(depths, panels, colors=colors, x_title="QAOA depth p", y_title="probability",
                                 hover_format=".1%", height=400)
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    fig.update_xaxes(dtick=1)
    fig.add_hline(y=guess, line=dict(color=charts.MUTED, width=1, dash="dot"), row=1, col=1,
                  annotation_text="random valid tour", annotation_position="top left",
                  annotation_font=dict(color=charts.MUTED, size=11))
    sec.figure(fig, "Left: share of the valid outcomes that are optimal, i.e. after discarding invalid ones; the "
                    "dotted line is picking a valid tour uniformly at random. Middle: probability of measuring any "
                    "valid tour. Right: probability of measuring an optimal tour.")
    fig = charts.lines(depths, series["ratio"],
                       colors=colors, x_title="QAOA depth p", y_title="expected length / optimal", height=320)
    fig.update_xaxes(dtick=1)
    sec.figure(fig, "Average length of the valid tours measured, relative to the optimum (1.0 = always optimal).")

    # Takeaways computed from the results
    uniform, research, xy = (final[v.name] for v in variants)
    sec.callout(
        f"At p = {args.depth}, a valid outcome is optimal {uniform['conditional']:.1%} of the time for the uniform "
        f"start, {research['conditional']:.1%} for the W start with X mixer (the research setting) and "
        f"{xy['conditional']:.1%} for the W start with XY mixer, against {guess:.1%} for a random valid tour. "
        f"Outcomes are valid {uniform['valid']:.1%}, {research['valid']:.1%} and {xy['valid']:.1%} of the time."
        + (" The W start raises validity, and only the XY mixer then lets QAOA favour short tours beyond guessing."
           if xy["conditional"] > max(research["conditional"], uniform["conditional"], guess) else ""),
        "info", "Takeaway.",
    )
    if all(abs(metrics(runs[(variants[1].name, p)].probabilities)["conditional"] - guess) < 0.02 for p in depths):
        sec.callout(
            "The W start with an X mixer does not move away from its starting distribution: any X rotation leaks "
            "probability into penalised states, so the optimiser keeps the mixer near zero. The W start pays off "
            "once the mixer respects the one-hot rows.", "warn", "Why (b) stays flat.")

    # Optimisation traces
    sec = page.section("Optimisation", f"Best expected cost found by COBYLA at p = {args.depth} "
                                       f"(best of {args.restarts} random starts plus a warm start).")
    fig = go.Figure()
    for v in variants:
        history = runs[(v.name, args.depth)].history
        fig.add_trace(go.Scatter(x=list(range(1, len(history) + 1)), y=history, name=v.name, mode="lines",
                                 line=dict(color=v.color), hovertemplate="%{y:.2f}<extra>" + v.name + "</extra>"))
    fig.add_hline(y=optimum, line=dict(color=charts.MUTED, width=1, dash="dot"), annotation_text="optimal length",
                  annotation_position="bottom right", annotation_font=dict(color=charts.MUTED, size=11))
    charts.style(fig, height=360, x_title="cost evaluation", y_title="expected cost ⟨C⟩", hovermode="x unified",
                 legend=True)
    sec.figure(fig, "Expected tour length plus penalty. (a) and (b) carry both penalties, (c) only the column "
                    "penalty, so the curves are not on the same objective; the dotted line is the optimal length.")

    # qiskit-optimization
    sec = page.section(
        "With qiskit-optimization",
        "The original code built a QuadraticProgram and solved it with QAOA inside MinimumEigenOptimizer. The same "
        f"route on today's stack (W start, X mixer, p = {min(args.depth, 2)}, 1024 shots per evaluation):",
    )
    sec.code(QISKIT_OPTIMIZATION_SNIPPET, "Outline; the full model is built in solve_with_qiskit_optimization().")
    is_optimal = modern["tour"] is not None and abs(tour_length(modern["tour"], distances) - optimum) < 1e-9
    sec.stats([
        Stat("Returned tour", tour_text(modern["tour"]), f"status {modern['status']}",
             "good" if is_optimal else "warn"),
        Stat("Objective", f"{modern['fval']:.2f}", f"optimal {optimum:.2f}", "good" if is_optimal else "warn"),
        Stat("Optimal tour in samples", f"{modern['p_optimal']:.1%}", "share of the final samples"),
        Stat("Runtime", f"{modern['seconds']:.1f} s", f"{modern['qp'].get_num_vars()} variables, "
                                                      f"{modern['qp'].get_num_linear_constraints()} constraints"),
    ])
    sec.text("MinimumEigenOptimizer returns the best sample it saw, so it can report the optimal tour even when "
             "QAOA puts little probability on it; the probability charts above show how much the circuit itself "
             "concentrates on good tours.")

    # Sampling the final circuits
    sec = page.section("Sampling the final circuits",
                       f"The p = {args.depth} circuits are compiled to gates and sampled ({args.shots} shots), "
                       f"ideally and with depolarizing noise (CX error {args.noise:g}).")
    body = []
    for v in variants:
        rates = sampled[v.name]
        row = [v.name, cx_counts[v.name], pct(final[v.name]["conditional"]), pct(rates["ideal"]["conditional"])]
        if "noisy" in rates:
            row += [pct(rates["noisy"]["conditional"]), pct(rates["noisy"]["valid"])]
        else:
            row += [HTML("—"), HTML("—")]
        body.append(row)
    sec.table(["setting", "CX gates", "P(opt | valid) exact", "P(opt | valid) sampled", "P(opt | valid) noisy",
               "P(valid) noisy"], body, numeric=[1, 2, 3, 4, 5], max_height=None)
    sec.text(f"Conditional rates are estimated from the valid shots only: with {args.shots} shots and a valid share "
             "below 1 %, a handful of shots decides them, so treat those entries as rough.")
    if args.noise > 0:
        noisy_best = max(variants, key=lambda v: sampled[v.name]["noisy"]["conditional"])
        sec.callout(
            f"With noise, the best setting is “{noisy_best.name}”: "
            f"{sampled[noisy_best.name]['noisy']['conditional']:.1%} of its valid outcomes are optimal "
            f"(only {sampled[noisy_best.name]['noisy']['valid']:.1%} of outcomes are valid), compared with "
            f"{guess:.1%} for a random valid tour. Its depth-{args.depth} circuit compiles "
            f"to {cx_counts[noisy_best.name]} CX gates, and every error can break the permutation structure.",
            "warn", "Noise.")

    # Most likely outcomes of every setting
    body = []
    for v in variants:
        probs = runs[(v.name, args.depth)].probabilities
        for k in np.argsort(probs)[::-1][:5]:
            tour = decode(int(k), n)
            body.append([v.name, format(int(k), f"0{num_qubits}b"), tour_text(tour),
                         tour_length(tour, distances) if tour is not None else HTML("—"),
                         bool(optimal[k]), pct(probs[k])])
    page.section("Most likely outcomes", f"Top 5 bitstrings of each setting at p = {args.depth} "
                                         "(qubit 0 on the right).").table(
        ["setting", "bitstring", "tour", "length", "optimal", "probability"], body, numeric=[3, 5])

    page.reproduce({"cities": source, "depth": args.depth, "restarts": args.restarts,
                    "penalty": f"{args.penalty_scale} × longest leg = {weight:.2f}", "shots": args.shots,
                    "CX error": args.noise, "seed": args.seed, "runtime": f"{time.time() - started:.0f} s"})
    path = page.save()
    print(f"\nReport: {path}")


if __name__ == "__main__":
    main()
