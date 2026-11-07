"""
Experiment: prepare W states of arbitrary size and compare the two constructions

For every requested qubit count n and method ("linear" / "tree") this script
checks the state against the exact W state, measures circuit size before and
after routing onto hardware-like connectivity, samples the circuit with gate
noise, and traces how the excitation spreads over the qubits. The results are
written to outputs/experiments/w_state_scaling/index.html.

Examples
--------
uv run python experiments/w_state_scaling.py
uv run python experiments/w_state_scaling.py --n 2-24 --noise 0.005
uv run python experiments/w_state_scaling.py --n 3,5,8,13 --method tree
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from onehot import w_state
from onehot.report import HTML, Page, Stat, charts

ROOT = Path(__file__).resolve().parents[1]
BASIS = ["cx", "rz", "sx", "x"]
METHODS = ("linear", "tree")
TOPOLOGIES = ("all-to-all", "line", "heavy-hex")
NOISY_TOPOLOGIES = ("all-to-all", "line")
MAX_STATEVECTOR = 22


@dataclass
class Row:
    """
    Class that holds every measurement for one (n, method) pair

    Parameters
    ----------
    n: int
        Number of qubits
    method: str
        Construction ("linear" or "tree")
    fidelity: float | None
        Fidelity with the exact W state (None when n is too large)
    depth: dict[str, int]
        Topology -> circuit depth after routing
    cx: dict[str, int]
        Topology -> CX count after routing
    one_hot_rate: dict[str, float]
        Topology -> share of noisy shots with exactly one '1'
    """

    n: int
    method: str
    fidelity: float | None
    depth: dict[str, int]
    cx: dict[str, int]
    one_hot_rate: dict[str, float]


def parse_ns(text: str) -> list[int]:
    """
    Function that parses qubit counts such as "2-16" or "3,5,8"

    Parameters
    ----------
    text: str
        Comma-separated counts and inclusive ranges

    Returns
    -------
    ns: list[int]
        Sorted, de-duplicated qubit counts
    """
    # Expand every range and collect single values
    values: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            low, high = (int(v) for v in part.split("-"))
            values.update(range(low, high + 1))
        elif part:
            values.add(int(part))

    # Reject empty input and counts below one
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("qubit counts must be >= 1")
    ns = sorted(values)

    return ns


def w_vector(n: int) -> np.ndarray:
    """
    Function that returns the exact W state as a vector

    Parameters
    ----------
    n: int
        Number of qubits

    Returns
    -------
    vector: np.ndarray
        State vector with 1/sqrt(n) on every one-hot basis state
    """
    # Put the same amplitude on |0..01>, |0..10>, ..., |10..0>
    vector = np.zeros(2**n, dtype=complex)
    vector[[1 << k for k in range(n)]] = 1 / np.sqrt(n)

    return vector


def coupling_map(n: int, topology: str) -> CouplingMap | None:
    """
    Function that returns the connectivity a circuit is routed onto

    Parameters
    ----------
    n: int
        Number of logical qubits
    topology: str
        "all-to-all", "line" or "heavy-hex"

    Returns
    -------
    coupling: CouplingMap | None
        Coupling map, or None when any two qubits may interact
    """
    # Any pair may interact, or only neighbours on a line of n qubits
    coupling = None
    if topology == "line" and n > 1:
        coupling = CouplingMap.from_line(n)

    # Use the smallest heavy-hex lattice (odd distance d) with at least n qubits
    elif topology == "heavy-hex":
        d = 3
        while (5 * d * d - 2 * d - 1) // 2 < n:
            d += 2
        coupling = CouplingMap.from_heavy_hex(d)

    return coupling


def route(qc: QuantumCircuit, topology: str, seed: int) -> QuantumCircuit:
    """
    Function that decomposes a circuit into basis gates and routes it

    Parameters
    ----------
    qc: QuantumCircuit
        Logical circuit
    topology: str
        Connectivity passed to coupling_map
    seed: int
        Seed of the transpiler

    Returns
    -------
    routed: QuantumCircuit
        Circuit in basis gates that respects the connectivity
    """
    # Optimisation level 3 gives each construction its best routed form
    routed = transpile(
        qc, basis_gates=BASIS, coupling_map=coupling_map(qc.num_qubits, topology),
        optimization_level=3, seed_transpiler=seed,
    )

    return routed


def noise_model(p2: float) -> NoiseModel:
    """
    Function that builds a gate-only depolarizing noise model

    Parameters
    ----------
    p2: float
        Depolarizing error of each CX; single-qubit gates get p2 / 10

    Returns
    -------
    model: NoiseModel
        Noise model for AerSimulator
    """
    # Attach depolarizing errors to every two-qubit and single-qubit gate
    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    model.add_all_qubit_quantum_error(depolarizing_error(p2 / 10, 1), ["sx", "x"])

    return model


def sample(routed: QuantumCircuit, n: int, sim: AerSimulator, shots: int, seed: int) -> dict[str, int]:
    """
    Function that measures the logical qubits of a routed circuit

    Parameters
    ----------
    routed: QuantumCircuit
        Circuit returned by route
    n: int
        Number of logical qubits
    sim: AerSimulator
        Simulator (with or without noise)
    shots: int
        Number of shots
    seed: int
        Seed of the simulator

    Returns
    -------
    counts: dict[str, int]
        Bitstring (logical qubit 0 on the right) -> count
    """
    # Find where routing finally placed each logical qubit
    qc = routed.copy()
    if qc.layout is not None:
        physical = qc.layout.final_index_layout()[:n]
    else:
        physical = list(range(n))

    # Measure logical qubit k into classical bit k
    creg = ClassicalRegister(n, "m")
    qc.add_register(creg)
    for k, q in enumerate(physical):
        qc.measure(q, creg[k])

    counts = sim.run(qc, shots=shots, seed_simulator=seed).result().get_counts()

    return counts


def one_hot_rate(counts: dict[str, int]) -> float:
    """
    Function that returns the share of outcomes with exactly one '1'

    Parameters
    ----------
    counts: dict[str, int]
        Bitstring -> count

    Returns
    -------
    rate: float
        Share of one-hot outcomes
    """
    # Count the shots that stayed in the one-hot subspace
    total = sum(counts.values())
    rate = sum(c for bits, c in counts.items() if bits.count("1") == 1) / total

    return rate


def excitation_flow(n: int, method: str) -> tuple[list[str], np.ndarray]:
    """
    Function that tracks which qubit holds the excitation after each stage

    A stage is one split for "linear" and one tree level for "tree", where all
    splits of a level act on disjoint qubits.

    Parameters
    ----------
    n: int
        Number of qubits
    method: str
        Construction ("linear" or "tree")

    Returns
    -------
    labels: list[str]
        Name of each stage, starting with "start"
    flow: np.ndarray
        Array of shape (stages, n): probability that qubit k is |1>
    """
    qc = w_state(n, method)

    # Group the instructions into stages: X first, then one stage per split or level
    stages: list[list] = []
    current: list = []
    busy: set[int] = set()
    for inst in qc.data:
        qubits = {qc.find_bit(q).index for q in inst.qubits}
        if inst.operation.name == "x":
            stages.append([inst])
            continue
        if inst.operation.name == "cry" and (method == "linear" or qubits & busy):
            stages.append(current)
            current, busy = [], set()
        busy |= qubits
        current.append(inst)
    if current:
        stages.append(current)

    # Evolve the state stage by stage and record P(qubit k = 1)
    labels, rows = [], []
    state = Statevector.from_label("0" * n)
    for i, stage in enumerate(stages):
        part = QuantumCircuit(n)
        for inst in stage:
            part.append(inst.operation, [qc.find_bit(q).index for q in inst.qubits])
        state = state.evolve(part)
        probs = state.probabilities()
        rows.append([probs[[idx for idx in range(2**n) if idx >> k & 1]].sum() for k in range(n)])
        labels.append("start" if i == 0 else f"{'split' if method == 'linear' else 'level'} {i}")
    flow = np.array(rows)

    return labels, flow


def run(args: argparse.Namespace) -> list[Row]:
    """
    Function that measures every (n, method) pair

    Parameters
    ----------
    args: argparse.Namespace
        Parsed command-line arguments

    Returns
    -------
    rows: list[Row]
        One row per (n, method)
    """
    sim = AerSimulator(noise_model=noise_model(args.noise))
    rows = []
    for n in args.n:
        for method in args.methods:
            qc = w_state(n, method)

            # Compare with the exact W state while the state vector is small enough
            fidelity = None
            if n <= MAX_STATEVECTOR:
                fidelity = float(abs(np.vdot(w_vector(n), Statevector(qc).data)) ** 2)

            # Route onto each topology, then sample the small ones with noise
            depth, cx, rate = {}, {}, {}
            for topology in TOPOLOGIES:
                routed = route(qc, topology, args.seed)
                depth[topology] = routed.depth()
                cx[topology] = routed.count_ops().get("cx", 0)
                if topology in NOISY_TOPOLOGIES and n <= args.noise_max_n:
                    rate[topology] = one_hot_rate(sample(routed, n, sim, args.shots, args.seed))

            rows.append(Row(n, method, fidelity, depth, cx, rate))
            print(f"n={n:>3} {method:<6} depth={depth} cx={cx}")

    return rows


def by_method(rows: list[Row], method: str, getter) -> list:
    """
    Function that picks one value per n for a given method

    Parameters
    ----------
    rows: list[Row]
        Every measured row
    method: str
        Construction to pick
    getter: Callable[[Row], Any]
        Value to read from each row

    Returns
    -------
    values: list
        Values ordered by n
    """
    values = [getter(r) for r in rows if r.method == method]

    return values


def build_report(args: argparse.Namespace, rows: list[Row]) -> Path:
    """
    Function that writes the HTML report

    Parameters
    ----------
    args: argparse.Namespace
        Parsed command-line arguments
    rows: list[Row]
        Results of run

    Returns
    -------
    path: Path
        Path of the written index.html
    """
    ns, methods = args.n, args.methods
    n_max = max(ns)
    both = set(METHODS) <= set(methods)
    table = {(r.n, r.method): r for r in rows}
    fidelities = [r.fidelity for r in rows if r.fidelity is not None]
    worst = min(fidelities) if fidelities else None
    noisy_ns = [n for n in ns if n <= args.noise_max_n]

    page = Page(
        "W-state scaling", category="experiments", slug="w_state_scaling", out_root=args.out,
        summary="How the linear and binary-tree W-state circuits compare in correctness, size, "
        "routing cost and noise as the number of qubits grows.",
        tags=["w_state", "linear vs tree", "routing", "noise"],
    )
    page.lead(
        r"The W state \(|W_n\rangle\) puts one excitation on exactly one of \(n\) qubits, with every "
        r"position equally likely. <code>onehot.w_state</code> builds it in two ways that use the "
        r"<em>same</em> gates in a different order. This report measures what that order changes."
    )

    # Headline numbers
    hero = []
    if worst is not None:
        hero.append(Stat("Exact W state", f"{worst:.6f}", "lowest fidelity over all runs",
                         "good" if worst > 1 - 1e-9 else "bad"))
    if both:
        lin, tree = table[(n_max, "linear")], table[(n_max, "tree")]
        hero.append(Stat(f"Depth at n = {n_max}", f"{lin.depth['all-to-all']} → {tree.depth['all-to-all']}",
                         "linear → tree, any qubit pair can interact"))
        hero.append(Stat(f"CX on a line, n = {n_max}", f"{lin.cx['line']} vs {tree.cx['line']}",
                         "linear vs tree after routing"))
        if noisy_ns:
            k = max(noisy_ns)
            lin_k, tree_k = table[(k, "linear")], table[(k, "tree")]
            hero.append(Stat(f"One-hot kept, n = {k}",
                             f"{lin_k.one_hot_rate['line']:.1%} vs {tree_k.one_hot_rate['line']:.1%}",
                             f"linear vs tree on a line, CX error {args.noise:g}"))
    page.headline(hero)

    # How the constructions work: excitation flow and circuits
    n_demo = args.detail_n or min(n_max, 8)
    sec = page.section(
        "How the two constructions work",
        r"Both start from \(|10\cdots0\rangle\) and repeatedly split the excitation between a qubit that "
        r"holds it and a fresh qubit, using a controlled-\(R_Y\) followed by a CNOT. Each heatmap shows, "
        rf"for \(n = {n_demo}\), the probability that each qubit holds the excitation after each stage. "
        r"The last row is the W state: every qubit at \(1/n\).",
    )
    for method in methods:
        labels, flow = excitation_flow(n_demo, method)
        fig = charts.heatmap(flow, x=[f"q{k}" for k in range(n_demo)], y=labels, x_title="qubit",
                             y_title="stage", z_title="P(excited)", zmax=1.0, height=90 + 34 * len(labels))
        fig.update_yaxes(autorange="reversed", type="category")
        if method == "linear":
            caption = ("<b>Linear</b>: the excitation is passed down the chain; qubit i keeps 1/(n−i) of what "
                       f"reaches it, so it takes {len(labels) - 1} sequential splits.")
        else:
            caption = ("<b>Tree</b>: each level halves every block at once, so the state is ready after "
                       f"{len(labels) - 1} levels (⌈log₂ n⌉).")
        sec.figure(fig, caption)
    sec.circuits([(w_state(n_demo, m), f"{m}, n = {n_demo}") for m in methods])
    sec.text("Each <code>cry</code>–<code>cx</code> pair is one split. The linear circuit only ever touches "
             "neighbours (i, i+1), while the tree reaches across blocks.")

    # Correctness
    sec = page.section("Correctness", "Fidelity between each prepared state and the exact W state.")
    if fidelities:
        exact = worst > 1 - 1e-9
        sec.callout(f"All {len(fidelities)} circuits prepare the W state exactly (lowest fidelity {worst:.12f})."
                    if exact else f"Lowest fidelity is {worst:.6f}: at least one circuit is not exact.",
                    "good" if exact else "bad")
    if n_max > MAX_STATEVECTOR:
        sec.callout(f"Fidelity is skipped for n > {MAX_STATEVECTOR} (state vector too large).", "info")

    # Circuit size with ideal connectivity
    sec = page.section(
        "Circuit size",
        "Depth after decomposing into the basis gates (cx, rz, sx, x) and optimising, "
        "assuming any two qubits can interact.",
    )
    sec.figure(
        charts.lines(ns, {m: by_method(rows, m, lambda r: r.depth["all-to-all"]) for m in methods},
                     colors=charts.METHOD_COLORS, x_title="qubits n", y_title="circuit depth"),
        "Depth grows linearly for the chain and logarithmically for the tree (it steps up after each power of two).",
    )
    if both:
        same_cx = all(table[(n, "linear")].cx["all-to-all"] == table[(n, "tree")].cx["all-to-all"] for n in ns)
        cx_max = table[(n_max, "linear")].cx["all-to-all"]
        sec.callout(
            f"The CX count is the same for both constructions at every n (e.g. {cx_max} at n = {n_max}): "
            "both perform n − 1 splits, so only the order, and hence the depth, differs."
            if same_cx else "The CX counts differ between the constructions; see the table below.",
            "info", "Same gates.",
        )

    # Routing onto hardware connectivity
    sec = page.section(
        "On hardware connectivity",
        "Real devices only couple some qubit pairs. Each circuit is routed (optimisation level 3) onto a line "
        "of n qubits and onto the smallest IBM heavy-hex lattice that fits; SWAPs show up as extra CX.",
    )
    for metric, label in (("cx", "CX gates"), ("depth", "Depth")):
        panels = {t: {m: by_method(rows, m, lambda r: getattr(r, metric)[t]) for m in methods} for t in TOPOLOGIES}
        sec.figure(charts.small_multiples(ns, panels, colors=charts.METHOD_COLORS, x_title="qubits n",
                                          y_title=label),
                   f"{label} after routing, one panel per connectivity on a shared scale. Where only one "
                   "line is visible the two constructions coincide.")
    if both:
        lin, tree = table[(n_max, "linear")], table[(n_max, "tree")]
        sec.callout(
            f"At n = {n_max} on a line, linear needs {lin.cx['line']} CX (no SWAPs) while tree needs "
            f"{tree.cx['line']}. The depth advantage of the tree ({tree.depth['all-to-all']} vs "
            f"{lin.depth['all-to-all']} with any pair coupled) shrinks to {tree.depth['line']} vs "
            f"{lin.depth['line']} once routed. Choose the construction for the device you run on.",
            "info", "Trade-off.",
        )

    # Noise
    if noisy_ns:
        sec = page.section(
            "Under noise",
            f"Each routed circuit is sampled ({args.shots} shots) with depolarizing noise: CX error {args.noise:g}, "
            f"single-qubit error {args.noise / 10:g}. An ideal W state always yields exactly one '1', so the share "
            "of such outcomes shows how much of the state stayed in the one-hot subspace.",
        )
        noisy_rows = [r for r in rows if r.one_hot_rate]
        panels = {t: {m: by_method(noisy_rows, m, lambda r: r.one_hot_rate[t]) for m in methods}
                  for t in NOISY_TOPOLOGIES}
        fig = charts.small_multiples(noisy_ns, panels, colors=charts.METHOD_COLORS, x_title="qubits n",
                                     y_title="one-hot outcomes", hover_format=".1%")
        fig.update_yaxes(tickformat=".0%")
        sec.figure(fig, "With gate-only noise the loss mostly follows the gate count: close with any pair "
                        "coupled, clearly worse for the tree on a line because of the SWAPs.")

        # Measured distribution of the largest noisy size on a line
        k = args.detail_n if args.detail_n and args.detail_n <= args.noise_max_n else max(noisy_ns)
        sim = AerSimulator(noise_model=noise_model(args.noise))
        dist = {}
        for m in methods:
            counts = sample(route(w_state(k, m), "line", args.seed), k, sim, args.shots, args.seed)
            total = sum(counts.values())
            values = [counts.get(format(1 << j, f"0{k}b"), 0) / total for j in range(k)]
            values.append(1 - sum(values))
            dist[m] = values
        fig = charts.bars([f"q{j}" for j in range(k)] + ["not one-hot"], dist, colors=charts.METHOD_COLORS,
                          x_title="qubit that read 1", y_title="probability", hover_format=".1%",
                          reference=1 / k, reference_label=f"ideal 1/{k}")
        fig.update_yaxes(tickformat=".0%")
        sec.figure(fig, f"Measured distribution for n = {k} on a line with noise. The last bar collects every "
                        "outcome with zero or several excitations.")

    # Full table
    sec = page.section("All results", "Click a column header to sort.")
    columns = ["n", "method", "fidelity"]
    columns += [f"depth · {t}" for t in TOPOLOGIES] + [f"CX · {t}" for t in TOPOLOGIES]
    columns += [f"one-hot · {t}" for t in NOISY_TOPOLOGIES]
    body = []
    for r in rows:
        line = [r.n, r.method, r.fidelity if r.fidelity is not None else HTML("—")]
        line += [r.depth[t] for t in TOPOLOGIES] + [r.cx[t] for t in TOPOLOGIES]
        line += [r.one_hot_rate.get(t, HTML("—")) for t in NOISY_TOPOLOGIES]
        body.append(line)
    sec.table(columns, body, numeric=[0, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    page.reproduce({"n": ", ".join(map(str, ns)), "methods": ", ".join(methods), "shots": args.shots,
                    "CX error": args.noise, "noise up to n": args.noise_max_n, "seed": args.seed})
    path = page.save()

    return path


def main() -> None:
    """
    Function that parses the arguments, runs the experiment and writes the report

    Returns
    -------
    None
    """
    # Read the command-line arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=parse_ns, default=parse_ns("1-16"),
                        help="qubit counts, e.g. '2-16' or '3,5,8' (default: 1-16)")
    parser.add_argument("--method", choices=["both", *METHODS], default="both")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--noise", type=float, default=0.01, help="CX depolarizing error (default: 0.01)")
    parser.add_argument("--noise-max-n", type=int, default=12, help="largest n sampled with noise (default: 12)")
    parser.add_argument("--detail-n", type=int, default=None, help="n for heatmaps, circuits and distribution")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    args.methods = list(METHODS) if args.method == "both" else [args.method]

    # Measure, then write the report
    rows = run(args)
    path = build_report(args, rows)
    print(f"\nReport: {path}\nIndex:  {args.out / 'index.html'}")


if __name__ == "__main__":
    main()
