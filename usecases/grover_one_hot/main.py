"""
Use case: Grover search over n items encoded as one-hot states

Amplitude amplification only needs a circuit A that prepares a superposition
over the candidates. Using A = W_n restricts the search space to the n one-hot
states instead of all 2^n bitstrings, and the oracle becomes a single Z gate on
each marked qubit. This script compares that search with standard Grover over
2^n states on the same qubits and writes outputs/usecases/grover_one_hot/index.html.

Examples
--------
uv run python usecases/grover_one_hot/main.py
uv run python usecases/grover_one_hot/main.py --items 12 --marked 3,5
uv run python usecases/grover_one_hot/main.py --items 6 --marked 0 --method linear
"""

import math
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ZGate
from qiskit.quantum_info import Statevector

from onehot import w_state
from onehot.report import HTML, Page, Stat, charts

# Make the repository root importable so that usecases.shared resolves when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from usecases.shared.common import base_parser, one_hot_index, sample  # noqa: E402


BASIS = ["cx", "rz", "sx", "x"]


def parse_marked(text: str) -> list[int]:
    """
    Function that parses the marked items such as "3" or "3,5"

    Parameters
    ----------
    text: str
        Comma-separated item indices

    Returns
    -------
    marked: list[int]
        Sorted, de-duplicated indices
    """
    # Split on commas and drop duplicates
    marked = sorted({int(v) for v in text.split(",") if v.strip()})

    return marked


def reflect_zero(n: int) -> QuantumCircuit:
    """
    Function that builds S0 = I - 2|0...0><0...0|

    Parameters
    ----------
    n: int
        Number of qubits

    Returns
    -------
    qc: QuantumCircuit
        Circuit that flips the sign of |0...0> only
    """
    # Map |0...0> to |1...1>, flip its sign with a multi-controlled Z, map back
    qc = QuantumCircuit(n, name="S0")
    qc.x(range(n))
    if n == 1:
        qc.z(0)
    else:
        qc.append(ZGate().control(n - 1), range(n))
    qc.x(range(n))

    return qc


def one_hot_oracle(n: int, marked: list[int]) -> QuantumCircuit:
    """
    Function that builds the oracle used inside the one-hot subspace

    On a one-hot state exactly one qubit is |1>, so a Z on each marked qubit
    flips the sign of exactly the marked one-hot states.

    Parameters
    ----------
    n: int
        Number of qubits (= number of items)
    marked: list[int]
        Indices of the marked items

    Returns
    -------
    qc: QuantumCircuit
        Circuit with one Z per marked qubit
    """
    # One single-qubit Z per marked item
    qc = QuantumCircuit(n, name="Oracle")
    for m in marked:
        qc.z(m)

    return qc


def full_oracle(n: int, marked: list[int]) -> QuantumCircuit:
    """
    Function that builds an oracle that flips the marked one-hot bitstrings among all 2^n

    Parameters
    ----------
    n: int
        Number of qubits
    marked: list[int]
        Indices of the marked items

    Returns
    -------
    qc: QuantumCircuit
        Circuit that flips the sign of each bitstring with a single 1 at a marked position
    """
    qc = QuantumCircuit(n, name="Oracle")
    for m in marked:
        # Turn the pattern 0..010..0 into 1...1, flip it, and undo the mapping
        others = [q for q in range(n) if q != m]
        if others:
            qc.x(others)
        if n == 1:
            qc.z(0)
        else:
            qc.append(ZGate().control(n - 1), range(n))
        if others:
            qc.x(others)

    return qc


def grover_circuit(prep: QuantumCircuit, oracle: QuantumCircuit, iterations: int, label: str) -> QuantumCircuit:
    """
    Function that builds A followed by `iterations` rounds of A S0 A^dagger O

    Parameters
    ----------
    prep: QuantumCircuit
        State preparation A
    oracle: QuantumCircuit
        Phase oracle O
    iterations: int
        Number of Grover iterations
    label: str
        Label drawn on the A box

    Returns
    -------
    qc: QuantumCircuit
        Full search circuit without measurements
    """
    n = prep.num_qubits
    prep_gate = prep.to_gate(label=label)
    prep_inv = prep.inverse().to_gate(label=f"{label}†")
    oracle_gate = oracle.to_gate(label="Oracle")
    s0_gate = reflect_zero(n).to_gate(label="S₀")

    # Prepare the superposition over the candidates
    qc = QuantumCircuit(n)
    qc.append(prep_gate, range(n))

    # Each iteration: mark with the oracle, then reflect about the prepared state
    for _ in range(iterations):
        qc.barrier()
        qc.append(oracle_gate, range(n))
        qc.append(prep_inv, range(n))
        qc.append(s0_gate, range(n))
        qc.append(prep_gate, range(n))

    return qc


def success_curve(prep: QuantumCircuit, oracle: QuantumCircuit, n: int, marked: list[int],
                  max_iterations: int) -> list[float]:
    """
    Function that simulates the success probability after 0, 1, ..., max_iterations rounds

    Parameters
    ----------
    prep: QuantumCircuit
        State preparation A
    oracle: QuantumCircuit
        Phase oracle O
    n: int
        Number of qubits
    marked: list[int]
        Indices of the marked items
    max_iterations: int
        Largest number of iterations

    Returns
    -------
    probs: list[float]
        Probability of reading a marked one-hot state after each number of rounds
    """
    # Build one Grover round as a reusable circuit
    round_qc = QuantumCircuit(n)
    round_qc.compose(oracle, inplace=True)
    round_qc.compose(prep.inverse(), inplace=True)
    round_qc.compose(reflect_zero(n), inplace=True)
    round_qc.compose(prep, inplace=True)

    # Evolve round by round and read the marked amplitudes
    targets = [1 << m for m in marked]
    state = Statevector(prep)
    probs = [float(state.probabilities()[targets].sum())]
    for _ in range(max_iterations):
        state = state.evolve(round_qc)
        probs.append(float(state.probabilities()[targets].sum()))

    return probs


def theory(num_marked: int, space: int, iterations: int) -> tuple[float, int, list[float]]:
    """
    Function that returns the textbook Grover curve sin^2((2k + 1) theta)

    Parameters
    ----------
    num_marked: int
        Number of marked states
    space: int
        Size of the search space
    iterations: int
        Largest number of iterations

    Returns
    -------
    theta: float
        Rotation angle with sin(theta) = sqrt(num_marked / space)
    best: int
        Number of iterations that maximises the success probability
    curve: list[float]
        Success probability after 0, 1, ..., iterations rounds
    """
    # One Grover round rotates by 2 theta in the plane of marked and unmarked states
    theta = math.asin(math.sqrt(num_marked / space))
    curve = [math.sin((2 * k + 1) * theta) ** 2 for k in range(iterations + 1)]

    # The best round count is the first one that reaches a quarter turn, (2k + 1) theta = pi / 2
    target = math.pi / (4 * theta) - 0.5
    candidates = {max(0, math.floor(target)), max(0, math.ceil(target))}
    best = max(candidates, key=lambda k: math.sin((2 * k + 1) * theta) ** 2)

    return theta, best, curve


def cx_count(qc: QuantumCircuit, seed: int) -> tuple[int, int]:
    """
    Function that returns the CX count and depth after decomposing into basis gates

    Parameters
    ----------
    qc: QuantumCircuit
        Circuit to measure
    seed: int
        Seed of the transpiler

    Returns
    -------
    cx: int
        Number of CX gates
    depth: int
        Circuit depth
    """
    # Decompose with light optimisation, as a device run would
    compiled = transpile(qc, basis_gates=BASIS, optimization_level=1, seed_transpiler=seed)
    cx = compiled.count_ops().get("cx", 0)
    depth = compiled.depth()

    return cx, depth


def main() -> None:
    """
    Function that runs both searches and writes the report

    Returns
    -------
    None
    """
    # Read the command-line arguments
    parser = base_parser(__doc__)
    parser.add_argument("--items", type=int, default=8, help="number of items n = qubits (default: 8)")
    parser.add_argument("--marked", type=parse_marked, default=parse_marked("3"),
                        help="marked item indices, e.g. '3' or '3,5' (default: 3)")
    parser.add_argument("--method", choices=["tree", "linear"], default="tree", help="W-state construction")

    parser.add_argument("--noisy-max-cx", type=int, default=6000,
                        help="skip the noisy runs when a circuit has more CX than this (default: 6000)")

    # Both searches contain deep multi-controlled Z gates, so default to a milder noise level
    parser.set_defaults(noise=0.001)
    args = parser.parse_args()
    n, marked = args.items, args.marked
    if n < 2 or not marked or min(marked) < 0 or max(marked) >= n or len(marked) >= n:
        parser.error("need 2 <= items and 1 <= len(marked) < items with indices in [0, items)")

    # Theory for both search spaces: n one-hot states vs all 2^n bitstrings
    m = len(marked)
    _, best_w, _ = theory(m, n, 0)
    _, best_h, _ = theory(m, 2**n, 0)

    # Show each curve past its first peak so the over-rotation is visible
    span_w = max(2 * best_w + 2, 4)
    span_h = max(2 * best_h + 2, span_w)
    theta_w, _, theory_w = theory(m, n, span_w)
    theta_h, _, theory_h = theory(m, 2**n, span_h)

    # Simulate both searches round by round
    prep_w, prep_h = w_state(n, args.method), QuantumCircuit(n)
    prep_h.h(range(n))
    oracle_w, oracle_h = one_hot_oracle(n, marked), full_oracle(n, marked)
    sim_w = success_curve(prep_w, oracle_w, n, marked, span_w)
    sim_h = success_curve(prep_h, oracle_h, n, marked, span_h)
    print(f"W search: best k={best_w} P={sim_w[best_w]:.4f}; standard: best k={best_h} P={sim_h[best_h]:.4f}")

    # Build the circuits at their best round counts and measure their cost
    circ_w = grover_circuit(prep_w, oracle_w, best_w, "W")
    circ_h = grover_circuit(prep_h, oracle_h, best_h, "H")
    cx_w, depth_w = cx_count(circ_w, args.seed)
    cx_h, depth_h = cx_count(circ_h, args.seed)

    # Sample both circuits ideally and, if requested, with gate noise
    # Noisy trajectories get slow for very deep circuits, so cap them by CX count
    runs = {"ideal": 0.0}
    noisy_skipped = args.noise > 0 and max(cx_w, cx_h) > args.noisy_max_cx
    if args.noise > 0 and not noisy_skipped:
        runs["noisy"] = args.noise
    hits: dict[tuple[str, str], float] = {}
    histograms: dict[str, list[float]] = {}
    for run, noise in runs.items():
        for name, circ in (("W search", circ_w), ("standard", circ_h)):
            counts = sample(circ, args.shots, args.seed, noise)
            total = sum(counts.values())
            per_item = [0.0] * (n + 1)
            for bits, c in counts.items():
                idx = one_hot_index(bits)
                per_item[n if idx is None else idx] += c / total
            hits[(run, name)] = sum(per_item[k] for k in marked)
            histograms[f"{name} · {run}"] = per_item

    # Page header and headline numbers
    page = Page(
        "Grover search over one-hot states", category="usecases", slug="grover_one_hot", out_root=args.out,
        summary="Amplitude amplification with the W state as the starting superposition: searching n items "
        "needs about √n rounds instead of √(2ⁿ).",
        tags=["grover", "amplitude amplification", "oracle"],
    )
    page.lead(
        r"Grover's algorithm rotates a starting superposition \(A|0\rangle\) towards the marked states. "
        r"With \(A = W_n\) the candidates are the \(n\) one-hot states, so the success angle obeys "
        rf"\(\sin\theta = \sqrt{{m/n}}\) instead of \(\sqrt{{m/2^n}}\). The oracle is just a \(Z\) gate on each "
        r"marked qubit, because in a one-hot state exactly one qubit is \(|1\rangle\)."
    )
    page.headline([
        Stat("Rounds needed", f"{best_w} vs {best_h}", f"W search vs standard Grover, n = {n}"),
        Stat("Success, ideal", f"{sim_w[best_w]:.1%}", f"W search after {best_w} round(s)",
             "good" if sim_w[best_w] > 0.5 else "warn"),
        Stat("CX gates", f"{cx_w:,} vs {cx_h:,}", "whole circuit, W search vs standard"),
        *([Stat(f"Success, CX error {args.noise:g}",
                f"{hits[('noisy', 'W search')]:.1%} vs {hits[('noisy', 'standard')]:.1%}",
                "W search vs standard",
                "good" if hits[("noisy", "W search")] > max(hits[("noisy", "standard")], 0.5) else "warn")]
          if "noisy" in runs else []),
    ])

    # The circuit
    sec = page.section(
        "The circuit",
        rf"One round is: oracle, then the reflection \(A S_0 A^\dagger\) about the starting state. "
        rf"Searching for item(s) {', '.join(map(str, marked))} among {n}, the W search uses {best_w} round(s).",
    )
    sec.circuit(circ_w, f"W search with {best_w} round(s). W and W† are w_state({n}) and its inverse; "
                        "S₀ flips the sign of |0…0⟩.")
    sec.circuits([(oracle_w, "One-hot oracle: one Z per marked qubit"),
                  (full_oracle(n, marked[:1]), "Standard oracle for one marked item: X gates + multi-controlled Z")])
    sec.callout(
        f"Inside the one-hot subspace the oracle costs {len(marked)} single-qubit Z gate(s). Over all 2ⁿ bitstrings "
        f"the same marking needs a multi-controlled Z on all {n} qubits per marked item.",
        "info", "Cheaper oracle.",
    )

    # Success probability vs rounds
    sec = page.section(
        "Success probability per round",
        "Simulated state-vector probability of reading a marked item after k rounds, against the textbook "
        r"curve \(\sin^2((2k+1)\theta)\).",
    )
    fig = go.Figure()
    for name, sim, curve, color in (("W search", sim_w, theory_w, charts.SERIES[0]),
                                    ("standard", sim_h, theory_h, charts.SERIES[1])):
        fig.add_trace(go.Scatter(x=list(range(len(sim))), y=sim, name=f"{name} · simulated", mode="lines+markers",
                                 line=dict(color=color), marker=dict(color=color),
                                 hovertemplate="%{y:.1%}<extra>" + name + " simulated</extra>"))
        fig.add_trace(go.Scatter(x=list(range(len(curve))), y=curve, name=f"{name} · theory", mode="lines",
                                 line=dict(color=color, dash="dot"),
                                 hovertemplate="%{y:.1%}<extra>" + name + " theory</extra>"))
    charts.style(fig, x_title="Grover rounds k", y_title="P(marked)", hovermode="x unified", legend=True)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
    sec.figure(fig, f"θ = {theta_w:.3f} rad for the W search vs {theta_h:.4f} rad for the standard one. "
                    "Simulation and theory overlap: the W search behaves like Grover on an n-element space. "
                    "Past the peak the state over-rotates and the success probability falls again.")
    gap = max(abs(a - b) for a, b in zip(sim_w, theory_w))
    sec.callout(f"Largest deviation between simulation and theory for the W search: {gap:.1e}.",
                "good" if gap < 1e-9 else "warn")

    # Sampled outcomes
    sec = page.section(
        "Sampled outcomes",
        f"{args.shots} shots of each search at its best round count. The last bar collects outcomes that are "
        "not one-hot (possible for the standard search, and for either search under noise).",
    )
    cats = [f"item {k}" + (" ★" if k in marked else "") for k in range(n)] + ["not one-hot"]
    colors = {"W search · ideal": charts.SERIES[0], "standard · ideal": charts.SERIES[1],
              "W search · noisy": charts.SERIES[2], "standard · noisy": charts.SERIES[3]}
    fig = charts.bars(cats, histograms, colors=colors, x_title="outcome", y_title="probability",
                      hover_format=".1%")
    fig.update_yaxes(tickformat=".0%")
    sec.figure(fig, "★ marks the searched item(s). Click legend entries to compare runs one at a time.")
    if noisy_skipped:
        sec.callout(f"Noisy runs were skipped: the standard search has {cx_h:,} CX, above --noisy-max-cx "
                    f"({args.noisy_max_cx:,}).", "info")
    if "noisy" in runs:
        hit_w, hit_h = hits[("noisy", "W search")], hits[("noisy", "standard")]
        sec.callout(
            f"With CX error {args.noise:g}, the W search finds a marked item {hit_w:.1%} of the time and the "
            f"standard search {hit_h:.1%}. The standard search needs {best_h} rounds and {cx_h:,} CX against "
            f"{best_w} rounds and {cx_w:,} CX; most of either cost is the multi-controlled Z inside S₀.",
            "good" if hit_w > max(hit_h, 0.5) else "warn", "Noise.",
        )

    # Cost comparison table
    sec = page.section("Cost comparison", "Both circuits decomposed into cx, rz, sx and x (optimisation level 1).")
    rows = [
        ["W search", n, f"{n}", best_w, sim_w[best_w], cx_w, depth_w, hits[("ideal", "W search")],
         hits.get(("noisy", "W search"), HTML("—"))],
        ["standard", n, f"2^{n} = {2**n:,}", best_h, sim_h[best_h], cx_h, depth_h, hits[("ideal", "standard")],
         hits.get(("noisy", "standard"), HTML("—"))],
    ]
    sec.table(["search", "qubits", "search space", "rounds", "P(success) theory", "CX", "depth",
               "hit rate ideal", "hit rate noisy"], rows, numeric=[1, 3, 4, 5, 6, 7, 8], max_height=None)
    sec.text(
        "The trade-off: the W search only covers n candidates with n qubits (one per item), while standard "
        "Grover covers 2ⁿ. It pays off when the problem is naturally one-hot, such as picking one option "
        "out of n, and the oracle can act qubit by qubit."
    )

    # Per-round table
    sec = page.section("Per-round values", "Click a column header to sort.")
    table = []
    for k in range(max(len(sim_w), len(sim_h))):
        table.append([k,
                      sim_w[k] if k < len(sim_w) else HTML("—"), theory_w[k] if k < len(theory_w) else HTML("—"),
                      sim_h[k] if k < len(sim_h) else HTML("—"), theory_h[k] if k < len(theory_h) else HTML("—")])
    sec.table(["round k", "W simulated", "W theory", "standard simulated", "standard theory"], table,
              numeric=[0, 1, 2, 3, 4])

    page.reproduce({"items": n, "marked": ", ".join(map(str, marked)), "method": args.method,
                    "shots": args.shots, "CX error": args.noise, "seed": args.seed})
    path = page.save()
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
