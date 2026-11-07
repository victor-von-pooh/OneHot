"""
Use case: superpose every "exactly one per row" assignment with one W state per row

Many combinatorial problems give each of k items exactly one of m choices
(tasks to machines, vertices to colours, cities to time slots). With one-hot
encoding that is a k x m grid of qubits where every row holds exactly one '1'.
Putting a W_m state on each row prepares the uniform superposition of all m^k
valid assignments at once, so every shot is a valid assignment, while a
Hadamard on every qubit almost never is. The script writes
outputs/usecases/one_hot_encoding/index.html.

Examples
--------
uv run python usecases/one_hot_encoding/main.py
uv run python usecases/one_hot_encoding/main.py --rows 4 --cols 3
uv run python usecases/one_hot_encoding/main.py --rows 3 --cols 5 --method linear
"""

import math
import sys
from itertools import product
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from qiskit import QuantumCircuit, transpile
from scipy.stats import chisquare

from onehot import w_state
from onehot.report import HTML, Page, Stat, charts

# Make the repository root importable so that usecases.shared resolves when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from usecases.shared.common import base_parser, one_hot_index, sample  # noqa: E402


BASIS = ["cx", "rz", "sx", "x"]


def row_encoding(rows: int, cols: int, method: str, boxed: bool = False) -> QuantumCircuit:
    """
    Function that puts one W state on every row of a rows x cols qubit grid

    Qubit r * cols + j means "item r takes choice j".

    Parameters
    ----------
    rows: int
        Number of items k
    cols: int
        Number of choices m per item
    method: str
        W-state construction
    boxed: bool = False
        Draw each W state as one labelled box instead of its gates

    Returns
    -------
    qc: QuantumCircuit
        Circuit on rows * cols qubits
    """
    qc = QuantumCircuit(rows * cols)

    # The rows act on disjoint qubits, so all W states run in parallel
    for r in range(rows):
        row = w_state(cols, method)
        qubits = [r * cols + j for j in range(cols)]
        if boxed:
            qc.append(row.to_gate(label=f"W (item {r})"), qubits)
        else:
            qc.compose(row, qubits=qubits, inplace=True)

    return qc


def hadamard_encoding(rows: int, cols: int) -> QuantumCircuit:
    """
    Function that puts a Hadamard on every qubit, the usual unconstrained start

    Parameters
    ----------
    rows: int
        Number of items k
    cols: int
        Number of choices m per item

    Returns
    -------
    qc: QuantumCircuit
        Circuit preparing the uniform superposition of all 2^(k m) bitstrings
    """
    # Every bitstring, valid or not, gets the same amplitude
    qc = QuantumCircuit(rows * cols)
    qc.h(range(rows * cols))

    return qc


def decode(bits: str, rows: int, cols: int) -> tuple[int, ...] | None:
    """
    Function that turns a measured bitstring into an assignment

    Parameters
    ----------
    bits: str
        Bitstring with qubit 0 on the right (Qiskit order)
    rows: int
        Number of items k
    cols: int
        Number of choices m per item

    Returns
    -------
    assignment: tuple[int, ...] | None
        Choice of each item, or None when some row is not one-hot
    """
    # Cut the string into rows (qubit order) and read the '1' of each row
    clean = bits.replace(" ", "")[::-1]
    choices = []
    for r in range(rows):
        row = clean[r * cols:(r + 1) * cols]
        index = one_hot_index(row[::-1])
        if index is None:
            choices = None
            break
        choices.append(index)
    assignment = tuple(choices) if choices is not None else None

    return assignment


def summarise(counts: dict[str, int], rows: int, cols: int) -> tuple[dict[tuple[int, ...], int], float]:
    """
    Function that decodes every outcome and returns the valid share

    Parameters
    ----------
    counts: dict[str, int]
        Bitstring -> count
    rows: int
        Number of items k
    cols: int
        Number of choices m per item

    Returns
    -------
    assignments: dict[tuple[int, ...], int]
        Valid assignment -> count
    valid: float
        Share of shots that decoded to a valid assignment
    """
    # Keep only outcomes where every row is one-hot
    assignments: dict[tuple[int, ...], int] = {}
    for bits, c in counts.items():
        a = decode(bits, rows, cols)
        if a is not None:
            assignments[a] = assignments.get(a, 0) + c
    valid = sum(assignments.values()) / sum(counts.values())

    return assignments, valid


def injective_share(rows: int, cols: int) -> float:
    """
    Function that returns the share of assignments in which no choice repeats

    Parameters
    ----------
    rows: int
        Number of items k
    cols: int
        Number of choices m per item

    Returns
    -------
    share: float
        m! / (m - k)! / m^k, or 0 when k > m
    """
    # Count injective maps among all m^k maps
    share = math.perm(cols, rows) / cols**rows if rows <= cols else 0.0

    return share


def main() -> None:
    """
    Function that samples both encodings and writes the report

    Returns
    -------
    None
    """
    # Read the command-line arguments
    parser = base_parser(__doc__)
    parser.add_argument("--rows", type=int, default=3, help="items k, one W state each (default: 3)")
    parser.add_argument("--cols", type=int, default=3, help="choices m per item (default: 3)")
    parser.add_argument("--method", choices=["tree", "linear"], default="tree", help="W-state construction")
    parser.add_argument("--grid-rows", type=int, default=6, help="largest k in the comparison grid (default: 6)")
    parser.add_argument("--grid-cols", type=int, default=8, help="largest m in the comparison grid (default: 8)")
    args = parser.parse_args()
    k, m = args.rows, args.cols
    if k < 1 or m < 2 or k * m > 24:
        parser.error("need rows >= 1, cols >= 2 and rows * cols <= 24")
    space = m**k

    # Sample the W encoding ideally and with noise, and the Hadamard start ideally
    circuit = row_encoding(k, m, args.method)
    ideal, ideal_valid = summarise(sample(circuit, args.shots, args.seed), k, m)
    noisy, noisy_valid = ({}, None)
    if args.noise > 0:
        noisy, noisy_valid = summarise(sample(circuit, args.shots, args.seed, args.noise), k, m)
    _, hadamard_valid = summarise(sample(hadamard_encoding(k, m), args.shots, args.seed), k, m)
    hadamard_theory = (m / 2**m) ** k
    print(f"valid: W={ideal_valid:.4f} noisy={noisy_valid} H={hadamard_valid:.4f} (theory {hadamard_theory:.4g})")

    # Uniformity of the sampled assignments
    all_assignments = list(product(range(m), repeat=k))
    observed = np.array([ideal.get(a, 0) for a in all_assignments])
    p_value = float(chisquare(observed).pvalue) if space > 1 else 1.0
    seen = int((observed > 0).sum())
    injective_sampled = sum(c for a, c in ideal.items() if len(set(a)) == len(a)) / max(1, sum(ideal.values()))

    # Circuit cost compared with a single row
    compiled = transpile(circuit, basis_gates=BASIS, optimization_level=1, seed_transpiler=args.seed)
    single = transpile(w_state(m, args.method), basis_gates=BASIS, optimization_level=1, seed_transpiler=args.seed)
    cx, depth = compiled.count_ops().get("cx", 0), compiled.depth()
    single_depth = single.depth()

    # Page header and headline numbers
    page = Page(
        "One-hot encoding of assignments", category="usecases", slug="one_hot_encoding", out_root=args.out,
        summary=f"One W state per row prepares all {space:,} valid assignments of {k} items to {m} choices "
        "in superposition; every shot is valid, unlike a Hadamard start.",
        tags=["one-hot", "constraints", "initial state", "assignment"],
    )
    page.lead(
        rf"Give each of \(k = {k}\) items exactly one of \(m = {m}\) choices. In one-hot form that is a "
        rf"\({k} \times {m}\) grid of qubits in which every row holds a single '1'. A W state per row prepares "
        rf"\(\bigotimes_{{r}} |W_{m}\rangle\), the uniform superposition of all \(m^k = {space:,}\) valid "
        rf"assignments, using \(k \cdot m = {k * m}\) qubits."
    )
    hero = [
        Stat("Valid shots, W rows", f"{ideal_valid:.1%}", "ideal simulator", "good" if ideal_valid == 1 else "bad"),
        Stat("Valid shots, Hadamard", f"{hadamard_valid:.2%}", f"theory (m/2^m)^k = {hadamard_theory:.2%}"),
        Stat("Assignments seen", f"{seen:,} / {space:,}", f"in {args.shots:,} shots, χ² p = {p_value:.2f}",
             "good" if seen == space else "warn"),
    ]
    if noisy_valid is not None:
        hero.append(Stat(f"Valid shots, CX error {args.noise:g}", f"{noisy_valid:.1%}",
                         "invalid shots are detectable and can be discarded"))
    page.headline(hero)

    # The circuit
    sec = page.section(
        "The circuit",
        rf"Qubit \(r \cdot m + j\) means “item \(r\) takes choice \(j\)”. The rows share no qubits, so the "
        rf"whole grid is as deep as one row: depth {depth} here vs {single_depth} for a single \(W_{m}\).",
    )
    sec.circuit(row_encoding(k, m, args.method, boxed=True), "One W box per item.")
    sec.circuit(circuit, f"The same circuit expanded into gates: {cx} CX in total.")

    # Sampled assignments
    sec = page.section(
        "Sampled assignments",
        f"{args.shots:,} shots decoded row by row. Every valid assignment should appear with probability "
        f"1/{space:,}.",
    )
    if space <= 125:
        labels = ["-".join(map(str, a)) for a in all_assignments]
        series = {"ideal": [ideal.get(a, 0) / args.shots for a in all_assignments]}
        if noisy:
            series[f"CX error {args.noise:g}"] = [noisy.get(a, 0) / args.shots for a in all_assignments]
        fig = charts.bars(labels, series, colors={"ideal": charts.SERIES[0],
                                                  f"CX error {args.noise:g}": charts.SERIES[2]},
                          x_title="assignment (choice of item 0-1-…)", y_title="probability",
                          hover_format=".2%", reference=1 / space, reference_label=f"1/{space}")
        fig.update_yaxes(tickformat=".1%")
        fig.update_xaxes(tickangle=-60 if space > 12 else 0)
        sec.figure(fig, f"Label a-b-c means item 0 → choice a, item 1 → choice b, and so on. χ² test against "
                        f"uniform: p = {p_value:.2f} ({'consistent with' if p_value > 0.01 else 'unlikely under'} "
                        "a uniform distribution). With noise the bars shrink because some shots become invalid.")
    else:
        sec.callout(f"{space:,} assignments are too many to draw; see the table below.", "info")
    rows_table = [["-".join(map(str, a)), ", ".join(f"item {i} → {c}" for i, c in enumerate(a)), n_a,
                   n_a / args.shots, len(set(a)) == len(a)]
                  for a, n_a in sorted(ideal.items(), key=lambda kv: -kv[1])]
    sec.table(["assignment", "meaning", "shots", "probability", "all choices distinct"], rows_table,
              numeric=[2, 3], caption="Ideal run. Click a header to sort.")
    sec.code(
        "bits = '...'                       # measured string, qubit 0 on the right\n"
        "q = bits[::-1]                     # qubit order\n"
        f"rows = [q[r*{m}:(r+1)*{m}] for r in range({k})]\n"
        "assignment = [row.index('1') for row in rows]   # valid iff every row has one '1'",
        "Decoding a shot.",
    )

    # Comparison with a Hadamard start
    sec = page.section(
        "Compared with a Hadamard start",
        r"A Hadamard on every qubit spreads the amplitude over all \(2^{km}\) bitstrings, of which only "
        r"\(m^k\) are valid, so the valid share is \((m/2^m)^k\). The W rows put all of it on valid ones.",
    )
    ks = list(range(1, args.grid_rows + 1))
    ms = list(range(2, args.grid_cols + 1))
    z = [[math.log10(2**mm / mm) * kk for mm in ms] for kk in ks]
    fig = charts.heatmap(z, x=ms, y=ks, x_title="choices m", y_title="items k", z_title="log₁₀(shots per valid)",
                         hover_format=".1f", height=380)
    fig.update_traces(customdata=[[1 / (mm / 2**mm) ** kk for mm in ms] for kk in ks],
                      hovertemplate="k = %{y}, m = %{x}<br>Hadamard needs ~%{customdata:,.0f} shots per valid "
                                    "assignment<extra></extra>")
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(dtick=1, autorange="reversed")
    sec.figure(fig, "Shots a Hadamard start needs, on average, for one valid assignment (log scale). "
                    "The W encoding needs exactly one at every grid point.")
    sizes = list(range(1, args.grid_rows + 1))
    fig = charts.lines(sizes, {"valid assignments m^k": [m**kk for kk in sizes],
                               "all bitstrings 2^(km)": [2 ** (kk * m) for kk in sizes]},
                       colors={"valid assignments m^k": charts.SERIES[0], "all bitstrings 2^(km)": charts.SERIES[1]},
                       x_title=f"items k (m = {m})", y_title="number of states", log_y=True, hover_format=",.0f")
    sec.figure(fig, "The valid space is a vanishing fraction of the Hilbert space, which is why starting inside it "
                    "matters for search and optimisation.")
    sec.table(["start", "qubits", "space it covers", "valid share (ideal)", "valid share (theory)"],
              [["W per row", k * m, f"{space:,}", ideal_valid, 1.0],
               ["Hadamard", k * m, f"{2 ** (k * m):,}", hadamard_valid, hadamard_theory]],
              numeric=[1, 3, 4], max_height=None)

    # Limits
    sec = page.section(
        "What it does not enforce",
        "Each row is constrained, the columns are not: two items may take the same choice.",
    )
    sec.callout(
        f"Only {injective_share(k, m):.1%} of the {space:,} assignments use distinct choices for every item "
        f"(sampled: {injective_sampled:.1%}). Problems that need a permutation, such as the travelling salesman "
        "with one city per time slot, still need a penalty or a mixer for the column constraint.",
        "warn", "Rows only.",
    )
    if noisy_valid is not None:
        sec.callout(
            f"Noise can break a row: with CX error {args.noise:g}, {1 - noisy_valid:.1%} of the shots had a row with "
            "zero or several '1's. Those shots are recognisable and can be discarded before decoding.",
            "info", "Errors are detectable.",
        )

    page.reproduce({"rows (items)": k, "cols (choices)": m, "method": args.method, "shots": args.shots,
                    "CX error": args.noise, "grid": f"k ≤ {args.grid_rows}, m ≤ {args.grid_cols}",
                    "seed": args.seed})
    path = page.save()
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
