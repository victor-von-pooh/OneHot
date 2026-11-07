"""
Use case: W vs GHZ, how entanglement survives losing or measuring a qubit

The W state and the GHZ state are the two inequivalent kinds of genuine
three-qubit entanglement, and they generalise to n qubits. The GHZ state
(|0...0> + |1...1>) / sqrt(2) is "all or nothing": lose or measure one qubit
and what remains is not entangled at all. The W state spreads its entanglement
over every pair, so the rest stays entangled. This script quantifies that
with exact state-vector calculations and writes the results to
outputs/usecases/entanglement_robustness/index.html.

Examples
--------
uv run python usecases/entanglement_robustness/main.py
uv run python usecases/entanglement_robustness/main.py --qubits 6 --sweep 3-10
"""

import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, concurrence, entropy, negativity, partial_trace, state_fidelity

from onehot import w_state
from onehot.report import Page, Stat, charts

# Make the repository root importable so that usecases.shared resolves when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from usecases.shared.common import base_parser  # noqa: E402


W_COLOR = charts.SERIES[0]
GHZ_COLOR = charts.SERIES[1]
COLORS = {"W": W_COLOR, "GHZ": GHZ_COLOR}
TOLERANCE = 1e-9


def parse_range(text: str) -> list[int]:
    """
    Function that parses a range of qubit counts such as "3-8"

    Parameters
    ----------
    text: str
        Inclusive range "low-high" or comma-separated values

    Returns
    -------
    values: list[int]
        Sorted qubit counts (each >= 3)
    """
    # Expand ranges and single values
    found: set[int] = set()
    for part in text.split(","):
        if "-" in part:
            low, high = (int(v) for v in part.split("-"))
            found.update(range(low, high + 1))
        elif part.strip():
            found.add(int(part))
    values = sorted(v for v in found if v >= 3)

    return values


def ghz_state(n: int) -> QuantumCircuit:
    """
    Function that builds the n-qubit GHZ state (|0...0> + |1...1>) / sqrt(2)

    Parameters
    ----------
    n: int
        Number of qubits

    Returns
    -------
    qc: QuantumCircuit
        Circuit preparing the GHZ state from |0...0>
    """
    # Put qubit 0 in |+> and copy it down the chain with CNOTs
    qc = QuantumCircuit(n, name=f"GHZ_{n}")
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)

    return qc


def states(n: int) -> dict[str, Statevector]:
    """
    Function that prepares both states as exact state vectors

    Parameters
    ----------
    n: int
        Number of qubits

    Returns
    -------
    prepared: dict[str, Statevector]
        "W" and "GHZ" state vectors
    """
    prepared = {"W": Statevector(w_state(n)), "GHZ": Statevector(ghz_state(n))}

    return prepared


def pair_concurrence(state: Statevector, a: int, b: int) -> float:
    """
    Function that returns the entanglement between two qubits once all others are ignored

    Parameters
    ----------
    state: Statevector
        n-qubit state
    a: int
        First qubit
    b: int
        Second qubit

    Returns
    -------
    value: float
        Concurrence of the two-qubit reduced state (0 = not entangled, 1 = maximal)
    """
    # Trace out every other qubit and measure what is left between a and b
    others = [q for q in range(state.num_qubits) if q not in (a, b)]
    value = max(0.0, float(concurrence(partial_trace(state, others))))

    return value


def negativity_after_loss(state: Statevector, lost: int) -> float:
    """
    Function that returns the entanglement left after the last `lost` qubits are lost

    Parameters
    ----------
    state: Statevector
        n-qubit state
    lost: int
        Number of qubits traced out (the remaining state keeps at least 2 qubits)

    Returns
    -------
    value: float
        Negativity across the split "first half | second half" of the remaining qubits
    """
    # Trace out the lost qubits, then test the remaining ones across a balanced split
    n = state.num_qubits
    remaining = partial_trace(state, list(range(n - lost, n))) if lost else state
    half = list(range((n - lost) // 2))
    value = max(0.0, float(negativity(remaining, half)))

    return value


def measure_last(state: Statevector, outcome: int) -> tuple[float, Statevector | None]:
    """
    Function that measures the last qubit and returns the state of the others

    Parameters
    ----------
    state: Statevector
        n-qubit state
    outcome: int
        Measured value 0 or 1 of the last qubit

    Returns
    -------
    probability: float
        Probability of the outcome
    remaining: Statevector | None
        Normalised state of the first n - 1 qubits (None if the outcome is impossible)
    """
    # The last qubit is the most significant bit, so it selects one half of the vector
    n = state.num_qubits
    half = 2 ** (n - 1)
    branch = state.data[outcome * half:(outcome + 1) * half]
    probability = float(np.vdot(branch, branch).real)
    remaining = Statevector(branch / np.sqrt(probability)) if probability > TOLERANCE else None

    return probability, remaining


def is_entangled(state: Statevector) -> bool:
    """
    Function that tells whether a pure state is entangled (not a product of single qubits)

    Parameters
    ----------
    state: Statevector
        Pure multi-qubit state

    Returns
    -------
    entangled: bool
        True when some qubit is entangled with the rest
    """
    # A pure state is a full product state exactly when every single-qubit reduction is pure
    n = state.num_qubits
    entangled = any(
        entropy(partial_trace(state, [q for q in range(n) if q != k])) > 1e-7 for k in range(n)
    )

    return entangled


def analyse(n: int) -> dict:
    """
    Function that computes every robustness figure for one qubit count

    Parameters
    ----------
    n: int
        Number of qubits

    Returns
    -------
    row: dict
        Figures for both states, keyed by measure and state name
    """
    row: dict = {"n": n}
    for name, state in states(n).items():
        # Entanglement of one qubit with all the others (pure state, so entropy of its reduction)
        row[f"entropy_{name}"] = float(entropy(partial_trace(state, list(range(1, n)))))

        # Entanglement between a pair of qubits after ignoring the others
        row[f"concurrence_{name}"] = pair_concurrence(state, 0, 1)

        # Entanglement left among the others after one qubit is lost
        row[f"lost_{name}"] = negativity_after_loss(state, 1)

        # Probability that measuring one qubit leaves the others entangled
        kept = 0.0
        for outcome in (0, 1):
            probability, remaining = measure_last(state, outcome)
            if remaining is not None and is_entangled(remaining):
                kept += probability
        row[f"measured_{name}"] = kept

    return row


def main() -> None:
    """
    Function that compares the W and GHZ states and writes the report

    Returns
    -------
    None
    """
    # Read the command-line arguments
    parser = base_parser(__doc__)
    parser.add_argument("--qubits", type=int, default=5, help="qubit count for the detailed views (default: 5)")
    parser.add_argument("--sweep", type=parse_range, default=parse_range("3-8"),
                        help="qubit counts to compare (default: 3-8)")
    args = parser.parse_args()
    n = max(3, args.qubits)

    # Compute every figure over the sweep and for the detailed qubit count
    sweep = [analyse(k) for k in args.sweep]
    detail = analyse(n)
    prepared = states(n)
    for row in sweep:
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})

    page = Page(
        "Entanglement robustness", category="usecases", slug="entanglement_robustness", out_root=args.out,
        summary="Lose or measure one qubit: the GHZ state's entanglement vanishes, the W state's survives.",
        tags=["w_state", "GHZ", "entanglement", "quantum_info"],
    )
    page.lead(
        r"\(|W_n\rangle = \tfrac{1}{\sqrt n}(|10\cdots0\rangle + \cdots + |0\cdots01\rangle)\) and "
        r"\(|\mathrm{GHZ}_n\rangle = \tfrac{1}{\sqrt 2}(|0\cdots0\rangle + |1\cdots1\rangle)\) are both entangled "
        r"across all \(n\) qubits, but in different ways. GHZ ties every qubit to one shared bit: impressive while "
        r"intact, gone as soon as any qubit leaks away. W spreads its entanglement over every <em>pair</em> of "
        r"qubits, so the rest keeps it. All numbers below are exact state-vector calculations."
    )

    # Headline numbers for the detailed qubit count
    page.headline([
        Stat(f"Pair entanglement, n = {n}", f"{detail['concurrence_W']:.2f} vs {detail['concurrence_GHZ']:.2f}",
             f"concurrence of two qubits, W vs GHZ (theory 2/n = {2 / n:.2f})",
             "good" if detail["concurrence_W"] > TOLERANCE else "bad"),
        Stat("After losing a qubit", f"{detail['lost_W']:.3f} vs {detail['lost_GHZ']:.3f}",
             "negativity of the rest, W vs GHZ", "good" if detail["lost_W"] > TOLERANCE else "bad"),
        Stat("After measuring a qubit", f"{detail['measured_W']:.0%} vs {detail['measured_GHZ']:.0%}",
             "chance the rest is still entangled, W vs GHZ",
             "good" if detail["measured_W"] > TOLERANCE else "bad"),
        Stat("One qubit vs the rest", f"{detail['entropy_W']:.2f} vs {detail['entropy_GHZ']:.2f}",
             "entanglement entropy in bits, W vs GHZ"),
    ])

    # The two circuits
    sec = page.section(
        "The two states",
        "Both need n − 1 two-qubit operations from |0…0⟩. GHZ copies one superposed bit down a chain of CNOTs; "
        "W splits one excitation over all qubits.",
    )
    sec.circuits([(w_state(n), f"W state, n = {n}"), (ghz_state(n), f"GHZ state, n = {n}")])

    # Who is entangled with whom
    sec = page.section(
        "Who is entangled with whom",
        "Concurrence between every pair of qubits after ignoring all the others (0 = not entangled, "
        "1 = a maximally entangled pair).",
    )
    for name, state in prepared.items():
        matrix = [[0.0 if a == b else pair_concurrence(state, a, b) for b in range(n)] for a in range(n)]
        fig = charts.heatmap(matrix, x=[f"q{k}" for k in range(n)], y=[f"q{k}" for k in range(n)],
                             x_title="qubit", y_title="qubit", z_title="concurrence", zmax=1.0,
                             height=120 + 44 * n)
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(scaleanchor="y", constrain="domain")
        caption = (f"<b>W</b>: every pair shares concurrence 2/n = {2 / n:.3f}; the entanglement is spread evenly."
                   if name == "W" else
                   "<b>GHZ</b>: no pair is entangled on its own; the entanglement exists only among all n qubits "
                   "together.")
        sec.figure(fig, caption)
    xs = [row["n"] for row in sweep]
    fig = charts.lines(xs, {"2/n": [2 / k for k in xs],
                            "W": [row["concurrence_W"] for row in sweep],
                            "GHZ": [row["concurrence_GHZ"] for row in sweep]},
                       colors={**COLORS, "2/n": "#898781"}, dash={"2/n": "dot"}, x_title="qubits n",
                       y_title="pair concurrence", hover_format=".3f")
    sec.figure(fig, "Pair entanglement over n: W follows 2/n exactly; GHZ is always 0.")

    # Losing qubits
    sec = page.section(
        "Losing qubits",
        "A qubit that leaks into the environment is traced out. What entanglement is left among the remaining "
        "qubits? Measured by the negativity across a split of the remaining qubits into two halves "
        "(0 = no detectable entanglement).",
    )
    lost_range = list(range(0, n - 1))
    losses = {name: [negativity_after_loss(state, k) for k in lost_range] for name, state in prepared.items()}
    fig = charts.bars([str(k) for k in lost_range], losses, colors=COLORS, x_title="qubits lost",
                      y_title="negativity of the rest", hover_format=".3f")

    # Describe what the bars show from the computed values
    ghz_gone = all(v <= TOLERANCE for v in losses["GHZ"][1:])
    w_kept = all(v > TOLERANCE for v in losses["W"])
    caption = f"Starting from n = {n}. "
    caption += "GHZ drops to zero after the first loss. " if ghz_gone else "GHZ keeps some entanglement. "
    caption += ("W keeps entanglement until only two qubits remain, and those two are still entangled."
                if w_kept else "W loses its entanglement before only two qubits remain.")
    sec.figure(fig, caption)
    fig = charts.lines(xs, {"W": [row["lost_W"] for row in sweep], "GHZ": [row["lost_GHZ"] for row in sweep]},
                       colors=COLORS, x_title="qubits n", y_title="negativity after losing one qubit",
                       hover_format=".3f")
    sec.figure(fig, "Entanglement left after losing a single qubit, for every n in the sweep.")
    sec.callout(
        f"Losing one of {n} qubits leaves the GHZ state as a classical mixture of |0…0⟩ and |1…1⟩ "
        f"(negativity {detail['lost_GHZ']:.3f}), while the W state keeps negativity {detail['lost_W']:.3f}. "
        "If qubits may be lost, W is the safer resource.",
        "good", "Robust.",
    )

    # Measuring a qubit
    sec = page.section(
        "Measuring a qubit",
        f"Measure the last of {n} qubits and look at the other {n - 1}.",
    )
    rows = []
    for name, state in prepared.items():
        for outcome in (0, 1):
            probability, remaining = measure_last(state, outcome)
            if remaining is None:
                continue
            target = Statevector(w_state(n - 1))
            fidelity = state_fidelity(remaining, target) if name == "W" else None
            rows.append([name, outcome, probability, "entangled" if is_entangled(remaining) else "product",
                         f"{fidelity:.3f}" if fidelity is not None else "—"])
    sec.table(["state", "outcome", "probability", f"other {n - 1} qubits", f"fidelity with W_{n - 1}"], rows,
              numeric=[1, 2], max_height=None)
    fig = charts.lines(xs, {"W": [row["measured_W"] for row in sweep],
                            "GHZ": [row["measured_GHZ"] for row in sweep]},
                       colors=COLORS, x_title="qubits n", y_title="P(rest still entangled)", hover_format=".1%")
    fig.update_yaxes(tickformat=".0%", range=[-0.03, 1.03])
    sec.figure(fig, "Probability that the unmeasured qubits are still entangled. For W it is (n − 1)/n: only "
                    "the outcome 1, which pins the excitation, destroys it.")
    sec.callout(
        f"With probability {detail['measured_W']:.0%} the W state answers 0 and the others are left in exactly "
        f"W<sub>{n - 1}</sub>, ready to use again. Any measurement outcome on GHZ leaves a product state.",
        "info", "Reusable.",
    )

    # The price
    sec = page.section(
        "The price",
        "Robustness is not free: per qubit, GHZ is the more strongly entangled state while it is intact.",
    )
    fig = charts.lines(xs, {"W": [row["entropy_W"] for row in sweep], "GHZ": [row["entropy_GHZ"] for row in sweep]},
                       colors=COLORS, x_title="qubits n", y_title="entropy of one qubit (bits)", hover_format=".3f")
    fig.update_yaxes(range=[0, 1.05])
    sec.figure(fig, "Entanglement entropy between one qubit and all the others. GHZ always gives the maximum of "
                    "1 bit; W gives the binary entropy of 1/n, which shrinks as n grows.")
    sec.callout(
        f"At n = {n}, one GHZ qubit carries {detail['entropy_GHZ']:.2f} bit of entanglement with the rest and a W "
        f"qubit {detail['entropy_W']:.2f} bit. GHZ is the better resource when every qubit survives (e.g. "
        "metrology); W when some may not.",
        "info", "Trade-off.",
    )

    # Table
    sec = page.section("All results", "Click a column header to sort.")
    columns = ["qubits n", "pair conc. W", "pair conc. GHZ", "neg. after loss W", "neg. after loss GHZ",
               "P(entangled after measure) W", "P(entangled after measure) GHZ", "entropy W", "entropy GHZ"]
    body = [[row["n"], row["concurrence_W"], row["concurrence_GHZ"], row["lost_W"], row["lost_GHZ"],
             row["measured_W"], row["measured_GHZ"], row["entropy_W"], row["entropy_GHZ"]] for row in sweep]
    sec.table(columns, body, numeric=list(range(9)))

    page.reproduce({"qubits": n, "sweep": f"{min(xs)}-{max(xs)}"})
    path: Path = page.save()
    print(f"\nReport: {path}")


if __name__ == "__main__":
    main()
