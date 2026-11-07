"""
Use case: a fair n-sided die from a W state

Hadamard gates only make uniform distributions over 2^m outcomes, so a 6-sided
die built from a binary register has to throw away the outcomes 6 and 7 and
roll again (rejection sampling). The W state is uniform over exactly n
outcomes for any n, so every shot is a valid roll. The price is n qubits and
2(n - 1) CX gates instead of ceil(log2 n) qubits and no entangling gates. The
results are written to outputs/usecases/fair_dice/index.html.

Examples
--------
uv run python usecases/fair_dice/main.py
uv run python usecases/fair_dice/main.py --sides 12 --sweep 2-24
uv run python usecases/fair_dice/main.py --sides 6 --noise 0.02
"""

import math
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from scipy.stats import chisquare

from onehot import w_state
from onehot.report import Page, Stat, charts

# Make the repository root importable so that usecases.shared resolves when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from usecases.shared.common import base_parser, one_hot_index, sample  # noqa: E402


W_COLOR = charts.SERIES[0]
BINARY_COLOR = charts.SERIES[1]


def parse_range(text: str) -> list[int]:
    """
    Function that parses a range of side counts such as "2-20"

    Parameters
    ----------
    text: str
        Inclusive range "low-high" or comma-separated values

    Returns
    -------
    values: list[int]
        Sorted side counts (each >= 2)
    """
    # Expand ranges and single values
    found: set[int] = set()
    for part in text.split(","):
        if "-" in part:
            low, high = (int(v) for v in part.split("-"))
            found.update(range(low, high + 1))
        elif part.strip():
            found.add(int(part))
    values = sorted(v for v in found if v >= 2)

    return values


def binary_circuit(n: int) -> QuantumCircuit:
    """
    Function that builds the binary die: a uniform register of ceil(log2 n) qubits

    Parameters
    ----------
    n: int
        Number of faces

    Returns
    -------
    qc: QuantumCircuit
        Circuit with a Hadamard on each of ceil(log2 n) qubits
    """
    # A Hadamard on each qubit gives every m-bit value probability 1 / 2^m
    m = max(1, math.ceil(math.log2(n)))
    qc = QuantumCircuit(m, name="binary")
    qc.h(range(m))

    return qc


def w_faces(counts: dict[str, int], n: int) -> tuple[np.ndarray, int]:
    """
    Function that reads die faces from W-state counts

    Parameters
    ----------
    counts: dict[str, int]
        Bitstring -> count from the W-state die
    n: int
        Number of faces

    Returns
    -------
    faces: np.ndarray
        Rolls per face (face k + 1 when qubit k reads 1)
    rejected: int
        Shots that were not one-hot (only possible with noise)
    """
    # A one-hot outcome names the face; anything else is an error and is rolled again
    faces = np.zeros(n, dtype=int)
    rejected = 0
    for bits, c in counts.items():
        index = one_hot_index(bits)
        if index is None:
            rejected += c
        else:
            faces[index] += c

    return faces, rejected


def binary_faces(counts: dict[str, int], n: int) -> tuple[np.ndarray, int]:
    """
    Function that reads die faces from binary-register counts with rejection

    Parameters
    ----------
    counts: dict[str, int]
        Bitstring -> count from the binary die
    n: int
        Number of faces

    Returns
    -------
    faces: np.ndarray
        Rolls per face (face v + 1 for register value v < n)
    rejected: int
        Shots with a value >= n, which must be rolled again
    """
    # Keep values below n and reject the rest
    faces = np.zeros(n, dtype=int)
    rejected = 0
    for bits, c in counts.items():
        value = int(bits.replace(" ", ""), 2)
        if value < n:
            faces[value] += c
        else:
            rejected += c

    return faces, rejected


def first_rolls(qc: QuantumCircuit, n: int, count: int, seed: int) -> list[int]:
    """
    Function that rolls the W-state die shot by shot and returns the faces in order

    Parameters
    ----------
    qc: QuantumCircuit
        W-state circuit without measurements
    n: int
        Number of faces
    count: int
        Number of rolls
    seed: int
        Seed of the simulator

    Returns
    -------
    rolls: list[int]
        Faces 1..n in the order they were rolled
    """
    # Keep the per-shot memory so the rolls can be shown in sequence
    circuit = qc.copy()
    circuit.measure_all()
    sim = AerSimulator()
    memory = sim.run(transpile(circuit, sim), shots=count, memory=True, seed_simulator=seed).result().get_memory()
    rolls = [one_hot_index(bits) + 1 for bits in memory]

    return rolls


def die_svg(face: int) -> str:
    """
    Function that draws one face of a 6-sided die as an inline SVG

    Parameters
    ----------
    face: int
        Face 1..6

    Returns
    -------
    svg: str
        SVG markup with the pips of the face
    """
    # Pip positions on a 3 x 3 grid for each face
    spots = {1: [(1, 1)], 2: [(0, 0), (2, 2)], 3: [(0, 0), (1, 1), (2, 2)],
             4: [(0, 0), (0, 2), (2, 0), (2, 2)], 5: [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)],
             6: [(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 2)]}[face]

    # Draw the pips in the text colour so the die follows the page theme
    pips = "".join(f'<circle cx="{8 + 9 * c}" cy="{8 + 9 * r}" r="2.6" fill="currentColor"/>' for r, c in spots)
    svg = f'<svg viewBox="0 0 34 34" width="30" height="30" aria-label="{face}">{pips}</svg>'

    return svg


def roll_chips(rolls: list[int], n: int) -> str:
    """
    Function that renders a sequence of rolls as small chips

    Parameters
    ----------
    rolls: list[int]
        Faces 1..n
    n: int
        Number of faces (a 6-sided die is drawn with pips)

    Returns
    -------
    markup: str
        HTML block with one chip per roll
    """
    # Draw pips for a 6-sided die and numbers otherwise
    chips = "".join(
        f'<span title="face {r}" style="display:inline-grid;place-items:center;min-width:36px;height:36px;'
        f'padding:0 4px;border-radius:9px;border:1px solid var(--line);background:var(--surface);'
        f'color:var(--ink);font-size:15px;font-weight:600;font-variant-numeric:tabular-nums">'
        f'{die_svg(r) if n == 6 else r}</span>'
        for r in rolls
    )
    markup = f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:16px">{chips}</div>'

    return markup


def main() -> None:
    """
    Function that rolls both dice and writes the report

    Returns
    -------
    None
    """
    # Read the command-line arguments
    parser = base_parser(__doc__)
    parser.add_argument("--sides", type=int, default=6, help="number of faces n (default: 6)")
    parser.add_argument("--sweep", type=parse_range, default=parse_range("2-20"),
                        help="side counts for the resource comparison (default: 2-20)")
    parser.add_argument("--method", choices=["tree", "linear"], default="tree")
    args = parser.parse_args()
    n = args.sides

    # Roll both dice for the requested number of shots
    w_qc = w_state(n, args.method)
    bin_qc = binary_circuit(n)
    w_counts, w_rejected = w_faces(sample(w_qc, args.shots, args.seed), n)
    b_counts, b_rejected = binary_faces(sample(bin_qc, args.shots, args.seed), n)
    w_noisy, w_noisy_rejected = (w_faces(sample(w_qc, args.shots, args.seed, args.noise), n)
                                 if args.noise > 0 else (None, 0))

    # Test both dice for fairness over the accepted rolls
    w_p = float(chisquare(w_counts).pvalue)
    b_p = float(chisquare(b_counts).pvalue)

    # Compare resources over many side counts
    sweep = []
    for k in args.sweep:
        m = max(1, math.ceil(math.log2(k)))
        cx = transpile(w_state(k, args.method), basis_gates=["cx", "rz", "sx", "x"],
                       optimization_level=3).count_ops().get("cx", 0)
        sweep.append({"n": k, "w_qubits": k, "b_qubits": m, "w_cx": cx, "b_accept": k / 2**m,
                      "b_shots": 2**m / k})
    print(f"W: accepted {args.shots - w_rejected}/{args.shots}, p={w_p:.3f}; "
          f"binary: accepted {args.shots - b_rejected}/{args.shots}, p={b_p:.3f}")

    page = Page(
        "Fair dice", category="usecases", slug="fair_dice", out_root=args.out,
        summary="Roll a fair die with any number of faces in one shot, without the rejection sampling a binary "
        "register needs.",
        tags=["w_state", "sampling", "random numbers"],
    )
    m = bin_qc.num_qubits
    page.lead(
        rf"A register of \(m\) qubits under Hadamard gates is uniform over \(2^m\) values, so a {n}-sided die "
        rf"built from it uses \(m = \lceil\log_2 {n}\rceil = {m}\) qubits and must reject the "
        rf"{2**m - n} value{'s' if 2**m - n != 1 else ''} above {n - 1} and roll again. "
        r"\(|W_n\rangle\) is uniform over exactly \(n\) one-hot outcomes for <em>any</em> \(n\), so the position "
        r"of the single 1 is the face and every shot counts."
    )

    # Headline numbers
    page.headline([
        Stat("W die: shots accepted", f"{1 - w_rejected / args.shots:.0%}", f"ideal, n = {n}",
             "good" if w_rejected == 0 else "warn"),
        Stat("Binary die: shots accepted", f"{1 - b_rejected / args.shots:.1%}",
             f"theory {n}/{2**m} = {n / 2**m:.1%}"),
        Stat("Qubits", f"{n} vs {m}", "W die vs binary die"),
        Stat("Fairness (χ² p-value)", f"{w_p:.2f} · {b_p:.2f}", "W · binary, accepted rolls",
             "good" if min(w_p, b_p) > 0.05 else "warn"),
    ])

    # The two circuits
    sec = page.section(
        "Two ways to roll",
        f"The W-state die uses one qubit per face. The binary die uses {m} qubits under Hadamards, whose "
        f"value is the face when it is below {n}.",
    )
    w_meas, b_meas = w_qc.copy(), bin_qc.copy()
    w_meas.measure_all()
    b_meas.measure_all()
    sec.circuits([(w_meas, f"W-state die, {n} qubits"), (b_meas, f"binary die, {m} qubits")])
    sec.html(roll_chips(first_rolls(w_qc, n, 40, args.seed), n))
    sec.text("The first 40 rolls of the W-state die, one shot each.")

    # Distribution of the faces
    sec = page.section(
        "Faces rolled",
        f"Share of accepted rolls that landed on each face over {args.shots} shots. A fair die puts every bar "
        f"at 1/{n}.",
    )
    shares = {"W state": list(w_counts / w_counts.sum()), "binary + rejection": list(b_counts / b_counts.sum())}
    colors = {"W state": W_COLOR, "binary + rejection": BINARY_COLOR}
    fig = charts.bars([f"{k + 1}" for k in range(n)], shares, colors=colors, x_title="face",
                      y_title="share of accepted rolls", hover_format=".2%", reference=1 / n,
                      reference_label=f"fair 1/{n}")
    fig.update_yaxes(tickformat=".0%")
    sec.figure(fig, "Both dice are fair once rejected shots are discarded; they differ in how many shots that takes.")

    shots_used = {"W state": [args.shots - w_rejected, w_rejected],
                  "binary + rejection": [args.shots - b_rejected, b_rejected]}
    fig = charts.bars(["accepted", "rejected"], shots_used, colors=colors, y_title="shots", hover_format=",d")
    sec.figure(fig, f"Where the {args.shots} shots went. Every rejected shot of the binary die is wasted work.")
    if w_noisy is not None:
        sec.callout(
            f"With CX error {args.noise:g}, the W-state die rejects {w_noisy_rejected / args.shots:.1%} of shots "
            "because they are not one-hot. That is a feature: noise that breaks the one-hot structure is "
            "<em>detected</em> and the roll is repeated, instead of silently producing a biased face. The binary "
            "die has no such check.",
            "info", "Errors announce themselves.",
        )

    # Resources over many side counts
    sec = page.section(
        "What it costs",
        "Resources per accepted roll as the number of faces grows. Neither die is better everywhere.",
    )
    xs = [row["n"] for row in sweep]
    fig = charts.lines(xs, {"W state": [row["w_qubits"] for row in sweep],
                            "binary + rejection": [row["b_qubits"] for row in sweep]},
                       colors=colors, x_title="faces n", y_title="qubits")
    sec.figure(fig, "Qubits: n for the W state, ⌈log₂ n⌉ for the binary register.")
    fig = charts.lines(xs, {"W state": [1.0] * len(xs), "binary + rejection": [row["b_shots"] for row in sweep]},
                       colors=colors, x_title="faces n", y_title="expected shots per accepted roll",
                       hover_format=".3f")
    fig.update_yaxes(range=[0.9, 2.05])
    sec.figure(fig, "Expected shots per accepted roll (2^m / n). The binary die is exact at powers of two and "
                    "worst just above them, where it approaches 2 shots per roll.")
    fig = charts.lines(xs, {"W state": [row["w_cx"] for row in sweep], "binary + rejection": [0] * len(xs)},
                       colors=colors, x_title="faces n", y_title="CX gates")
    sec.figure(fig, "Entangling gates after optimisation: the W state needs CX gates, the binary die none.")
    worst = max(sweep, key=lambda row: row["b_shots"])
    sec.callout(
        f"The binary die wastes up to {1 - 1 / worst['b_shots']:.0%} of shots (at n = {worst['n']}) but needs "
        f"only {worst['b_qubits']} qubits and no entangling gates. The W-state die never wastes an ideal shot "
        f"and detects one-hot-breaking errors, at the cost of {worst['n']} qubits and {worst['w_cx']} CX gates. "
        "Prefer the W state when every shot is expensive (each roll feeds a larger circuit) and the binary die "
        "when qubits or two-qubit gates are the scarce resource.",
        "info", "Trade-off.",
    )

    # Table
    sec = page.section("All results", "Click a column header to sort.")
    columns = ["faces n", "W qubits", "W CX", "binary qubits", "binary acceptance", "binary shots / roll"]
    body = [[row["n"], row["w_qubits"], row["w_cx"], row["b_qubits"], row["b_accept"], row["b_shots"]]
            for row in sweep]
    sec.table(columns, body, numeric=[0, 1, 2, 3, 4, 5])

    page.reproduce({"sides": n, "sweep": f"{min(xs)}-{max(xs)}", "method": args.method, "shots": args.shots,
                    "CX error": args.noise, "seed": args.seed})
    path: Path = page.save()
    print(f"\nReport: {path}")


if __name__ == "__main__":
    main()
