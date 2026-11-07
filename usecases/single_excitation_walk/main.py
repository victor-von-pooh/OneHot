"""
Use case: a single excitation hopping between qubits (quantum walk, state transfer)

The XY Hamiltonian H = sum_(i,j) J_ij (X_i X_j + Y_i Y_j) / 2 moves an
excitation from qubit i to qubit j without creating or destroying one, so a
one-hot state stays one-hot forever. Inside that subspace H is just the
weighted adjacency matrix of the coupling graph, which makes the W state the
natural reference point: it is the stationary state of the complete graph,
and it spreads on a line. The script simulates four scenarios exactly and with
a Trotter circuit, samples the circuit with noise, and writes
outputs/usecases/single_excitation_walk/index.html.

Examples
--------
uv run python usecases/single_excitation_walk/main.py
uv run python usecases/single_excitation_walk/main.py --sites 6 --time 6
uv run python usecases/single_excitation_walk/main.py --dt 0.02 --noise 0.005
"""

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import scipy.linalg
from plotly.subplots import make_subplots
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import Statevector

from onehot import w_state
from onehot.report import HTML, Page, Stat, charts

# Make the repository root importable so that usecases.shared resolves when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from usecases.shared.common import base_parser, one_hot_index, sample  # noqa: E402


@dataclass
class Scenario:
    """
    Class that describes one walk: couplings, starting state and its results

    Parameters
    ----------
    name: str
        Label used in charts and tables
    edges: list[tuple[int, int, float]]
        Couplings (i, j, J_ij)
    start: str
        "local" (excitation on qubit 0) or "w" (W state)
    color: str
        Fixed colour of the scenario
    exact: np.ndarray
        Array (times, sites): exact P(site k excited)
    one_hot: list[float]
        One-hot probability of the Trotter state at each time
    fidelity: list[float]
        Fidelity between the Trotter state and the exact state at each time
    """

    name: str
    edges: list[tuple[int, int, float]]
    start: str
    color: str
    exact: np.ndarray = field(default=None)
    one_hot: list[float] = field(default_factory=list)
    fidelity: list[float] = field(default_factory=list)


def scenarios(n: int) -> list[Scenario]:
    """
    Function that defines the four walks shown in the report

    Parameters
    ----------
    n: int
        Number of sites (qubits)

    Returns
    -------
    items: list[Scenario]
        Local excitation on a line, W on a line, W on a complete graph, perfect-transfer chain
    """
    # Uniform couplings on a line and on the complete graph
    line = [(i, i + 1, 1.0) for i in range(n - 1)]
    complete = [(i, j, 1.0) for i in range(n) for j in range(i + 1, n)]

    # Couplings sqrt(k (n - k)) send site 0 to site n - 1 perfectly at t = pi / 2
    pst = [(k - 1, k, math.sqrt(k * (n - k))) for k in range(1, n)]

    items = [
        Scenario("local · line", line, "local", charts.SERIES[0]),
        Scenario("W · line", line, "w", charts.SERIES[1]),
        Scenario("W · complete", complete, "w", charts.SERIES[2]),
        Scenario("transfer chain", pst, "local", charts.SERIES[3]),
    ]

    return items


def initial_circuit(n: int, start: str, method: str) -> QuantumCircuit:
    """
    Function that prepares the starting one-hot state

    Parameters
    ----------
    n: int
        Number of sites
    start: str
        "local" puts the excitation on qubit 0, "w" prepares the W state
    method: str
        W-state construction

    Returns
    -------
    qc: QuantumCircuit
        Preparation circuit
    """
    # A single X for a local excitation, otherwise the W state as one box
    qc = QuantumCircuit(n)
    if start == "local":
        qc.x(0)
    else:
        qc.append(w_state(n, method).to_gate(label="W"), range(n))

    return qc


def trotter_step(n: int, edges: list[tuple[int, int, float]], dt: float) -> QuantumCircuit:
    """
    Function that builds one first-order Trotter step of exp(-i H dt)

    Edges are packed greedily into layers of disjoint pairs, so a line takes two
    layers (even bonds, then odd bonds).

    Parameters
    ----------
    n: int
        Number of sites
    edges: list[tuple[int, int, float]]
        Couplings (i, j, J_ij)
    dt: float
        Time step

    Returns
    -------
    qc: QuantumCircuit
        Circuit of XXPlusYY gates for one step
    """
    # Pack edges into layers whose pairs share no qubit
    layers: list[list[tuple[int, int, float]]] = []
    for edge in edges:
        for layer in layers:
            if all({edge[0], edge[1]}.isdisjoint({e[0], e[1]}) for e in layer):
                layer.append(edge)
                break
        else:
            layers.append([edge])

    # exp(-i J dt (XX + YY) / 2) is XXPlusYYGate(theta = 2 J dt)
    qc = QuantumCircuit(n)
    for layer in layers:
        for i, j, coupling in layer:
            qc.append(XXPlusYYGate(2 * coupling * dt), [i, j])

    return qc


def exact_walk(n: int, sc: Scenario, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Function that evolves the walk exactly inside the one-hot subspace

    Parameters
    ----------
    n: int
        Number of sites
    sc: Scenario
        Walk to evolve
    times: np.ndarray
        Times to record

    Returns
    -------
    probs: np.ndarray
        Array (times, sites): P(site k excited)
    states: np.ndarray
        Array (times, sites): amplitudes on the one-hot states
    """
    # Inside the one-hot subspace H is the weighted adjacency matrix
    adjacency = np.zeros((n, n))
    for i, j, coupling in sc.edges:
        adjacency[i, j] = adjacency[j, i] = coupling

    # Start from e_0 or from the uniform vector (the W state)
    psi0 = np.zeros(n, dtype=complex)
    if sc.start == "local":
        psi0[0] = 1
    else:
        psi0[:] = 1 / math.sqrt(n)

    # Apply exp(-i A t) for every recorded time
    states = np.array([scipy.linalg.expm(-1j * adjacency * t) @ psi0 for t in times])
    probs = np.abs(states) ** 2

    return probs, states


def trotter_walk(n: int, sc: Scenario, times: np.ndarray, dt: float, method: str,
                 exact_states: np.ndarray) -> tuple[list[float], list[float]]:
    """
    Function that runs the Trotter circuit on all 2^n states and checks it

    Parameters
    ----------
    n: int
        Number of sites
    sc: Scenario
        Walk to evolve
    times: np.ndarray
        Times to record, spaced by dt
    dt: float
        Trotter step
    method: str
        W-state construction
    exact_states: np.ndarray
        Exact one-hot amplitudes from exact_walk

    Returns
    -------
    one_hot: list[float]
        Probability of the one-hot subspace at each time
    fidelity: list[float]
        Fidelity with the exact state at each time
    """
    one_hot_idx = [1 << k for k in range(n)]
    step = trotter_step(n, sc.edges, dt)
    state = Statevector(initial_circuit(n, sc.start, method))

    # Step forward and compare with the exact state embedded in 2^n dimensions
    one_hot, fidelity = [], []
    for i in range(len(times)):
        if i > 0:
            state = state.evolve(step)
        amplitudes = state.data[one_hot_idx]
        one_hot.append(float(np.sum(np.abs(amplitudes) ** 2)))
        fidelity.append(float(abs(np.vdot(exact_states[i], amplitudes)) ** 2))

    return one_hot, fidelity


def noisy_walk(n: int, sc: Scenario, t_max: float, dt: float, points: int, method: str,
               args) -> list[dict]:
    """
    Function that samples coarse Trotter circuits with noise at a few times

    Parameters
    ----------
    n: int
        Number of sites
    sc: Scenario
        Walk to sample
    t_max: float
        Last time
    dt: float
        Trotter step of the noisy circuits
    points: int
        Number of sampled times
    method: str
        W-state construction
    args: argparse.Namespace
        Shots, noise and seed

    Returns
    -------
    results: list[dict]
        Per time: t, steps, CX count, one-hot rate and distances to the exact distribution
    """
    results = []
    step = trotter_step(n, sc.edges, dt)
    for t in np.linspace(0, t_max, points):
        # Build the circuit for this time with whole Trotter steps
        steps = int(round(t / dt))
        qc = initial_circuit(n, sc.start, method)
        for _ in range(steps):
            qc.compose(step, inplace=True)
        exact, _ = exact_walk(n, sc, np.array([steps * dt]))

        # Sample with noise and read site marginals with and without post-selection
        counts = sample(qc, args.shots, args.seed, args.noise)
        total = sum(counts.values())
        raw = np.zeros(n)
        kept = np.zeros(n)
        for bits, c in counts.items():
            clean = bits.replace(" ", "")[::-1]
            raw += np.array([int(b) for b in clean]) * c / total
            idx = one_hot_index(bits)
            if idx is not None:
                kept[idx] += c
        rate = kept.sum() / total
        kept = kept / kept.sum() if kept.sum() else kept

        # Total variation distance to the exact distribution over sites
        raw_tvd = 0.5 * float(np.abs(raw / raw.sum() - exact[0]).sum()) if raw.sum() else 1.0
        kept_tvd = 0.5 * float(np.abs(kept - exact[0]).sum())
        cx = qc.decompose(reps=3).count_ops().get("cx", 0)
        results.append({"t": steps * dt, "steps": steps, "cx": cx, "rate": rate,
                        "raw_tvd": raw_tvd, "kept_tvd": kept_tvd})

    return results


def main() -> None:
    """
    Function that runs every walk and writes the report

    Returns
    -------
    None
    """
    # Read the command-line arguments
    parser = base_parser(__doc__)
    parser.add_argument("--sites", type=int, default=8, help="number of sites = qubits (default: 8)")
    parser.add_argument("--time", type=float, default=4.0, help="last simulated time (default: 4.0)")
    parser.add_argument("--dt", type=float, default=0.05, help="Trotter step of the noiseless check (default: 0.05)")
    parser.add_argument("--noisy-dt", type=float, default=0.25, help="Trotter step of the noisy runs (default: 0.25)")
    parser.add_argument("--method", choices=["tree", "linear"], default="tree", help="W-state construction")
    args = parser.parse_args()
    n = args.sites
    if n < 3:
        parser.error("--sites must be >= 3")
    t_max = max(args.time, math.pi / 2)
    times = np.arange(0, t_max + 1e-9, args.dt)

    # Exact and Trotter evolution of every scenario
    items = scenarios(n)
    for sc in items:
        sc.exact, states = exact_walk(n, sc, times)
        sc.one_hot, sc.fidelity = trotter_walk(n, sc, times, args.dt, args.method, states)
        print(f"{sc.name:<15} min one-hot={min(sc.one_hot):.12f} final fidelity={sc.fidelity[-1]:.4f}")
    by_name = {sc.name: sc for sc in items}

    # Key quantities quoted in the report
    min_one_hot = min(min(sc.one_hot) for sc in items)
    w_complete = by_name["W · complete"]
    stationary_dev = float(np.abs(w_complete.exact - 1 / n).max())
    _, pst_states = exact_walk(n, by_name["transfer chain"], np.array([math.pi / 2]))
    pst_peak = float(abs(pst_states[0, n - 1]) ** 2)
    uniform_last = by_name["local · line"].exact[:, n - 1]
    uniform_peak, uniform_peak_t = float(uniform_last.max()), float(times[int(uniform_last.argmax())])
    w_line_spread = float(np.abs(by_name["W · line"].exact - 1 / n).max())

    # Noisy sampling of the local walk on a line
    noisy = []
    if args.noise > 0:
        noisy = noisy_walk(n, by_name["local · line"], t_max, args.noisy_dt, 9, args.method, args)

    # Page header and headline numbers
    page = Page(
        "Single-excitation walk", category="usecases", slug="single_excitation_walk", out_root=args.out,
        summary="An excitation hopping between qubits under the XY Hamiltonian never leaves the one-hot "
        "subspace: the W state is stationary on a complete graph, spreads on a line, and engineered "
        "couplings transfer a state end to end.",
        tags=["quantum walk", "XY model", "state transfer", "Trotter"],
    )
    page.lead(
        r"The XY coupling \(\tfrac{J}{2}(X_iX_j + Y_iY_j)\) swaps \(|01\rangle \leftrightarrow |10\rangle\) and "
        r"leaves \(|00\rangle\), \(|11\rangle\) alone, so it moves an excitation without creating or destroying one. "
        r"Inside the one-hot subspace the Hamiltonian is simply the coupling graph's adjacency matrix "
        r"\(A\), and \(|\psi(t)\rangle = e^{-iAt}|\psi(0)\rangle\). The W state is the all-ones vector of "
        r"that subspace."
    )
    hero = [
        Stat("One-hot probability", f"{min_one_hot:.6f}", "minimum over every Trotter state",
             "good" if min_one_hot > 1 - 1e-9 else "bad"),
        Stat("W on complete graph", f"±{stationary_dev:.1e}", "largest deviation from 1/n over time",
             "good" if stationary_dev < 1e-9 else "warn"),
        Stat("Transfer at t = π/2", f"{pst_peak:.1%}", f"site 0 → site {n - 1}, engineered chain",
             "good" if pst_peak > 0.999 else "warn"),
    ]
    if noisy:
        hero.append(Stat(f"One-hot kept at t = {noisy[-1]['t']:g}", f"{noisy[-1]['rate']:.1%}",
                         f"local walk with CX error {args.noise:g}"))
    page.headline(hero)

    # Heatmaps of the four walks
    sec = page.section(
        "Four walks",
        f"Exact probability that each of the {n} sites holds the excitation, from t = 0 (top) to "
        f"t = {t_max:g} (bottom). Every row sums to 1.",
    )
    fig = make_subplots(rows=1, cols=4, shared_yaxes=True, subplot_titles=[sc.name for sc in items],
                        horizontal_spacing=0.03)
    for col, sc in enumerate(items, start=1):
        fig.add_trace(go.Heatmap(z=sc.exact, x=list(range(n)), y=times, coloraxis="coloraxis", xgap=1,
                                 hovertemplate="site %{x}<br>t = %{y:.2f}<br>P = %{z:.3f}<extra>"
                                 + sc.name + "</extra>"), row=1, col=col)
    charts.style(fig, height=520)
    fig.update_layout(coloraxis=dict(colorscale=charts.SEQUENTIAL, cmin=0, cmax=1,
                                     colorbar=dict(title=dict(text="P(excited)", side="right"), thickness=12)),
                      margin=dict(t=56))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_yaxes(title_text="time t", row=1, col=1)
    fig.update_xaxes(title_text="site", dtick=1)
    fig.update_annotations(font=dict(size=13))
    sec.figure(fig, "Hover for exact values. The local excitation spreads ballistically and bounces off the ends; "
                    "the W state on a line drifts away from uniform; on the complete graph it does not move at all; "
                    "the engineered chain refocuses the excitation on the last site.")
    sec.callout(
        f"On the complete graph every site couples to every other, so the uniform vector is an eigenvector of A "
        f"(eigenvalue {n - 1}): the W state only picks up a global phase, and its site probabilities stay at 1/{n} "
        f"(largest deviation {stationary_dev:.1e}). On a line the same state is not an eigenvector and its "
        f"probabilities move by up to {w_line_spread:.3f}.",
        "info", "Why the W state matters here.",
    )

    # Time slider over the site distributions
    sec = page.section("Step through time", "Drag the slider or press play to watch all four walks together.")
    frame_idx = list(range(0, len(times), max(1, len(times) // 60)))
    fig = go.Figure(
        data=[go.Bar(x=list(range(n)), y=sc.exact[0], name=sc.name, marker=dict(color=sc.color),
                     hovertemplate="site %{x}: %{y:.3f}<extra>" + sc.name + "</extra>") for sc in items],
        frames=[go.Frame(data=[go.Bar(y=sc.exact[i]) for sc in items], name=f"{times[i]:.2f}") for i in frame_idx],
    )
    charts.style(fig, x_title="site", y_title="P(excited)", height=440, legend=True)
    fig.update_layout(
        barmode="group", yaxis=dict(range=[0, 1]), margin=dict(b=120),
        updatemenus=[dict(type="buttons", showactive=False, x=0, y=-0.32, xanchor="left", yanchor="top",
                          buttons=[dict(label="▶ play", method="animate",
                                        args=[None, dict(frame=dict(duration=120, redraw=True), fromcurrent=True)]),
                                   dict(label="❚❚ pause", method="animate",
                                        args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
        sliders=[dict(active=0, x=0.16, len=0.84, y=-0.24, yanchor="top", currentvalue=dict(prefix="t = "),
                      steps=[dict(label=f"{times[i]:.2f}", method="animate",
                                  args=[[f"{times[i]:.2f}"], dict(mode="immediate", frame=dict(duration=0))])
                             for i in frame_idx])],
    )
    fig.update_xaxes(dtick=1)
    sec.figure(fig, "Each frame is one moment of the exact evolution.")

    # Perfect state transfer
    sec = page.section(
        "Perfect state transfer",
        rf"With couplings \(J_k = \sqrt{{k(n-k)}}\) between sites \(k-1\) and \(k\), the chain acts like a spin "
        r"rotated by \(\pi\) at \(t = \pi/2\): the excitation leaves site 0 and arrives intact at the last site.",
    )
    fig = charts.lines(times, {"transfer chain": by_name["transfer chain"].exact[:, n - 1],
                               "uniform line": uniform_last},
                       colors={"transfer chain": charts.SERIES[3], "uniform line": charts.SERIES[0]},
                       x_title="time t", y_title=f"P(site {n - 1})", hover_format=".3f")
    fig.update_traces(mode="lines")
    fig.add_vline(x=math.pi / 2, line=dict(color="#898781", width=1, dash="dot"),
                  annotation_text="t = π/2", annotation_position="top left")
    fig.update_yaxes(range=[0, 1.05])
    sec.figure(fig, f"The engineered chain reaches {pst_peak:.4f} at t = π/2; with uniform couplings the best "
                    f"arrival in this window is {uniform_peak:.3f} at t = {uniform_peak_t:.2f}.")
    fig = charts.bars([f"{k - 1}–{k}" for k in range(1, n)], {"J_k": [math.sqrt(k * (n - k)) for k in range(1, n)]},
                      colors={"J_k": charts.SERIES[3]}, x_title="bond", y_title="coupling J", height=280)
    sec.figure(fig, "Coupling profile of the transfer chain: strongest in the middle, symmetric about the centre.")

    # The circuit and its validation
    sec = page.section(
        "As a circuit",
        rf"On hardware the evolution is approximated by Trotter steps of length dt = {args.dt:g}. Each bond becomes "
        r"one <code>XXPlusYYGate</code>\((2J\,dt)\), which also preserves the number of excitations, so the "
        r"circuit never leaves the one-hot subspace either.",
    )
    sec.circuit(initial_circuit(n, "w", args.method).compose(trotter_step(n, by_name["W · line"].edges, args.dt)),
                f"W-state preparation followed by one Trotter step on a line of {n} sites: even bonds, then odd bonds.")
    fig = charts.lines(times, {sc.name: sc.fidelity for sc in items}, colors={sc.name: sc.color for sc in items},
                       x_title="time t", y_title="fidelity with exact state", hover_format=".4f")
    fig.update_traces(mode="lines")
    sec.figure(fig, "Trotter error accumulates with time and grows with the coupling strength (the transfer chain "
                    "has the largest J). It never shows up as leakage: the one-hot probability never drops below "
                    f"{min_one_hot:.12f}.")
    w_trotter_error = 1 - min(w_complete.fidelity)
    sec.callout(
        f"For the W state on the complete graph the Trotter circuit matches the exact evolution to "
        f"{w_trotter_error:.1e} in fidelity, so the stationary state survives the discretisation; the other walks "
        f"lose up to {max(1 - min(sc.fidelity) for sc in items):.1e}, which shrinks with dt.",
        "good" if w_trotter_error < 1e-6 else "info",
    )

    # Noise
    if noisy:
        sec = page.section(
            "With noise: errors become detectable",
            f"The local walk on a line, sampled ({args.shots} shots) with coarser Trotter steps (dt = "
            f"{args.noisy_dt:g}) and CX error {args.noise:g}. A perfect run always yields exactly one '1', so any "
            "other outcome is a detected error and can be thrown away (post-selection).",
        )
        fig = charts.lines([r["t"] for r in noisy], {"one-hot rate": [r["rate"] for r in noisy]},
                           colors={"one-hot rate": charts.SERIES[0]}, x_title="time t",
                           y_title="one-hot outcomes", hover_format=".1%")
        fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
        sec.figure(fig, "The longer the walk, the more CX gates, the more shots leave the subspace.")
        fig = charts.lines([r["t"] for r in noisy], {"all shots": [r["raw_tvd"] for r in noisy],
                                                     "post-selected": [r["kept_tvd"] for r in noisy]},
                           colors={"all shots": charts.SERIES[1], "post-selected": charts.SERIES[0]},
                           x_title="time t", y_title="distance to exact distribution", hover_format=".3f")
        sec.figure(fig, "Total variation distance between the measured site distribution and the exact one. "
                        "Keeping only one-hot shots removes much of the noise (remaining distance includes the "
                        "coarse Trotter error).")
        sec.table(["t", "Trotter steps", "CX", "one-hot rate", "distance · all shots", "distance · post-selected"],
                  [[r["t"], r["steps"], r["cx"], r["rate"], r["raw_tvd"], r["kept_tvd"]] for r in noisy],
                  max_height=None)

    # Summary table
    sec = page.section("Summary", "Click a column header to sort.")
    rows = []
    for sc in items:
        last = sc.exact[:, n - 1]
        rows.append([sc.name, sc.start, len(sc.edges), min(sc.one_hot), sc.fidelity[-1],
                     float(np.abs(sc.exact - 1 / n).max()), float(last.max()), float(times[int(last.argmax())])])
    sec.table(["walk", "start", "couplings", "min one-hot", "final Trotter fidelity", "max |P − 1/n|",
               f"max P(site {n - 1})", "at t"], rows, numeric=[2, 3, 4, 5, 6, 7], max_height=None)

    page.reproduce({"sites": n, "time": t_max, "dt": args.dt, "noisy dt": args.noisy_dt, "method": args.method,
                    "shots": args.shots, "CX error": args.noise, "seed": args.seed})
    path = page.save()
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
