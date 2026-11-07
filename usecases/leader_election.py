"""
Use case: fair leader election with a shared W state

n parties each hold one qubit of the W state and measure it. Because every
outcome of the W state contains exactly one '1', exactly one party reads 1 and
becomes the leader, and every party is equally likely to win. A classical
"everyone flips a coin and hopes exactly one gets heads" scheme succeeds only
with probability n / 2^n per round. The results are written to
outputs/usecases/leader_election/index.html.

Examples
--------
uv run python usecases/leader_election.py
uv run python usecases/leader_election.py --parties 7 --sweep 2-14
uv run python usecases/leader_election.py --noise 0.005 --method linear
"""

from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from scipy.stats import chisquare

from _common import base_parser, one_hot_index, sample
from onehot import w_state
from onehot.report import Page, Stat, charts


def parse_range(text: str) -> list[int]:
    """
    Function that parses a range of party counts such as "2-12"

    Parameters
    ----------
    text: str
        Inclusive range "low-high" or comma-separated values

    Returns
    -------
    values: list[int]
        Sorted party counts (each >= 2)
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


def coin_flip_circuit(n: int) -> QuantumCircuit:
    """
    Function that builds the naive scheme: every party flips an independent fair coin

    Parameters
    ----------
    n: int
        Number of parties

    Returns
    -------
    qc: QuantumCircuit
        Circuit that puts every qubit in |+>
    """
    # A Hadamard on each qubit is a fair, independent coin per party
    qc = QuantumCircuit(n, name="coins")
    qc.h(range(n))

    return qc


def election_stats(counts: dict[str, int], n: int) -> dict:
    """
    Function that summarises the election rounds in a set of counts

    Parameters
    ----------
    counts: dict[str, int]
        Bitstring -> number of rounds
    n: int
        Number of parties

    Returns
    -------
    stats: dict
        "wins": rounds won per party, "valid": share of rounds with exactly one
        leader, "none": share with no leader, "many": share with several leaders
    """
    # Tally a win for the only party that read 1, and classify the other rounds
    total = sum(counts.values())
    wins = np.zeros(n, dtype=int)
    none = many = 0
    for bits, c in counts.items():
        index = one_hot_index(bits)
        if index is not None:
            wins[index] += c
        elif bits.count("1") == 0:
            none += c
        else:
            many += c

    stats = {"wins": wins, "valid": wins.sum() / total, "none": none / total, "many": many / total,
             "total": total}

    return stats


def main() -> None:
    """
    Function that runs the elections and writes the report

    Returns
    -------
    None
    """
    # Read the command-line arguments
    parser = base_parser(__doc__)
    parser.add_argument("--parties", type=int, default=5, help="number of parties n (default: 5)")
    parser.add_argument("--sweep", type=parse_range, default=parse_range("2-12"),
                        help="party counts for the scaling sweep (default: 2-12)")
    parser.add_argument("--method", choices=["tree", "linear"], default="tree")
    args = parser.parse_args()
    n = args.parties

    # Elect a leader once per shot, ideally and with gate noise
    qc = w_state(n, args.method)
    ideal = election_stats(sample(qc, args.shots, args.seed), n)
    noisy = election_stats(sample(qc, args.shots, args.seed, args.noise), n) if args.noise > 0 else None

    # Test whether the ideal winners are uniform over the parties
    p_value = float(chisquare(ideal["wins"]).pvalue)

    # Sweep the number of parties: W state (ideal / noisy) against independent coins
    sweep = []
    for k in args.sweep:
        w_ideal = election_stats(sample(w_state(k, args.method), args.shots, args.seed), k)["valid"]
        w_noisy = (election_stats(sample(w_state(k, args.method), args.shots, args.seed, args.noise), k)["valid"]
                   if args.noise > 0 else None)
        coins = election_stats(sample(coin_flip_circuit(k), args.shots, args.seed), k)["valid"]
        sweep.append({"n": k, "w_ideal": w_ideal, "w_noisy": w_noisy, "coins": coins, "coins_theory": k / 2**k})
        print(f"n={k:>2} W ideal={w_ideal:.3f} W noisy={w_noisy} coins={coins:.3f}")

    page = Page(
        "Leader election", category="usecases", slug="leader_election", out_root=args.out,
        summary="Pick exactly one of n parties, uniformly at random, in a single round by measuring a shared "
        "W state.",
        tags=["w_state", "distributed", "sampling"],
    )
    page.lead(
        r"Each of \(n\) parties holds one qubit of \(|W_n\rangle\) and measures it. Every term of the W state has "
        r"exactly one \(|1\rangle\), so <b>exactly one party reads 1</b>: that party is the leader, and each party "
        r"wins with probability \(1/n\). No messages, no retries. The naive alternative, where every party flips "
        r"its own coin and the round counts only if exactly one gets heads, succeeds with probability "
        r"\(n/2^n\) and needs ever more rounds as \(n\) grows."
    )

    # Headline numbers
    hero = [
        Stat("Rounds with one leader", f"{ideal['valid']:.1%}", f"ideal, {args.shots} rounds, n = {n}",
             "good" if ideal["valid"] == 1 else "bad"),
        Stat("Fairness (χ² p-value)", f"{p_value:.2f}", "uniform winners if p is not small",
             "good" if p_value > 0.05 else "warn"),
        Stat("Coin flips succeed", f"{n / 2**n:.1%}", f"n/2ⁿ per round for n = {n}"),
    ]
    if noisy:
        hero.append(Stat("With noise", f"{noisy['valid']:.1%}", f"one leader per round, CX error {args.noise:g}"))
    page.headline(hero)

    # The protocol and its circuit
    sec = page.section(
        "The protocol",
        "Prepare the W state, hand one qubit to each party, and let everyone measure. The measured bitstring is "
        "one-hot, and the position of the 1 names the leader.",
    )
    measured = qc.copy()
    measured.measure_all()
    sec.circuit(measured, f"W state on {n} qubits ({args.method} construction) followed by a measurement of every "
                          "qubit; qubit k belongs to party k.")

    # Who won
    sec = page.section(
        "Who won",
        f"How often each party became leader over {args.shots} rounds. A fair election puts every bar at 1/{n}.",
    )
    shares = {"ideal": list(ideal["wins"] / ideal["total"])}
    if noisy:
        shares["noisy"] = list(noisy["wins"] / noisy["total"])
    fig = charts.bars([f"party {k}" for k in range(n)], shares,
                      colors={"ideal": charts.SERIES[0], "noisy": charts.SERIES[1]},
                      y_title="share of rounds won", hover_format=".2%", reference=1 / n,
                      reference_label=f"fair 1/{n}")
    fig.update_yaxes(tickformat=".0%")
    sec.figure(fig, "Ideal wins scatter around 1/n only by sampling error. With noise, some rounds are lost to "
                    "zero or several leaders, so every party's share drops a little.")
    sec.callout(
        f"A chi-square test of the ideal winners against the uniform distribution gives p = {p_value:.3f} "
        f"(χ² over {n} parties). " + ("Nothing suggests any party is favoured."
                                      if p_value > 0.05 else "This is unusually uneven; try more shots."),
        "good" if p_value > 0.05 else "warn", "Fair.",
    )
    if noisy:
        sec.callout(
            f"With CX error {args.noise:g}, {noisy['valid']:.1%} of rounds still have exactly one leader; "
            f"{noisy['none']:.1%} have none and {noisy['many']:.1%} have several. Those rounds are detectable "
            "(everyone sees the bitstring is not one-hot) and can simply be repeated.",
            "info", "Noise is detectable.",
        )

    # Scaling with the number of parties
    sec = page.section(
        "More parties",
        "Share of rounds that produce exactly one leader, as the number of parties grows.",
    )
    xs = [row["n"] for row in sweep]
    series = {"W state (ideal)": [row["w_ideal"] for row in sweep]}
    colors = {"W state (ideal)": charts.SERIES[0], "independent coins": charts.SERIES[1]}
    dash = {}
    if args.noise > 0:
        series[f"W state (CX error {args.noise:g})"] = [row["w_noisy"] for row in sweep]
        colors[f"W state (CX error {args.noise:g})"] = charts.SERIES[0]
        dash[f"W state (CX error {args.noise:g})"] = "dash"
    series["independent coins"] = [row["coins"] for row in sweep]
    fig = charts.lines(xs, series, colors=colors, dash=dash, x_title="parties n",
                       y_title="rounds with one leader", hover_format=".1%")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
    lowest = min(row["w_ideal"] for row in sweep)
    sec.figure(fig, f"Coins collapse as n/2ⁿ; the ideal W state never drops below {lowest:.0%}"
                    + (" and degrades slowly with gate noise." if args.noise > 0 else "."))

    rounds = {"W state (ideal)": [1 / row["w_ideal"] for row in sweep]}
    if args.noise > 0:
        rounds[f"W state (CX error {args.noise:g})"] = [1 / row["w_noisy"] for row in sweep]
    rounds["independent coins"] = [2 ** row["n"] / row["n"] for row in sweep]
    fig = charts.lines(xs, rounds, colors=colors, dash=dash, x_title="parties n",
                       y_title="expected rounds per election", log_y=True, hover_format=".2f")
    fig.update_yaxes(dtick=1)
    sec.figure(fig, "Expected number of rounds until exactly one leader emerges (1 / success rate), on a log scale.")
    last = sweep[-1]
    sec.callout(
        f"At n = {last['n']}, independent coins need on average {2 ** last['n'] / last['n']:.0f} rounds; "
        f"the W state needs {1 / last['w_ideal']:.0f} ideally"
        + (f" and {1 / last['w_noisy']:.2f} with CX error {args.noise:g}." if last["w_noisy"] else "."),
        "info", "Takeaway.",
    )

    # Table
    sec = page.section("All results", "Click a column header to sort.")
    columns = ["parties n", "W ideal", "W noisy", "coins (sampled)", "coins n/2ⁿ", "rounds · coins"]
    body = [[row["n"], row["w_ideal"], row["w_noisy"] if row["w_noisy"] is not None else 0.0, row["coins"],
             row["coins_theory"], 2 ** row["n"] / row["n"]] for row in sweep]
    sec.table(columns, body, numeric=[0, 1, 2, 3, 4, 5])

    page.reproduce({"parties": n, "sweep": f"{min(xs)}-{max(xs)}", "method": args.method, "shots": args.shots,
                    "CX error": args.noise, "seed": args.seed})
    path: Path = page.save()
    print(f"\nReport: {path}")


if __name__ == "__main__":
    main()
