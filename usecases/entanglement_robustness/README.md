# Entanglement robustness

Lose or measure one qubit: the GHZ state's entanglement vanishes, the W state's survives.

## The idea

The W state and the GHZ state are the two fundamentally different kinds of entanglement shared by
three or more qubits:

```
|W_n>   = ( |10...0> + |01...0> + ... + |00...1> ) / sqrt(n)
|GHZ_n> = ( |00...0> + |11...1> ) / sqrt(2)
```

GHZ is "all or nothing": if one qubit is lost or measured, the rest is no longer entangled. The W
state spreads its entanglement over every pair of qubits, so whatever remains after losing or
measuring a qubit is still entangled. That robustness is the W state's best-known property. The
price is that each single qubit is less strongly entangled with the rest than in GHZ.

## How it works

All quantities are computed exactly from state vectors with `qiskit.quantum_info`:

1. **Pairwise entanglement**: trace out all but two qubits and compute their concurrence
   (W: 2/n for every pair, GHZ: 0).
2. **Losing qubits**: trace out one or more qubits and compute the negativity across a split of the
   remaining qubits.
3. **Measuring a qubit**: measure one qubit and check whether the rest is still entangled; when the
   W state's qubit reads 0, the rest is exactly a W state on n − 1 qubits.
4. **The price**: entanglement entropy between one qubit and the rest (GHZ is higher).
5. All of this is repeated for a range of qubit counts.

## Run it

```bash
uv run python usecases/entanglement_robustness/main.py
```

Takes about a second.

| Option | Default | Meaning |
|---|---|---|
| `--qubits` | `5` | qubit count for the detailed views |
| `--sweep` | `3-8` | qubit counts compared |
| `--seed` | `7` | seed (kept for a uniform interface; the calculation is exact) |
| `--out` | `outputs/` | where the report is written |

`--shots` and `--noise` are accepted for a uniform interface but not used: nothing is sampled.

## What you get

The terminal prints one line of measures per qubit count, followed by the path of the report.

The report is written to `outputs/usecases/entanglement_robustness/index.html`:

| Section | What it shows |
|---|---|
| The two states | both preparation circuits |
| Who is entangled with whom | heatmaps of the concurrence of every qubit pair, W vs GHZ, and its comparison with 2/n |
| Losing qubits | entanglement left after tracing out 1, 2, ... qubits, and its trend with n |
| Measuring a qubit | what remains after measuring one qubit, outcome by outcome |
| The price | one-qubit entanglement entropy, where GHZ is ahead |
| All results | every number in a sortable table |

## Results with the default settings

- Every pair of qubits in W₅ has concurrence 0.40 (= 2/5); in GHZ₅ it is 0.
- After losing one qubit, W₅ keeps negativity 0.312; GHZ₅ keeps 0.
- After measuring one qubit, the rest of W₅ is still entangled with probability 80% (= 4/5), and
  the outcome 0 leaves exactly W₄ (fidelity 1.000); GHZ₅ is never left entangled.
- The price: one qubit of W₅ shares 0.72 bit of entanglement entropy with the rest, against 1.00
  bit for GHZ₅.

## Try next

```bash
uv run python usecases/entanglement_robustness/main.py --qubits 6 --sweep 3-10
```

Watch the pairwise concurrence fall as 2/n while the entanglement left after losing a qubit stays
positive.
