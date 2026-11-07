# Single-excitation walk

One excitation hopping between qubits never leaves the one-hot subspace: the W state stays still on
a complete graph, spreads on a line, and engineered couplings move a state from one end of a chain
to the other.

## The idea

The XY Hamiltonian

```
H = Σ_(i,j) J_ij (X_i X_j + Y_i Y_j) / 2
```

moves an excitation from qubit i to qubit j without creating or destroying one. A one-hot state
therefore stays one-hot for all time, and inside the one-hot subspace H is simply the weighted
adjacency matrix of the coupling graph: this is a continuous-time quantum walk of one particle.

The W state is the natural reference point. It is the stationary state of the complete graph, it
spreads on a line, and the one-hot structure makes errors detectable: any shot that is not
one-hot must have been hit by noise.

## How it works

Four scenarios are simulated exactly (matrix exponential) and as a Trotter circuit made of
`XXPlusYYGate`s:

1. **Local excitation on a line**: start with only qubit 0 excited and watch it spread.
2. **W state on a line**: start from `w_state(n)` with nearest-neighbour couplings.
3. **W state on a complete graph**: every pair coupled; the W state does not move.
4. **Perfect state transfer**: couplings J_k = √(k(N − k)) move an excitation from the first qubit
   to the last at t = π/2, compared with a uniform chain.

The Trotter circuits are also sampled with gate noise, and discarding non-one-hot shots
(post-selection) is shown to bring the result closer to the exact one.

## Run it

```bash
uv run python usecases/single_excitation_walk/main.py
```

Takes a few seconds.

| Option | Default | Meaning |
|---|---|---|
| `--sites` | `8` | number of sites (= qubits) |
| `--time` | `4.0` | last simulated time |
| `--dt` | `0.05` | Trotter step of the noiseless check |
| `--noisy-dt` | `0.25` | Trotter step of the noisy runs (coarser, so the circuits stay short) |
| `--method` | `tree` | W-state construction (`tree` or `linear`) |
| `--shots` | `4096` | shots per sampled circuit |
| `--noise` | `0.01` | CX depolarizing error of the noisy runs (`0` disables them) |
| `--seed` | `7` | seed for the simulator and transpiler |
| `--out` | `outputs/` | where the report is written |

## What you get

The terminal prints, for each scenario, the lowest one-hot probability over time and the fidelity
of the Trotter circuit with the exact state at the last time, followed by the path of the report.

The report is written to `outputs/usecases/single_excitation_walk/index.html`:

| Section | What it shows |
|---|---|
| Four walks | heatmaps of "probability that site k is excited" over time for the four scenarios |
| Step through time | the same distributions with a time slider |
| Perfect state transfer | arrival probability at the far end, engineered vs uniform couplings |
| As a circuit | the Trotter circuit and its fidelity with the exact evolution |
| With noise: errors become detectable | one-hot share over time under noise, with and without post-selection |
| Summary | every number in a sortable table |

## Results with the default settings

- In all four scenarios the one-hot probability stays at 1.000000000000 throughout.
- On the complete graph the W state deviates from 1/n by at most 7.4e-16: it does not move.
- The engineered chain transfers the excitation end to end with probability 1.0000 at t = π/2; the
  uniform chain reaches at most 0.359 within t ≤ 4.
- The Trotter circuit matches the exact evolution with fidelity at least 0.9923.
- With CX error 0.01 the one-hot share of the local walk falls to 21.6% at t = 4, and
  post-selecting the one-hot shots reduces the error at every time step.

## Try next

```bash
uv run python usecases/single_excitation_walk/main.py --sites 6 --time 6
uv run python usecases/single_excitation_walk/main.py --dt 0.02 --noise 0.005
```

A smaller `--dt` makes the Trotter circuit more accurate but deeper, so it also suffers more from
noise.
