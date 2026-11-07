# Grover search over one-hot states

Amplitude amplification with the W state as the starting superposition: searching n items needs
about √n rounds instead of √(2ⁿ).

## The idea

Grover search does not need a Hadamard start. Amplitude amplification works with any circuit A
that prepares a superposition over the candidates. If the n candidates are encoded as one-hot
states (item i = "only qubit i is 1"), then A = W_n prepares exactly those n states, and

- the search space has n states instead of all 2^n bitstrings on the same qubits,
- the oracle is a single Z gate on each marked qubit, because Z_i flips the sign of exactly the
  one-hot state of item i.

The optimal number of rounds drops from about (π/4)·√(2^n) to about (π/4)·√n.

## How it works

1. **Preparation**: A = `w_state(n)`.
2. **Oracle**: Z on every marked qubit.
3. **Diffusion**: A · S₀ · A†, where S₀ flips the sign of |0...0⟩ (X on every qubit, a
   multi-controlled Z, X again).
4. Repeat oracle + diffusion k times. The success probability is sin²((2k + 1)θ) with
   sin θ = √(m/n) for m marked items; the script checks the simulation against this formula and
   picks the first peak.
5. The same search is run as standard Grover over 2^n states (Hadamard start, oracle on the full
   one-hot pattern) for comparison, ideally and with noise.

## Run it

```bash
uv run python usecases/grover_one_hot/main.py
```

Takes a few seconds.

| Option | Default | Meaning |
|---|---|---|
| `--items` | `8` | number of items n (= qubits) |
| `--marked` | `3` | marked item indices, e.g. `3` or `3,5` |
| `--method` | `tree` | W-state construction (`tree` or `linear`) |
| `--noise` | `0.001` | CX depolarizing error of the noisy runs (`0` disables them) |
| `--noisy-max-cx` | `6000` | skip the noisy runs of circuits with more CX gates than this |
| `--shots` | `4096` | shots per sampled circuit |
| `--seed` | `7` | seed for the simulator and transpiler |
| `--out` | `outputs/` | where the report is written |

The default noise is lower than in the other use cases, because at 0.01 both searches drown in
noise and nothing can be compared.

## What you get

The terminal prints the best number of rounds and its success probability for both searches,
followed by the path of the report.

The report is written to `outputs/usecases/grover_one_hot/index.html`:

| Section | What it shows |
|---|---|
| The circuit | one round of the W-state search |
| Success probability per round | simulated vs theoretical success probability for every k, both searches |
| Sampled outcomes | measured counts at the optimal k, ideal and noisy |
| Cost comparison | rounds, CX gates and depth of both searches |
| Per-round values | every number in a sortable table |

## Results with the default settings

8 items, item 3 marked:

| | W-state search | standard Grover (2⁸ = 256 states) |
|---|---|---|
| optimal rounds | 2 | 12 |
| ideal success | 94.5% | 99.99% |
| CX gates | 445 | 4,080 |
| success with CX error 0.001 | 65.5% | 3.5% |

The simulated success probabilities match the formula to within 1e-14.

## Try next

```bash
uv run python usecases/grover_one_hot/main.py --items 12 --marked 3,5
uv run python usecases/grover_one_hot/main.py --items 6 --marked 0 --method linear
```

With several marked items, fewer rounds are needed (sin θ = √(m/n)).
