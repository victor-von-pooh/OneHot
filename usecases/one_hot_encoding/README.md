# One-hot encoding of assignments

One W state per row prepares every "exactly one choice per item" assignment at once, so every shot
is a valid assignment, unlike a Hadamard start.

## The idea

Many combinatorial problems give each of k items exactly one of m choices: tasks to machines,
vertices to colours, cities to time slots. With one-hot encoding this is a k × m grid of qubits in
which every row contains exactly one 1.

Putting a W_m state on every row prepares the uniform superposition of all m^k valid assignments
in one step. A Hadamard on every qubit instead spreads the amplitude over all 2^(km) bitstrings,
and only (m / 2^m)^k of them are valid. This makes the W-state rows a natural starting state for
optimisation algorithms such as QAOA (see the graph colouring and TSP use cases).

## How it works

1. Build a k × m grid of qubits; row i is item i, column j is choice j.
2. Put `w_state(m)` on every row. All rows are prepared in parallel, so the whole circuit is as
   deep as a single W_m.
3. Sample the circuit, decode every shot into an assignment ("item i → choice j"), and test that
   all m^k assignments are equally likely.
4. Compare the valid share with a Hadamard start over a grid of (k, m), and sample both ideally
   and with noise.

The W rows enforce "exactly one per row" only. They do not enforce "each choice used at most once"
(permutations); the report shows how often that happens by chance.

## Run it

```bash
uv run python usecases/one_hot_encoding/main.py
```

Takes about a second.

| Option | Default | Meaning |
|---|---|---|
| `--rows` | `3` | number of items k (one W state each) |
| `--cols` | `3` | number of choices m per item |
| `--grid-rows` | `6` | largest k in the comparison grid |
| `--grid-cols` | `8` | largest m in the comparison grid |
| `--method` | `tree` | W-state construction (`tree` or `linear`) |
| `--shots` | `4096` | shots per sampled circuit |
| `--noise` | `0.01` | CX depolarizing error of the noisy run (`0` disables it) |
| `--seed` | `7` | seed for the simulator and transpiler |
| `--out` | `outputs/` | where the report is written |

## What you get

The terminal prints the valid share for the W rows (ideal and noisy) and for the Hadamard start,
followed by the path of the report.

The report is written to `outputs/usecases/one_hot_encoding/index.html`:

| Section | What it shows |
|---|---|
| The circuit | one W state per row of the grid |
| Sampled assignments | how often every assignment was drawn, with a χ² uniformity test |
| Compared with a Hadamard start | valid share over the (k, m) grid, ideal and noisy |
| What it does not enforce | how many sampled assignments happen to use every choice at most once |

## Results with the default settings

3 items, 3 choices each (9 qubits):

- 100% of ideal shots are valid assignments; with CX error 0.01, 89.8% are.
- All 27 assignments appear, uniformly (χ² p-value 0.72).
- A Hadamard start gives a valid assignment in 5.13% of shots (theory 5.27%).
- The circuit depth is 18, the same as a single W₃.
- 21.8% of the sampled assignments also use every choice once (theory 22.2%).

## Try next

```bash
uv run python usecases/one_hot_encoding/main.py --rows 4 --cols 3
uv run python usecases/one_hot_encoding/main.py --rows 3 --cols 5 --method linear
```

The valid share of the Hadamard start shrinks exponentially with k, while the W rows stay at 100%.
