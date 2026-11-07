# Use cases

Eight circuits built on top of `onehot.w_state`, each in its own directory with a `main.py` that
runs it and a `README.md` that explains it. Every script writes an interactive HTML report to
`outputs/usecases/<name>/index.html` and adds it to `outputs/index.html`.

| Use case | What it shows | Runtime |
|---|---|---|
| [leader_election](leader_election/) | Elect exactly one of n parties fairly in a single round | seconds |
| [fair_dice](fair_dice/) | Roll an n-sided die (n need not be a power of two) without rejection | seconds |
| [entanglement_robustness](entanglement_robustness/) | Why W-state entanglement survives losing a qubit while GHZ does not | ~1 s |
| [grover_one_hot](grover_one_hot/) | Grover search restricted to the n one-hot states | seconds |
| [single_excitation_walk](single_excitation_walk/) | One excitation moving through qubits: quantum walk and state transfer | seconds |
| [one_hot_encoding](one_hot_encoding/) | One W state per row gives every "exactly one per row" assignment | ~1 s |
| [graph_coloring_qaoa](graph_coloring_qaoa/) | Graph colouring with QAOA, W-state starts and an XY mixer | ~15 s |
| [tsp_qaoa](tsp_qaoa/) | The travelling salesman problem with QAOA and W-state starts | ~2 min |

They are ordered from simple to advanced: the first three use the W state directly, the next three
use it as a building block, and the last two use it as the starting state of an optimisation.

## Running

Run from the repository root:

```bash
uv run python usecases/leader_election/main.py
uv run python usecases/leader_election/main.py --help     # options of this use case
```

Run all of them (works in both bash and zsh):

```bash
for f in usecases/*/main.py; do uv run python "$f"; done
```

## Options shared by every use case

| Option | Default | Meaning |
|---|---|---|
| `--shots` | `4096` | shots per sampled circuit |
| `--noise` | `0.01` | CX depolarizing error of the noisy runs; `0` disables them |
| `--seed` | `7` | seed for simulators, transpiler and optimisers |
| `--out` | `outputs/` | root directory of the reports |

Each use case adds its own options; see its README.

## Layout

```
usecases/
├── README.md                 # this file
├── shared/
│   ├── common.py             # shared options, sampling, noise model, one-hot decoding
│   └── qaoa.py               # QUBO and QAOA helpers for the optimisation use cases
└── <name>/
    ├── README.md             # what the use case is and what running it produces
    └── main.py               # the script
```

To add a use case, copy one of the directories, rename it, and give its `Page` a new `slug`; the
report then appears on `outputs/index.html` automatically.
