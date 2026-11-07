# OneHot

Prepare W states with Qiskit, and explore what they are good for.

The n-qubit **W state** is the equal superposition of every one-hot basis state:

```
|W_n> = ( |10...0> + |01...0> + ... + |00...1> ) / sqrt(n)
```

Measuring it always yields exactly one `1`, at a uniformly random position. This
repository provides a small library that builds W-state circuits, an experiment
that compares the two constructions at any size, and a set of use cases that
show circuits built on top of the W state. Experiments and use cases write
interactive HTML reports.

## Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [Using the library](#using-the-library)
- [Two constructions](#two-constructions)
- [Experiments](#experiments)
- [Use cases](#use-cases)
- [Reports](#reports)
- [Project layout](#project-layout)
- [Maintaining the project with uv](#maintaining-the-project-with-uv)
- [License](#license)

## Requirements

- [uv](https://docs.astral.sh/uv/) (manages Python, the virtual environment and dependencies)
- Python 3.10 or later for the library; the experiments and use cases use Python 3.12
  (pinned in `.python-version`, installed by uv automatically)

Install uv if you do not have it yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# or: brew install uv
```

## Setup

```bash
git clone https://github.com/victor-von-pooh/OneHot.git
cd OneHot
uv sync
```

`uv sync` creates `.venv/`, installs the library in editable mode and installs every
dependency group (`experiments`, `usecases`) at the versions recorded in `uv.lock`.
Run anything inside that environment with `uv run`:

```bash
uv run python -c "from onehot import w_state; print(w_state(4))"
```

### Using OneHot from another project

Only the library is installed this way; it depends on `qiskit` alone.

```bash
uv add git+https://github.com/victor-von-pooh/OneHot.git            # library only
uv add "onehot[report] @ git+https://github.com/victor-von-pooh/OneHot.git"  # + HTML report helpers
```

## Using the library

```python
from qiskit.quantum_info import Statevector
from onehot import w_state

qc = w_state(5)                     # binary-tree construction (default)
qc_line = w_state(5, method="linear")

print(Statevector(qc).probabilities_dict())
# {'00001': 0.2, '00010': 0.2, '00100': 0.2, '01000': 0.2, '10000': 0.2}
```

`w_state(num_qubits, method="tree" | "linear")` returns a `QuantumCircuit` without
measurements, so it can be composed into larger circuits:

```python
from qiskit import QuantumCircuit

big = QuantumCircuit(9)
big.compose(w_state(3), qubits=[0, 1, 2], inplace=True)
big.compose(w_state(3), qubits=[3, 4, 5], inplace=True)
big.compose(w_state(3), qubits=[6, 7, 8], inplace=True)
```

## Two constructions

Both constructions put one excitation on qubit 0 and then split it n − 1 times with a
controlled-RY followed by a CNOT. They use the same gates and differ only in the order
of the splits.

| | `"linear"` | `"tree"` (default) |
|---|---|---|
| Idea | pass the excitation down a chain | halve blocks of qubits level by level |
| Depth | O(n) | O(log n) |
| CX count | same | same |
| Two-qubit gates act on | neighbours (i, i + 1) only | distant qubits too |
| Best for | hardware with limited connectivity (no SWAPs on a line) | simulators and all-to-all hardware |

The [experiment](#experiments) measures this trade-off, including routing onto a line and
onto IBM's heavy-hex lattice.

## Experiments

```bash
uv run python experiments/w_state_scaling.py                     # n = 1..16, both methods
uv run python experiments/w_state_scaling.py --n 2-24 --noise 0.005
uv run python experiments/w_state_scaling.py --n 3,5,8,13 --method tree
uv run python experiments/w_state_scaling.py --help              # every option
```

| Option | Default | Meaning |
|---|---|---|
| `--n` | `1-16` | qubit counts, as ranges and/or a comma-separated list |
| `--method` | `both` | `linear`, `tree` or `both` |
| `--shots` | `4096` | shots per noisy simulation |
| `--noise` | `0.01` | CX depolarizing error (single-qubit gates get a tenth) |
| `--noise-max-n` | `12` | largest n simulated with noise |
| `--detail-n` | min(max n, 8) | n used for the heatmaps, circuit diagrams and distribution |
| `--seed` | `7` | seed for the transpiler and simulator |
| `--out` | `outputs/` | where reports are written |

The report covers correctness (fidelity with the exact W state), how the excitation spreads
stage by stage, depth and CX count, the cost of routing onto hardware connectivity, and how
much of the state survives in the one-hot subspace under noise.

## Use cases

Each script builds a circuit on top of `w_state` and writes its own report.

| Script | What it shows |
|---|---|
| `usecases/leader_election.py` | Elect exactly one of n parties fairly |
| `usecases/fair_dice.py` | Roll an n-sided die (n need not be a power of two) without rejection |
| `usecases/entanglement_robustness.py` | Why W-state entanglement survives losing a qubit while GHZ does not |
| `usecases/grover_one_hot.py` | Grover search restricted to the n one-hot states |
| `usecases/single_excitation_walk.py` | One excitation moving through a chain: quantum walk and state transfer |
| `usecases/one_hot_encoding.py` | One W state per row gives every "exactly one per row" assignment |
| `usecases/graph_coloring_qaoa.py` | Graph colouring with QAOA, W initial states and an XY mixer |
| `usecases/tsp_qaoa.py` | The travelling salesman problem with QAOA and a W-state initial state |

```bash
uv run python usecases/leader_election.py
uv run python usecases/tsp_qaoa.py --help        # options differ per script
```

Every use case accepts `--shots`, `--noise`, `--seed` and `--out`. To generate all reports:

```bash
for f in experiments/*.py usecases/[!_]*.py; do uv run python "$f"; done
```

## Reports

Reports are written under `outputs/` (ignored by git):

```
outputs/
├── index.html                          # overview of every generated report
├── experiments/w_state_scaling/index.html
└── usecases/<name>/index.html
```

Open `outputs/index.html` in a browser (`open outputs/index.html` on macOS). Charts are
interactive (hover, zoom, click legend entries to hide series), tables sort by clicking a
column header, and the button in the top-right corner switches between light and dark mode.
Plotly and KaTeX are loaded from a CDN, so an internet connection is needed to view them.

## Project layout

```
OneHot/
├── pyproject.toml          # package metadata, dependencies, dependency groups
├── uv.lock                 # exact versions used for the experiments and use cases
├── .python-version         # Python version uv uses for this project
├── src/onehot/
│   ├── w_state.py          # the library: w_state(n, method)
│   └── report/             # HTML report helpers (optional extra "report")
├── experiments/
│   └── w_state_scaling.py
└── usecases/
    ├── _common.py          # shared CLI options, sampling and noise helpers
    ├── _qaoa.py            # shared QAOA helpers for the optimisation use cases
    └── *.py                # one script per use case
```

## Maintaining the project with uv

| Task | Command |
|---|---|
| Install / update the environment after pulling | `uv sync` |
| Add a library dependency | `uv add numpy` |
| Add a dependency only for the use cases | `uv add --group usecases networkx` |
| Remove a dependency | `uv remove numpy` |
| Upgrade every dependency to the latest allowed version | `uv lock --upgrade && uv sync` |
| Upgrade one package | `uv lock --upgrade-package qiskit && uv sync` |
| Bump the version | `uv version --bump minor` (or `patch` / `major`) |
| Run a script | `uv run python <script>` |

`uv add`, `uv remove` and `uv version` edit `pyproject.toml` and `uv.lock` for you; commit
both files. The library itself only declares `qiskit>=2.6`; the use cases and experiments
additionally use `qiskit-aer`, `qiskit-optimization`, `scipy`, `plotly`, `matplotlib` and
`pylatexenc`.

## License

[MIT](LICENSE) © 2026 Hiroki Akita
