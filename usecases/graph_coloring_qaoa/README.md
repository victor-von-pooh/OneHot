# Graph colouring with QAOA

Colour a graph with QAOA using one-hot variables; one W state per vertex plus an XY mixer keeps
every amplitude on valid colourings.

## The idea

Each vertex v gets one qubit per colour c, and x_{v,c} = 1 means "v has colour c". A valid
colouring is one-hot inside every vertex, and the cost is the number of edges whose two ends share
a colour. QAOA alternates a cost layer and a mixer layer, and the choice of starting state and
mixer decides whether the search stays among valid colourings.

Three settings are compared:

| | Start | Mixer | One-hot constraint |
|---|---|---|---|
| (a) | Hadamard on every qubit | X | penalty term |
| (b) | one W state per vertex | X | penalty term |
| (c) | one W state per vertex | XY ring inside each vertex | not needed: the mixer never leaves the one-hot subspace |

## How it works

1. Build the cost (plus the one-hot penalty for (a) and (b)) as a diagonal QUBO.
2. For each setting and each depth p = 1..P, optimise the QAOA angles with COBYLA from several
   random starts and from the best angles of depth p − 1. The expectation is computed exactly
   from the state vector.
3. Report the probability of a proper colouring, the probability of a valid one-hot state and the
   expected cost.
4. Build the optimised circuits, sample them ideally and with noise, and show the most likely
   colourings.

The shared QAOA helpers live in [`usecases/shared/qaoa.py`](../shared/qaoa.py).

## Run it

```bash
uv run python usecases/graph_coloring_qaoa/main.py
```

Takes about 15 seconds.

| Option | Default | Meaning |
|---|---|---|
| `--graph` | `paw` | graph to colour: `triangle`, `paw` (triangle + one pendant vertex) or `cycle4` |
| `--colors` | `3` | number of colours K |
| `--depth` | `3` | largest QAOA depth p |
| `--restarts` | `4` | random starts per depth |
| `--penalty` | `1.0` | weight of the one-hot penalty in (a) and (b) |
| `--shots` | `4096` | shots per sampled circuit |
| `--noise` | `0.01` | CX depolarizing error of the noisy runs (`0` disables them) |
| `--seed` | `7` | seed for the optimiser, simulator and transpiler |
| `--out` | `outputs/` | where the report is written |

## What you get

The terminal prints, for every setting and depth, the probability of a proper colouring and of a
valid one-hot state, followed by the path of the report.

The report is written to `outputs/usecases/graph_coloring_qaoa/index.html`:

| Section | What it shows |
|---|---|
| Problem and answer | the graph with the most likely colouring |
| Circuits | the W-state start and one QAOA layer |
| Results versus depth | proper-colouring and valid-state probabilities for p = 1..P |
| Optimisation | how the expected cost converged |
| Sampling the final circuits | sampled results, ideal and noisy, with compiled CX counts |
| Most likely outcomes | the top bitstrings decoded into colourings |

## Results with the default settings

Paw graph, 3 colours, 12 qubits, at p = 3:

| Setting | P(proper colouring) | P(valid one-hot) |
|---|---|---|
| (a) Hadamard + X mixer + penalty | 18.0% | 45% |
| (b) W + X mixer + penalty | 14.8% | 100% |
| (c) W + XY mixer | 81.1% | 100% |

Setting (b) lands exactly on the share of proper colourings among all valid ones (14.8%): the
optimiser switches the X mixer off because any mixing would leave the one-hot subspace. The W start
pays off only together with a mixer that respects it, as in (c). With CX error 0.01, (c) drops to
20.5%.

## Try next

```bash
uv run python usecases/graph_coloring_qaoa/main.py --graph cycle4 --depth 4
uv run python usecases/graph_coloring_qaoa/main.py --graph triangle --colors 3 --restarts 6
```
