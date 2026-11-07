# Travelling salesman with QAOA

Solve a small travelling salesman problem (TSP) with QAOA using one-hot city/time variables,
comparing a uniform start, a W-state start, and a W-state start with an XY mixer.

## The idea

x_{i,t} = 1 means "city i is visited at time t", so a tour is a permutation matrix: one-hot in
every row (each city once) and in every column (one city per time). City 0 is fixed at time 0 and
every other city starts in a W state over the times 1..N−1, which leaves (N−1)² qubits. The W
start therefore satisfies "each city once" from the beginning; only "one city per time" is left to
the optimisation.

Three settings are compared:

| | Start | Mixer | Constraints handled by |
|---|---|---|---|
| (a) | Hadamard on every qubit | X | row and column penalties |
| (b) | one W state per city | X | row and column penalties |
| (c) | one W state per city | XY ring inside each city row | column penalty only (the mixer keeps rows one-hot) |

The same W-start problem is also solved through qiskit-optimization (`QuadraticProgram`, `QAOA`
with a `StatevectorSampler` and the W initial state, `MinimumEigenOptimizer`), the standard Qiskit
route to the same calculation.

## How it works

1. Generate N random cities (or read a file), compute the distance matrix and brute-force the
   optimal tour for reference.
2. Build the tour length plus penalties as a QUBO on the (N−1)² free qubits.
3. For each setting and each depth p = 1..P, optimise the QAOA angles with COBYLA from several
   random starts and from the best angles of depth p − 1, using exact state-vector expectations.
4. Report the probability of the optimal tour, the probability of a valid tour, the share of valid
   outcomes that are optimal and the approximation ratio.
5. Sample the optimised circuits ideally and with noise, and decode the most likely bitstrings
   into tours.
6. Solve the W-start problem once more with qiskit-optimization.

The shared QAOA helpers live in [`usecases/shared/qaoa.py`](../shared/qaoa.py).

## Run it

```bash
uv run python usecases/tsp_qaoa/main.py
```

Takes about 2 minutes with 5 cities (16 qubits). With `--cities 4` (9 qubits) it takes about
15 seconds.

| Option | Default | Meaning |
|---|---|---|
| `--cities` | `5` | number of random cities N (≥ 3) |
| `--input` | — | read the problem from a file instead (see below) |
| `--depth` | `3` | largest QAOA depth p |
| `--restarts` | `3` | random starts per depth |
| `--penalty-scale` | `1.5` | penalty weight as a multiple of the longest leg |
| `--shots` | `4096` | shots per sampled circuit |
| `--noise` | `0.01` | CX depolarizing error of the noisy runs (`0` disables them) |
| `--seed` | `7` | seed for the cities, optimiser, simulator and transpiler |
| `--out` | `outputs/` | where the report is written |

### Input file format

```
V E
s t d
s t d
...
```

The first line gives the number of cities V and of directed edges E; each following line is an
edge from city s to city t with distance d. A missing edge is treated as a long detour (ten times
the longest given edge), so tours avoid it.

## What you get

The terminal prints, for every setting and depth, the probability of the optimal tour and of a
valid tour, then the tour found by qiskit-optimization, followed by the path of the report.

The report is written to `outputs/usecases/tsp_qaoa/index.html`:

| Section | What it shows |
|---|---|
| Cities and tours | the cities with the optimal tour and the most likely sampled tour, plus the distance matrix |
| Circuits | the W-state start and one QAOA layer |
| Results versus depth | optimal-tour, valid-tour and approximation-ratio curves for p = 1..P |
| Optimisation | how the expected cost converged |
| With qiskit-optimization | the tour and timing of the qiskit-optimization run |
| Sampling the final circuits | sampled results, ideal and noisy |
| Most likely outcomes | the top bitstrings decoded into tours |

## Results with the default settings

5 random cities (seed 7), 16 qubits, at p = 3. Picking a valid tour at random finds the optimum
8.3% of the time (2 of 24 tours).

| Setting | Valid outcomes that are optimal | Outcomes that are valid tours |
|---|---|---|
| (a) Hadamard + X mixer + penalties | 6.8% | 0.1% |
| (b) W + X mixer + penalties | 8.3% | 9.4% |
| (c) W + XY mixer + column penalty | 11.8% | 18.5% |

- The W start raises the share of valid tours from 0.1% to 9.4%, but with the X mixer the
  optimiser cannot improve on it (b equals random guessing among valid tours).
- With the XY mixer, deeper circuits help: the valid share grows from 10.8% to 18.5% and the
  optimal share to 11.8%. The gain is real but modest at this size.
- qiskit-optimization returns the optimal tour 0 → 2 → 3 → 1 → 4 → 0 (length 23.06).

Few outcomes are valid tours, so the sampled and noisy rates rest on a small number of shots; the
report says so next to those numbers.

## Try next

```bash
uv run python usecases/tsp_qaoa/main.py --cities 4 --depth 4 --restarts 6
uv run python usecases/tsp_qaoa/main.py --input path/to/tsp.txt
```

With 4 cities random guessing already finds the optimum a third of the time, so differences between
the settings are easier to see with 5 cities.
