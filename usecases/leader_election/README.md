# Leader election

Pick exactly one of n parties, uniformly at random, in a single round by measuring a shared W state.

## The idea

n parties want to elect one leader fairly. Each party holds one qubit of the W state

```
|W_n> = ( |10...0> + |01...0> + ... + |00...1> ) / sqrt(n)
```

and measures it. Every outcome of the W state contains exactly one `1`, so exactly one party
reads 1 and becomes the leader, and every party wins with probability 1/n. One round is always
enough.

The classical baseline is "everyone flips a fair coin and we hope exactly one gets heads". That
succeeds only with probability n / 2^n per round, so it needs more and more rounds as n grows.

## How it works

1. `w_state(n)` prepares the W state on n qubits (one qubit per party).
2. Every qubit is measured; the index of the qubit that reads 1 is the leader.
3. The same circuit is sampled ideally and with gate noise. With noise some shots contain zero or
   several 1s; those rounds fail and must be repeated, but they are detected, never silently wrong.
4. The coin-flip baseline puts a Hadamard on every qubit instead.

## Run it

```bash
uv run python usecases/leader_election/main.py
```

Takes a few seconds.

| Option | Default | Meaning |
|---|---|---|
| `--parties` | `5` | number of parties n for the detailed views |
| `--sweep` | `2-12` | party counts compared in the scaling section |
| `--method` | `tree` | W-state construction (`tree` or `linear`) |
| `--shots` | `4096` | shots per sampled circuit (one shot = one election round) |
| `--noise` | `0.01` | CX depolarizing error of the noisy runs (`0` disables them) |
| `--seed` | `7` | seed for the simulator and transpiler |
| `--out` | `outputs/` | where the report is written |

## What you get

The terminal prints, for every n in the sweep, the share of successful rounds for the ideal W
state, the noisy W state and the coin flips, followed by the path of the report.

The report is written to `outputs/usecases/leader_election/index.html`:

| Section | What it shows |
|---|---|
| The protocol | the circuit every party shares |
| Who won | how often each party became leader, against the fair share 1/n, with a χ² test |
| More parties | success per round and expected rounds per election as n grows, W state vs coin flips, ideal and noisy |
| All results | every number in a sortable table |

## Results with the default settings

- Every ideal round elects exactly one leader (100%), and the leaders are uniform (χ² p-value 0.81).
- With CX error 0.01, 92.5% of rounds still elect exactly one leader.
- Coin flips succeed in 15.6% of rounds for 5 parties and in 0.2% for 12 parties, i.e. about 341
  rounds per election, while the W state needs 1 round (about 1.27 with noise).

## Try next

```bash
uv run python usecases/leader_election/main.py --parties 7 --sweep 2-14
uv run python usecases/leader_election/main.py --noise 0.005 --method linear
```

Compare `--method tree` and `--method linear`: the leader distribution is the same, but the noisy
success rate depends on the circuit each construction produces.
