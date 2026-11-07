# Fair dice

Roll a fair die with any number of faces in one shot, without the rejection sampling a binary
register needs.

## The idea

Hadamard gates on m qubits give a uniform distribution over 2^m values. A 6-sided die built that
way uses 3 qubits (8 values) and has to throw away the values 6 and 7 and roll again. That is
rejection sampling: 2 out of 8 shots are wasted.

The W state on n qubits is uniform over exactly n outcomes for any n, so the position of its single
1 is the face. Every shot is a valid roll. The price is n qubits and some CX gates instead of
ceil(log2 n) qubits and no entangling gates at all, so neither die is better everywhere.

## How it works

1. **W die**: `w_state(n)` on n qubits, measured; the qubit that reads 1 is the face.
2. **Binary die**: Hadamard on ceil(log2 n) qubits, measured; the value is the face if it is
   below n, otherwise the shot is rejected.
3. Both dice are sampled; the W die is also sampled with gate noise. A noisy W-die shot that is not
   one-hot is detected and rejected, instead of silently producing a biased face.
4. A sweep over n compares qubits, CX gates and expected shots per accepted roll.

## Run it

```bash
uv run python usecases/fair_dice/main.py
```

Takes a few seconds.

| Option | Default | Meaning |
|---|---|---|
| `--sides` | `6` | number of faces n |
| `--sweep` | `2-20` | face counts compared in the cost section |
| `--method` | `tree` | W-state construction (`tree` or `linear`) |
| `--shots` | `4096` | shots per sampled circuit (one shot = one roll attempt) |
| `--noise` | `0.01` | CX depolarizing error of the noisy run (`0` disables it) |
| `--seed` | `7` | seed for the simulator and transpiler |
| `--out` | `outputs/` | where the report is written |

## What you get

The terminal prints how many shots each die accepted and the χ² p-value of its faces, followed by
the path of the report.

The report is written to `outputs/usecases/fair_dice/index.html`:

| Section | What it shows |
|---|---|
| Two ways to roll | both circuits and the first 40 rolls of the W die, drawn as dice |
| Faces rolled | the share of each face for both dice, and where all shots went (accepted or rejected) |
| What it costs | qubits, shots per accepted roll and CX gates for every n in the sweep |
| All results | every number in a sortable table |

## Results with the default settings

- The W die accepts all 4096 shots; the binary die accepts 74.6% (theory 6/8 = 75%).
- Both dice are fair once rejected shots are discarded (χ² p-values 0.94 and 0.30).
- The W die uses 6 qubits, the binary die 3.
- With CX error 0.01 the W die rejects 9.6% of shots as "not one-hot", so noise is detected.
- Across 2 to 20 faces, the binary die wastes up to 47% of its shots (at 17 faces) with 5 qubits and
  no CX gates, while the W die wastes none but needs 17 qubits and 32 CX gates.

## Try next

```bash
uv run python usecases/fair_dice/main.py --sides 12 --sweep 2-24
uv run python usecases/fair_dice/main.py --sides 6 --noise 0.02
```

Faces just above a power of two (5, 9, 17) are where the binary die wastes the most shots.
