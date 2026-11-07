import argparse
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def base_parser(doc: str | None) -> argparse.ArgumentParser:
    """
    Function that creates an argument parser with the options every use case shares

    Parameters
    ----------
    doc: str | None
        Module docstring shown by --help

    Returns
    -------
    parser: argparse.ArgumentParser
        Parser with --shots, --noise, --seed and --out
    """
    # Show the module docstring (with its examples) in --help
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Options shared by every use case
    parser.add_argument("--shots", type=int, default=4096, help="shots per sampled circuit (default: 4096)")
    parser.add_argument("--noise", type=float, default=0.01,
                        help="CX depolarizing error for noisy runs; 0 disables them (default: 0.01)")
    parser.add_argument("--seed", type=int, default=7, help="seed for simulators and transpiler (default: 7)")
    parser.add_argument("--out", type=Path, default=OUTPUTS, help="output root (default: outputs/)")

    return parser


def noise_model(p2: float) -> NoiseModel:
    """
    Function that builds a gate-only depolarizing noise model

    Parameters
    ----------
    p2: float
        Depolarizing error of each CX; single-qubit gates get p2 / 10

    Returns
    -------
    model: NoiseModel
        Noise model for AerSimulator
    """
    # Attach depolarizing errors to every two-qubit and single-qubit basis gate
    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    model.add_all_qubit_quantum_error(depolarizing_error(p2 / 10, 1), ["sx", "x"])

    return model


def sample(qc: QuantumCircuit, shots: int, seed: int, noise: float = 0.0) -> dict[str, int]:
    """
    Function that samples a circuit, measuring every qubit if it has no measurements

    Parameters
    ----------
    qc: QuantumCircuit
        Circuit to sample
    shots: int
        Number of shots
    seed: int
        Seed of the simulator and transpiler
    noise: float = 0.0
        CX depolarizing error; 0 means an ideal simulator

    Returns
    -------
    counts: dict[str, int]
        Bitstring (qubit 0 on the right) -> count
    """
    # Measure every qubit when the circuit has no classical bits
    circuit = qc.copy()
    if circuit.num_clbits == 0:
        circuit.measure_all()

    # Run ideally or with gate noise; noisy runs use the basis the noise model knows
    if noise > 0:
        sim = AerSimulator(noise_model=noise_model(noise))
        circuit = transpile(circuit, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1,
                            seed_transpiler=seed)
    else:
        sim = AerSimulator()
        circuit = transpile(circuit, sim, seed_transpiler=seed)
    counts = sim.run(circuit, shots=shots, seed_simulator=seed).result().get_counts()

    return counts


def one_hot_index(bits: str) -> int | None:
    """
    Function that returns which qubit is |1> in a one-hot bitstring

    Parameters
    ----------
    bits: str
        Bitstring with qubit 0 on the right (Qiskit order); spaces are ignored

    Returns
    -------
    index: int | None
        Index of the only '1', or None when the string is not one-hot
    """
    # Qiskit prints qubit 0 last, so reverse before searching
    clean = bits.replace(" ", "")
    index = clean[::-1].index("1") if clean.count("1") == 1 else None

    return index
