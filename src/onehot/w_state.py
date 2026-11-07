from typing import Literal

import numpy as np
from qiskit import QuantumCircuit

Method = Literal["tree", "linear"]

__all__ = ["Method", "w_state"]


def w_state(num_qubits: int, method: Method = "tree") -> QuantumCircuit:
    """
    Function that builds a circuit preparing the W state from |0...0>

    The n-qubit W state is the equal superposition of every one-hot basis
    state, (|10...0> + |01...0> + ... + |00...1>) / sqrt(n). Both methods put
    a single excitation on qubit 0 and then split it n - 1 times, so they use
    the same gates and differ only in the order of the splits.

    Parameters
    ----------
    num_qubits: int
        Number of qubits n (n >= 1). For n = 1 the W state is |1>
    method: Method = "tree"
        "tree" halves blocks of qubits level by level: O(log n) depth, but
        two-qubit gates may connect distant qubits.
        "linear" passes the excitation down a chain: O(n) depth, but every
        two-qubit gate acts on neighbouring qubits (i, i + 1)

    Returns
    -------
    qc: QuantumCircuit
        Circuit on num_qubits qubits that prepares the W state (no measurements)
    """
    # Reject sizes and methods that cannot be built
    if num_qubits < 1:
        raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
    if method not in ("tree", "linear"):
        raise ValueError(f"method must be 'tree' or 'linear', got {method!r}")

    # Start from |10...0>, where the single excitation sits on qubit 0
    qc = QuantumCircuit(num_qubits, name=f"W_{num_qubits}")
    qc.x(0)

    # Spread the excitation over all qubits in the requested order
    if method == "linear":
        build_linear(qc)
    else:
        build_tree(qc)

    return qc


def split_excitation(qc: QuantumCircuit, src: int, dst: int, keep: float) -> None:
    """
    Function that shares the excitation on src with dst, which must be in |0>

    On the branch where src is |1>, the state becomes
    sqrt(keep) |1>_src |0>_dst + sqrt(1 - keep) |0>_src |1>_dst.
    On the branch where src is |0>, nothing happens.

    Parameters
    ----------
    qc: QuantumCircuit
        Circuit to append the gates to
    src: int
        Qubit that may hold the excitation
    dst: int
        Qubit in |0> that receives part of the excitation
    keep: float
        Share of the excitation (probability) that stays on src

    Returns
    -------
    None
    """
    # Choose the angle so that cos(theta / 2) = sqrt(keep)
    theta = 2 * np.arccos(np.sqrt(keep))

    # Rotate dst only when src is excited: |1>|0> -> cos|1>|0> + sin|1>|1>
    qc.cry(theta, src, dst)

    # Clear src on the branch where dst took the excitation: |1>|1> -> |0>|1>
    qc.cx(dst, src)


def build_linear(qc: QuantumCircuit) -> None:
    """
    Function that spreads the excitation along a chain of neighbouring qubits

    Parameters
    ----------
    qc: QuantumCircuit
        Circuit whose excitation sits on qubit 0

    Returns
    -------
    None
    """
    n = qc.num_qubits

    # The excitation reaches qubit i with probability (n - i) / n, so keeping
    # 1 / (n - i) of it leaves exactly 1 / n on qubit i and passes the rest on
    for i in range(n - 1):
        split_excitation(qc, i, i + 1, keep=1 / (n - i))


def build_tree(qc: QuantumCircuit) -> None:
    """
    Function that spreads the excitation by halving blocks of qubits

    A block [start, end) is owned by its first qubit, which carries the whole
    share of the block. Splitting the block at mid hands the share of the right
    half to qubit mid, the owner of the right half. Splits on one level touch
    disjoint qubits, so they run in parallel and the depth is O(log n).

    Parameters
    ----------
    qc: QuantumCircuit
        Circuit whose excitation sits on qubit 0

    Returns
    -------
    None
    """
    # Begin with one block that covers every qubit
    blocks = [(0, qc.num_qubits)]

    # Halve every block of the current level until all blocks are single qubits
    while blocks:
        next_level = []
        for start, end in blocks:
            size = end - start
            if size == 1:
                continue

            # Keep the left half's share on start and move the rest to mid
            mid = start + size // 2
            split_excitation(qc, start, mid, keep=(mid - start) / size)
            next_level += [(start, mid), (mid, end)]

        blocks = next_level
