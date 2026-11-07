from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize


@dataclass
class Qubo:
    """
    Class that holds a quadratic cost over binary variables x_0 ... x_{n-1}

    Parameters
    ----------
    num_qubits: int
        Number of binary variables (one qubit each)
    constant: float
        Constant offset
    linear: dict[int, float]
        Variable index -> coefficient of x_i
    quadratic: dict[tuple[int, int], float]
        Pair (i, j) with i < j -> coefficient of x_i x_j
    """

    num_qubits: int
    constant: float = 0.0
    linear: dict[int, float] = field(default_factory=dict)
    quadratic: dict[tuple[int, int], float] = field(default_factory=dict)

    def add_linear(self, i: int, value: float) -> None:
        """
        Function that adds value * x_i

        Parameters
        ----------
        i: int
            Variable index
        value: float
            Coefficient to add

        Returns
        -------
        None
        """
        self.linear[i] = self.linear.get(i, 0.0) + value

    def add_quadratic(self, i: int, j: int, value: float) -> None:
        """
        Function that adds value * x_i x_j (x_i x_i is folded into x_i)

        Parameters
        ----------
        i: int
            First variable index
        j: int
            Second variable index
        value: float
            Coefficient to add

        Returns
        -------
        None
        """
        # x_i * x_i = x_i for binary variables
        if i == j:
            self.add_linear(i, value)
        else:
            key = (min(i, j), max(i, j))
            self.quadratic[key] = self.quadratic.get(key, 0.0) + value

    def add_one_hot_penalty(self, indices: list[int], weight: float) -> None:
        """
        Function that adds weight * (1 - sum_k x_k)^2, which is 0 only when exactly one x_k is 1

        Parameters
        ----------
        indices: list[int]
            Variables that must contain exactly one 1
        weight: float
            Penalty weight

        Returns
        -------
        None
        """
        # (1 - sum x)^2 = 1 - sum x + 2 sum_{k<l} x_k x_l for binary x
        self.constant += weight
        for k in indices:
            self.add_linear(k, -weight)
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                self.add_quadratic(indices[a], indices[b], 2 * weight)

    def values(self) -> np.ndarray:
        """
        Function that evaluates the cost on every basis state

        Returns
        -------
        values: np.ndarray
            Array of length 2^n; entry k is the cost of the bitstring of k (qubit i = bit i)
        """
        # Bit i of every basis index is the value of x_i
        index = np.arange(2**self.num_qubits)
        bits = [(index >> i) & 1 for i in range(self.num_qubits)]

        # Sum the constant, linear and quadratic terms
        values = np.full(index.shape, self.constant, dtype=float)
        for i, a in self.linear.items():
            values += a * bits[i]
        for (i, j), b in self.quadratic.items():
            values += b * bits[i] * bits[j]

        return values

    def ising(self) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
        """
        Function that rewrites the cost with x_i = (1 - Z_i) / 2, dropping the constant

        Returns
        -------
        z: dict[int, float]
            Qubit -> coefficient of Z_i
        zz: dict[tuple[int, int], float]
            Pair -> coefficient of Z_i Z_j
        """
        z: dict[int, float] = {}
        zz: dict[tuple[int, int], float] = {}

        # a x_i = a/2 - a/2 Z_i
        for i, a in self.linear.items():
            z[i] = z.get(i, 0.0) - a / 2

        # b x_i x_j = b/4 (1 - Z_i - Z_j + Z_i Z_j)
        for (i, j), b in self.quadratic.items():
            z[i] = z.get(i, 0.0) - b / 4
            z[j] = z.get(j, 0.0) - b / 4
            zz[(i, j)] = zz.get((i, j), 0.0) + b / 4

        return z, zz


@dataclass
class Variant:
    """
    Class that describes one QAOA setting to compare

    Parameters
    ----------
    name: str
        Label shown in the report
    init: QuantumCircuit
        Circuit that prepares the initial state
    qubo: Qubo
        Cost the optimiser minimises (objective plus any penalty)
    mixer: str
        "x" for the transverse-field mixer, "xy" for XY rings inside groups
    groups: list[list[int]]
        Qubit groups that form XY rings (only used when mixer is "xy")
    color: str
        Colour of the variant in every chart
    """

    name: str
    init: QuantumCircuit
    qubo: Qubo
    mixer: str
    groups: list[list[int]] = field(default_factory=list)
    color: str = "#2a78d6"


@dataclass
class Run:
    """
    Class that holds the optimised result of one variant at one depth

    Parameters
    ----------
    variant: Variant
        Setting that was optimised
    p: int
        Number of QAOA layers
    params: np.ndarray
        Optimal angles [gamma_1..gamma_p, beta_1..beta_p] (gamma in scaled units)
    probabilities: np.ndarray
        Final probability of every basis state
    energy: float
        Expected cost (objective plus penalty) of the final state
    history: list[float]
        Best expected cost found so far after each evaluation of the best restart
    evaluations: int
        Number of cost evaluations over all restarts
    """

    variant: Variant
    p: int
    params: np.ndarray
    probabilities: np.ndarray
    energy: float
    history: list[float]
    evaluations: int


def ring_pairs(group: list[int]) -> list[tuple[int, int]]:
    """
    Function that lists the neighbouring pairs of a ring of qubits

    Parameters
    ----------
    group: list[int]
        Qubits of the ring, in order

    Returns
    -------
    pairs: list[tuple[int, int]]
        (q_k, q_{k+1}) for every k, closing the ring when it has more than two qubits
    """
    # Two qubits form a single pair; longer groups close back to the first qubit
    pairs = [(group[k], group[k + 1]) for k in range(len(group) - 1)]
    if len(group) > 2:
        pairs.append((group[-1], group[0]))

    return pairs


def apply_rx_all(state: np.ndarray, beta: float, n: int) -> np.ndarray:
    """
    Function that applies RX(2 beta) = exp(-i beta X) to every qubit

    Parameters
    ----------
    state: np.ndarray
        State vector of length 2^n
    beta: float
        Mixing angle
    n: int
        Number of qubits

    Returns
    -------
    state: np.ndarray
        Updated state vector
    """
    c, s = np.cos(beta), -1j * np.sin(beta)
    for q in range(n):
        # Amplitudes whose index differs only in bit q form the pairs (a, b)
        pairs = state.reshape(-1, 2, 2**q)
        out = np.empty_like(pairs)
        out[:, 0] = c * pairs[:, 0] + s * pairs[:, 1]
        out[:, 1] = s * pairs[:, 0] + c * pairs[:, 1]
        state = out.reshape(-1)

    return state


def apply_xy(state: np.ndarray, beta: float, q1: int, q2: int, n: int) -> np.ndarray:
    """
    Function that applies exp(-i beta (XX + YY) / 2) to qubits q1 and q2

    The gate swaps amplitude between |01> and |10> and leaves |00> and |11>
    alone, so it never changes the number of 1s.

    Parameters
    ----------
    state: np.ndarray
        State tensor of shape (2,) * n
    beta: float
        Mixing angle
    q1: int
        First qubit
    q2: int
        Second qubit
    n: int
        Number of qubits

    Returns
    -------
    state: np.ndarray
        Updated state tensor
    """
    # Select the |..0..1..> and |..1..0..> slices of the two axes
    a1, a2 = n - 1 - q1, n - 1 - q2
    s01 = [slice(None)] * n
    s10 = [slice(None)] * n
    s01[a1], s01[a2] = 0, 1
    s10[a1], s10[a2] = 1, 0
    v01, v10 = state[tuple(s01)].copy(), state[tuple(s10)].copy()

    # Rotate inside the {|01>, |10>} subspace
    state = state.copy()
    state[tuple(s01)] = np.cos(beta) * v01 - 1j * np.sin(beta) * v10
    state[tuple(s10)] = np.cos(beta) * v10 - 1j * np.sin(beta) * v01

    return state


def evolve(variant: Variant, values: np.ndarray, init: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Function that runs the QAOA layers exactly on a state vector

    Parameters
    ----------
    variant: Variant
        Setting (mixer and groups)
    values: np.ndarray
        Scaled cost of every basis state
    init: np.ndarray
        Initial state vector
    params: np.ndarray
        [gamma_1..gamma_p, beta_1..beta_p]

    Returns
    -------
    state: np.ndarray
        Final state vector
    """
    n = variant.init.num_qubits
    p = len(params) // 2
    gammas, betas = params[:p], params[p:]
    state = init.copy()

    for gamma, beta in zip(gammas, betas):
        # Cost layer: a phase exp(-i gamma C) on every basis state
        state = state * np.exp(-1j * gamma * values)

        # X mixer: the same RX rotation on every qubit
        if variant.mixer == "x":
            state = apply_rx_all(state, beta, n)

        # XY mixer: rotate every neighbouring pair around each ring
        else:
            tensor = state.reshape((2,) * n)
            for group in variant.groups:
                for q1, q2 in ring_pairs(group):
                    tensor = apply_xy(tensor, beta, q1, q2, n)
            state = tensor.reshape(-1)

    return state


def cost_scale(objective: np.ndarray, feasible: np.ndarray) -> float:
    """
    Function that picks the unit in which gamma is measured

    Every variant divides its cost by the spread of the objective over feasible
    states, so the same gamma means the same phase spread in every variant.

    Parameters
    ----------
    objective: np.ndarray
        Objective (without penalty) of every basis state
    feasible: np.ndarray
        Boolean mask of feasible basis states

    Returns
    -------
    scale: float
        Positive scale factor
    """
    # Use the spread of feasible objective values, or 1 if they are all equal
    spread = float(objective[feasible].max() - objective[feasible].min())
    scale = spread if spread > 0 else 1.0

    return scale


def optimize(variant: Variant, p: int, scale: float, restarts: int, seed: int,
             warm_start: np.ndarray | None = None, maxiter: int = 300) -> Run:
    """
    Function that optimises the QAOA angles of one variant at depth p

    Parameters
    ----------
    variant: Variant
        Setting to optimise
    p: int
        Number of layers
    scale: float
        Cost unit (see cost_scale)
    restarts: int
        Number of random starting points
    seed: int
        Seed for the starting points
    warm_start: np.ndarray | None = None
        Optimal angles at depth p - 1, extended by one layer as an extra start
    maxiter: int = 300
        Iteration limit of COBYLA per start

    Returns
    -------
    run: Run
        Best result over all starts
    """
    rng = np.random.default_rng(seed)
    values = variant.qubo.values()
    scaled = values / scale
    init = Statevector(variant.init).data

    # Random starting points, plus the previous depth's optimum with one more layer
    starts = [np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2, p)]) for _ in range(restarts)]
    if warm_start is not None:
        q = len(warm_start) // 2
        starts.append(np.concatenate([warm_start[:q], [warm_start[q - 1]], warm_start[q:], [warm_start[-1]]]))

    best = None
    evaluations = 0
    for start in starts:
        # Track the best expected cost after each evaluation of this start
        history: list[float] = []

        def energy(params: np.ndarray) -> float:
            probs = np.abs(evolve(variant, scaled, init, params)) ** 2
            value = float(probs @ values)
            history.append(min(value, history[-1]) if history else value)
            return value

        result = minimize(energy, start, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.4})
        evaluations += len(history)
        if best is None or result.fun < best[0]:
            best = (result.fun, result.x, history)

    # Recompute the final state of the best start
    energy_value, params, history = best
    probabilities = np.abs(evolve(variant, scaled, init, params)) ** 2
    run = Run(variant, p, params, probabilities, float(energy_value), history, evaluations)

    return run


def qaoa_circuit(variant: Variant, p: int, scale: float, params: np.ndarray | None = None) -> QuantumCircuit:
    """
    Function that builds the QAOA circuit as gates (for drawing and sampling)

    Parameters
    ----------
    variant: Variant
        Setting to build
    p: int
        Number of layers
    scale: float
        Cost unit used by optimize, so bound angles reproduce the optimised state
    params: np.ndarray | None = None
        Angles to bind; symbolic gamma / beta parameters when None

    Returns
    -------
    qc: QuantumCircuit
        Initial state followed by p cost and mixer layers
    """
    n = variant.init.num_qubits
    gammas, betas = ParameterVector("γ", p), ParameterVector("β", p)
    z, zz = variant.qubo.ising()

    # Round the coefficients of a symbolic circuit so its drawing stays readable
    digits = 15 if params is not None else 2

    # Prepare the initial state
    qc = QuantumCircuit(n)
    qc.compose(variant.init, inplace=True)

    for layer in range(p):
        # Separate the layers visually (no leading barrier when there is no initial state)
        if layer > 0 or variant.init.size() > 0:
            qc.barrier()

        # Cost layer: exp(-i gamma C) = product of RZ and RZZ rotations (global phase dropped)
        for (i, j), b in sorted(zz.items()):
            qc.rzz(round(2 * b / scale, digits) * gammas[layer], i, j)
        for i, a in sorted(z.items()):
            if abs(a) > 1e-12:
                qc.rz(round(2 * a / scale, digits) * gammas[layer], i)

        # Mixer layer: RX on every qubit, or XY rotations around each ring
        if variant.mixer == "x":
            for q in range(n):
                qc.rx(2 * betas[layer], q)
        else:
            for group in variant.groups:
                for q1, q2 in ring_pairs(group):
                    qc.append(XXPlusYYGate(2 * betas[layer]), [q1, q2])

    # Bind the optimised angles when they are given
    if params is not None:
        qc = qc.assign_parameters({gammas: params[:p], betas: params[p:]})

    return qc
