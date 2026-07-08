"""Fully twirled checkerboard QeMCMC circuits.

This module implements a quantum-enhanced MCMC block proposal for
larger 2D Ising spin glasses by combining three ingredients:

* checkerboard domain decomposition;
* frozen-boundary effective fields for each active block;
* Layden's SPAM and RZX gate-twirling prescriptions.

Default geometry:

* 10 x 12 open-boundary nearest-neighbor Edwards-Anderson lattice;
* 5 x 3 checkerboard blocks, so each quantum proposal acts on 15 spins;
* one 120-qubit half-sweep circuit for each checkerboard color, with same-color
  blocks placed on disjoint qubits and therefore executable in parallel.

The Trotterized block proposal uses the endpoint-cancelled product formula

    V_tilde = exp(-i H1 dt) [exp(-i H2 dt) exp(-i H1 dt)] ** (r - 1)

The randomized schedule follows Layden's experimental Trotter-circuit grid:
gamma in [0.25, 0.6] with 10 midpoint values, and total time
t = r * 0.8 for integer r in [2, 25].

Checkerboard substitution:

    h_eff[j] = h[j] + sum_{i outside block} (J + J.T)[j, i] s[i]

Frozen neighboring spins are folded into the block field before the usual
Layden proposal is built.  The global model uses the local codebase convention
E(s) = s.T J s + s.T h, so the circuit angles are translated to make the
proposal Hamiltonian diagonal match this sign convention.

SPAM twirling:

    c in {-1, +1} ** block_size
    input_c[j] = input[j] * c[j]
    h_eff_c[j] = h_eff[j] * c[j]
    J_eff_c[j, k] = J_eff[j, k] * c[j] * c[k]
    output[j] = measured_c[j] * c[j]

This is the SPAM twirl written in the spin convention used here:
XOR is multiplication by the twirl key.  The circuit is executed in the
twirled frame, but the proposed block returned to Metropolis is always the
untwirled physical spin block.

Gate twirling:

    RZZ(theta) = H_target RZX(theta) H_target

Every RZX occurrence is independently Pauli twirled as in Fig. S12.  If the
random two-qubit Pauli commutes with Z tensor X, the same theta is used; if it
anticommutes, -theta is used.  In ideal statevector simulation this leaves the
logical unitary unchanged, but it matches Layden's full twirling prescription
for suppressing preferred directions in two-qubit gate noise.

The sampler path uses exact 15-qubit statevector probabilities and draws one
proposal sample from that distribution.  No finite-shot backend execution is
used here.  The 120-qubit circuit builders remain available for inspecting the
full checkerboard half-sweep circuits.  Counts from those circuits must be
untwirled with the returned block plans before they are interpreted as physical
outputs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import math
import sys
from typing import Iterable

import numpy as np


DISPLAY_RNG_OFFSET = 104729
REPORT_SEPARATOR = "=" * 88


# =============================================================================
# Data models
# =============================================================================


@dataclass(frozen=True)
class IsingProblem:
    """Upper-triangular Ising instance with energy E(s) = s.T J s + s.T h."""

    J: np.ndarray
    h: np.ndarray
    T: float

    @property
    def n(self) -> int:
        return int(self.h.size)

    def energy(self, spin_config: np.ndarray) -> float:
        spin_config = np.asarray(spin_config, dtype=float)
        return float(spin_config @ self.J @ spin_config + spin_config @ self.h)


@dataclass(frozen=True)
class LatticeSpec:
    """Open rectangular lattice and rectangular checkerboard block shape."""

    rows: int = 10
    cols: int = 12
    block_rows: int = 5
    block_cols: int = 3

    @property
    def n(self) -> int:
        return self.rows * self.cols

    @property
    def block_size(self) -> int:
        return self.block_rows * self.block_cols


@dataclass(frozen=True)
class TrotterSpec:
    """One fixed Layden-style product-formula setting for a block proposal."""

    gamma: float = 0.425
    delta_t: float = 0.8
    trotter_steps: int = 4

    @property
    def total_time(self) -> float:
        return self.delta_t * self.trotter_steps


@dataclass(frozen=True)
class ParameterSchedule:
    """Randomized grid from which block-level Trotter settings are sampled.

    This is the experimental Trotter-circuit setting in Layden Eq. S17/S18:
    10 gamma midpoints on [0.25, 0.6], delta_t = 0.8, and integer r in
    [2, 25], giving t in {1.6, 2.4, ..., 20}.
    """

    gamma_min: float = 0.25
    gamma_max: float = 0.6
    gamma_grid_size: int = 10
    r_min: int = 2
    r_max: int = 25
    delta_t: float = 0.8


@dataclass(frozen=True)
class EffectiveBlockProblem:
    """Block Hamiltonian after freezing outside spins into local fields."""

    J_eff: np.ndarray
    h_eff: np.ndarray
    alpha: float
    J_quantum: np.ndarray
    h_quantum: np.ndarray

    @property
    def n(self) -> int:
        return int(self.h_eff.size)


@dataclass(frozen=True)
class OneQubitTerm:
    """Compiled H1 single-qubit gate parameters for one block spin."""

    local_qubit: int
    global_qubit: int
    a: float
    b: float
    theta: float
    phi: float
    lam: float


@dataclass(frozen=True)
class RZZTerm:
    """Compiled H2 ZZ gate parameters for one internal block coupling."""

    local_j: int
    local_k: int
    global_j: int
    global_k: int
    coupling: float
    theta: float


@dataclass(frozen=True)
class GateTwirl:
    """Random Pauli frame for one RZX(theta) occurrence."""

    pauli_z_qubit: str
    pauli_x_qubit: str
    theta_sign: int
    anticommutes: bool


@dataclass(frozen=True)
class BlockPlan:
    """All data needed to build or simulate one effective block proposal."""

    color: str
    color_index: int
    trotter: TrotterSpec
    global_indices: np.ndarray
    current_spins: np.ndarray
    circuit_input_spins: np.ndarray
    spam_key: np.ndarray
    physical_effective_problem: EffectiveBlockProblem
    effective_problem: EffectiveBlockProblem
    gate_twirling: bool
    gate_twirl_seed: int | None
    one_qubit_terms: tuple[OneQubitTerm, ...]
    rzz_layers: tuple[tuple[RZZTerm, ...], ...]


@dataclass(frozen=True)
class AcceptedMove:
    """Accepted block move recorded during a checkerboard simulation."""

    block_index: int
    spins_changed: int
    dE: float
    gamma: float
    trotter_steps: int
    total_time: float


@dataclass(frozen=True)
class PhaseResult:
    """Summary of one black or white checkerboard half-sweep."""

    sweep: int
    color: str
    accepted_blocks: int
    total_blocks: int
    self_proposals: int
    accepted_spin_changes: int
    accepted_moves: tuple[AcceptedMove, ...]
    energy: float


# =============================================================================
# Lattice and checkerboard construction
# =============================================================================


def validate_lattice_spec(spec: LatticeSpec) -> None:
    """Validate the lattice/block sizes supported by this circuit builder."""

    if spec.rows < 2 or spec.cols < 2:
        raise ValueError("rows and cols must both be at least 2.")
    if spec.rows % spec.block_rows != 0 or spec.cols % spec.block_cols != 0:
        raise ValueError("block_rows and block_cols must divide rows and cols.")
    if spec.block_size != 15:
        raise ValueError(
            "This script is configured for 15-qubit blocks; use a block shape "
            "whose area is 15."
        )


def lattice_index(row: int, col: int, cols: int) -> int:
    """Map a row/column coordinate to a flat lattice index."""

    return row * cols + col


def build_2d_open_spin_glass(
    spec: LatticeSpec,
    rng: np.random.Generator,
    *,
    temperature: float,
    coupling_scale: float = 1.0,
) -> IsingProblem:
    """Build a zero-field nearest-neighbor bimodal Edwards-Anderson instance."""
    validate_lattice_spec(spec)

    J = np.zeros((spec.n, spec.n), dtype=float)
    h = np.zeros(spec.n, dtype=float)

    for row in range(spec.rows):
        for col in range(spec.cols):
            i = lattice_index(row, col, spec.cols)
            neighbors = []
            if col + 1 < spec.cols:
                neighbors.append(lattice_index(row, col + 1, spec.cols))
            if row + 1 < spec.rows:
                neighbors.append(lattice_index(row + 1, col, spec.cols))

            for neighbor in neighbors:
                left, right = sorted((i, neighbor))
                J[left, right] = coupling_scale * float(rng.choice((-1.0, 1.0)))

    return IsingProblem(J=J, h=h, T=float(temperature))


def build_problem_and_initial_state(
    spec: LatticeSpec,
    *,
    seed: int,
    temperature: float,
) -> tuple[np.random.Generator, IsingProblem, np.ndarray]:
    """Build the seeded problem instance and initial state used by all baselines."""
    rng = np.random.default_rng(seed)
    problem = build_2d_open_spin_glass(spec, rng, temperature=temperature)
    initial_state = rng.choice(np.array([-1, 1], dtype=int), size=problem.n)
    return rng, problem, initial_state


def fingerprint_array(values: np.ndarray) -> str:
    """Return a short reproducibility fingerprint for an array."""

    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()[:16]


def problem_fingerprint(problem: IsingProblem) -> str:
    """Return a short reproducibility fingerprint for J, h, and T."""

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(problem.J).tobytes())
    digest.update(np.ascontiguousarray(problem.h).tobytes())
    digest.update(np.array([problem.T], dtype=np.float64).tobytes())
    return digest.hexdigest()[:16]


def get_rectangular_checkerboard_blocks(
    spec: LatticeSpec,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Partition the open-boundary lattice into rectangular checkerboard blocks."""
    validate_lattice_spec(spec)
    block_grid_rows = spec.rows // spec.block_rows
    block_grid_cols = spec.cols // spec.block_cols
    black_blocks: list[np.ndarray] = []
    white_blocks: list[np.ndarray] = []

    for block_row in range(block_grid_rows):
        for block_col in range(block_grid_cols):
            indices = []
            row_start = block_row * spec.block_rows
            col_start = block_col * spec.block_cols
            for row in range(row_start, row_start + spec.block_rows):
                for col in range(col_start, col_start + spec.block_cols):
                    indices.append(lattice_index(row, col, spec.cols))

            block = np.array(indices, dtype=int)
            if (block_row + block_col) % 2 == 0:
                black_blocks.append(block)
            else:
                white_blocks.append(block)

    return black_blocks, white_blocks


def validate_same_color_independence(
    problem: IsingProblem,
    blocks: list[np.ndarray],
    *,
    tol: float = 1e-12,
) -> None:
    """Require same-color checkerboard blocks to be conditionally independent."""
    coupling = np.abs(problem.J + problem.J.T)
    for left_index, left in enumerate(blocks):
        for right in blocks[left_index + 1 :]:
            if np.any(coupling[np.ix_(left, right)] > tol):
                raise ValueError(
                    "Same-color checkerboard blocks are coupled. Use a different "
                    "lattice/block shape."
                )


# =============================================================================
# Effective block model and Layden gate parameters
# =============================================================================


def validate_spin_config(spin_config: np.ndarray, n: int) -> None:
    """Validate a spin vector in the +/-1 convention."""

    spin_config = np.asarray(spin_config)
    if spin_config.shape != (n,):
        raise ValueError(f"Expected spin configuration of shape ({n},).")
    if np.any((spin_config != -1) & (spin_config != 1)):
        raise ValueError("Spin configurations must contain only -1 and +1.")


def validate_trotter_spec(trotter: TrotterSpec) -> None:
    """Validate one fixed product-formula setting."""

    if not 0.0 <= trotter.gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    if trotter.delta_t <= 0.0:
        raise ValueError("delta_t must be positive.")
    if trotter.trotter_steps < 1:
        raise ValueError("trotter_steps must be at least 1.")


def validate_parameter_schedule(schedule: ParameterSchedule) -> None:
    """Validate the randomized product-formula parameter grid."""

    if not 0.0 <= schedule.gamma_min <= schedule.gamma_max <= 1.0:
        raise ValueError("Layden gamma bounds must satisfy 0 <= min <= max <= 1.")
    if schedule.gamma_min == schedule.gamma_max:
        raise ValueError("Layden gamma_min and gamma_max must differ.")
    if schedule.gamma_grid_size < 1:
        raise ValueError("Layden gamma grid size must be at least 1.")
    if schedule.r_min < 1 or schedule.r_max < schedule.r_min:
        raise ValueError("Layden r bounds must satisfy 1 <= r_min <= r_max.")
    if schedule.delta_t <= 0.0:
        raise ValueError("Layden delta_t must be positive.")


def layden_gamma_grid(schedule: ParameterSchedule) -> np.ndarray:
    """Midpoint grid used in Layden's hardware Trotter implementation."""
    step = (schedule.gamma_max - schedule.gamma_min) / schedule.gamma_grid_size
    return np.array(
        [
            schedule.gamma_min + (index + 0.5) * step
            for index in range(schedule.gamma_grid_size)
        ],
        dtype=float,
    )


def sample_trotter_spec(schedule: ParameterSchedule, rng: np.random.Generator) -> TrotterSpec:
    """Draw one Trotter parameter set for one quantum block proposal."""
    validate_parameter_schedule(schedule)

    gamma = float(rng.choice(layden_gamma_grid(schedule)))
    trotter_steps = int(rng.integers(schedule.r_min, schedule.r_max + 1))
    return TrotterSpec(
        gamma=gamma,
        delta_t=schedule.delta_t,
        trotter_steps=trotter_steps,
    )


def sample_trotter_specs(
    schedule: ParameterSchedule,
    count: int,
    rng: np.random.Generator,
) -> list[TrotterSpec]:
    """Draw independent Trotter settings for a list of blocks."""

    return [sample_trotter_spec(schedule, rng) for _ in range(count)]


def effective_block_problem(
    problem: IsingProblem,
    active_indices: np.ndarray,
    full_state_config: np.ndarray,
) -> EffectiveBlockProblem:
    """Build the effective problem seen by one checkerboard block.

    Spins outside ``active_indices`` are held fixed.  Their couplings to active
    spins become an induced local field, so energy differences inside the block
    match global energy differences with the boundary frozen.
    """
    active_indices = np.asarray(active_indices, dtype=int)
    if active_indices.size == 0:
        raise ValueError("A checkerboard block cannot be empty.")
    if np.any(active_indices < 0) or np.any(active_indices >= problem.n):
        raise ValueError("A checkerboard block contains an out-of-range spin.")
    if np.unique(active_indices).size != active_indices.size:
        raise ValueError("A checkerboard block contains duplicate spins.")
    validate_spin_config(full_state_config, problem.n)

    active_mask = np.zeros(problem.n, dtype=bool)
    active_mask[active_indices] = True

    inactive_config = np.asarray(full_state_config, dtype=float).copy()
    inactive_config[active_mask] = 0.0

    h_interaction = (problem.J + problem.J.T) @ inactive_config
    h_eff = problem.h[active_indices] + h_interaction[active_indices]
    J_eff = problem.J[np.ix_(active_indices, active_indices)]

    denominator = float(np.sum(J_eff**2) + np.sum(h_eff**2))
    alpha = 1.0 if denominator == 0.0 else float(np.sqrt(active_indices.size / denominator))

    return EffectiveBlockProblem(
        J_eff=J_eff,
        h_eff=h_eff,
        alpha=alpha,
        J_quantum=alpha * J_eff,
        h_quantum=alpha * h_eff,
    )


def validate_spam_key(spam_key: np.ndarray, n: int) -> None:
    """Validate a SPAM-twirl key in the +/-1 spin convention."""

    spam_key = np.asarray(spam_key)
    if spam_key.shape != (n,):
        raise ValueError(f"Expected SPAM twirl key of shape ({n},).")
    if np.any((spam_key != -1) & (spam_key != 1)):
        raise ValueError("SPAM twirl keys must contain only -1 and +1.")


def sample_spam_key(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a uniform Fig. S11 SPAM-twirl key in {-1, +1}^n."""

    if n < 1:
        raise ValueError("SPAM twirl key length must be positive.")
    return rng.choice(np.array([-1, 1], dtype=int), size=n)


def sample_spam_keys(
    blocks: list[np.ndarray],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Draw one independent SPAM-twirl key for each checkerboard block."""

    return [sample_spam_key(len(block), rng) for block in blocks]


def spam_keys_for_blocks(
    blocks: list[np.ndarray],
    rng: np.random.Generator,
    *,
    enabled: bool,
) -> list[np.ndarray]:
    """Return random or identity SPAM keys for a checkerboard phase."""

    if enabled:
        return sample_spam_keys(blocks, rng)
    return [identity_spam_key(len(block)) for block in blocks]


def sample_gate_twirl_seed(rng: np.random.Generator) -> int:
    """Draw a reproducible seed for independently twirling RZX occurrences."""

    return int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))


def gate_twirl_seeds_for_blocks(
    blocks: list[np.ndarray],
    rng: np.random.Generator,
    *,
    enabled: bool,
) -> list[int | None]:
    """Return one gate-twirl seed for each checkerboard block."""

    if enabled:
        return [sample_gate_twirl_seed(rng) for _ in blocks]
    return [None for _ in blocks]


def identity_spam_key(n: int) -> np.ndarray:
    """Return the no-twirl key, useful for exact equivalence checks."""

    if n < 1:
        raise ValueError("SPAM twirl key length must be positive.")
    return np.ones(n, dtype=int)


def spam_twirl_effective_problem(
    effective_problem: EffectiveBlockProblem,
    spam_key: np.ndarray,
) -> EffectiveBlockProblem:
    """Move the random X frame into the block Hamiltonian coefficients.

    In the spin convention used here, XOR by c is multiplication by c.  The
    circuit-frame Hamiltonian must therefore use h_j -> h_j c_j and
    J_jk -> J_jk c_j c_k.  The normalization alpha is unchanged because this
    sign change preserves sum(J^2) + sum(h^2).
    """

    q = effective_problem.n
    spam_key = np.asarray(spam_key, dtype=int)
    validate_spam_key(spam_key, q)
    key_outer = np.outer(spam_key, spam_key)

    return EffectiveBlockProblem(
        J_eff=effective_problem.J_eff * key_outer,
        h_eff=effective_problem.h_eff * spam_key,
        alpha=effective_problem.alpha,
        J_quantum=effective_problem.J_quantum * key_outer,
        h_quantum=effective_problem.h_quantum * spam_key,
    )


def untwirl_block_spins(
    circuit_frame_spins: np.ndarray,
    spam_key: np.ndarray,
) -> np.ndarray:
    """Convert measured circuit-frame spins back to physical block spins."""

    circuit_frame_spins = np.asarray(circuit_frame_spins, dtype=int)
    validate_spin_config(circuit_frame_spins, circuit_frame_spins.size)
    validate_spam_key(spam_key, circuit_frame_spins.size)
    return circuit_frame_spins * np.asarray(spam_key, dtype=int)


def untwirl_phase_spins(
    circuit_frame_spins: np.ndarray,
    plans: list[BlockPlan],
) -> np.ndarray:
    """Untwirl a full-register phase measurement using the block plans."""

    circuit_frame_spins = np.asarray(circuit_frame_spins, dtype=int)
    physical_spins = circuit_frame_spins.copy()
    validate_spin_config(physical_spins, physical_spins.size)

    for plan in plans:
        if np.any(plan.global_indices < 0) or np.any(plan.global_indices >= physical_spins.size):
            raise ValueError(
                "Block plan contains a global index outside the measured spin vector."
            )
        physical_spins[plan.global_indices] = untwirl_block_spins(
            physical_spins[plan.global_indices],
            plan.spam_key,
        )
    return physical_spins


PAULI_LABELS = ("i", "x", "y", "z")


def pauli_anticommutes_with_axis(pauli: str, axis: str) -> bool:
    """Return whether a one-qubit Pauli anticommutes with X or Z."""

    pauli = pauli.lower()
    axis = axis.lower()
    if pauli not in PAULI_LABELS:
        raise ValueError(f"Unknown Pauli label: {pauli}")
    if axis == "z":
        return pauli in ("x", "y")
    if axis == "x":
        return pauli in ("y", "z")
    raise ValueError(f"Unsupported Pauli axis: {axis}")


def gate_twirl_from_seed(
    seed: int,
    repeat_index: int,
    layer_index: int,
    term_index: int,
) -> GateTwirl:
    """Generate the Fig. S12 Pauli twirl for one RZX occurrence."""

    sequence = np.random.SeedSequence(
        [int(seed), int(repeat_index), int(layer_index), int(term_index)]
    )
    rng = np.random.default_rng(sequence)
    pauli_z_qubit = PAULI_LABELS[int(rng.integers(0, len(PAULI_LABELS)))]
    pauli_x_qubit = PAULI_LABELS[int(rng.integers(0, len(PAULI_LABELS)))]

    anticommutes = pauli_anticommutes_with_axis(
        pauli_z_qubit,
        "z",
    ) ^ pauli_anticommutes_with_axis(pauli_x_qubit, "x")
    return GateTwirl(
        pauli_z_qubit=pauli_z_qubit,
        pauli_x_qubit=pauli_x_qubit,
        theta_sign=-1 if anticommutes else 1,
        anticommutes=anticommutes,
    )


def apply_pauli(qc, pauli: str, qubit: int) -> None:
    """Append a one-qubit Pauli if it is not identity."""

    if pauli == "i":
        return
    if pauli == "x":
        qc.x(qubit)
    elif pauli == "y":
        qc.y(qubit)
    elif pauli == "z":
        qc.z(qubit)
    else:
        raise ValueError(f"Unknown Pauli label: {pauli}")


def normalize_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def xz_rotation_to_u_parameters(
    a: float,
    b: float,
    *,
    eps: float = 1e-12,
) -> tuple[float, float, float]:
    r"""Return Qiskit U(theta, phi, lambda) parameters for exp[-i(a X + b Z)].

    Qiskit's U gate matches this single-qubit unitary up to a global phase,
    which has no effect on the measurement statistics used by QeMCMC.
    """
    radius = float(np.hypot(a, b))
    if radius < eps:
        return 0.0, 0.0, 0.0

    c = float(np.cos(radius))
    s_over_r = float(np.sin(radius) / radius)
    A = c - 1j * b * s_over_r
    B = -1j * a * s_over_r
    C = B
    D = c + 1j * b * s_over_r

    if abs(B) < eps and abs(C) < eps:
        theta = 0.0
        phi = 0.0
        lam = np.angle(D / A) if abs(A) >= eps else 0.0
        return theta, normalize_angle(phi), normalize_angle(lam)

    if abs(A) >= eps:
        ratio_b = B / A
        ratio_c = C / A
        theta = 2.0 * np.arctan(abs(ratio_b))
        phi = np.angle(ratio_c)
        lam = np.angle(-ratio_b)
        return float(theta), normalize_angle(phi), normalize_angle(lam)

    theta = np.pi
    phi = np.angle(C)
    lam = np.angle(-B)
    return float(theta), normalize_angle(phi), normalize_angle(lam)


def nonzero_upper_couplings(
    J_matrix: np.ndarray,
    *,
    tol: float = 1e-12,
) -> Iterable[tuple[int, int, float]]:
    """Yield nonzero upper-triangular ZZ couplings."""

    q = J_matrix.shape[0]
    for j in range(q):
        for k in range(j + 1, q):
            coupling = float(J_matrix[j, k])
            if abs(coupling) > tol:
                yield j, k, coupling


def greedy_disjoint_rzz_layers(terms: list[RZZTerm]) -> tuple[tuple[RZZTerm, ...], ...]:
    """Greedily color ZZ edges so each layer has disjoint qubits."""
    layers: list[list[RZZTerm]] = []
    layer_vertices: list[set[int]] = []

    for term in terms:
        placed = False
        for layer, used_vertices in zip(layers, layer_vertices):
            if term.local_j not in used_vertices and term.local_k not in used_vertices:
                layer.append(term)
                used_vertices.update((term.local_j, term.local_k))
                placed = True
                break
        if not placed:
            layers.append([term])
            layer_vertices.append({term.local_j, term.local_k})

    return tuple(tuple(layer) for layer in layers)


def build_block_plan(
    problem: IsingProblem,
    block: np.ndarray,
    full_state_config: np.ndarray,
    trotter: TrotterSpec,
    *,
    color: str,
    color_index: int,
    spam_key: np.ndarray | None = None,
    gate_twirling: bool = False,
    gate_twirl_seed: int | None = None,
) -> BlockPlan:
    """Compile one effective block problem into Qiskit gate parameters."""

    validate_trotter_spec(trotter)
    physical_eff = effective_block_problem(problem, block, full_state_config)
    current_spins = np.asarray(full_state_config, dtype=int)[block]
    if spam_key is None:
        spam_key = identity_spam_key(current_spins.size)
    else:
        spam_key = np.asarray(spam_key, dtype=int).copy()
    validate_spam_key(spam_key, current_spins.size)

    circuit_input_spins = current_spins * spam_key
    eff = spam_twirl_effective_problem(physical_eff, spam_key)
    if gate_twirling and gate_twirl_seed is None:
        raise ValueError("Provide gate_twirl_seed when gate_twirling is enabled.")

    one_qubit_terms = []
    for local_qubit, global_qubit in enumerate(block):
        # H1 contains gamma * X plus the effective local-field Z term.
        a = trotter.gamma * trotter.delta_t
        b = (1.0 - trotter.gamma) * eff.h_quantum[local_qubit] * trotter.delta_t
        theta, phi, lam = xz_rotation_to_u_parameters(a, b)
        one_qubit_terms.append(
            OneQubitTerm(
                local_qubit=local_qubit,
                global_qubit=int(global_qubit),
                a=float(a),
                b=float(b),
                theta=theta,
                phi=phi,
                lam=lam,
            )
        )

    rzz_terms = []
    for local_j, local_k, coupling in nonzero_upper_couplings(eff.J_quantum):
        # Qiskit's rzz(theta) implements exp[-i theta/2 Z_j Z_k].
        theta = 2.0 * (1.0 - trotter.gamma) * coupling * trotter.delta_t
        rzz_terms.append(
            RZZTerm(
                local_j=local_j,
                local_k=local_k,
                global_j=int(block[local_j]),
                global_k=int(block[local_k]),
                coupling=float(coupling),
                theta=float(theta),
            )
        )

    return BlockPlan(
        color=color,
        color_index=color_index,
        trotter=trotter,
        global_indices=np.asarray(block, dtype=int),
        current_spins=current_spins,
        circuit_input_spins=circuit_input_spins,
        spam_key=spam_key,
        physical_effective_problem=physical_eff,
        effective_problem=eff,
        gate_twirling=bool(gate_twirling),
        gate_twirl_seed=int(gate_twirl_seed) if gate_twirling else None,
        one_qubit_terms=tuple(one_qubit_terms),
        rzz_layers=greedy_disjoint_rzz_layers(rzz_terms),
    )


def build_phase_plans(
    problem: IsingProblem,
    blocks: list[np.ndarray],
    full_state_config: np.ndarray,
    trotter: TrotterSpec | None = None,
    *,
    color: str,
    trotter_specs: list[TrotterSpec] | None = None,
    spam_keys: list[np.ndarray] | None = None,
    gate_twirling: bool = False,
    gate_twirl_seeds: list[int | None] | None = None,
) -> list[BlockPlan]:
    """Compile all blocks in one checkerboard phase into block plans."""

    if trotter_specs is None:
        if trotter is None:
            raise ValueError("Provide either trotter or trotter_specs.")
        trotter_specs = [trotter for _ in blocks]
    if len(trotter_specs) != len(blocks):
        raise ValueError("trotter_specs must have one entry per block.")
    if spam_keys is None:
        spam_keys = [identity_spam_key(len(block)) for block in blocks]
    if len(spam_keys) != len(blocks):
        raise ValueError("spam_keys must have one entry per block.")
    if gate_twirl_seeds is None:
        gate_twirl_seeds = [None for _ in blocks]
    if len(gate_twirl_seeds) != len(blocks):
        raise ValueError("gate_twirl_seeds must have one entry per block.")

    return [
        build_block_plan(
            problem,
            block,
            full_state_config,
            block_trotter,
            color=color,
            color_index=index,
            spam_key=spam_key,
            gate_twirling=gate_twirling,
            gate_twirl_seed=gate_twirl_seed,
        )
        for index, (block, block_trotter, spam_key, gate_twirl_seed) in enumerate(
            zip(blocks, trotter_specs, spam_keys, gate_twirl_seeds)
        )
    ]


# =============================================================================
# Qiskit circuit construction
# =============================================================================


def import_qiskit():
    """Import Qiskit lazily so non-circuit utilities remain importable."""

    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise ImportError(
            "Qiskit is not importable. Install qiskit or run this script in the "
            "project environment that contains qiskit."
        ) from exc

    return QuantumCircuit


def active_qubits(plans: list[BlockPlan]) -> list[int]:
    """Return global qubits touched by a collection of block plans."""

    qubits: list[int] = []
    for plan in plans:
        qubits.extend(int(index) for index in plan.global_indices)
    return qubits


def barrier_on(qc, qubits: list[int]) -> None:
    """Insert a barrier when there are active qubits to separate layers."""

    if qubits:
        qc.barrier(*qubits)


def append_h1_layer(qc, plans: list[BlockPlan], *, local: bool) -> None:
    """Append the single-qubit H1 layer for one or more blocks."""

    for plan in plans:
        for term in plan.one_qubit_terms:
            qubit = term.local_qubit if local else term.global_qubit
            qc.u(term.theta, term.phi, term.lam, qubit)


def append_rzx(qc, theta: float, qz: int, qx: int) -> None:
    """Append Qiskit's RZX(theta), with a fallback for older Qiskit builds."""

    if hasattr(qc, "rzx"):
        qc.rzx(theta, qz, qx)
        return
    try:
        from qiskit.circuit.library import RZXGate
    except ImportError as exc:
        raise ImportError("Qiskit RZXGate is required for gate twirling.") from exc
    qc.append(RZXGate(theta), [qz, qx])


def append_logical_rzz_via_rzx(
    qc,
    plan: BlockPlan,
    term: RZZTerm,
    *,
    local: bool,
    repeat_index: int,
    layer_index: int,
    term_index: int,
) -> None:
    """Append logical RZZ(theta) through the Fig. S12-twirled RZX gate."""

    qz = term.local_j if local else term.global_j
    qx = term.local_k if local else term.global_k
    theta = term.theta

    qc.h(qx)
    if plan.gate_twirling:
        assert plan.gate_twirl_seed is not None
        twirl = gate_twirl_from_seed(
            plan.gate_twirl_seed,
            repeat_index,
            layer_index,
            term_index,
        )
        apply_pauli(qc, twirl.pauli_z_qubit, qz)
        apply_pauli(qc, twirl.pauli_x_qubit, qx)
        append_rzx(qc, twirl.theta_sign * theta, qz, qx)
        apply_pauli(qc, twirl.pauli_z_qubit, qz)
        apply_pauli(qc, twirl.pauli_x_qubit, qx)
    else:
        append_rzx(qc, theta, qz, qx)
    qc.h(qx)


def append_h2_step(
    qc,
    plans: list[BlockPlan],
    *,
    local: bool,
    repeat_index: int,
) -> None:
    """Append one ZZ-coupling H2 step, layered over twirled RZX gates."""

    max_layers = max((len(plan.rzz_layers) for plan in plans), default=0)
    qubits = (
        list(range(plans[0].effective_problem.n))
        if local and plans
        else active_qubits(plans)
    )

    for layer_index in range(max_layers):
        for plan in plans:
            if layer_index >= len(plan.rzz_layers):
                continue
            for term_index, term in enumerate(plan.rzz_layers[layer_index]):
                append_logical_rzz_via_rzx(
                    qc,
                    plan,
                    term,
                    local=local,
                    repeat_index=repeat_index,
                    layer_index=layer_index,
                    term_index=term_index,
                )
        barrier_on(qc, qubits)


def prepare_block_qubits(qc, plan: BlockPlan, *, local: bool) -> None:
    """Prepare circuit-frame basis spins using +1 -> |0>, -1 -> |1>."""

    for local_qubit, spin in enumerate(plan.circuit_input_spins):
        if spin == -1:
            qubit = local_qubit if local else int(plan.global_indices[local_qubit])
            qc.x(qubit)


def build_block_circuit(
    plan: BlockPlan,
    *,
    include_measurements: bool = True,
):
    """Build the 15-qubit product-formula circuit for one effective block."""
    trotter = plan.trotter
    if trotter.trotter_steps < 1:
        raise ValueError("trotter_steps must be at least 1.")
    QuantumCircuit = import_qiskit()
    q = plan.effective_problem.n
    qc = QuantumCircuit(
        q,
        q if include_measurements else 0,
        name=f"{plan.color}_{plan.color_index}",
    )

    prepare_block_qubits(qc, plan, local=True)
    barrier_on(qc, list(range(q)))
    append_h1_layer(qc, [plan], local=True)

    for repeat_index in range(trotter.trotter_steps - 1):
        barrier_on(qc, list(range(q)))
        append_h2_step(qc, [plan], local=True, repeat_index=repeat_index)
        append_h1_layer(qc, [plan], local=True)

    if include_measurements:
        barrier_on(qc, list(range(q)))
        qc.measure(range(q), range(q))

    return qc


def build_phase_circuit(
    problem: IsingProblem,
    blocks: list[np.ndarray],
    full_state_config: np.ndarray,
    trotter: TrotterSpec | None = None,
    *,
    color: str,
    trotter_specs: list[TrotterSpec] | None = None,
    spam_keys: list[np.ndarray] | None = None,
    gate_twirling: bool = False,
    gate_twirl_seeds: list[int | None] | None = None,
    include_measurements: bool = True,
):
    """Build one full-register checkerboard half-sweep circuit.

    All same-color blocks are placed on their global lattice qubits and appended
    layer-by-layer, so disjoint block operations can run in parallel.  The raw
    measured output is in the SPAM-twirled frame and should be converted with
    ``untwirl_phase_spins`` before using it as a physical spin proposal.
    """
    QuantumCircuit = import_qiskit()
    plans = build_phase_plans(
        problem,
        blocks,
        full_state_config,
        trotter,
        color=color,
        trotter_specs=trotter_specs,
        spam_keys=spam_keys,
        gate_twirling=gate_twirling,
        gate_twirl_seeds=gate_twirl_seeds,
    )
    phase_qubits = active_qubits(plans)
    qc = QuantumCircuit(
        problem.n,
        problem.n if include_measurements else 0,
        name=f"{color}_phase",
    )

    for plan in plans:
        prepare_block_qubits(qc, plan, local=False)
    barrier_on(qc, phase_qubits)
    append_h1_layer(qc, plans, local=False)

    max_trotter_steps = max((plan.trotter.trotter_steps for plan in plans), default=1)
    for step_index in range(max_trotter_steps - 1):
        active_step_plans = [
            plan for plan in plans if step_index < plan.trotter.trotter_steps - 1
        ]
        active_step_qubits = active_qubits(active_step_plans)
        barrier_on(qc, active_step_qubits)
        append_h2_step(qc, active_step_plans, local=False, repeat_index=step_index)
        append_h1_layer(qc, active_step_plans, local=False)

    if include_measurements:
        barrier_on(qc, phase_qubits)
        qc.measure(phase_qubits, phase_qubits)

    return qc, plans


# =============================================================================
# Statevector sampler and checkerboard runner
# =============================================================================


def basis_index_to_spins_little_endian(index: int, n: int) -> np.ndarray:
    """Convert a Qiskit basis index to spins ordered by qubit index."""

    return np.array(
        [1 if ((index >> qubit) & 1) == 0 else -1 for qubit in range(n)],
        dtype=int,
    )


def spins_to_basis_index_little_endian(spins: np.ndarray) -> int:
    """Convert +/-1 spins ordered by qubit index to a Qiskit basis index."""

    spins = np.asarray(spins, dtype=int)
    validate_spin_config(spins, spins.size)
    index = 0
    for qubit, spin in enumerate(spins):
        if spin == -1:
            index |= 1 << qubit
    return index


def block_output_probabilities_statevector(
    plan: BlockPlan,
    *,
    physical_output: bool = True,
) -> np.ndarray:
    """Return exact statevector output probabilities for one block plan.

    If ``physical_output`` is true, probabilities are indexed by untwirled
    physical output states.  If false, they are indexed by raw circuit-frame
    states, matching direct Qiskit measurement bits.
    """

    try:
        from qiskit.quantum_info import Statevector
    except ImportError as exc:
        raise ImportError(
            "qiskit.quantum_info.Statevector is required for local "
            "simulation. Use --no-statevector-run to only build circuits."
        ) from exc

    circuit = build_block_circuit(plan, include_measurements=False)
    state = Statevector.from_instruction(circuit)
    raw_probabilities = np.asarray(state.probabilities(), dtype=float)
    raw_probabilities = raw_probabilities / raw_probabilities.sum()

    if not physical_output:
        return raw_probabilities

    physical_probabilities = np.zeros_like(raw_probabilities)
    for raw_index, probability in enumerate(raw_probabilities):
        raw_spins = basis_index_to_spins_little_endian(raw_index, plan.effective_problem.n)
        physical_spins = untwirl_block_spins(raw_spins, plan.spam_key)
        physical_index = spins_to_basis_index_little_endian(physical_spins)
        physical_probabilities[physical_index] += probability
    return physical_probabilities


def sample_block_output_statevector(
    plan: BlockPlan,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one physical block output from the SPAM-twirled circuit."""
    probabilities = block_output_probabilities_statevector(plan, physical_output=False)
    outcome = int(rng.choice(probabilities.size, p=probabilities))
    circuit_frame_output = basis_index_to_spins_little_endian(outcome, plan.effective_problem.n)
    return untwirl_block_spins(circuit_frame_output, plan.spam_key)


def metropolis_accept(dE: float, temperature: float, rng: np.random.Generator) -> bool:
    """Metropolis accept/reject rule for a symmetric proposal kernel."""

    if dE <= 0.0:
        return True
    if temperature == 0.0:
        return False
    if math.isinf(temperature):
        return True
    return bool(rng.random() <= np.exp(-dE / temperature))


def run_checkerboard_qemcmc(
    problem: IsingProblem,
    black_blocks: list[np.ndarray],
    white_blocks: list[np.ndarray],
    initial_state: np.ndarray,
    schedule: ParameterSchedule,
    *,
    sweeps: int,
    rng: np.random.Generator,
    spam_twirling: bool = True,
    gate_twirling: bool = True,
) -> tuple[np.ndarray, list[PhaseResult]]:
    """Run checkerboard sweeps using 15-qubit statevectors for each block."""
    if sweeps < 1:
        raise ValueError("sweeps must be at least 1.")
    validate_parameter_schedule(schedule)

    state = np.asarray(initial_state, dtype=int).copy()
    validate_spin_config(state, problem.n)
    history: list[PhaseResult] = []

    for sweep in range(1, sweeps + 1):
        for color, blocks in (("black", black_blocks), ("white", white_blocks)):
            phase_reference = state.copy()
            trotter_specs = sample_trotter_specs(schedule, len(blocks), rng)
            spam_keys = spam_keys_for_blocks(blocks, rng, enabled=spam_twirling)
            gate_twirl_seeds = gate_twirl_seeds_for_blocks(
                blocks,
                rng,
                enabled=gate_twirling,
            )
            plans = build_phase_plans(
                problem,
                blocks,
                phase_reference,
                color=color,
                trotter_specs=trotter_specs,
                spam_keys=spam_keys,
                gate_twirling=gate_twirling,
                gate_twirl_seeds=gate_twirl_seeds,
            )
            accepted_moves: list[AcceptedMove] = []
            self_proposals = 0

            for plan in plans:
                proposed_block = sample_block_output_statevector(plan, rng)
                spins_changed = int(np.count_nonzero(proposed_block != plan.current_spins))
                if spins_changed == 0:
                    self_proposals += 1
                    continue

                old_energy = problem.energy(phase_reference)
                proposed_state = phase_reference.copy()
                proposed_state[plan.global_indices] = proposed_block
                dE = problem.energy(proposed_state) - old_energy

                if metropolis_accept(dE, problem.T, rng):
                    state[plan.global_indices] = proposed_block
                    accepted_moves.append(
                        AcceptedMove(
                            block_index=plan.color_index,
                            spins_changed=spins_changed,
                            dE=float(dE),
                            gamma=plan.trotter.gamma,
                            trotter_steps=plan.trotter.trotter_steps,
                            total_time=plan.trotter.total_time,
                        )
                    )

            history.append(
                PhaseResult(
                    sweep=sweep,
                    color=color,
                    accepted_blocks=len(accepted_moves),
                    total_blocks=len(blocks),
                    self_proposals=self_proposals,
                    accepted_spin_changes=sum(move.spins_changed for move in accepted_moves),
                    accepted_moves=tuple(accepted_moves),
                    energy=problem.energy(state),
                )
            )

    return state, history


# =============================================================================
# Reporting and CLI
# =============================================================================


def safe_print(text) -> None:
    """Print text safely even when the terminal encoding is not UTF-8."""

    output = str(text)
    encoding = sys.stdout.encoding or "utf-8"
    print(output.encode(encoding, errors="replace").decode(encoding))


def configure_stdout_for_circuits() -> None:
    """Prefer UTF-8 so Qiskit's text circuit drawer is readable on Windows."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def format_vector(values: np.ndarray) -> str:
    """Format a numeric vector compactly for console reports."""

    return "[" + ", ".join(f"{float(value): .6g}" for value in values) + "]"


def print_block_report(
    plan: BlockPlan,
    circuit,
    *,
    fold: int,
    draw_circuit: bool = True,
) -> None:
    """Print the block-level effective fields, gate parameters, and circuit."""

    eff = plan.effective_problem
    physical_eff = plan.physical_effective_problem
    selected = [index for index, value in enumerate(plan.spam_key) if value == -1]
    print(REPORT_SEPARATOR)
    print(f"{plan.color.upper()} block {plan.color_index}")
    print(
        "proposal parameters: "
        f"gamma={plan.trotter.gamma:.8g}, "
        f"delta_t={plan.trotter.delta_t:.8g}, "
        f"r={plan.trotter.trotter_steps}, "
        f"t={plan.trotter.total_time:.8g}"
    )
    print(f"global spin indices: {plan.global_indices.tolist()}")
    print(
        f"current block spins: {plan.current_spins.tolist()}  "
        "(+1 -> |0>, -1 -> |1>)"
    )
    print(
        f"SPAM twirl key c: {plan.spam_key.tolist()}  "
        f"(selected local qubits: {selected})"
    )
    if plan.gate_twirling:
        print(f"gate twirling: enabled, RZX twirl seed={plan.gate_twirl_seed}")
    else:
        print("gate twirling: disabled")
    print(f"circuit-frame input spins s*c: {plan.circuit_input_spins.tolist()}")
    print(f"alpha: {physical_eff.alpha:.12g}")
    print(f"physical h_eff: {format_vector(physical_eff.h_eff)}")
    print(f"circuit-frame h_eff*c: {format_vector(eff.h_eff)}")
    print(f"circuit-frame h_quantum*c: {format_vector(eff.h_quantum)}")
    print(f"internal ZZ couplings: {sum(len(layer) for layer in plan.rzz_layers)}")
    print(f"entangling layers per H2 step: {len(plan.rzz_layers)}")

    print(
        "\nH1 gates use Qiskit u(theta, phi, lambda) "
        "for exp[-i(a X + b Z)]:"
    )
    for term in plan.one_qubit_terms:
        print(
            f"  q[{term.local_qubit:>2}] global={term.global_qubit:>3} "
            f"a={term.a: .8g} b={term.b: .8g} "
            f"u=({term.theta: .8g}, {term.phi: .8g}, {term.lam: .8g})"
        )

    print(
        "\nH2 gates use logical RZZ(theta) = "
        "H(target) RZX(theta) H(target):"
    )
    for layer_index, layer in enumerate(plan.rzz_layers):
        print(f"  layer {layer_index}:")
        for term in layer:
            print(
                f"    q[{term.local_j}], q[{term.local_k}] "
                f"global=({term.global_j}, {term.global_k}) "
                f"Jq={term.coupling: .8g} theta={term.theta: .8g} "
                f"RZX=({term.local_j}->{term.local_k})"
            )

    if draw_circuit:
        print("\nOne-block circuit:")
        safe_print(circuit.draw(output="text", fold=fold))
    else:
        print("\nOne-block circuit drawing skipped.")
    print(f"\noperation counts: {dict(circuit.count_ops())}")


def twirling_equivalence_error(
    problem: IsingProblem,
    block: np.ndarray,
    full_state_config: np.ndarray,
    trotter: TrotterSpec,
    *,
    color: str,
    color_index: int,
    rng: np.random.Generator,
    checks: int,
    spam_twirling: bool = True,
    gate_twirling: bool = True,
) -> float:
    """Compare random twirled block probabilities to the bare ideal circuit."""

    if checks < 1:
        raise ValueError("checks must be at least 1.")

    bare_plan = build_block_plan(
        problem,
        block,
        full_state_config,
        trotter,
        color=color,
        color_index=color_index,
        spam_key=identity_spam_key(len(block)),
        gate_twirling=False,
    )
    bare_probabilities = block_output_probabilities_statevector(bare_plan, physical_output=True)

    max_error = 0.0
    for _ in range(checks):
        spam_key = (
            sample_spam_key(len(block), rng)
            if spam_twirling
            else identity_spam_key(len(block))
        )
        gate_twirl_seed = sample_gate_twirl_seed(rng) if gate_twirling else None
        twirled_plan = build_block_plan(
            problem,
            block,
            full_state_config,
            trotter,
            color=color,
            color_index=color_index,
            spam_key=spam_key,
            gate_twirling=gate_twirling,
            gate_twirl_seed=gate_twirl_seed,
        )
        twirled_probabilities = block_output_probabilities_statevector(
            twirled_plan,
            physical_output=True,
        )
        error = float(np.max(np.abs(twirled_probabilities - bare_probabilities)))
        max_error = max(max_error, error)

    return max_error


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for circuit inspection and smoke runs."""

    parser = argparse.ArgumentParser(
        description=(
            "Build Layden-style checkerboard QeMCMC circuits with effective "
            "fields for a 10x12 open-boundary lattice and 15-qubit blocks."
        )
    )
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--cols", type=int, default=12)
    parser.add_argument("--block-rows", type=int, default=5)
    parser.add_argument("--block-cols", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sweeps", type=int, default=1)
    parser.add_argument("--show-color", choices=("black", "white"), default="black")
    parser.add_argument("--show-block", type=int, default=0)
    parser.add_argument("--fold", type=int, default=140)
    parser.add_argument(
        "--no-spam-twirling",
        action="store_true",
        help="Use identity SPAM keys. By default this file applies random SPAM twirling.",
    )
    parser.add_argument(
        "--no-gate-twirling",
        action="store_true",
        help="Do not insert Fig. S12 Pauli twirls around each RZX gate.",
    )
    parser.add_argument(
        "--twirl-equivalence-checks",
        type=int,
        default=3,
        help=(
            "Number of random full-twirl frames to compare against the bare "
            "ideal statevector distribution for the displayed block."
        ),
    )
    parser.add_argument(
        "--no-circuit-drawing",
        action="store_true",
        help="Print circuit metadata and operation counts without drawing the text circuit.",
    )
    parser.add_argument(
        "--no-statevector-run",
        action="store_true",
        help="Only build and print circuits; do not run the 15-qubit block simulator.",
    )
    return parser


def main() -> None:
    """CLI entry point."""

    configure_stdout_for_circuits()
    args = build_arg_parser().parse_args()
    spec = LatticeSpec(
        rows=args.rows,
        cols=args.cols,
        block_rows=args.block_rows,
        block_cols=args.block_cols,
    )
    schedule = ParameterSchedule()

    validate_parameter_schedule(schedule)
    spam_twirling = not args.no_spam_twirling
    gate_twirling = not args.no_gate_twirling

    rng, problem, initial_state = build_problem_and_initial_state(
        spec,
        seed=args.seed,
        temperature=args.temperature,
    )
    display_rng = np.random.default_rng(args.seed + DISPLAY_RNG_OFFSET)
    black_blocks, white_blocks = get_rectangular_checkerboard_blocks(spec)
    validate_same_color_independence(problem, black_blocks)
    validate_same_color_independence(problem, white_blocks)

    print("Checkerboard QeMCMC circuit build")
    print(f"lattice: {spec.rows}x{spec.cols} open boundary, n={spec.n}")
    print(f"block shape: {spec.block_rows}x{spec.block_cols}, block size={spec.block_size}")
    print(f"blocks: {len(black_blocks)} black, {len(white_blocks)} white")
    print(f"problem fingerprint: {problem_fingerprint(problem)}")
    print(f"initial state fingerprint: {fingerprint_array(initial_state)}")
    print(f"initial energy: {problem.energy(initial_state): .8g}")
    print(f"SPAM twirling: {'enabled' if spam_twirling else 'disabled'}")
    print(f"gate twirling: {'enabled' if gate_twirling else 'disabled'}")
    print(
        "Layden experimental Trotter settings: randomized per block, "
        f"gamma midpoint grid in [{schedule.gamma_min:g}, {schedule.gamma_max:g}] "
        f"with {schedule.gamma_grid_size} points, "
        f"r in [{schedule.r_min}, {schedule.r_max}], "
        f"delta_t={schedule.delta_t:g}"
    )

    show_blocks = black_blocks if args.show_color == "black" else white_blocks
    if args.show_block < 0 or args.show_block >= len(show_blocks):
        raise ValueError(f"--show-block must be in [0, {len(show_blocks) - 1}].")

    show_trotter = sample_trotter_spec(schedule, display_rng)
    show_spam_key = (
        sample_spam_key(len(show_blocks[args.show_block]), display_rng)
        if spam_twirling
        else identity_spam_key(len(show_blocks[args.show_block]))
    )
    show_gate_twirl_seed = sample_gate_twirl_seed(display_rng) if gate_twirling else None
    show_plan = build_block_plan(
        problem,
        show_blocks[args.show_block],
        initial_state,
        show_trotter,
        color=args.show_color,
        color_index=args.show_block,
        spam_key=show_spam_key,
        gate_twirling=gate_twirling,
        gate_twirl_seed=show_gate_twirl_seed,
    )
    show_circuit = build_block_circuit(show_plan, include_measurements=True)
    print_block_report(
        show_plan,
        show_circuit,
        fold=args.fold,
        draw_circuit=not args.no_circuit_drawing,
    )

    black_trotter_specs = sample_trotter_specs(schedule, len(black_blocks), display_rng)
    black_spam_keys = spam_keys_for_blocks(black_blocks, display_rng, enabled=spam_twirling)
    black_gate_twirl_seeds = gate_twirl_seeds_for_blocks(
        black_blocks,
        display_rng,
        enabled=gate_twirling,
    )
    black_phase_circuit, black_plans = build_phase_circuit(
        problem,
        black_blocks,
        initial_state,
        color="black",
        trotter_specs=black_trotter_specs,
        spam_keys=black_spam_keys,
        gate_twirling=gate_twirling,
        gate_twirl_seeds=black_gate_twirl_seeds,
        include_measurements=True,
    )
    print(REPORT_SEPARATOR)
    print("120-qubit black half-sweep circuit")
    print(f"active qubits: {len(active_qubits(black_plans))}")
    print(f"operation counts: {dict(black_phase_circuit.count_ops())}")
    print(f"circuit depth: {black_phase_circuit.depth()}")

    if args.no_statevector_run:
        return

    equivalence_error = twirling_equivalence_error(
        problem,
        show_blocks[args.show_block],
        initial_state,
        show_trotter,
        color=args.show_color,
        color_index=args.show_block,
        rng=display_rng,
        checks=args.twirl_equivalence_checks,
        spam_twirling=spam_twirling,
        gate_twirling=gate_twirling,
    )
    print(
        "ideal full-twirl equivalence max probability error "
        f"({args.twirl_equivalence_checks} random frame(s)): {equivalence_error:.3e}"
    )

    final_state, history = run_checkerboard_qemcmc(
        problem,
        black_blocks,
        white_blocks,
        initial_state,
        schedule,
        sweeps=args.sweeps,
        rng=rng,
        spam_twirling=spam_twirling,
        gate_twirling=gate_twirling,
    )
    print(REPORT_SEPARATOR)
    print(
        f"Checkerboard simulation "
        f"({args.sweeps} sweep(s), Layden-randomized parameters)"
    )
    for item in history:
        rejected = item.total_blocks - item.accepted_blocks - item.self_proposals
        if item.accepted_moves:
            move_text = "; ".join(
                "block "
                f"{move.block_index}: {move.spins_changed} spins, "
                f"dE={move.dE:.6g}, gamma={move.gamma:.5g}, "
                f"r={move.trotter_steps}, t={move.total_time:.5g}"
                for move in item.accepted_moves
            )
        else:
            move_text = "-"
        print(
            f"sweep {item.sweep:>2} {item.color:>5}: "
            f"accepted {item.accepted_blocks}/{item.total_blocks}, "
            f"rejected {rejected}, "
            f"self {item.self_proposals}, "
            f"changed spins total={item.accepted_spin_changes}, "
            f"moves=[{move_text}], "
            f"energy={item.energy: .8g}"
        )
    print(f"final energy: {problem.energy(final_state): .8g}")


if __name__ == "__main__":
    main()
