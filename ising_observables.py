"""Ferromagnetic Ising observables from the untwirled checkerboard QeMCMC circuit.

This script reuses the non-SPAM-twirled Qiskit block proposal implemented in
qiskit_checkerboard_circuits.py.  The generated model is a 120-spin open-boundary
ferromagnetic Ising lattice by default:

    E(s) = s.T J s, with J_ij = -coupling for nearest-neighbor bonds and h = 0.

No physical external field is added.  The only fields seen by a block proposal
are the checkerboard frozen-boundary effective fields constructed by the circuit
implementation.

Two proposal-accounting traces are produced from the same Markov chain:

* color_round: one proposal unit is a parallel checkerboard color phase
  (black or white), so one full checkerboard sweep contains two proposal units.
* block_proposal: one proposal unit is one 15-spin block proposal, so one full
  default 10x12 checkerboard sweep contains eight proposal units.

Outputs:
  proposal_trace.csv      observables after every post-burn-in proposal unit
  summary.csv             means, autocorrelation times, and acceptance stats
  metadata.json           run settings and definitions
  plots/*.png             optional navy/orange diagnostic traces
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import shutil

import numpy as np

from checkerboard import (
    IsingProblem,
    LatticeSpec,
    ParameterSchedule,
    build_phase_plans,
    get_rectangular_checkerboard_blocks,
    lattice_index,
    metropolis_accept,
    problem_fingerprint,
    sample_block_output_statevector,
    sample_trotter_specs,
    validate_lattice_spec,
    validate_parameter_schedule,
    validate_same_color_independence,
    validate_spin_config,
)


# Plot colors used throughout the project.
NAVY = "#0b1f3a"
ORANGE = "#f28e2b"

# Proposal-accounting bases written to proposal_trace.csv and summary.csv.
BASIS_COLOR_ROUND = "color_round"
BASIS_BLOCK_PROPOSAL = "block_proposal"

# Script defaults.  The default temperature is Onsager's infinite-volume
# critical temperature for the 2D square-lattice ferromagnetic Ising model
# with coupling strength |J| = 1.
DEFAULT_TEMPERATURE = 2.269185314213022
DEFAULT_COUPLING = 1.0
DEFAULT_BURN_IN_SWEEPS = 25
DEFAULT_PRODUCTION_SWEEPS = 50
DEFAULT_SEED = 1234
DEFAULT_GAMMA_MIN = 0.25
DEFAULT_GAMMA_MAX = 0.6
DEFAULT_GAMMA_GRID_SIZE = 20
DEFAULT_R_MIN = 2
DEFAULT_R_MAX = 25
DEFAULT_DELTA_T = 0.8
DEFAULT_PROGRESS_INTERVAL = 10


@dataclass(frozen=True)
class UnitStats:
    """Acceptance statistics for one proposal-accounting unit."""

    proposed_blocks: int = 0
    accepted_blocks: int = 0
    rejected_blocks: int = 0
    self_proposals: int = 0
    accepted_spin_changes: int = 0
    accepted_downhill: int = 0
    accepted_uphill: int = 0
    accepted_neutral: int = 0

    def __add__(self, other: "UnitStats") -> "UnitStats":
        return UnitStats(
            proposed_blocks=self.proposed_blocks + other.proposed_blocks,
            accepted_blocks=self.accepted_blocks + other.accepted_blocks,
            rejected_blocks=self.rejected_blocks + other.rejected_blocks,
            self_proposals=self.self_proposals + other.self_proposals,
            accepted_spin_changes=self.accepted_spin_changes + other.accepted_spin_changes,
            accepted_downhill=self.accepted_downhill + other.accepted_downhill,
            accepted_uphill=self.accepted_uphill + other.accepted_uphill,
            accepted_neutral=self.accepted_neutral + other.accepted_neutral,
        )


@dataclass(frozen=True)
class ProposalRecord:
    """Observable row after one proposal-accounting unit."""

    basis: str
    proposal_index: int
    sweep: int
    color: str
    block_index: int | str
    energy: float
    energy_per_spin: float
    magnetization: float
    abs_magnetization: float
    proposed_blocks: int
    accepted_blocks: int
    rejected_blocks: int
    self_proposals: int
    accepted_spin_changes: int
    accepted_downhill: int
    accepted_uphill: int
    accepted_neutral: int


def build_2d_open_ferromagnetic_ising(
    spec: LatticeSpec,
    *,
    temperature: float,
    coupling: float = 1.0,
) -> IsingProblem:
    """Build an open-boundary nearest-neighbor ferromagnetic Ising model."""

    validate_lattice_spec(spec)
    if coupling <= 0.0:
        raise ValueError("--coupling must be positive for the ferromagnetic model.")

    J = np.zeros((spec.n, spec.n), dtype=float)
    h = np.zeros(spec.n, dtype=float)
    bond_value = -float(coupling)

    for row in range(spec.rows):
        for col in range(spec.cols):
            site = lattice_index(row, col, spec.cols)
            if col + 1 < spec.cols:
                neighbor = lattice_index(row, col + 1, spec.cols)
                left, right = sorted((site, neighbor))
                J[left, right] = bond_value
            if row + 1 < spec.rows:
                neighbor = lattice_index(row + 1, col, spec.cols)
                left, right = sorted((site, neighbor))
                J[left, right] = bond_value

    return IsingProblem(J=J, h=h, T=float(temperature))


def stable_rng(seed: int, *keys: int) -> np.random.Generator:
    """Create a reproducible generator from a master seed and integer keys."""

    return np.random.default_rng(np.random.SeedSequence([int(seed), *map(int, keys)]))


def ising_energy(problem: IsingProblem, state: np.ndarray) -> float:
    return problem.energy(state)


def magnetization_per_spin(state: np.ndarray) -> float:
    return float(np.mean(np.asarray(state, dtype=int)))


def naive_standard_error(values: list[float]) -> float:
    """Return the uncorrected-by-autocorrelation standard error."""

    if len(values) <= 1:
        return float("nan")
    array = np.asarray(values, dtype=float)
    return float(np.std(array, ddof=1) / np.sqrt(array.size))


def integrated_autocorrelation_time(values: list[float], *, window_constant: float = 5.0) -> float:
    """Estimate tau_int with an FFT autocorrelation and automatic positive window."""

    series = np.asarray(values, dtype=float)
    n = series.size
    if n < 4:
        return float("nan")

    centered = series - np.mean(series)
    variance = float(np.dot(centered, centered) / n)
    if variance <= 0.0:
        return 0.5

    fft_size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.fft(centered, fft_size)
    autocov = np.fft.ifft(spectrum * np.conjugate(spectrum)).real[:n]
    autocov /= np.arange(n, 0, -1)
    autocorr = autocov / autocov[0]

    tau = 0.5
    for lag in range(1, n):
        if autocorr[lag] <= 0.0:
            break
        tau += float(autocorr[lag])
        if lag >= window_constant * tau:
            break
    return float(tau)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def proposal_unit_label(basis: str) -> str:
    """Human-readable description of one proposal-accounting unit."""

    if basis == BASIS_COLOR_ROUND:
        return "parallel checkerboard color phase"
    if basis == BASIS_BLOCK_PROPOSAL:
        return "single 15-spin block proposal"
    raise ValueError(f"Unknown proposal basis: {basis}")


def proposals_per_checkerboard_sweep(basis: str, blocks_per_sweep: int) -> int:
    """Return how many proposal-accounting units make one checkerboard sweep."""

    if basis == BASIS_COLOR_ROUND:
        return 2
    if basis == BASIS_BLOCK_PROPOSAL:
        return blocks_per_sweep
    raise ValueError(f"Unknown proposal basis: {basis}")


def observe_state(
    *,
    basis: str,
    proposal_index: int,
    sweep: int,
    color: str,
    block_index: int | str,
    problem: IsingProblem,
    state: np.ndarray,
    stats: UnitStats,
) -> ProposalRecord:
    """Measure observables for the current spin state."""

    energy = ising_energy(problem, state)
    magnetization = magnetization_per_spin(state)
    return ProposalRecord(
        basis=basis,
        proposal_index=proposal_index,
        sweep=sweep,
        color=color,
        block_index=block_index,
        energy=energy,
        energy_per_spin=energy / problem.n,
        magnetization=magnetization,
        abs_magnetization=abs(magnetization),
        proposed_blocks=stats.proposed_blocks,
        accepted_blocks=stats.accepted_blocks,
        rejected_blocks=stats.rejected_blocks,
        self_proposals=stats.self_proposals,
        accepted_spin_changes=stats.accepted_spin_changes,
        accepted_downhill=stats.accepted_downhill,
        accepted_uphill=stats.accepted_uphill,
        accepted_neutral=stats.accepted_neutral,
    )


def run_checkerboard_sweep_with_proposal_records(
    problem: IsingProblem,
    black_blocks: list[np.ndarray],
    white_blocks: list[np.ndarray],
    state: np.ndarray,
    schedule: ParameterSchedule,
    rng: np.random.Generator,
    *,
    sweep: int,
    record: bool,
    next_color_proposal_index: int,
    next_block_proposal_index: int,
) -> tuple[np.ndarray, list[ProposalRecord], list[ProposalRecord], int, int]:
    """Run one sweep and optionally record both proposal-accounting bases."""

    state = np.asarray(state, dtype=int).copy()
    validate_spin_config(state, problem.n)
    color_records: list[ProposalRecord] = []
    block_records: list[ProposalRecord] = []

    for color, blocks in (("black", black_blocks), ("white", white_blocks)):
        phase_reference = state.copy()
        # Same-color blocks are uncoupled.  All plans in a color phase are
        # therefore built from the same frozen opposite-color boundary.
        trotter_specs = sample_trotter_specs(schedule, len(blocks), rng)
        plans = build_phase_plans(
            problem,
            blocks,
            phase_reference,
            color=color,
            trotter_specs=trotter_specs,
        )
        phase_stats = UnitStats()

        for plan in plans:
            proposed_block = sample_block_output_statevector(plan, rng)
            spins_changed = int(np.count_nonzero(proposed_block != plan.current_spins))
            block_stats = UnitStats(proposed_blocks=1)

            if spins_changed == 0:
                block_stats = block_stats + UnitStats(self_proposals=1)
            else:
                proposed_state = state.copy()
                proposed_state[plan.global_indices] = proposed_block
                current_energy = ising_energy(problem, state)
                dE = ising_energy(problem, proposed_state) - current_energy

                if metropolis_accept(dE, problem.T, rng):
                    state = proposed_state
                    block_stats = block_stats + UnitStats(
                        accepted_blocks=1,
                        accepted_spin_changes=spins_changed,
                        accepted_downhill=1 if dE < 0.0 else 0,
                        accepted_uphill=1 if dE > 0.0 else 0,
                        accepted_neutral=1 if dE == 0.0 else 0,
                    )
                else:
                    block_stats = block_stats + UnitStats(rejected_blocks=1)

            phase_stats = phase_stats + block_stats
            if record:
                block_records.append(
                    observe_state(
                        basis=BASIS_BLOCK_PROPOSAL,
                        proposal_index=next_block_proposal_index,
                        sweep=sweep,
                        color=color,
                        block_index=plan.color_index,
                        problem=problem,
                        state=state,
                        stats=block_stats,
                    )
                )
                next_block_proposal_index += 1

        if record:
            color_records.append(
                observe_state(
                    basis=BASIS_COLOR_ROUND,
                    proposal_index=next_color_proposal_index,
                    sweep=sweep,
                    color=color,
                    block_index="all",
                    problem=problem,
                    state=state,
                    stats=phase_stats,
                )
            )
            next_color_proposal_index += 1

    return (
        state,
        color_records,
        block_records,
        next_color_proposal_index,
        next_block_proposal_index,
    )


def summarize_records(
    records: list[ProposalRecord],
    *,
    basis: str,
    problem: IsingProblem,
    production_sweeps: int,
    blocks_per_sweep: int,
) -> dict[str, float | int | str]:
    """Summarize observables and tau_int in the units of one proposal basis."""

    if not records:
        raise ValueError("Cannot summarize an empty proposal trace.")

    energy_per_spin = [record.energy_per_spin for record in records]
    magnetization = [record.magnetization for record in records]
    abs_magnetization = [record.abs_magnetization for record in records]

    stats = UnitStats()
    for record in records:
        stats = stats + UnitStats(
            proposed_blocks=record.proposed_blocks,
            accepted_blocks=record.accepted_blocks,
            rejected_blocks=record.rejected_blocks,
            self_proposals=record.self_proposals,
            accepted_spin_changes=record.accepted_spin_changes,
            accepted_downhill=record.accepted_downhill,
            accepted_uphill=record.accepted_uphill,
            accepted_neutral=record.accepted_neutral,
        )

    proposals_per_sweep = 2 if basis == BASIS_COLOR_ROUND else blocks_per_sweep
    tau_energy = integrated_autocorrelation_time(energy_per_spin)
    tau_m = integrated_autocorrelation_time(magnetization)
    tau_abs_m = integrated_autocorrelation_time(abs_magnetization)
    proposals_per_sweep = proposals_per_checkerboard_sweep(basis, blocks_per_sweep)

    return {
        "basis": basis,
        "proposal_unit": proposal_unit_label(basis),
        "n": problem.n,
        "temperature": problem.T,
        "production_sweeps": production_sweeps,
        "proposals_per_checkerboard_sweep": proposals_per_sweep,
        "recorded_proposals": len(records),
        "energy_per_spin_mean": float(np.mean(energy_per_spin)),
        "energy_per_spin_stderr_naive": naive_standard_error(energy_per_spin),
        "magnetization_per_spin_mean": float(np.mean(magnetization)),
        "magnetization_per_spin_stderr_naive": naive_standard_error(magnetization),
        "abs_magnetization_mean": float(np.mean(abs_magnetization)),
        "abs_magnetization_stderr_naive": naive_standard_error(abs_magnetization),
        "tau_int_energy_per_spin_proposals": tau_energy,
        "tau_int_magnetization_proposals": tau_m,
        "tau_int_abs_magnetization_proposals": tau_abs_m,
        "tau_int_energy_per_spin_checkerboard_sweeps": tau_energy / proposals_per_sweep,
        "tau_int_magnetization_checkerboard_sweeps": tau_m / proposals_per_sweep,
        "tau_int_abs_magnetization_checkerboard_sweeps": tau_abs_m / proposals_per_sweep,
        "proposed_blocks": stats.proposed_blocks,
        "accepted_blocks": stats.accepted_blocks,
        "rejected_blocks": stats.rejected_blocks,
        "self_proposals": stats.self_proposals,
        "raw_block_acceptance_rate": safe_ratio(stats.accepted_blocks, stats.proposed_blocks),
        "nonself_block_acceptance_rate": safe_ratio(
            stats.accepted_blocks,
            stats.proposed_blocks - stats.self_proposals,
        ),
        "self_proposal_rate": safe_ratio(stats.self_proposals, stats.proposed_blocks),
        "accepted_spin_changes": stats.accepted_spin_changes,
        "accepted_spin_changes_per_checkerboard_sweep": safe_ratio(
            stats.accepted_spin_changes,
            production_sweeps,
        ),
        "accepted_spin_changes_per_proposal": safe_ratio(
            stats.accepted_spin_changes,
            len(records),
        ),
        "accepted_downhill": stats.accepted_downhill,
        "accepted_uphill": stats.accepted_uphill,
        "accepted_neutral": stats.accepted_neutral,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def prepare_output_dir(path: Path, *, overwrite: bool) -> Path:
    path = path.resolve()
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Output path exists and is not a directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_output_dir() -> Path:
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("ferromagnetic_ising_observables") / f"qemcmc_ising_{stamp}"


def proposal_record_fieldnames() -> list[str]:
    return [
        "basis",
        "proposal_index",
        "sweep",
        "color",
        "block_index",
        "energy",
        "energy_per_spin",
        "magnetization",
        "abs_magnetization",
        "proposed_blocks",
        "accepted_blocks",
        "rejected_blocks",
        "self_proposals",
        "accepted_spin_changes",
        "accepted_downhill",
        "accepted_uphill",
        "accepted_neutral",
    ]


def summary_fieldnames() -> list[str]:
    return [
        "basis",
        "proposal_unit",
        "n",
        "temperature",
        "production_sweeps",
        "proposals_per_checkerboard_sweep",
        "recorded_proposals",
        "energy_per_spin_mean",
        "energy_per_spin_stderr_naive",
        "magnetization_per_spin_mean",
        "magnetization_per_spin_stderr_naive",
        "abs_magnetization_mean",
        "abs_magnetization_stderr_naive",
        "tau_int_energy_per_spin_proposals",
        "tau_int_magnetization_proposals",
        "tau_int_abs_magnetization_proposals",
        "tau_int_energy_per_spin_checkerboard_sweeps",
        "tau_int_magnetization_checkerboard_sweeps",
        "tau_int_abs_magnetization_checkerboard_sweeps",
        "proposed_blocks",
        "accepted_blocks",
        "rejected_blocks",
        "self_proposals",
        "raw_block_acceptance_rate",
        "nonself_block_acceptance_rate",
        "self_proposal_rate",
        "accepted_spin_changes",
        "accepted_spin_changes_per_checkerboard_sweep",
        "accepted_spin_changes_per_proposal",
        "accepted_downhill",
        "accepted_uphill",
        "accepted_neutral",
    ]


def make_plots(
    *,
    output_dir: Path,
    color_records: list[ProposalRecord],
    block_records: list[ProposalRecord],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("warning: matplotlib is not installed; skipping plots.", flush=True)
        return

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(
        [record.proposal_index for record in color_records],
        [record.energy_per_spin for record in color_records],
        color=NAVY,
        linewidth=1.2,
        label="color round",
    )
    ax.plot(
        [record.proposal_index for record in block_records],
        [record.energy_per_spin for record in block_records],
        color=ORANGE,
        linewidth=0.8,
        alpha=0.75,
        label="block proposal",
    )
    ax.set_xlabel("Proposal index")
    ax.set_ylabel("Energy per spin")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "energy_per_spin_trace.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(
        [record.proposal_index for record in color_records],
        [record.abs_magnetization for record in color_records],
        color=NAVY,
        linewidth=1.2,
        label="color round",
    )
    ax.plot(
        [record.proposal_index for record in block_records],
        [record.abs_magnetization for record in block_records],
        color=ORANGE,
        linewidth=0.8,
        alpha=0.75,
        label="block proposal",
    )
    ax.set_xlabel("Proposal index")
    ax.set_ylabel("Absolute magnetization")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "abs_magnetization_trace.png", dpi=180)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--cols", type=int, default=12)
    parser.add_argument("--block-rows", type=int, default=5)
    parser.add_argument("--block-cols", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--coupling", type=float, default=DEFAULT_COUPLING)
    parser.add_argument("--burn-in-sweeps", type=int, default=DEFAULT_BURN_IN_SWEEPS)
    parser.add_argument("--production-sweeps", type=int, default=DEFAULT_PRODUCTION_SWEEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--gamma-min", type=float, default=DEFAULT_GAMMA_MIN)
    parser.add_argument("--gamma-max", type=float, default=DEFAULT_GAMMA_MAX)
    parser.add_argument("--gamma-grid-size", type=int, default=DEFAULT_GAMMA_GRID_SIZE)
    parser.add_argument("--r-min", type=int, default=DEFAULT_R_MIN)
    parser.add_argument("--r-max", type=int, default=DEFAULT_R_MAX)
    parser.add_argument("--delta-t", type=float, default=DEFAULT_DELTA_T)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=DEFAULT_PROGRESS_INTERVAL)
    return parser


def validate_args(args: argparse.Namespace, spec: LatticeSpec, schedule: ParameterSchedule) -> None:
    validate_lattice_spec(spec)
    validate_parameter_schedule(schedule)
    if math.isnan(args.temperature) or args.temperature < 0.0:
        raise ValueError("--temperature must be a non-negative number.")
    if args.coupling <= 0.0:
        raise ValueError("--coupling must be positive.")
    if args.burn_in_sweeps < 0:
        raise ValueError("--burn-in-sweeps must be non-negative.")
    if args.production_sweeps < 1:
        raise ValueError("--production-sweeps must be at least 1.")


def main() -> None:
    args = build_arg_parser().parse_args()
    spec = LatticeSpec(
        rows=args.rows,
        cols=args.cols,
        block_rows=args.block_rows,
        block_cols=args.block_cols,
    )
    schedule = ParameterSchedule(
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        gamma_grid_size=args.gamma_grid_size,
        r_min=args.r_min,
        r_max=args.r_max,
        delta_t=args.delta_t,
    )
    validate_args(args, spec, schedule)

    output_dir = prepare_output_dir(args.output_dir or default_output_dir(), overwrite=args.overwrite)
    problem = build_2d_open_ferromagnetic_ising(
        spec,
        temperature=args.temperature,
        coupling=args.coupling,
    )
    black_blocks, white_blocks = get_rectangular_checkerboard_blocks(spec)
    validate_same_color_independence(problem, black_blocks)
    validate_same_color_independence(problem, white_blocks)

    rng = stable_rng(args.seed, 1000)
    state = rng.choice(np.array([-1, 1], dtype=int), size=problem.n)

    next_color_proposal_index = 1
    next_block_proposal_index = 1
    color_records: list[ProposalRecord] = []
    block_records: list[ProposalRecord] = []
    total_sweeps = args.burn_in_sweeps + args.production_sweeps

    print("QeMCMC ferromagnetic Ising observable run")
    print(f"output_dir: {output_dir}")
    print(f"lattice: {spec.rows}x{spec.cols} open boundary, n={spec.n}")
    print(f"block shape: {spec.block_rows}x{spec.block_cols}, block size={spec.block_size}")
    print(f"blocks: {len(black_blocks)} black, {len(white_blocks)} white")
    print(f"temperature: {args.temperature:g}")
    print(f"coupling: {args.coupling:g}; code convention uses J_ij=-coupling")
    print(f"external field norm: {np.linalg.norm(problem.h):.6g}")
    print(f"problem fingerprint: {problem_fingerprint(problem)}")
    print(
        "proposal accounting: color_round has 2 units/sweep; "
        f"block_proposal has {len(black_blocks) + len(white_blocks)} units/sweep"
    )

    for sweep in range(1, total_sweeps + 1):
        record = sweep > args.burn_in_sweeps
        (
            state,
            new_color_records,
            new_block_records,
            next_color_proposal_index,
            next_block_proposal_index,
        ) = run_checkerboard_sweep_with_proposal_records(
            problem,
            black_blocks,
            white_blocks,
            state,
            schedule,
            rng,
            sweep=sweep,
            record=record,
            next_color_proposal_index=next_color_proposal_index,
            next_block_proposal_index=next_block_proposal_index,
        )
        color_records.extend(new_color_records)
        block_records.extend(new_block_records)

        if args.progress_interval > 0 and sweep % args.progress_interval == 0:
            print(
                f"  sweep={sweep}/{total_sweeps} "
                f"recorded_color={len(color_records)} "
                f"recorded_block={len(block_records)}",
                flush=True,
            )

    summary_rows = [
        summarize_records(
            color_records,
            basis=BASIS_COLOR_ROUND,
            problem=problem,
            production_sweeps=args.production_sweeps,
            blocks_per_sweep=len(black_blocks) + len(white_blocks),
        ),
        summarize_records(
            block_records,
            basis=BASIS_BLOCK_PROPOSAL,
            problem=problem,
            production_sweeps=args.production_sweeps,
            blocks_per_sweep=len(black_blocks) + len(white_blocks),
        ),
    ]

    trace_rows = [asdict(record) for record in color_records + block_records]
    write_csv(output_dir / "proposal_trace.csv", trace_rows, proposal_record_fieldnames())
    write_csv(output_dir / "summary.csv", summary_rows, summary_fieldnames())

    metadata = {
        "script": Path(__file__).name,
        "sampler": (
            "Untwirled Trotterized Qiskit checkerboard QeMCMC, simulated "
            "block-by-block with exact 15-qubit statevector probabilities."
        ),
        "lattice": asdict(spec),
        "schedule": asdict(schedule),
        "temperature": args.temperature,
        "coupling": args.coupling,
        "burn_in_sweeps": args.burn_in_sweeps,
        "production_sweeps": args.production_sweeps,
        "seed": args.seed,
        "problem_fingerprint": problem_fingerprint(problem),
        "energy_convention": (
            "E(s)=s.T J s + s.T h.  Ferromagnetic nearest-neighbor bonds use "
            "J_ij=-coupling; h is exactly zero."
        ),
        "effective_field_note": (
            "No physical external field is added.  Frozen neighboring spins are "
            "folded into h_eff inside qiskit_checkerboard_circuits.effective_block_problem."
        ),
        "proposal_bases": {
            BASIS_COLOR_ROUND: proposal_unit_label(BASIS_COLOR_ROUND),
            BASIS_BLOCK_PROPOSAL: proposal_unit_label(BASIS_BLOCK_PROPOSAL),
        },
        "observable_definitions": {
            "energy_per_spin": "E(s) / N.",
            "magnetization": "(1 / N) sum_i s_i.",
            "abs_magnetization": "Absolute value of magnetization.",
            "tau_int": (
                "Integrated autocorrelation time estimated from consecutive "
                "post-burn-in proposal records with no thinning."
            ),
            "stderr_naive": (
                "Standard error ignoring autocorrelation; use tau_int for "
                "production error analysis."
            ),
        },
        "outputs": {
            "proposal_trace.csv": "Post-burn-in observables after each proposal unit.",
            "summary.csv": "Means, tau_int values, and acceptance statistics by proposal basis.",
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    if not args.no_plots:
        make_plots(
            output_dir=output_dir,
            color_records=color_records,
            block_records=block_records,
        )

    print("done")
    print(f"wrote: {output_dir}")
    for row in summary_rows:
        print(
            f"{row['basis']}: E/N={float(row['energy_per_spin_mean']):.6g}, "
            f"<m>={float(row['magnetization_per_spin_mean']):.6g}, "
            f"<|m|>={float(row['abs_magnetization_mean']):.6g}, "
            f"tau_|m|={float(row['tau_int_abs_magnetization_proposals']):.6g} proposals"
        )


if __name__ == "__main__":
    main()

