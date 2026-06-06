import numpy as np
import scipy.linalg as la
from itertools import product
import multiprocessing
import time

# =============================================================================
# QUANTUM MCMC CORE MATHEMATICS
# =============================================================================

class ProblemInstance:
    """
    Encodes a problem instance, which is defined on n spins by an n*n coupling matrix J, an 
    n-dimensional local field vector h, and a non-negative temperature T.
    """
    def __init__(self, J, h, T=None, precomp_E = True):
        J = np.asarray(J, dtype=float)
        h = np.asarray(h, dtype=float)
        if h.ndim != 1:
            raise ValueError('h must be a one-dimensional array.')
        if J.shape != (h.size, h.size):
            raise ValueError('J must be a square matrix with dimensions matching h.')
        if np.any(np.triu(J, k=1) != J):
            raise ValueError('J must be upper-triangular with 0s along the diagonal')

        self.J = J
        self.h = h
        self.T = T 
        self.n = self.h.size
        
        # Scale for quantum Hamiltonian (Prevents variance from blowing up or shrinking)
        scale_denominator = la.norm(self.J, ord='fro')**2 + la.norm(self.h, ord=2)**2
        scale = 1.0 if scale_denominator == 0 else ( self.n / scale_denominator )**0.5
        self.J_quantum = self.J * scale 
        self.h_quantum = self.h * scale

        if precomp_E:
            self.E_arr = np.zeros((2**self.n))
            for i in range(2**self.n):
                config = int2spinconf(i, self.n)
                self.E_arr[i] = config @ self.J @ config + config @ self.h 

def RandomProblemInstance(n, connectivity, T=None, precomp_E=True):
    h = np.random.randn(n)
    if connectivity=='line':
        J = np.diag(np.random.randn(n-1), k=1)
    elif connectivity=='full':
        J = np.triu(np.random.randn(n,n), k=1)
    else:
        raise ValueError("connectivity must be either 'line' or 'full'")
    return ProblemInstance(J, h, T=T, precomp_E=precomp_E)

def my_kron(arr_list):
    if len(arr_list) == 1:
        return arr_list[0]
    else:
        return np.kron(arr_list[0], my_kron(arr_list[1:]))

def X(i, n):
    if i<0 or i>=n or n<1: raise ValueError('Bad value of i and/or n.')
    X_list = [np.array([[0,1],[1,0]]) if j==i else np.eye(2) for j in range(n)]
    return my_kron(X_list)

def Y(i, n):
    if i<0 or i>=n or n<1: raise ValueError('Bad value of i and/or n.')
    Y_list = [np.array([[0,-1j],[1j,0]]) if j==i else np.eye(2) for j in range(n)]
    return my_kron(Y_list)

def Z(i, n):
    if i<0 or i>=n or n<1: raise ValueError('Bad value of i and/or n.')
    Z_list = [np.array([[1,0],[0,-1]]) if j==i else np.eye(2) for j in range(n)]
    return my_kron(Z_list)

def int2spinconf(i, n):
    if i<0 or i>2**n-1: raise ValueError('i out of range')
    bin_list = list( np.binary_repr(i,n))
    return np.array([1 if bit=='0' else -1 for bit in bin_list])

def spinconf2int(config):
    bin_str = ''.join(['0' if spin==1 else '1' for spin in config])
    return int(bin_str, 2)

def hamming_dist(int1, int2):
    diff_int = np.bitwise_xor(int1, int2)
    diff_bin = [int(bit) for bit in np.binary_repr(diff_int)]
    return sum(diff_bin)

def local_proposal_mat(n):
    return sum([X(i,n) for i in range(n)])/n

def uniform_proposal_mat(n):
    return np.ones((2**n, 2**n))/2**n

def clean_column_stochastic_matrix(matrix, tol=1e-12):
    """
    Removes tiny numerical artifacts from a column-stochastic matrix and
    renormalizes columns. Raises if the matrix contains meaningful negatives.
    """
    matrix = np.real_if_close(matrix).astype(float)
    matrix[np.abs(matrix) < tol] = 0.0
    if np.any(matrix < -tol):
        raise ValueError('Matrix contains negative probabilities beyond numerical tolerance.')
    matrix = np.maximum(matrix, 0.0)
    col_sums = matrix.sum(axis=0)
    if np.any(col_sums <= 0):
        raise ValueError('Matrix contains a zero-probability column.')
    return matrix / col_sums


def clean_symmetric_stochastic_matrix(matrix, tol=1e-10):
    """
    Cleans an ideal quantum proposal matrix while preserving the two properties
    needed by the Layden Metropolis simplification: symmetry and stochasticity.
    This function checks the quantum walker; it does not reshape a bad proposal
    into a different Markov kernel.
    """
    matrix = np.real_if_close(matrix).astype(float)
    matrix = (matrix + matrix.T)/2
    matrix[np.abs(matrix) < tol] = 0.0
    if np.any(matrix < -tol):
        raise ValueError('Quantum proposal contains negative probabilities beyond numerical tolerance.')
    matrix = np.maximum(matrix, 0.0)

    col_sums = matrix.sum(axis=0)
    if np.any(col_sums <= 0):
        raise ValueError('Quantum proposal contains a zero-probability column.')
    col_err = np.max(np.abs(matrix.sum(axis=0) - 1))
    sym_err = np.max(np.abs(matrix - matrix.T))
    if col_err > tol or sym_err > tol:
        raise ValueError('Quantum proposal is not symmetric and stochastic within tolerance.')
    return matrix


def clean_transition_matrix(matrix, tol=1e-12):
    """
    Removes tiny numerical artifacts from a transition matrix while preserving
    off-diagonal Metropolis probabilities. Any column-sum residual is assigned
    to the self-transition probability.
    """
    matrix = np.real_if_close(matrix).astype(float)
    matrix[np.abs(matrix) < tol] = 0.0
    if np.any(matrix < -tol):
        raise ValueError('Transition matrix contains negative probabilities beyond numerical tolerance.')
    matrix = np.maximum(matrix, 0.0)
    residual = 1.0 - matrix.sum(axis=0)
    matrix = matrix + np.diag(residual)
    if np.any(np.diag(matrix) < -tol):
        raise ValueError('Transition matrix has invalid self-transition probabilities.')
    matrix[np.abs(matrix) < tol] = 0.0
    col_err = np.max(np.abs(matrix.sum(axis=0) - 1))
    if col_err > tol:
        raise ValueError('Transition matrix is not column-stochastic within tolerance.')
    return matrix


def quantum_proposal_mat_ideal(problem_inst):
    def cont_eig(Dlambda):
        t_0, t_f = 2, 20
        x = np.sin(Dlambda*t_f) - np.sin(Dlambda*t_0) 
        return np.divide(2*x/(t_f-t_0), Dlambda, out=np.ones_like(Dlambda), where=(Dlambda!=0) ) 

    J_Q = problem_inst.J_quantum 
    h_Q = problem_inst.h_quantum
    n = problem_inst.n 

    H_z = sum([J_Q[i,j]*Z(i,n) @ Z(j,n) for i in range(n) for j in range(n)]) + sum([h_Q[i]*Z(i,n) for i in range(n)])
    H_x = sum([X(i,n) for i in range(n)])

    d = 2**n
    a = np.arange(d**2)
    mask = (a//d >= a%d)
    ones = np.ones(d)

    c_lims = [0.25, 0.6] 
    c_steps = 20 
    c_starts, step_size = np.linspace(c_lims[0], c_lims[1], num=c_steps, endpoint=False, retstep=True)
    c_mids = c_starts + step_size/2

    prop_list = [None]*c_mids.size
    for c_ind, c in enumerate(c_mids):
        H = (1-c)*H_z + c*H_x
        vals, vecs = la.eigh(H)

        vals_diff = (np.kron(vals, ones) - np.kron(ones, vals))[mask]
        M = la.khatri_rao(vecs.T, vecs.T)[mask]
        prop_list[c_ind] = M.T * cont_eig(vals_diff) @ M 
    
    proposal_mat = sum(prop_list)/len(c_mids)
    return clean_symmetric_stochastic_matrix(proposal_mat)

def make_transition_mat(problem_inst, proposal_mat, acceptance='metropolis'):
    if problem_inst.T is None: raise TypeError('Temperature T is undefined.')
    if problem_inst.T < 0: raise ValueError('Temperature T cannot be negative.')
    if not hasattr(problem_inst, 'E_arr'):
        raise TypeError('ProblemInstance must have precomputed E_arr to build a dense transition matrix.')

    n = problem_inst.n
    T = problem_inst.T
    E_rowstack = np.tile(problem_inst.E_arr, (2**n,1))
    E_diff = E_rowstack.T - E_rowstack

    uphill_or_level_moves = (E_diff >= 0)
    uphill_moves = (E_diff > 0)
    downhill_moves = (E_diff < 0)
    level_moves = (E_diff == 0)

    if acceptance=='metropolis':
        if T>0:
            pi_ratio = np.exp(-E_diff/T, where=uphill_moves, out=np.ones_like(E_diff))
        if T==0:
            pi_ratio = np.where(uphill_moves, 0., 1.)
        A = pi_ratio

    elif acceptance=='glauber':
        if T>0:
            pi_ratio = np.exp(-E_diff/T, where=uphill_or_level_moves, out=E_diff*np.nan) 
            pi_ratio_inv = np.exp(E_diff/T, where=downhill_moves, out=E_diff*np.nan)
        if T==0:
            A = np.where(uphill_moves, 0., np.where(level_moves, 0.5, 1.))
        else:
            A = np.where(uphill_or_level_moves, pi_ratio/(1+pi_ratio), (1+pi_ratio_inv)**(-1) )
    else:
        raise ValueError("acceptance must be either 'metropolis' or 'glauber'")

    P = A * proposal_mat
    np.fill_diagonal(P, 0)
    diag = np.ones(2**n) - np.sum(P, axis=0)
    P = P + np.diag(diag)
    return clean_transition_matrix(P)

def abs_spectral_gap(transition_mat):
    dist = np.sort( 1-np.abs(la.eigvals(transition_mat)) )
    delta = np.min(dist[1:])
    dist_lazy = np.sort( 1-np.abs(1/2 + la.eigvals(transition_mat)/2) )
    delta_lazy = np.min(dist_lazy[1:])
    return delta, delta_lazy

def generate_move(transition_mat, state):
    probabilities = np.real_if_close(transition_mat[:, state]).astype(float)
    probabilities = np.where(np.abs(probabilities) < 1e-14, 0.0, probabilities)
    if np.any(probabilities < 0):
        raise ValueError('Transition matrix column contains negative probabilities.')
    total = probabilities.sum()
    if total <= 0:
        raise ValueError('Transition matrix column has zero total probability.')
    return np.random.choice(transition_mat.shape[0], p=probabilities/total)

# =============================================================================
# NEW PARALLEL CHECKERBOARD ARCHITECTURE
# =============================================================================

class _ProblemView:
    """Lightweight view of a global problem that avoids copying J into J_quantum in workers."""
    def __init__(self, J, h, T):
        self.J = J
        self.h = h
        self.T = T
        self.n = h.size


def ising_energy(problem_inst, config):
    """Energy convention used by this file: E(s) = s.T J s + s.T h."""
    config = np.asarray(config, dtype=float)
    return config @ problem_inst.J @ config + config @ problem_inst.h


def _validate_spin_config(config, expected_size):
    config = np.asarray(config)
    if config.shape != (expected_size,):
        raise ValueError(f'Expected a spin configuration of shape ({expected_size},).')
    if np.any((config != -1) & (config != 1)):
        raise ValueError('Spin configuration entries must all be -1 or +1.')


def _validate_checkerboard_partition(black_blocks, white_blocks, n):
    blocks = list(black_blocks) + list(white_blocks)
    if len(blocks) == 0:
        raise ValueError('At least one checkerboard block is required.')
    flat = np.concatenate([np.asarray(block, dtype=int) for block in blocks])
    if flat.size != n:
        raise ValueError('Checkerboard blocks must cover every spin exactly once.')
    if np.any(flat < 0) or np.any(flat >= n):
        raise ValueError('Checkerboard blocks contain out-of-range spin indices.')
    if np.unique(flat).size != n:
        raise ValueError('Checkerboard blocks contain duplicate spin indices.')


def _validate_parallel_block_independence(problem_inst, blocks, tol=1e-12):
    """
    Same-color blocks may be updated in parallel only when there are no couplings
    between distinct blocks of that color.
    """
    coupling = np.abs(problem_inst.J + problem_inst.J.T)
    for left_ind, left in enumerate(blocks):
        left = np.asarray(left, dtype=int)
        for right in blocks[left_ind + 1:]:
            right = np.asarray(right, dtype=int)
            if np.any(coupling[np.ix_(left, right)] > tol):
                raise ValueError(
                    'Same-color checkerboard blocks are coupled; parallel block updates '
                    'would not be conditionally independent for this problem.'
                )


def get_effective_problem(problem_inst, active_indices, full_state_config, precomp_E=None):
    r"""
    Creates the improved-local-group problem for one checkerboard block.

    The inactive spins are frozen and folded into the block field, so energy
    differences inside the block exactly match global energy differences with
    the boundary held fixed:
    
        h_eff[j] = h[j] + sum_{i not in block} J_sym[j, i] s[i]

    This is the coarse-grained strategy from Ferguson and Wallden, adapted to
    this file's sign convention E(s) = s.T J s + s.T h.
    """
    active_indices = np.sort(np.asarray(active_indices, dtype=int))
    if active_indices.size == 0:
        raise ValueError('A checkerboard block cannot be empty.')
    if np.any(active_indices < 0) or np.any(active_indices >= problem_inst.n):
        raise ValueError('A checkerboard block contains an out-of-range spin index.')
    if np.unique(active_indices).size != active_indices.size:
        raise ValueError('A checkerboard block contains duplicate spin indices.')

    _validate_spin_config(full_state_config, problem_inst.n)
    full_state_config = np.asarray(full_state_config, dtype=float)

    active_mask = np.zeros(problem_inst.n, dtype=bool)
    active_mask[active_indices] = True

    inactive_config = full_state_config.copy()
    inactive_config[active_mask] = 0.0 

    h_interaction = (problem_inst.J + problem_inst.J.T) @ inactive_config
    h_eff = problem_inst.h[active_indices] + h_interaction[active_indices]
    J_eff = problem_inst.J[np.ix_(active_indices, active_indices)]

    if precomp_E is None:
        precomp_E = len(active_indices) <= 12
    return ProblemInstance(J_eff, h_eff, T=problem_inst.T, precomp_E=precomp_E)

def build_2d_lattice(L):
    """
    Builds a 2D Edwards-Anderson nearest-neighbor square lattice of size L x L.
    Returns a ProblemInstance with upper triangular J (couplings) and h (local fields).
    """
    n = L * L
    J = np.zeros((n, n))
    
    # Initialize random local magnetic fields h_i ~ N(0,1)
    h = np.random.randn(n) 
    
    # Populate the nearest-neighbor spin-glass couplings J_{ij} ~ N(0,1)
    for r in range(L):
        for c in range(L):
            i = r * L + c  # Convert 2D coordinate to 1D index
            
            # Connect to Right neighbor (if not on the right edge)
            if c + 1 < L:
                j = r * L + (c + 1)
                # Maintain upper-triangular structure for ProblemInstance compatibility
                if i < j: J[i, j] = np.random.randn()
                else: J[j, i] = np.random.randn()
                
            # Connect to Bottom neighbor (if not on the bottom edge)
            if r + 1 < L:
                j = (r + 1) * L + c
                if i < j: J[i, j] = np.random.randn()
                else: J[j, i] = np.random.randn()
                
    return ProblemInstance(J, h, precomp_E=False)

def get_checkerboard_blocks(L, B):
    """
    Partitions an L x L grid into B x B blocks.
    Returns black_blocks, white_blocks (lists of active_indices arrays).
    This strictly bipartite geometry ensures conditional independence during parallel updates.
    """
    if L <= 0 or B <= 0:
        raise ValueError('L and B must be positive integers.')
    if L % B != 0:
        raise ValueError('B must divide L exactly; otherwise some spins would be dropped.')

    black_blocks = []
    white_blocks = []
    
    blocks_per_row = L // B
    
    # Iterate over the grid of blocks
    for br in range(blocks_per_row):
        for bc in range(blocks_per_row):
            block_indices = []
            
            # Map the block coordinates back to the global 1D spin indices
            for r in range(br * B, (br + 1) * B):
                for c in range(bc * B, (bc + 1) * B):
                    block_indices.append(r * L + c)
                    
            # Use parity (row + col) % 2 to alternate colors, creating the checkerboard
            if (br + bc) % 2 == 0:
                black_blocks.append(np.array(block_indices))
            else:
                white_blocks.append(np.array(block_indices))
                
    return black_blocks, white_blocks

def dynamic_local_mcmc(problem_inst, current_block_config, num_moves=100, acceptance='metropolis'):
    """
    Performs standard local Metropolis MCMC on a ProblemInstance. 
    Unlike make_transition_mat, this does not build a dense 2^n * 2^n transition 
    matrix, making it safe and efficient for classically simulating large blocks 
    (e.g., n=100) where exact diagonalization would cause an OOM crash.
    """
    if problem_inst.T is None:
        raise TypeError('Temperature T is undefined.')
    if problem_inst.T < 0:
        raise ValueError('Temperature T cannot be negative.')
    _validate_spin_config(current_block_config, problem_inst.n)

    config = current_block_config.copy()
    n = problem_inst.n
    
    # Symmetrize J for fast O(1) energy differential calculations
    J_sym = problem_inst.J + problem_inst.J.T
    h = problem_inst.h
    T = problem_inst.T
    
    for _ in range(num_moves):
        # Select a random spin within the block to propose a flip
        i = np.random.randint(n)
        
        # Calculate the energy difference \Delta E of flipping spin i.
        # \Delta E = -2 * s_i * ( \sum_j J_{ij} s_j + h_i )
        dE = -2 * config[i] * (np.dot(J_sym[i, :], config) + h[i])
        
        if acceptance == 'metropolis':
            if dE <= 0:
                accept_prob = 1.0
            elif T == 0:
                accept_prob = 0.0
            else:
                accept_prob = np.exp(-dE / T)
        elif acceptance == 'glauber':
            if T == 0:
                accept_prob = 0.0 if dE > 0 else (0.5 if dE == 0 else 1.0)
            elif dE >= 0:
                ratio = np.exp(-dE / T)
                accept_prob = ratio/(1 + ratio)
            else:
                ratio_inv = np.exp(dE / T)
                accept_prob = 1/(1 + ratio_inv)
        else:
            raise ValueError("acceptance must be either 'metropolis' or 'glauber'")

        if np.random.rand() <= accept_prob:
            config[i] *= -1 # Execute the spin flip
            
    return config


def _update_block_worker(args):
    """
    Multiprocessing worker for one improved-local checkerboard group.

    The worker freezes all spins outside block_indices, folds them into h_eff
    using get_effective_problem, then applies the ideal Layden quantum walker.
    A local-MCMC path exists only when the caller explicitly allows a classical
    fallback for oversized exact-simulation blocks.
    """
    (
        block_indices,
        global_J,
        global_h,
        global_T,
        frozen_config,
        use_quantum,
        acceptance,
        max_quantum_block_size,
        classical_sweeps_per_block,
        allow_classical_fallback,
    ) = args

    block_indices = np.sort(np.asarray(block_indices, dtype=int))
    n_block = len(block_indices)
    if not use_quantum and not allow_classical_fallback:
        raise ValueError('Checkerboard QeMCMC requires use_quantum=True unless a classical fallback is explicitly allowed.')
    use_exact_quantum = use_quantum and n_block <= max_quantum_block_size
    if use_quantum and not use_exact_quantum and not allow_classical_fallback:
        raise ValueError(
            f'Quantum checkerboard block has {n_block} spins, which exceeds '
            f'max_quantum_block_size={max_quantum_block_size}. Increase the limit, '
            'use smaller checkerboard blocks, or explicitly allow the classical fallback.'
        )

    prob_global = _ProblemView(global_J, global_h, global_T)
    eff_prob = get_effective_problem(
        prob_global,
        block_indices,
        frozen_config,
        precomp_E=use_exact_quantum,
    )
    current_block_config = np.asarray(frozen_config)[block_indices]

    if use_exact_quantum:
        prop_mat = quantum_proposal_mat_ideal(eff_prob)
        trans_mat = make_transition_mat(eff_prob, prop_mat, acceptance=acceptance)
        current_block_int = spinconf2int(current_block_config)
        new_block_int = generate_move(trans_mat, current_block_int)
        new_block_config = int2spinconf(new_block_int, n_block)
    else:
        num_moves = max(1, int(np.ceil(classical_sweeps_per_block * n_block)))
        new_block_config = dynamic_local_mcmc(
            eff_prob,
            current_block_config,
            num_moves=num_moves,
            acceptance=acceptance,
        )

    return block_indices, new_block_config

def parallel_checkerboard_update(
    problem_inst,
    current_config,
    black_blocks,
    white_blocks,
    pool,
    use_quantum=True,
    acceptance='metropolis',
    max_quantum_block_size=12,
    classical_sweeps_per_block=1,
    allow_classical_fallback=False,
):
    """
    Performs one full parallel sweep of the checkerboard lattice.
    Updates all Black blocks simultaneously (conditionally on the static White blocks), 
    synchronizes the global state, and then updates all White blocks.

    This is a block MCMC kernel: each block uses the improved local group
    Hamiltonian induced by the frozen spins outside that block. Same-color
    blocks may be dispatched in parallel when they do not directly interact,
    which is true for build_2d_lattice nearest-neighbor checkerboards.

    By default this function requires the Layden quantum walker for every
    block. Set allow_classical_fallback=True only for explicit debugging or
    large-block classical sanity checks.
    """
    if problem_inst.T is None:
        raise TypeError('Temperature T is undefined.')
    if not use_quantum and not allow_classical_fallback:
        raise ValueError('Checkerboard QeMCMC requires use_quantum=True unless a classical fallback is explicitly allowed.')
    if max_quantum_block_size < 1:
        raise ValueError('max_quantum_block_size must be at least 1.')
    if classical_sweeps_per_block <= 0:
        raise ValueError('classical_sweeps_per_block must be positive.')
    _validate_spin_config(current_config, problem_inst.n)
    _validate_checkerboard_partition(black_blocks, white_blocks, problem_inst.n)
    _validate_parallel_block_independence(problem_inst, black_blocks)
    _validate_parallel_block_independence(problem_inst, white_blocks)

    current_config = np.asarray(current_config).copy()

    black_reference = current_config.copy()
    args_black = [
        (
            block,
            problem_inst.J,
            problem_inst.h,
            problem_inst.T,
            black_reference,
            use_quantum,
            acceptance,
            max_quantum_block_size,
            classical_sweeps_per_block,
            allow_classical_fallback,
        )
        for block in black_blocks
    ]
    results_black = pool.map(_update_block_worker, args_black)

    for block_indices, new_block_config in results_black:
        current_config[block_indices] = new_block_config

    white_reference = current_config.copy()
    args_white = [
        (
            block,
            problem_inst.J,
            problem_inst.h,
            problem_inst.T,
            white_reference,
            use_quantum,
            acceptance,
            max_quantum_block_size,
            classical_sweeps_per_block,
            allow_classical_fallback,
        )
        for block in white_blocks
    ]
    results_white = pool.map(_update_block_worker, args_white)

    for block_indices, new_block_config in results_white:
        current_config[block_indices] = new_block_config
        
    return current_config
