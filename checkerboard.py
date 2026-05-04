# the code in this document uses code from Laydens original QeMCMC algorithm
# the actual quantum proposal model is orchestrated to run in 
# a parallel checkerboard decomposition on the EA model
# the paper treats the constant spins surrounding each block as 
# constants which are factored into the hamiltonian through the 
# external field term. the reacceptance logic is independant for 
# each block as they do not interact. testing has not yet been 
# carried out to see how the advantage manifests in this formulation.
# it has been thoeretically shown to follow all of the rules of mcmc. 

# only the functions are provided here, not the actual code to run it
# although the code to do is quite straighforward and has effectively been done. 

# the code in this was created in part with ai
# the exact mathematical formulation was provided, including the mathematical proof  
# of efficacy. the relevant quantum proposal functions Laydens group used were 
# also provided. ai was then told to just execute on all of the math it already had.
# classical parallel checkerboard domain decomposition was also given. 

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
        if np.any(np.triu(J, k=1) != J):
            raise ValueError('J must be upper-triangular with 0s along the diagonal')

        self.J = J
        self.h = h
        self.T = T 
        self.n = h.size
        
        # Scale for quantum Hamiltonian (Prevents variance from blowing up or shrinking)
        scale = ( self.n / (la.norm(J, ord='fro')**2 + la.norm(h, ord=2)**2) )**0.5
        self.J_quantum = self.J * scale 
        self.h_quantum = self.h * scale

        if precomp_E:
            self.E_arr = np.zeros((2**self.n))
            for i in range(2**self.n):
                config = int2spinconf(i, self.n)
                self.E_arr[i] = config @ J @ config + config @ h 

def RandomProblemInstance(n, connectivity, T=None, precomp_E=True):
    h = np.random.randn(n)
    if connectivity=='line':
        J = np.diag(np.random.randn(n-1), k=1)
    if connectivity=='full':
        J = np.triu(np.random.randn(n,n), k=1)
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
    return proposal_mat

def make_transition_mat(problem_inst, proposal_mat, acceptance='metropolis'):
    if problem_inst.T is None: raise TypeError('Temperature T is undefined.')
    if problem_inst.T < 0: raise ValueError('Temperature T cannot be negative.')

    n = problem_inst.n
    T = problem_inst.T
    E_rowstack = np.tile(problem_inst.E_arr, (2**n,1))
    E_diff = E_rowstack.T - E_rowstack

    uphill_moves = (E_diff >= 0) 
    downhill_moves = np.invert(uphill_moves)

    if acceptance=='metropolis':
        if T>0:
            pi_ratio = np.exp(-E_diff/T, where=uphill_moves, out=np.ones_like(E_diff)) 
        if T==0:
            pi_ratio = np.where(uphill_moves, 0., 1.)
        A = pi_ratio

    if acceptance=='glauber':
        if T>0:
            pi_ratio = np.exp(-E_diff/T, where=uphill_moves, out=E_diff*np.nan) 
            pi_ratio_inv = np.exp(E_diff/T, where=downhill_moves, out=E_diff*np.nan)
        if T==0:
            pi_ratio = np.where(uphill_moves, 0., 1.)
            pi_ratio_inv = np.where(downhill_moves, 0., 1.)
        A = np.where(uphill_moves, pi_ratio/(1+pi_ratio), (1+pi_ratio_inv)**(-1) )

    P = A * proposal_mat
    np.fill_diagonal(P, 0)
    diag = np.ones(2**n) - np.sum(P, axis=0)
    P = P + np.diag(diag)
    return P

def abs_spectral_gap(transition_mat):
    dist = np.sort( 1-np.abs(la.eigvals(transition_mat)) )
    delta = np.min(dist[1:])
    dist_lazy = np.sort( 1-np.abs(1/2 + la.eigvals(transition_mat)/2) )
    delta_lazy = np.min(dist_lazy[1:])
    return delta, delta_lazy

def generate_move(transition_mat, state):
    return np.random.choice(transition_mat.shape[0], p=transition_mat[:,state])

# =============================================================================
# NEW PARALLEL CHECKERBOARD ARCHITECTURE
# =============================================================================

def get_effective_problem(problem_inst, active_indices, full_state_config):
    r"""
    Creates an effective ProblemInstance representing a localized subsystem (e.g., a 
    checkerboard block). The unselected boundary spins are frozen and mathematically 
    factored into an external field applied to the active spins. This allows us to 
    simulate the sub-block dynamically without breaking detailed balance.
    
    Implements Equation 14: \tilde{h}_j = h_j + \sum_{i \notin g_l} J_{ji} s_i
    """
    # Ensure indices are sorted to maintain proper matrix alignment
    active_indices = np.sort(active_indices)
    
    # -------------------------------------------------------------------------
    # 1. Isolate the Inactive (Boundary) Environment
    # -------------------------------------------------------------------------
    # Create a boolean mask where True indicates an active spin inside the block
    active_mask = np.zeros(problem_inst.n, dtype=bool)
    active_mask[active_indices] = True
    
    # Extract the full configuration and zero out the active spins. 
    # This leaves only the fixed environmental boundary spins.
    inactive_config = full_state_config.copy()
    inactive_config[active_mask] = 0.0 
    
    # -------------------------------------------------------------------------
    # 2. Compute the Effective Field Term (\tilde{h}_j)
    # -------------------------------------------------------------------------
    # The interaction between the active block and the frozen boundary acts as 
    # a local magnetic field. Since J is upper triangular, (J + J.T) gives the 
    # full symmetric coupling matrix. Multiplying by the inactive configuration 
    # perfectly calculates \sum_{i \notin g_l} J_{ji} s_i for all spins.
    h_interaction = (problem_inst.J + problem_inst.J.T) @ inactive_config
    
    # Add the base field h_j to the boundary interaction field
    h_eff = problem_inst.h[active_indices] + h_interaction[active_indices]
    
    # -------------------------------------------------------------------------
    # 3. Preserve Internal Entanglement Couplings (J_eff)
    # -------------------------------------------------------------------------
    # Extract the sub-matrix of J containing only connections *within* the block
    J_eff = problem_inst.J[np.ix_(active_indices, active_indices)]
    
    # -------------------------------------------------------------------------
    # 4. Construct the Localized Problem Instance
    # -------------------------------------------------------------------------
    # WARNING: Precomputing E_arr constructs a 2^n length array. 
    # For a 10x10 block, 2^100 states would immediately cause an Out-Of-Memory crash.
    # Therefore, we only precompute energies if the block is small enough (n <= 12).
    precomp = len(active_indices) <= 12
    return ProblemInstance(J_eff, h_eff, T=problem_inst.T, precomp_E=precomp)

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

def dynamic_local_mcmc(problem_inst, current_block_config, num_moves=100):
    """
    Performs standard local Metropolis MCMC on a ProblemInstance. 
    Unlike make_transition_mat, this does not build a dense 2^n * 2^n transition 
    matrix, making it safe and efficient for classically simulating large blocks 
    (e.g., n=100) where exact diagonalization would cause an OOM crash.
    """
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
        
        # Metropolis Acceptance Criterion:
        # Always accept downhill moves (dE <= 0).
        # Accept uphill moves with probability exp(-\Delta E / T).
        if dE <= 0 or (T > 0 and np.random.rand() < np.exp(-dE / T)):
            config[i] *= -1 # Execute the spin flip
            
    return config

    """
    Worker function for multiprocessing. 
    Evaluates a single block independently conditionally on its frozen boundary fields. 
    If the block is small enough, it uses the exact theoretical quantum proposal matrix. 
    For large blocks, it gracefully falls back to classical dynamic MCMC to avoid 
    memory crashes.
    """
    # Unpack the serialized arguments from the multiprocessing pool
    block_indices, global_J, global_h, global_T, current_config, use_quantum = args
    
    # Reconstruct the global ProblemInstance (acting as a lightweight container)
    prob_global = ProblemInstance(global_J, global_h, T=global_T, precomp_E=False)
    
    # Dynamically generate the localized Hamiltonian, folding the boundary spins 
    # into the effective field term \tilde{h}_j
    eff_prob = get_effective_problem(prob_global, block_indices, current_config)
    
    n_block = len(block_indices)
    current_block_config = current_config[block_indices]
    
    # -------------------------------------------------------------------------
    # STATE TRANSITION LOGIC
    # -------------------------------------------------------------------------
    # For a 10x10 block, n=100. Generating the quantum proposal requires exact 
    # diagonalization of a 2^100 x 2^100 matrix, which is physically impossible classically.
    # Therefore, we branch logic based on the block size.
    if use_quantum and n_block <= 12:
        # EXACT QUANTUM PROPOSAL (Theoretical Hardware Simulation)
        prop_mat = quantum_proposal_mat_ideal(eff_prob)
        trans_mat = make_transition_mat(eff_prob, prop_mat, acceptance='metropolis')
        
        # Map boolean configuration to integer state, apply transition, and map back
        current_block_int = spinconf2int(current_block_config)
        new_block_int = generate_move(trans_mat, current_block_int)
        new_block_config = int2spinconf(new_block_int, n_block)
    else:
        # CLASSICAL FALLBACK (For large blocks / classical sanity testing)
        new_block_config = dynamic_local_mcmc(eff_prob, current_block_config, num_moves=n_block)
        
    # Return the block indices alongside the new state so the master process can re-sync
    return block_indices, new_block_config

def parallel_checkerboard_update(problem_inst, current_config, black_blocks, white_blocks, pool, use_quantum=False):
    """
    Performs one full parallel sweep of the checkerboard lattice.
    Updates all Black blocks simultaneously (conditionally on the static White blocks), 
    synchronizes the global state, and then updates all White blocks.
    """
    # -------------------------------------------------------------------------
    # PHASE 1: BLACK BLOCKS
    # -------------------------------------------------------------------------
    # Serialize arguments and dispatch all black blocks to the CPU pool
    args_black = [(block, problem_inst.J, problem_inst.h, problem_inst.T, current_config, use_quantum) for block in black_blocks]
    results_black = pool.map(_update_block_worker, args_black)
    
    # Barrier synchronization: Wait for all Black blocks to finish, then merge their 
    # new states into the global configuration
    for block_indices, new_block_config in results_black:
        current_config[block_indices] = new_block_config
        
    # -------------------------------------------------------------------------
    # PHASE 2: WHITE BLOCKS
    # -------------------------------------------------------------------------
    # Now that the Black blocks have updated, dispatch all White blocks. 
    # They will calculate their new boundary fields based on the updated Black spins.
    args_white = [(block, problem_inst.J, problem_inst.h, problem_inst.T, current_config, use_quantum) for block in white_blocks]
    results_white = pool.map(_update_block_worker, args_white)
    
    # Barrier synchronization: Merge White blocks into the global configuration
    for block_indices, new_block_config in results_white:
        current_config[block_indices] = new_block_config
        
    return current_config
