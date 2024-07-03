from qetu_sim.wmi_backend_grid import *
from qetu_sim.wmi_decompositions import *

import numpy as np
import scipy.linalg
import itertools

from qiskit import *
from qiskit.transpiler import Layout
from qiskit.quantum_info.operators import Operator
from qiskit.circuit.library import Permutation
from qiskit.visualization import *
from qiskit_aer import AerSimulator

from qetu_sim.util import *
from qetu_sim.qetu import *

from qetu_sim.fermi_hubbard_ref import ref_fh_hamiltonian, ref_fh_op
from qetu_sim.fermi_hubbard_trotter import add_trotter_steps

def calculate_shift_params(u, t):
    """"
    Calculates the parameters c1 and c2 that can be used to shift a Fermi-Hubbard Hamiltonian
    such that its eigenvalues are within the interval [0+η, π/2-η] where η=0.05.

    Args:
        u: float
            on-site Couloumb repulsion energy of the Hamiltonian
        t: float
            hopping energy of the Hamiltonian

    Returns:
        c1
        c2 
    """
    ref_H_matrix = ref_fh_hamiltonian(u=u, t=t)
    λ, v = np.linalg.eigh(ref_H_matrix)
    λ = λ.real
    λ_min = λ.min()
    λ_max = λ.max()
    η = 0.05
    c1 = (np.pi/2 - 2*η) / (λ_max - λ_min)
    c2 = η - c1 * λ_min
    return c1, c2

def construct_trotter_V(u, t, delta_t, trotter_steps, shift=False):
    """
    Construct the trotterized controlled forward and backward time evolution operator V.

    Args:
        u: float
            on-site Couloumb repulsion energy of the Hamiltonian
        t: float
            hopping energy of the Hamiltonian
        delta_t : float
            time step
        trotter_steps: int
            number of Trotter steps
        shift: boolean
            Indicated if the Hamiltonian should be shifted
    
    Returns:
        V: QuantumCircuit
    """
    ref_H_matrix = ref_fh_hamiltonian(u=u, t=t)
    num_sites = 4
    spin_up = QuantumRegister(num_sites, '↑')
    spin_down = QuantumRegister(num_sites, '↓')
    aux = QuantumRegister(1, 'aux')
    V_trotter_qc = QuantumCircuit(spin_down, spin_up, aux)
    if shift:
        c1, c2 = calculate_shift_params(u=u, t=t)      
        tau = delta_t*c1
        V_trotter_qc.rz(-2*c2, aux[0])
    else:
        tau = delta_t
    add_trotter_steps(V_trotter_qc, spin_up, spin_down, aux, 4, u, t, tau, trotter_steps, True, False)
    return V_trotter_qc
    
def construct_QETU_circ(u, t, degree, trotter_steps, phi_vec):
    """
    Construct the overall QETU circuit.

    Args:
        u: float
            on-site Couloumb repulsion energy of the Hamiltonian
        t: float
            hopping energy of the Hamiltonian
        degree: int
            the degree of the polynomial
        trotter_steps: int
            number of Trotter steps
        phi_vec: numpy array
            vector of the phase angles

    Returns:
        QETU_circ: QuantumCircuit
            A qiskit circuit representation of the QETU algorithm
    """
    num_sites = 4
    spin_up = QuantumRegister(num_sites, '↑')
    spin_down = QuantumRegister(num_sites, '↓')
    aux = QuantumRegister(1, 'aux')
    QETU_circ = QuantumCircuit(spin_down, spin_up, aux)
    V_trotter_sh_qc = construct_trotter_V(u, t, 1, trotter_steps, True)
    QETU_circ.rx(-2*phi_vec[0], aux)
    for phi in phi_vec[1:]:
        QETU_circ.compose(V_trotter_sh_qc, inplace=True)
        QETU_circ.rx(-2*phi, aux)
    return QETU_circ

def transpile_QETU_to_WMI(QETU_circ):
    """
    Transpile the QETU circuit to the WMI hardware model.

    Args:
        QETU_circ: QuantumCircuit
            A qiskit representation of the QETU circuit

    Returns:
        QETU_circ_transpiled: QuantumCircuit
            A transpiled version of the QETU circuit
    """
    spin_down, spin_up, aux = QETU_circ.qregs
    initial_layout = Layout({
        spin_up[0]   : 2,
        spin_up[1]   : 7,
        spin_up[2]   : 1,
        spin_up[3]   : 6,
        aux[0]       : 4,
        spin_down[0] : 5,
        spin_down[1] : 8,
        spin_down[2] : 0,
        spin_down[3] : 3,
    })
    backend = WMIBackendGrid()
    QETU_circ_transpiled = transpile(
        QETU_circ,
        backend=backend,
        optimization_level=1,
        initial_layout=initial_layout
    )
    return QETU_circ_transpiled

def qetu_sim(QETU_circ, initial_state):
    """
    Perform a simulation of the QETU ground state preparation.

    Args:
        QETU_circ: QuantumCircuit
            Qiskit representation of the QETU circuit for the WMI hardware
        initial_state: Statevector
            initial quantum state that should be evolved by the circuit

    Returns:
        final_state: numpy array
            A numpy array representing the final state in textbook qubit ordering
    """
    num_qubits = QETU_circ.num_qubits - 1
    final_state = initial_state.evolve(QETU_circ, qargs=[2,6,4,3,8,0,7,5,1])
    final_state = final_state.data.real * -1
    final_state = final_state[0:2**num_qubits]
    return final_state

def qetu_sim_noise(QETU_circ, initial_state, noise_model):
    """
    Perform a noisy simulation of the QETU ground state preparation.

    Args:
        QETU_circ: QuantumCircuit
            Qiskit representation of the QETU circuit for the WMI hardware
        initial_state: Statevector
            initial quantum state that should be evolved by the circuit
        noise_model:
            Noise model of the WMI hardware

    Returns:
        final_state: numpy array
            A numpy array representing the final state in textbook qubit ordering
    """
    noise_model.add_basis_gates(['unitary'])     
    sim_noise = AerSimulator(noise_model=noise_model)   
    QETU_circ.save_unitary()
    QETU_circ = add_pswap_labels(QETU_circ)
    QETU_circ = add_sy_labels(QETU_circ)
    QETU_circ = transpile(QETU_circ, sim_noise, basis_gates=['pswap', 'cp', 'rz', 'sx', 'sy', 'x', 'y', 'save_unitary'])
    # Run on the simulator with noise
    num_qubits = QETU_circ.num_qubits - 1
    noise_result = sim_noise.run(QETU_circ).result()
    noise_unitary = noise_result.get_unitary(QETU_circ)
    final_state_noise = initial_state.evolve(noise_unitary, qargs=[2,6,4,3,8,0,7,5,1])
    final_state_noise = final_state_noise.data.real * -1
    final_state_noise = final_state_noise[0:2**num_qubits]
    return final_state_noise

def calculate_reference_ground_state(u, t, shift=False):
    """
    Numerically calculate the ground state energy and ground state vector of the Hamiltonian.

    Args:
        u: float
            on-site Couloumb repulsion energy of the Hamiltonian
        t: float
            hopping energy of the Hamiltonian
        shift: boolean
            Indicates if the Hamiltonian should be shifted

    Returns:
        ground_state_energy: float
        ground_state_vector: numpy array
    """
    ref_H_matrix = ref_fh_hamiltonian(u=u, t=t)
    if shift:
        c1, c2 = calculate_shift_params(u=u, t=t)
        H_sh = c1*ref_H_matrix + c2*np.eye(ref_H_matrix.shape[0])
        H = H_sh
    else:
        H = ref_H_matrix
    λ, v = np.linalg.eigh(H)
    ground_state_index = np.argmin(np.cos(λ))
    ground_state_energy = λ[ground_state_index]
    ground_state_vector = v[:, ground_state_index]
    return ground_state_energy, ground_state_vector