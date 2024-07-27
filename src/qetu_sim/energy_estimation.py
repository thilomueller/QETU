import numpy as np
import scipy.linalg
import itertools

from qetu_sim.qetu import *
from qetu_sim.wmi_backend_grid import *
from qetu_sim.util import *

from qiskit import *
from qiskit_aer import AerSimulator

def calculate_propability(counts, q1_pos, q2_pos, q1, q2):
    """
    Given the measurement results, this function calculates the probability
    of measuring the qubits at position q1_pos and q2_pos as q1 and q2, respectively

    Args:
        counts: dict
            Measurement results including the bitstrings and their counts
        q1_pos: int
            Index of the first qubit
        q2_pos: int
            Index of the second qubit
        q1: int
            Expected measurement result of the first qubit
        q2: int
            Expected measurement result of the second qubit

    Returns:
        probability: float
    """
    # only include successful measurements
    counts = {bitstring: count for (bitstring, count) in counts.items() if int(bitstring[4]) == 0}
    num_shots = np.sum([count for _, count in counts.items()])
    occurrences = 0

    for bitstring, count in counts.items():
        measured_q1 = int(bitstring[q1_pos])
        measured_q2 = int(bitstring[q2_pos])
        if (measured_q1 == q1 and measured_q2 == q2):
            occurrences += count

    probability = occurrences / num_shots
    return probability

def add_transform_to_XX_YY_basis(circ, q1, q2):
    circ.cx(q2, q1)
    circ.ch(q1, q2)
    circ.cx(q2, q1)
    return circ

def estimate_ground_state_energy(prepared_state, u=1, t=1, num_shots=1_000, noise_model=None):
    """
    Estimate the ground state energy for a given QETU circuit.

    Args:
        prepared_state: numpy array
            Approximation of the ground state prepared by the QETU circuit (needs to be normalized)
        num_shots: int
            Number of shots to repeat each experiment
        noise_model: NoiseModel
            A noise model which should be applied to the energy estimation process
    
    Returns:
        E_0: float
            Reconstructed ground state energy
    """
    simulator = AerSimulator()

    # onsite term
    expectation_value_onsite = 0
    H_onsite = QuantumCircuit(9)   
    H_onsite.initialize(prepared_state)
    H_onsite.measure_all()

    result_onsite = simulator.run(H_onsite, shots=num_shots).result()
    counts_onsite = result_onsite.get_counts(0)

    probes_onsite = [(0,0), (0,1), (1,0), (1,1)]
    onsite_pairs = [(2,5), (7,8), (1,0), (6,3)]
    
    for qubit_pair in onsite_pairs:
        for probe in probes_onsite:
            probability = calculate_propability(counts_onsite, qubit_pair[0], qubit_pair[1], probe[0], probe[1])
            expectation_value_onsite += (-1)**probe[0] * (-1)**probe[1] * probability
    
    # hopping term 1
    expectation_value_hop_1 = 0

    H_hop_1 = QuantumCircuit(9)
    H_hop_1.initialize(prepared_state)

    probes_hop_1 = [(0,1), (1,0)]
    hop_1_pairs = [(2,1), (7,6), (5,8), (0,3)]

    for qubit_pair in hop_1_pairs:
        H_hop_1 = add_transform_to_XX_YY_basis(H_hop_1, qubit_pair[0], qubit_pair[1])

    H_hop_1.measure_all()

    result_hop_1 = simulator.run(transpile(H_hop_1, simulator), shots=num_shots).result()
    counts_hop_1 = result_hop_1.get_counts(0)

    for qubit_pair in hop_1_pairs:
        probability_00 = calculate_propability(counts_hop_1, qubit_pair[0], qubit_pair[0], 0, 0)
        probability_01 = calculate_propability(counts_hop_1, qubit_pair[0], qubit_pair[1], 0, 1)
        probability_10 = calculate_propability(counts_hop_1, qubit_pair[0], qubit_pair[1], 1, 0)
        probability_11 = calculate_propability(counts_hop_1, qubit_pair[0], qubit_pair[1], 1, 1)

        expectation_value_hop_1 += 0*probability_00
        expectation_value_hop_1 += 2*probability_01
        expectation_value_hop_1 += -2*probability_10
        expectation_value_hop_1 += 0*probability_11

    # hopping term 2
    expectation_value_hop_2 = 0

    H_hop_2 = QuantumCircuit(9)
    H_hop_2.initialize(prepared_state)

    meas_trans = QuantumCircuit(9)
    swaps = [(2,5), (7,8), (1,0), (6,3)]
    for swap in swaps:
        meas_trans.append(fSwap(), swap)

    probes_hop_2 = [(0,1), (1,0)]
    hop_2_pairs = [(2,1), (7,6), (5,8), (0,3)]

    for qubit_pair in hop_2_pairs:
        meas_trans = add_transform_to_XX_YY_basis(meas_trans, qubit_pair[0], qubit_pair[1])

    backend = WMIBackendGrid()
    meas_trans = transpile(meas_trans, backend, optimization_level=0)

    meas_trans = add_pswap_labels(meas_trans)
    meas_trans = add_sy_labels(meas_trans)
    meas_trans = transpile(meas_trans, simulator, basis_gates=['pswap', 'cp', 'rz', 'sx', 'sy', 'x', 'y', 'unitary'])

    H_hop_2.compose(meas_trans, inplace=True)
    H_hop_2.measure_all()

    result_hop_2 = simulator.run(transpile(H_hop_2, simulator),
                                 shots=num_shots,
                                 basis_gates=['pswap', 'cp', 'rz', 'sx', 'sy', 'x', 'y', 'unitary']).result()
    counts_hop_2 = result_hop_2.get_counts(0)

    for qubit_pair in hop_2_pairs:
        probability_00 = calculate_propability(counts_hop_2, qubit_pair[0], qubit_pair[0], 0, 0)
        probability_01 = calculate_propability(counts_hop_2, qubit_pair[0], qubit_pair[1], 0, 1)
        probability_10 = calculate_propability(counts_hop_2, qubit_pair[0], qubit_pair[1], 1, 0)
        probability_11 = calculate_propability(counts_hop_2, qubit_pair[0], qubit_pair[1], 1, 1)

        expectation_value_hop_2 += 0*probability_00
        expectation_value_hop_2 += 2*probability_01
        expectation_value_hop_2 += -2*probability_10
        expectation_value_hop_2 += 0*probability_11

    # hopping parity
    expectation_value_hop_parity = 0

    H_hop_parity = QuantumCircuit(9)
    H_hop_parity.initialize(prepared_state)
    H_hop_parity.measure_all()

    result_hop_parity = simulator.run(H_hop_parity, shots=num_shots).result()
    counts_hop_pairty = result_hop_parity.get_counts(0)

    probes_hop_parity = [(0,0), (0,1), (1,0), (1,1)]
    hop_parity_pairs = [(5,8), (0,3), (2,1), (7,6)]

    for qubit_pair in hop_parity_pairs:
        for probe in probes_hop_parity:
            probability = calculate_propability(counts_hop_pairty, qubit_pair[0], qubit_pair[1], probe[0], probe[1])
            expectation_value_hop_parity += (-1)**probe[0] * (-1)**probe[1] * probability
    
    E0_meas = 0.25*u*expectation_value_onsite -2*t*(expectation_value_hop_1 + expectation_value_hop_2)
    return E0_meas