import numpy as np
import scipy.linalg
import itertools
import json

from qetu_sim.fh_2x2_sim import *
from qetu_sim.qetu import *
from qetu_sim.wmi_backend_grid import *
from qetu_sim.util import *

from qiskit import *
from qiskit_aer import AerSimulator

def calculate_propability(counts, q1_pos, q2_pos, q1, q2, use_num_conservation=False):
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
        use_num_conservation: boolean
            Indicates whether invalid shots that violate the
            number symmetry of fermions should be discarded

    Returns:
        probability: float
    """
    # only include successful measurements
    counts = {bitstring: count for (bitstring, count) in counts.items() if int(bitstring[4]) == 0}
    if use_num_conservation:
        counts = {bitstring: count for (bitstring, count) in counts.items()
            if (int(bitstring[0]) + int(bitstring[1]) + int(bitstring[2]) + int(bitstring[3])
                + int(bitstring[5]) + int(bitstring[6]) + int(bitstring[7]) + int(bitstring[8])) == 4}
    num_shots = np.sum([count for _, count in counts.items()])
    if num_shots == 0:
        print("No successful measurement!")
        return 0.0
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

def estimate_ground_state_energy(prepared_state, u=1, t=1, num_shots=1_000, noise_model=None, use_num_conservation=False):
    """
    Estimate the ground state energy for a given QETU circuit.

    Args:
        prepared_state: numpy array
            Approximation of the ground state prepared by the QETU circuit (needs to be normalized)
        num_shots: int
            Number of shots to repeat each experiment
        noise_model: NoiseModel
            A noise model which should be applied to the energy estimation process
        use_num_conservation: boolean
            Indicates whether invalid shots that violate the number symmetry of fermions should be discarded
    
    Returns:
        E_0: float
            Reconstructed ground state energy
    """
    if noise_model is not None:
        simulator = AerSimulator(noise_model=noise_model)
    else:
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
            probability = calculate_propability(counts_onsite, qubit_pair[0], qubit_pair[1], probe[0], probe[1], use_num_conservation)
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
        probability_00 = calculate_propability(counts_hop_1, qubit_pair[0], qubit_pair[0], 0, 0, use_num_conservation)
        probability_01 = calculate_propability(counts_hop_1, qubit_pair[0], qubit_pair[1], 0, 1, use_num_conservation)
        probability_10 = calculate_propability(counts_hop_1, qubit_pair[0], qubit_pair[1], 1, 0, use_num_conservation)
        probability_11 = calculate_propability(counts_hop_1, qubit_pair[0], qubit_pair[1], 1, 1, use_num_conservation)

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
        probability_00 = calculate_propability(counts_hop_2, qubit_pair[0], qubit_pair[0], 0, 0, use_num_conservation)
        probability_01 = calculate_propability(counts_hop_2, qubit_pair[0], qubit_pair[1], 0, 1, use_num_conservation)
        probability_10 = calculate_propability(counts_hop_2, qubit_pair[0], qubit_pair[1], 1, 0, use_num_conservation)
        probability_11 = calculate_propability(counts_hop_2, qubit_pair[0], qubit_pair[1], 1, 1, use_num_conservation)

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
            probability = calculate_propability(counts_hop_pairty, qubit_pair[0], qubit_pair[1], probe[0], probe[1], use_num_conservation)
            expectation_value_hop_parity += (-1)**probe[0] * (-1)**probe[1] * probability
    
    E0_meas = 0.25*u*expectation_value_onsite +0.5*t*(expectation_value_hop_1 + expectation_value_hop_2)
    return E0_meas

def estimate_ground_state_energy_from_statevector_list(prepared_states, u=1, t=1, noise_model=None, use_num_conservation=False):
    """
    Estimate the ground state energy for a given QETU circuit from a list of given statevectors.
    These statevectors are the outcome of a noisy simulation run.

    Args:
        prepared_states: list of numpy array
            List of the statevectors
        noise_model: NoiseModel
            A noise model which should be applied to the energy estimation process
        use_num_conservation: boolean
            Indicates whether invalid shots that violate the number symmetry of fermions should be discarded
    
    Returns:
        E_0: float
            Reconstructed ground state energy
    """
    def combine_dicts(a_dict, b_dict):
        return {k: a_dict.get(k, 0) + b_dict.get(k, 0) for k in a_dict.keys() | b_dict.keys()}
    
    num_shots = 100

    if noise_model is not None:
        simulator = AerSimulator(noise_model=noise_model)
    else:
        simulator = AerSimulator()

    # onsite term
    expectation_value_onsite = 0
    H_onsite_circs = []
    for init_state in prepared_states:
        H_onsite = QuantumCircuit(9)   
        H_onsite.initialize(init_state)
        H_onsite.measure_all()
        H_onsite_circs.append(H_onsite)

    counts_onsite = {}
    for circ in H_onsite_circs:
        result_onsite = simulator.run(circ, shots=num_shots).result()
        counts_onsite = combine_dicts(counts_onsite, result_onsite.get_counts(0))

    probes_onsite = [(0,0), (0,1), (1,0), (1,1)]
    onsite_pairs = [(2,5), (7,8), (1,0), (6,3)]
    
    for qubit_pair in onsite_pairs:
        for probe in probes_onsite:
            probability = calculate_propability(counts_onsite, qubit_pair[0], qubit_pair[1], probe[0], probe[1], use_num_conservation)
            expectation_value_onsite += (-1)**probe[0] * (-1)**probe[1] * probability
    
    # hopping term 1
    expectation_value_hop_1 = 0
    probes_hop_1 = [(0,1), (1,0)]
    hop_1_pairs = [(2,1), (7,6), (5,8), (0,3)]
    H_hop_1_circs = []
    for init_state in prepared_states:
        H_hop_1 = QuantumCircuit(9)
        H_hop_1.initialize(init_state)
        for qubit_pair in hop_1_pairs:
            H_hop_1 = add_transform_to_XX_YY_basis(H_hop_1, qubit_pair[0], qubit_pair[1])
        H_hop_1.measure_all()
        H_hop_1 = transpile(H_hop_1, simulator)
        H_hop_1_circs.append(H_hop_1)

    counts_hop_1 = {}
    for circ in H_hop_1_circs:
        result_hop_1 = simulator.run(circ, shots=num_shots).result()
        counts_hop_1 = combine_dicts(counts_hop_1, result_hop_1.get_counts(0))

    for qubit_pair in hop_1_pairs:
        probability_00 = calculate_propability(counts_hop_1, qubit_pair[0], qubit_pair[0], 0, 0, use_num_conservation)
        probability_01 = calculate_propability(counts_hop_1, qubit_pair[0], qubit_pair[1], 0, 1, use_num_conservation)
        probability_10 = calculate_propability(counts_hop_1, qubit_pair[0], qubit_pair[1], 1, 0, use_num_conservation)
        probability_11 = calculate_propability(counts_hop_1, qubit_pair[0], qubit_pair[1], 1, 1, use_num_conservation)

        expectation_value_hop_1 += 0*probability_00
        expectation_value_hop_1 += 2*probability_01
        expectation_value_hop_1 += -2*probability_10
        expectation_value_hop_1 += 0*probability_11

    # hopping term 2
    expectation_value_hop_2 = 0
    probes_hop_2 = [(0,1), (1,0)]
    hop_2_pairs = [(2,1), (7,6), (5,8), (0,3)]
    H_hop_2_circs = []
    for init_state in prepared_states:
        H_hop_2 = QuantumCircuit(9)
        H_hop_2.initialize(init_state)
        meas_trans = QuantumCircuit(9)
        swaps = [(2,5), (7,8), (1,0), (6,3)]
        for swap in swaps:
            meas_trans.append(fSwap(), swap)
        for qubit_pair in hop_2_pairs:
            meas_trans = add_transform_to_XX_YY_basis(meas_trans, qubit_pair[0], qubit_pair[1])
        backend = WMIBackendGrid()
        meas_trans = transpile(meas_trans, backend, optimization_level=0)
        meas_trans = add_pswap_labels(meas_trans)
        meas_trans = add_sy_labels(meas_trans)
        meas_trans = transpile(meas_trans, simulator, basis_gates=['pswap', 'cp', 'rz', 'sx', 'sy', 'x', 'y', 'unitary'])
        H_hop_2.compose(meas_trans, inplace=True)
        H_hop_2.measure_all()
        H_hop_2 = transpile(H_hop_2, simulator)
        H_hop_2_circs.append(H_hop_2)

    counts_hop_2 = {}
    for circ in H_hop_2_circs:
        result_hop_2 = simulator.run(circ,
                                    shots=num_shots,
                                    basis_gates=['pswap', 'cp', 'rz', 'sx', 'sy', 'x', 'y', 'unitary']).result()
        counts_hop_2 = combine_dicts(counts_hop_2, result_hop_2.get_counts(0))

    for qubit_pair in hop_2_pairs:
        probability_00 = calculate_propability(counts_hop_2, qubit_pair[0], qubit_pair[0], 0, 0, use_num_conservation)
        probability_01 = calculate_propability(counts_hop_2, qubit_pair[0], qubit_pair[1], 0, 1, use_num_conservation)
        probability_10 = calculate_propability(counts_hop_2, qubit_pair[0], qubit_pair[1], 1, 0, use_num_conservation)
        probability_11 = calculate_propability(counts_hop_2, qubit_pair[0], qubit_pair[1], 1, 1, use_num_conservation)

        expectation_value_hop_2 += 0*probability_00
        expectation_value_hop_2 += 2*probability_01
        expectation_value_hop_2 += -2*probability_10
        expectation_value_hop_2 += 0*probability_11
    
    E0_meas = 0.25*u*expectation_value_onsite +0.5*t*(expectation_value_hop_1 + expectation_value_hop_2)
    return E0_meas


def estimate_ground_state_energy_from_circ(onsite_circ, hop_1_circ, hop_2_circ, u=1, t=1, num_shots=1_000, noise_model=None, use_num_conservation=False):
    """
    Estimate the ground state energy for a given QETU circuit.

    Args:
        prepared_state: numpy array
            Approximation of the ground state prepared by the QETU circuit (needs to be normalized)
        num_shots: int
            Number of shots to repeat each experiment
        noise_model: NoiseModel
            A noise model which should be applied to the energy estimation process
        use_num_conservation: boolean
            Indicates whether invalid shots that violate the number symmetry of fermions should be discarded
    
    Returns:
        E_0: float
            Reconstructed ground state energy
    """
    if noise_model is not None:
        simulator = AerSimulator(noise_model=noise_model)
    else:
        simulator = AerSimulator()

    # onsite term
    expectation_value_onsite = 0
    result_onsite = simulator.run(onsite_circ, shots=num_shots).result()
    counts_onsite = result_onsite.get_counts(0)

    probes_onsite = [(0,0), (0,1), (1,0), (1,1)]
    onsite_pairs = [(2,5), (7,8), (1,0), (6,3)]
    
    for qubit_pair in onsite_pairs:
        for probe in probes_onsite:
            probability = calculate_propability(counts_onsite, qubit_pair[0], qubit_pair[1], probe[0], probe[1], use_num_conservation)
            expectation_value_onsite += (-1)**probe[0] * (-1)**probe[1] * probability
    
    # hopping term 1
    expectation_value_hop_1 = 0
    hop_1_pairs = [(2,1), (7,6), (5,8), (0,3)]

    result_hop_1 = simulator.run(transpile(hop_1_circ, simulator), shots=num_shots).result()
    counts_hop_1 = result_hop_1.get_counts(0)

    for qubit_pair in hop_1_pairs:
        probability_00 = calculate_propability(counts_hop_1, qubit_pair[0], qubit_pair[0], 0, 0, use_num_conservation)
        probability_01 = calculate_propability(counts_hop_1, qubit_pair[0], qubit_pair[1], 0, 1, use_num_conservation)
        probability_10 = calculate_propability(counts_hop_1, qubit_pair[0], qubit_pair[1], 1, 0, use_num_conservation)
        probability_11 = calculate_propability(counts_hop_1, qubit_pair[0], qubit_pair[1], 1, 1, use_num_conservation)

        expectation_value_hop_1 += 0*probability_00
        expectation_value_hop_1 += 2*probability_01
        expectation_value_hop_1 += -2*probability_10
        expectation_value_hop_1 += 0*probability_11

    # hopping term 2
    expectation_value_hop_2 = 0
    hop_2_pairs = [(2,1), (7,6), (5,8), (0,3)]

    result_hop_2 = simulator.run(transpile(hop_2_circ, simulator),
                                 shots=num_shots).result()
    counts_hop_2 = result_hop_2.get_counts(0)

    for qubit_pair in hop_2_pairs:
        probability_00 = calculate_propability(counts_hop_2, qubit_pair[0], qubit_pair[0], 0, 0, use_num_conservation)
        probability_01 = calculate_propability(counts_hop_2, qubit_pair[0], qubit_pair[1], 0, 1, use_num_conservation)
        probability_10 = calculate_propability(counts_hop_2, qubit_pair[0], qubit_pair[1], 1, 0, use_num_conservation)
        probability_11 = calculate_propability(counts_hop_2, qubit_pair[0], qubit_pair[1], 1, 1, use_num_conservation)

        expectation_value_hop_2 += 0*probability_00
        expectation_value_hop_2 += 2*probability_01
        expectation_value_hop_2 += -2*probability_10
        expectation_value_hop_2 += 0*probability_11

    E0_meas = 0.25*u*expectation_value_onsite -0.5*t*(expectation_value_hop_1 + expectation_value_hop_2)
    return E0_meas

class QobjEncoder(json.JSONEncoder):
    """
    Taken from: https://docs.quantum.ibm.com/api/qiskit/0.37/qiskit.qobj.Qobj#to_dict
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, complex):
            return (obj.real, obj.imag)
        return json.JSONEncoder.default(self, obj)

def create_qobj(ansatz_circ):
    """
    Create the qobj files for all circuits necessary to estimate the ground state energy.

    Args:
        ansatz_circuit: QuantumCircuit
            Quantum circuit that is used to prepare the ground state
    """

    # onsite term
    onsite_circ = QuantumCircuit(9)
    prepare_init_state(onsite_circ)
    onsite_circ.compose(ansatz_circ, inplace=True)
    onsite_circ.measure_all()
    onsite_circ = transpile(onsite_circ, basis_gates=['pswap', 'cp', 'rz', 'sx', 'sy', 'x', 'y'])
    onsite_qobj = assemble(onsite_circ)
    f = open("onsite_circ.qobj", "w+")
    f.write(json.dumps(onsite_qobj.to_dict(), cls=QobjEncoder))
    f.close()

    # hopping term 1
    hop_1_circ = QuantumCircuit(9)
    prepare_init_state(hop_1_circ)
    hop_1_circ.compose(ansatz_circ, inplace=True)
    hop_1_pairs = [(2,1), (7,6), (5,8), (0,3)]
    for qubit_pair in hop_1_pairs:
        hop_1_circ = add_transform_to_XX_YY_basis(hop_1_circ, qubit_pair[0], qubit_pair[1])
    hop_1_circ.measure_all()
    hop_1_circ = transpile(hop_1_circ, basis_gates=['pswap', 'cp', 'rz', 'sx', 'sy', 'x', 'y'])
    hop_1_qobj = assemble(hop_1_circ)
    f = open("hop_1_circ.qobj", "w+")
    f.write(json.dumps(hop_1_qobj.to_dict(), cls=QobjEncoder))
    f.close()

    # hopping term 2
    hop_2_circ = QuantumCircuit(9)
    hop_2_pairs = [(2,1), (7,6), (5,8), (0,3)]
    prepare_init_state(hop_2_circ)
    hop_2_circ.compose(ansatz_circ, inplace=True)
    swaps = [(2,5), (7,8), (1,0), (6,3)]
    for swap in swaps:
        hop_2_circ.append(fSwap(), swap)
    for qubit_pair in hop_2_pairs:
        hop_2_circ = add_transform_to_XX_YY_basis(hop_2_circ, qubit_pair[0], qubit_pair[1])
    hop_2_circ.measure_all()
    hop_2_circ = transpile(hop_2_circ, basis_gates=['pswap', 'cp', 'rz', 'sx', 'sy', 'x', 'y'])
    hop_2_qobj = assemble(hop_2_circ)
    f = open("hop_2_circ.qobj", "w+")
    f.write(json.dumps(hop_2_qobj.to_dict(), cls=QobjEncoder))
    f.close()



def create_energy_estimation_circuits(ansatz_circ):
    """
    Create the three circuits that are necessary to estimate the ground state energy.

    Args:
        ansatz_circuit: QuantumCircuit
            Quantum circuit that is used to prepare the ground state
    
    Returns:
        onsite_circ: QuantumCircuit
        hop_1_circ: QuantumCircuit
        hop_2_circ: QuantumCircuit
    """

    # onsite term
    onsite_circ = QuantumCircuit(9, 9)
    prepare_init_state(onsite_circ)
    onsite_circ.compose(ansatz_circ, inplace=True)
    onsite_circ.measure(4, 4)
    onsite_circ.measure_all(add_bits=False)
    onsite_circ = transpile(onsite_circ, basis_gates=['pswap', 'cp', 'rz', 'sx', 'sy', 'x', 'y', 'u3'])

    # hopping term 1
    hop_1_circ = QuantumCircuit(9, 9)
    prepare_init_state(hop_1_circ)
    hop_1_circ.compose(ansatz_circ, inplace=True)
    hop_1_pairs = [(2,1), (7,6), (5,8), (0,3)]
    for qubit_pair in hop_1_pairs:
        hop_1_circ = add_transform_to_XX_YY_basis(hop_1_circ, qubit_pair[0], qubit_pair[1])
    hop_1_circ.measure(4, 4)
    hop_1_circ.measure_all(add_bits=False)
    hop_1_circ = transpile(hop_1_circ, basis_gates=['pswap', 'cp', 'rz', 'sx', 'sy', 'x', 'y', 'u3'])

    # hopping term 2
    hop_2_circ = QuantumCircuit(9, 9)
    hop_2_pairs = [(2,1), (7,6), (5,8), (0,3)]
    prepare_init_state(hop_2_circ)
    hop_2_circ.compose(ansatz_circ, inplace=True)
    swaps = [(2,5), (7,8), (1,0), (6,3)]
    for swap in swaps:
        hop_2_circ.append(fSwap(), swap)
    for qubit_pair in hop_2_pairs:
        hop_2_circ = add_transform_to_XX_YY_basis(hop_2_circ, qubit_pair[0], qubit_pair[1])
    hop_2_circ.measure(4, 4)
    hop_2_circ.measure_all(add_bits=False)
    hop_2_circ = transpile(hop_2_circ, basis_gates=['pswap', 'cp', 'rz', 'sx', 'sy', 'x', 'y', 'u3'])

    return onsite_circ, hop_1_circ, hop_2_circ