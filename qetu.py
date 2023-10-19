import numpy as np
from qiskit import *

def QSP(phi_vec, x=0):
    nqubits = 1
    q = QuantumRegister(nqubits, 'q')
    circuit = QuantumCircuit(q)

    circuit.rx(-2*phi_vec[0], q[0])

    for phi in phi_vec[1:]:
        circuit.rz(-2*np.arccos(x), q[0]) # W_z
        circuit.rx(-2*phi, q[0])

    return circuit

def QETU(
        block_encoding_qc,
        phi_vec
):
    """
        Implements the Quantum Eigenvalue Transformation of Unitary Matrices.
    """
    n = block_encoding_qc.num_qubits-1
    anc = QuantumRegister(1, 'ancilla')
    q = QuantumRegister(n, 'q')
    c = ClassicalRegister(1, 'c')
    circuit = QuantumCircuit(anc, q, c)

    V = block_encoding_qc
    V_dagger = V.inverse()

    circuit.rx(-2*phi_vec[0], anc)

    for index, phi in enumerate(phi_vec[1:]):
        if index % 2 == 0:
            circuit.append(V, anc[:] + q[:])
        else:
            circuit.append(V_dagger, anc[:] + q[:])
        circuit.rx(-2*phi, anc)
    #circuit.measure(anc, 0)

    return circuit

def convert_SU2_to_Xrot(
    phi_vec_su2
):
    """
        Convert the phase factors from the SU2 convention to the RX convention.

        phi_su2 = (phi_0, phi_1, phi_2, ..., phi_{d-2}, phi_{d-1}, phi_d)
        phi_xrot = (phi_0 - pi/4, phi_1 + pi/2, phi_2 - pi/2, ..., phi_{d-2} - pi/2, phi_{d-1} + pi/2, phi_d - pi/4)        
        
        Input:
            phi_vec_su2
    """
    d = len(phi_vec_su2) - 1
    assert d % 2 == 0, "The degree of the phase vector must be even."
    phi_vec_xrot = np.array(phi_vec_su2)
    phi_vec_xrot[0] += np.pi/4
    phi_vec_xrot[-1] += np.pi/4
    phi_vec_xrot[0::2] -= np.pi/2
    phi_vec_xrot[1::2] += np.pi/2
    return phi_vec_xrot