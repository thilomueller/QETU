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
        phi_vec,
        anc = None
):
    """
        Implements the Quantum Eigenvalue Transformation of Unitary Matrices.
    """
    n = block_encoding_qc.num_qubits-1
    if anc is None:
        anc = QuantumRegister(1, 'ancilla')
    q = QuantumRegister(n, 'q')
    c = ClassicalRegister(1, 'c')
    circuit = QuantumCircuit(q, anc, c)

    V = block_encoding_qc

    circuit.rx(-2*phi_vec[0], anc)

    for phi in phi_vec[1:]:
        circuit.append(V, q[:] + anc[:])
        circuit.rx(-2*phi, anc)

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

def convert_Zrot_to_Xrot(
    phi_seq_zrot
):
    """
        Reference: https://github.com/qsppack/QETU/blob/main/qet_u.py
        
        Convert the phase factors from Z rotations to that of X rotations.
        d is even
        phi_su2 = (phi0, phi1, phi2, ..., phi(d-2), phi(d-1), phi(d))
        phi_zrot = (phi0+pi/4, phi1+pi/2, phi2+pi/2, ..., phi(d-2)+pi/2, phi(d-1)+pi/2, phi(d)+pi/4)
        phi_xrot = (phi0-pi/4, phi1+pi/2, phi2-pi/2, ..., phi(d-2)-pi/2, phi(d-1)+pi/2, phi(d)-pi/4)
    """
    deg = len(phi_seq_zrot)-1
    assert deg % 2 == 0
    phi_seq = np.array(phi_seq_zrot)
    phi_seq[::2] -= np.pi
    phi_seq[0] += np.pi/2
    phi_seq[-1] += np.pi/2
    return phi_seq