import numpy as np
from scipy.linalg import expm, sinm, cosm

from qiskit import *
from qiskit.quantum_info import partial_trace, Statevector


X_gate = np.array([[0, 1], [1, 0]], dtype = 'complex_')
Y_gate = np.array([[0, -1j], [1j, 0]], dtype = 'complex_')
Z_gate = np.array([[1, 0], [0, -1]], dtype = 'complex_')

def R_x(theta):
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype = 'complex_')

def R_y(theta):
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]], dtype = 'complex_')

def R_z(theta):
    return np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]], dtype = 'complex_')


def qiskit2normal(mat):
    """Converts a unitary matrix in Qiskit qubit ordering to textbook ordering."""
    import numpy as np

    orig_shape = mat.shape
    intlog2 = lambda x: x.bit_length() - 1
    qubits = intlog2(mat.shape[0])
    shape = [2] * (2 * qubits)
    mat = mat.reshape(shape)
    inputs = list(range(qubits))
    outputs = list(range(qubits, 2 * qubits))
    mat = np.transpose(mat, axes=inputs[::-1] + outputs[::-1])
    return mat.reshape(orig_shape)


def circuit2matrix(circ, keep_qiskit_ordering=True):
    """Converts a qiskit circuit into an unitary numpy array."""
    from qiskit.quantum_info import Operator

    mat = Operator(circ).data
    if keep_qiskit_ordering:
        return mat
    return qiskit2normal(mat)

def transform_eigenvals(M, a=0.0, b=np.pi):
    """
        Transforms a unitary matrix M so that its eigenvalues are between [a,b]
    """
    λ, v = np.linalg.eig(M)
    λ_min = λ.min()
    λ_max = λ.max()
    λ_t = (λ - λ_min) * (b - a) / (λ_max - λ_min) + a
    M_t = v @ np.diag(λ_t) @ v.conj().T
    return M_t

def spectral_gap(matrix):
    """
        Calculate the spectral gab of a matrix
    """
    eigenvalues = np.linalg.eigvals(matrix)
    sorted_eigenvalues = np.sort(np.real(eigenvalues))
    # Spectral gap is the difference between the two largest eigenvalues
    gap = sorted_eigenvalues[-1] - sorted_eigenvalues[-2]
    return gap

def compute_depth_no_rz(qc : QuantumCircuit) -> int:
    qc1 = qc.copy()
    rz_gates = [i for (i, data) in enumerate(qc1.data) if data[0].name == "rz"]
    for i in reversed(rz_gates):
        qc1.data.pop(i)
    return qc1.depth()

def add_sy_labels(qc : QuantumCircuit) -> QuantumCircuit:
    """
        This function takes a quantum circuit as an imput and replaces every Squared Y gate with a custom labeled ISwap gate.
        This is necessary so that the noise model can be applied to that gate.
    """
    import qiskit.quantum_info as qi
    # iSWAP matrix operator
    sy_op = qi.Operator([[0.5+0.5j, -0.5-0.5j],
                            [0.5+0.5j, 0.5+0.5j]])
    qc1 = qc.copy()
    for (i, data) in enumerate(qc1.data):
        if data[0].name == "sy":
            # get the list of indices the quantum gate is operating on
            list_of_indices = data[1]
            # create a new labeled iswap gate
            sy_gate = QuantumCircuit(qc1.num_qubits)
            sy_gate.unitary(sy_op, list_of_indices, label='sy')
            # remove the unlabel gate and replace it with a labeled one
            qc1.data.pop(i)
            qc1.data.insert(i, sy_gate.data[0])
    return qc1

def add_pswap_labels(qc : QuantumCircuit) -> QuantumCircuit:
    """
        This function takes a quantum circuit as an imput and replaces every Parametric Swap gate with a custom labeled gate.
        This is necessary so that the noise model can be applied to that gate.
    """
    import qiskit.quantum_info as qi
    qc1 = qc.copy()
    for (i, data) in enumerate(qc1.data):
        if data[0].name == "pswap":
            # get parameters
            theta, eta = data[0].params
            # construct operator
            pswap_op = qi.Operator([[1, 0,                                  0,                                 0],
                                    [0, np.cos(theta/2),                    1j*np.sin(theta/2)*np.exp(1j*eta), 0],
                                    [0, 1j*np.sin(theta/2)*np.exp(-1j*eta), np.cos(theta/2),                   0],
                                    [0, 0,                                  0,                                 1]])
            # get the list of indices the quantum gate is operating on
            list_of_indices = data[1]
            # create a new labeled iswap gate
            pswap_gate = QuantumCircuit(qc1.num_qubits)
            pswap_gate.unitary(pswap_op, list_of_indices, label='pswap')
            # remove the unlabel gate and replace it with a labeled one
            qc1.data.pop(i)
            qc1.data.insert(i, pswap_gate.data[0])
    return qc1
