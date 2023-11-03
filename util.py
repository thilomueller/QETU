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


def get_matrix(circ):
    """
        Takes a qiskit circuit and returns the corresponding matrix
    """
    backend = Aer.get_backend('unitary_simulator')
    job = execute(circ, backend)
    result = job.result()
    matrix = np.array(result.get_unitary(circ, decimals=15))
    return matrix

def get_state(circ):
    """
        Takes a qiskit circuit and returns the resulting statevector
    """
    backend = BasicAer.get_backend('statevector_simulator') # the device to run on
    result = backend.run(transpile(circ, backend)).result()
    full_statevector  = result.get_statevector(circ)
    return full_statevector

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