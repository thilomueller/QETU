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