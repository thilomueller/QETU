import scipy as sp
import numpy as np
import functools as ft

Id = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

def H_1(onsite_pairs, include_aux):
    """
    This function creates the onsite Hamiltonian for given qubit positions.

    Args:
        onsite_pairs: list of tuple
            A list of qubit pair positions indicating where the onsite terms should be inserted
        include_aux: boolean
            Indicates if the auxiliary qubit should be included
    
    Returns:
        H_1: numpy array
            Hamiltonian of the onsite terms.
    """
    if include_aux:
        num_qubits = 9
    else:
        num_qubits = 8
    H_1 = np.zeros((2**num_qubits, 2**num_qubits), dtype="complex128")
    for qubit_pair in onsite_pairs:
        q1 = qubit_pair[0]
        q2 = qubit_pair[1]
        ZZ_list = [Z if x==q1 else Z if x==q2 else Id for x in range(num_qubits)]
        ZZ = ft.reduce(np.kron, ZZ_list)
        H_1 += ZZ
    return H_1

def H_2(hopping_pairs_1, include_aux):
    """
    This function creates the Hamiltonian of horizontal spin-down and vertical spin-up hopping interaction terms
    for given qubit positions.
    
    Args:
        hopping_pairs_1: list of tuple
            A list of qubit pair positions indicating where the hopping terms should be inserted
        include_aux: boolean
            Indicates if the auxiliary qubit should be included
    
    Returns:
        H_1: numpy array
            Hamiltonian of the hopping terms.
    """
    if include_aux:
        num_qubits = 9
    else:
        num_qubits = 8
    H_2 = np.zeros((2**num_qubits, 2**num_qubits), dtype="complex128")
    for qubit_pair in hopping_pairs_1:
        q1 = qubit_pair[0]
        q2 = qubit_pair[1]
        XX_list = [X if x==q1 else X if x==q2 else Id for x in range(num_qubits)]
        XX = ft.reduce(np.kron, XX_list)
        YY_list = [Y if x==q1 else Y if x==q2 else Id for x in range(num_qubits)]
        YY = ft.reduce(np.kron, YY_list)
        H_2 += (XX + YY)
    return H_2

def H_3(hopping_pairs_2, include_aux):
    """
    This function creates the Hamiltonian of vertical spin-down and horizontal spin-up hopping interaction terms
    for given qubit positions.
    
    Args:
        hopping_pairs_1: list of tuple
            A list of qubit pair positions indicating where the hopping terms should be inserted
        include_aux: boolean
            Indicates if the auxiliary qubit should be included
    
    Returns:
        H_1: numpy array
            Hamiltonian of the hopping terms.
    """
    if include_aux:
        num_qubits = 9
    else:
        num_qubits = 8
    H_3 = np.zeros((2**num_qubits, 2**num_qubits), dtype="complex128")
    for qubit_pair in hopping_pairs_2:
        q1 = qubit_pair[0]
        q2 = qubit_pair[1]
        q3 = qubit_pair[2]
        q4 = qubit_pair[3]
        XX_list = [X if x==q1 else X if x==q2 else Id for x in range(num_qubits)]
        XX_list = [Z if x==q3 else Z if x==q4 else XX_list[x] for x in range(num_qubits)]
        XX = ft.reduce(np.kron, XX_list)
        YY_list = [Y if x==q1 else Y if x==q2 else Id for x in range(num_qubits)]
        YY_list = [Z if x==q3 else Z if x==q4 else YY_list[x] for x in range(num_qubits)]
        YY = ft.reduce(np.kron, YY_list)
        H_3 += (XX + YY)
    return H_3

def ref_fh_hamiltonian(u=2, t=1, WMI_qubit_layout=False, include_aux=False):
    """
    Creates a reference Hamiltonian of the 2x2 Fermi Hubbard model.

    Args:
        u: float
            Coulomb repulsion energy
        t: float
            kinetic hopping term
        WMI_qubit_layout: boolean
            Indicates if the qubit layout should follow the optimal mapping on the WMI grid-like architecture.
            If false, the qubit ordering will just follow the textbook convention
        include_aux: boolean
            Indicates if the auxiliary qubit should be included

    Returns:
        A numpy array of the reference Hamiltonian.
    """
    if WMI_qubit_layout:
        onsite_pairs = [(2,5), (7,8), (1,0), (6,3)]
        hopping_pairs_1 = [(2,1), (7,6), (5,8), (0,3)]
        hopping_pairs_2 = [(2,7,5,8), (1,6,0,3), (5,0,2,1), (8,3,7,6)]
        assert include_aux == True, "When using the WMI qubit layout, the auxiliary qubit must be included"
    else:
        onsite_pairs = [(0,4), (1,5), (2,6), (3,7)]
        hopping_pairs_1 = [(0,2), (1,3), (4,5), (6,7)]
        hopping_pairs_2 = [(0,1,4,5), (2,3,6,7), (4,6,0,2), (5,7,1,3)]
    H1 = H_1(onsite_pairs, include_aux)
    H2 = H_2(hopping_pairs_1, include_aux)
    H3 = H_3(hopping_pairs_2, include_aux)
    H = 0.25*u*H1 - 2*t*(H2 + H3)
    return H

def ref_fh_op(u=2, t=1, delta_t=1, WMI_qubit_layout=False):
    """
    Creates a reference time evolution operator of the 2x2 Fermi Hubbard model.

    Args:
        u: float
            Coulomb repulsion energy
        t: float
            kinetic hopping term
        delta_t: float
            time difference
        WMI_qubit_layout: boolean
            Indicates if the qubit layout should follow the optimal mapping on the WMI grid-like architecture.
            If false, the qubit ordering will just follow the textbook convention

    Returns:
        A numpy array of the reference time evolution operator.
    """
    if WMI_qubit_layout:
        onsite_pairs = [(2,5), (7,8), (1,0), (6,3)]
        hopping_pairs_1 = [(2,1), (7,6), (5,8), (0,3)]
        hopping_pairs_2 = [(2,7,5,8), (1,6,0,3), (5,0,2,1), (8,3,7,6)]
    else:
        onsite_pairs = [(0,4), (1,5), (2,6), (3,7)]
        hopping_pairs_1 = [(0,2), (1,3), (4,5), (6,7)]
        hopping_pairs_2 = [(0,1,4,5), (2,3,6,7), (4,6,0,2), (5,7,1,3)]
    H1 = H_1(onsite_pairs, False)
    H2 = H_2(hopping_pairs_1, False)
    H3 = H_3(hopping_pairs_2, False)
    U = sp.linalg.expm(-1j*0.25*u*H1 -1j*-2*t*(H2 + H3))
    return U