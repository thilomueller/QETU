import scipy as sp
import numpy as np

I = sp.sparse.csr_array(np.array([[1, 0], [0, 1]]))
X = sp.sparse.csr_array(np.array([[0, 1], [1, 0]]))
Y = sp.sparse.csr_array(np.array([[0, -1j], [1j, 0]]))
Z = sp.sparse.csr_array(np.array([[1, 0], [0, -1]]))

op_1 = sp.sparse.kron(X, sp.sparse.kron(I, sp.sparse.kron(X, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, I))))))) # X_1 X_3
op_2 = sp.sparse.kron(Y, sp.sparse.kron(I, sp.sparse.kron(Y, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, I))))))) # Y_1 Y_3
op_3 = sp.sparse.kron(I, sp.sparse.kron(X, sp.sparse.kron(I, sp.sparse.kron(X, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, I))))))) # X_2 X_4
op_4 = sp.sparse.kron(I, sp.sparse.kron(Y, sp.sparse.kron(I, sp.sparse.kron(Y, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, I))))))) # Y_2 Y_4
op_5 = sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(X, sp.sparse.kron(X, sp.sparse.kron(I, I))))))) # X_5 X_6
op_6 = sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(Y, sp.sparse.kron(Y, sp.sparse.kron(I, I))))))) # Y_5 Y_6
op_7 = sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(X, X))))))) # X_7 X_8
op_8 = sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(Y, Y))))))) # Y_7 Y_8
op_9 = sp.sparse.kron(X, sp.sparse.kron(X, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(Z, sp.sparse.kron(Z, sp.sparse.kron(I, I))))))) # X_1 X_2 Z_5 Z_6
op_10 = sp.sparse.kron(Y, sp.sparse.kron(Y, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(Z, sp.sparse.kron(Z, sp.sparse.kron(I, I))))))) # Y_1 Y_2 Z_5 Z_6
op_11 = sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(X, sp.sparse.kron(X, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(Z, Z))))))) # X_3 X_4 Z_7 Z_8
op_12 = sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(Y, sp.sparse.kron(Y, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(Z, Z))))))) # Y_3 Y_4 Z_7 Z_8
op_13 = sp.sparse.kron(Z, sp.sparse.kron(I, sp.sparse.kron(Z, sp.sparse.kron(I, sp.sparse.kron(X, sp.sparse.kron(I, sp.sparse.kron(X, I))))))) # Z_1 Z_3 X_5 X_7
op_14 = sp.sparse.kron(Z, sp.sparse.kron(I, sp.sparse.kron(Z, sp.sparse.kron(I, sp.sparse.kron(Y, sp.sparse.kron(I, sp.sparse.kron(Y, I))))))) # Z_1 Z_3 Y_5 Y_7
op_15 = sp.sparse.kron(I, sp.sparse.kron(Z, sp.sparse.kron(I, sp.sparse.kron(Z, sp.sparse.kron(I, sp.sparse.kron(X, sp.sparse.kron(I, X))))))) # Z_2 Z_4 X_6 X_8
op_16 = sp.sparse.kron(I, sp.sparse.kron(Z, sp.sparse.kron(I, sp.sparse.kron(Z, sp.sparse.kron(I, sp.sparse.kron(Y, sp.sparse.kron(I, Y))))))) # Z_2 Z_4 Y_6 Y_8

op_17 = sp.sparse.kron(X, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(X, sp.sparse.kron(I, sp.sparse.kron(I, I))))))) # X_1 X_5
op_18 = sp.sparse.kron(Y, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(Y, sp.sparse.kron(I, sp.sparse.kron(I, I))))))) # Y_1 Y_5
op_19 = sp.sparse.kron(Z, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(Z, sp.sparse.kron(I, sp.sparse.kron(I, I))))))) # Z_1 Z_5
op_20 = sp.sparse.kron(Z, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, I))))))) # Z_1
op_21 = sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(Z, sp.sparse.kron(I, sp.sparse.kron(I, I))))))) # Z_5

op_22 = sp.sparse.kron(I, sp.sparse.kron(X, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(X, sp.sparse.kron(I, I))))))) # X_2 X_6
op_23 = sp.sparse.kron(I, sp.sparse.kron(Y, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(Y, sp.sparse.kron(I, I))))))) # Y_2 Y_6
op_24 = sp.sparse.kron(I, sp.sparse.kron(Z, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(Z, sp.sparse.kron(I, I))))))) # Z_2 Z_6
op_25 = sp.sparse.kron(I, sp.sparse.kron(Z, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, I))))))) # Z_2
op_26 = sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(Z, sp.sparse.kron(I, I))))))) # Z_6

op_27 = sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(X, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(X, I))))))) # X_3 X_7
op_28 = sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(Y, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(Y, I))))))) # Y_3 Y_7
op_29 = sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(Z, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(Z, I))))))) # Z_3 Z_7
op_30 = sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(Z, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, I))))))) # Z_3
op_31 = sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(Z, I))))))) # Z_7

op_32 = sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(X, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, X))))))) # X_4 X_8
op_33 = sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(Y, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, Y))))))) # Y_4 Y_8
op_34 = sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(Z, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, Z))))))) # Z_4 Z_8
op_35 = sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(Z, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, I))))))) # Z_4
op_36 = sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, sp.sparse.kron(I, Z))))))) # Z_8

def ref_fh_hamiltonian(u=2, t=1):
    """
    Creates a reference Hamiltonian of the 2x2 Fermi Hubbard model.

    Args:
        u - Coulomb repulsion energy
        t - kinetic hopping term
        delta_t - time difference

    Returns:
        A numpy array of the reference Hamiltonian.
    """
    H = 0.25*u*(op_19 + op_24 + op_29 + op_34) -2*t*(op_1 + op_2 + op_3 + op_4 + op_5 + op_6 + op_7 + op_8
                                                                   + op_9 + op_10 + op_11 + op_12 + op_13 + op_14 + op_15 + op_16)
    return H.toarray()

def ref_fh_op(u=2, t=1, delta_t=1):
    """
    Creates a reference time evolution operator of the 2x2 Fermi Hubbard model.

    Args:
        u - Coulomb repulsion energy
        t - kinetic hopping term
        delta_t - time difference

    Returns:
        A numpy array of the reference time evolution operator.
    """
    U = sp.sparse.linalg.expm(-1j*0.25*u*delta_t*(op_19 + op_24 + op_29 + op_34)
                              -1j*-2*t*delta_t*(op_1 + op_2 + op_3 + op_4 + op_5 + op_6 + op_7 + op_8
                                                + op_9 + op_10 + op_11 + op_12 + op_13 + op_14 + op_15 + op_16))
    return U.toarray()