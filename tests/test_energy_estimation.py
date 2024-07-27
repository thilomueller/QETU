import unittest

from qiskit import *
import functools as ft

from qetu_sim import *

class TestEnergyEstimation(unittest.TestCase):
    def expectation_value_decomposition(self):
        """
        Tests if the decomposition of the ground state energy into its energy components is correct, e.g.
        tests if the ground state energy can be reconstructed by the single energy components.
        """
        Id = np.array([[1, 0], [0, 1]], dtype="complex128")
        X = np.array([[0, 1], [1, 0]], dtype="complex128")
        Y = np.array([[0, -1j], [1j, 0]], dtype="complex128")
        Z = np.array([[1, 0], [0, -1]], dtype="complex128")
        P0 = np.array([[1, 0], [0, 0]], dtype="complex128")
        P1 = np.array([[0, 0], [0, 1]], dtype="complex128")

        # test values
        u = 1
        t = t
    
        # reference values
        H_ref = ref_fh_hamiltonian(u=u, t=t, WMI_qubit_layout=True, include_aux=True)
        λ, v = np.linalg.eigh(H_ref)
        ground_state_energy = λ[0]
        E0_vec = v[:,0]

        # onsite energy
        P0 = np.array([[1, 0], [0, 0]])
        P1 = np.array([[0, 0], [0, 1]])

        expectation_value_onsite = 0

        onsite_pairs = [(2,5), (7,8), (1,0), (6,3)]
        for qubit_pair in onsite_pairs:
            q1, q2 = qubit_pair
            Proj_00_list = [P0 if x==q1 else P0 if x==q2 else Id for x in range(9)]
            Proj_00 = ft.reduce(np.kron, Proj_00_list)
            Proj_01_list = [P0 if x==q1 else P1 if x==q2 else Id for x in range(9)]
            Proj_01 = ft.reduce(np.kron, Proj_01_list)
            Proj_10_list = [P1 if x==q1 else P0 if x==q2 else Id for x in range(9)]
            Proj_10 = ft.reduce(np.kron, Proj_10_list)
            Proj_11_list = [P1 if x==q1 else P1 if x==q2 else Id for x in range(9)]
            Proj_11 = ft.reduce(np.kron, Proj_11_list)

            Pr_00 = E0_vec.conj().T @ Proj_00 @ E0_vec
            Pr_01 = E0_vec.conj().T @ Proj_01 @ E0_vec
            Pr_10 = E0_vec.conj().T @ Proj_10 @ E0_vec
            Pr_11 = E0_vec.conj().T @ Proj_11 @ E0_vec

            expectation_value_onsite += 1 * Pr_00
            expectation_value_onsite += -1 * Pr_01
            expectation_value_onsite += -1 * Pr_10
            expectation_value_onsite += 1 * Pr_11

        # hopping energy 1
        meas_trans = QuantumCircuit(9)
        swaps = [(2,5), (7,8), (1,0), (6,3)]
        for swap in swaps:
            meas_trans.append(fSwap(), swap)
        V = circuit2matrix(meas_trans, keep_qiskit_ordering=False)

        expectation_value_hop_1 = 0

        U = QuantumCircuit(9)

        horizontal_pairs = [(2,1), (7,6), (5,8), (0,3)]

        for qubit_pair in horizontal_pairs:
            U = add_transform_to_XX_YY_basis(U, qubit_pair[0], qubit_pair[1])
        U = circuit2matrix(U, keep_qiskit_ordering=False)

        for qubit_pair in horizontal_pairs:
            q1, q2 = qubit_pair
            Proj_00_list = [P0 if x==q1 else P0 if x==q2 else Id for x in range(9)]
            Proj_00 = ft.reduce(np.kron, Proj_00_list)
            Proj_01_list = [P0 if x==q1 else P1 if x==q2 else Id for x in range(9)]
            Proj_01 = ft.reduce(np.kron, Proj_01_list)
            Proj_10_list = [P1 if x==q1 else P0 if x==q2 else Id for x in range(9)]
            Proj_10 = ft.reduce(np.kron, Proj_10_list)
            Proj_11_list = [P1 if x==q1 else P1 if x==q2 else Id for x in range(9)]
            Proj_11 = ft.reduce(np.kron, Proj_11_list)

            Pr_00 = E0_vec.conj().T @ U.conj().T @ Proj_00 @ U @ E0_vec
            Pr_01 = E0_vec.conj().T @ U.conj().T @ Proj_01 @ U @ E0_vec
            Pr_10 = E0_vec.conj().T @ U.conj().T @ Proj_10 @ U @ E0_vec
            Pr_11 = E0_vec.conj().T @ U.conj().T @ Proj_11 @ U @ E0_vec


            #expectation_value_H_h += 1 * Pr_00
            expectation_value_hop_1 += 1 * Pr_01
            expectation_value_hop_1 += -1 * Pr_10
            #expectation_value_H_h += 1 * Pr_11

        expectation_value_hop_1 = 2*expectation_value_hop_1

        # hopping energy 2
        meas_trans = QuantumCircuit(9)
        swaps = [(2,5), (7,8), (1,0), (6,3)]
        for swap in swaps:
            meas_trans.append(fSwap(), swap)
        V = circuit2matrix(meas_trans, keep_qiskit_ordering=False)

        expectation_value_hop_2 = 0

        U = QuantumCircuit(9)

        horizontal_pairs = [(2,1), (7,6), (5,8), (0,3)]

        for qubit_pair in horizontal_pairs:
            U = add_transform_to_XX_YY_basis(U, qubit_pair[0], qubit_pair[1])
        U = circuit2matrix(U, keep_qiskit_ordering=False)


        for qubit_pair in horizontal_pairs:
            q1, q2 = qubit_pair
            Proj_00_list = [P0 if x==q1 else P0 if x==q2 else Id for x in range(9)]
            Proj_00 = ft.reduce(np.kron, Proj_00_list)
            Proj_01_list = [P0 if x==q1 else P1 if x==q2 else Id for x in range(9)]
            Proj_01 = ft.reduce(np.kron, Proj_01_list)
            Proj_10_list = [P1 if x==q1 else P0 if x==q2 else Id for x in range(9)]
            Proj_10 = ft.reduce(np.kron, Proj_10_list)
            Proj_11_list = [P1 if x==q1 else P1 if x==q2 else Id for x in range(9)]
            Proj_11 = ft.reduce(np.kron, Proj_11_list)

            Pr_00 = E0_vec.conj().T @ V.conj().T @ U.conj().T @ Proj_00 @ U @ V @ E0_vec
            Pr_01 = E0_vec.conj().T @ V.conj().T @ U.conj().T @ Proj_01 @ U @ V @ E0_vec
            Pr_10 = E0_vec.conj().T @ V.conj().T @ U.conj().T @ Proj_10 @ U @ V @ E0_vec
            Pr_11 = E0_vec.conj().T @ V.conj().T @ U.conj().T @ Proj_11 @ U @ V @ E0_vec


            #expectation_value_H_h += 1 * Pr_00
            expectation_value_hop_2 += 1 * Pr_01
            expectation_value_hop_2 += -1 * Pr_10
            #expectation_value_H_h += 1 * Pr_11

        expectation_value_hop_2 = 2*expectation_value_hop_2

        # parity terms
        expectation_value_hop_parity = 0

        onsite_pairs = [(5,8), (0,3), (2,1), (7,6)]
        for qubit_pair in onsite_pairs:
            q1, q2 = qubit_pair
            Proj_00_list = [P0 if x==q1 else P0 if x==q2 else Id for x in range(9)]
            Proj_00 = ft.reduce(np.kron, Proj_00_list)
            Proj_01_list = [P0 if x==q1 else P1 if x==q2 else Id for x in range(9)]
            Proj_01 = ft.reduce(np.kron, Proj_01_list)
            Proj_10_list = [P1 if x==q1 else P0 if x==q2 else Id for x in range(9)]
            Proj_10 = ft.reduce(np.kron, Proj_10_list)
            Proj_11_list = [P1 if x==q1 else P1 if x==q2 else Id for x in range(9)]
            Proj_11 = ft.reduce(np.kron, Proj_11_list)

            Pr_00 = E0_vec.conj().T @ Proj_00 @ E0_vec
            Pr_01 = E0_vec.conj().T @ Proj_01 @ E0_vec
            Pr_10 = E0_vec.conj().T @ Proj_10 @ E0_vec
            Pr_11 = E0_vec.conj().T @ Proj_11 @ E0_vec

            expectation_value_hop_parity += 1 * Pr_00
            expectation_value_hop_parity += -1 * Pr_01
            expectation_value_hop_parity += -1 * Pr_10
            expectation_value_hop_parity += 1 * Pr_11

        E0_recon = 0.25*u*expectation_value_onsite - 2*t*(expectation_value_hop_1 + expectation_value_hop_2)

        self.assertAlmostEqual(E0_recon, E0)

if __name__ == "__main__":
    unittest.main()