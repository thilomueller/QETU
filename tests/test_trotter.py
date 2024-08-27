import unittest

from qiskit import *

from qetu_sim.util import *
from qetu_sim.fh_2x2_sim import *

class TestTrotter(unittest.TestCase):
    def test_trotter_V_convergence(self):
        """
        Test the convergence of the controlled forward and backward time evolution operator.
        """
        # increasing number of Trotter steps
        print("Test trotter convergence")
        u = 1
        t = 1
        delta_t = 1
        U_ref = ref_fh_op(u, t, delta_t)
        V_ref = scipy.linalg.block_diag(U_ref.conjugate().transpose(), U_ref)
        trotter_steps_list = range(1,50,5)
        previous_error = 2.0
        for trotter_steps in trotter_steps_list:
            V = construct_trotter_V(u, t, delta_t, trotter_steps, shift=False)
            V_matrix = circuit2matrix(V, keep_qiskit_ordering=True)
            trotter_error = np.linalg.norm(V_matrix - V_ref, 2)
            print(trotter_error)
            self.assertLess(trotter_error, previous_error,
                            "The Trotter approximaton should improve with increasing number of steps.")
            previous_error = trotter_error

    def test_trotter_V_sh_convergence(self):
        """
        Test the convergence of the controlled forward and backward time evolution operator for the shifted Hamiltonian.
        """
        print("Test trotter approximation of shifted Hamiltonian")
        u = 1
        t = 1
        delta_t = 1
        H_ref = ref_fh_hamiltonian(u, t)
        c1, c2 = calculate_shift_params(u, t) 
        H_sh_ref = c1*H_ref + c2*np.eye(H_ref.shape[0])
        U_sh_ref = scipy.linalg.expm(-1j*H_sh_ref)
        V_sh_ref = scipy.linalg.block_diag(U_sh_ref.conjugate().transpose(), U_sh_ref)
        trotter_steps_list = range(1,10)
        previous_error = 2.0
        for trotter_steps in trotter_steps_list:
            V_sh = construct_trotter_V(u, t, delta_t, trotter_steps, shift=True)
            V_sh_matrix = circuit2matrix(V_sh, keep_qiskit_ordering=True)
            trotter_sh_error = np.linalg.norm(V_sh_matrix - V_sh_ref, 2)
            print(trotter_sh_error)
            self.assertLess(trotter_sh_error, previous_error,
                            "The Trotter approximaton should improve with increasing number of steps.")
            previous_error = trotter_sh_error

if __name__ == "__main__":
    unittest.main()