import unittest

from qiskit import *

from qetu_sim.util import *
from qetu_sim.fh_2x2_sim import *
from qetu_sim.qsp_phase_matlab import *

class TestQETU(unittest.TestCase):
    def test_ground_state_preparation(self):
        """
        Test the ground state preparation of the QETU circuit.
        """
        print("Test ground state preparation")
        u = 1
        t = 1
        degree = 250
        trotter_steps = 10
        #mat_step_func = scipy.io.loadmat('phase_angles/step_function_02_d' + str(degree) + '.mat')['phi_proc']
        #step_function_qsp_angles = list(itertools.chain.from_iterable(mat_step_func))
        E_min, E_mu_m, E_mu_p, E_max = calculate_qsp_params(u, t)
        qsp = QSPPhase()
        phi_seq_su2 = qsp.cvx_qsp_heaviside(
            degree, 
            E_min,
            E_mu_m, 
            E_mu_p, 
            E_max
        )
        phi_vec = convert_Zrot_to_Xrot(phi_seq_su2)
        QETU_circ = construct_QETU_circ(u, t, trotter_steps, phi_vec)
        QETU_circ_WMI = transpile_QETU_to_WMI(QETU_circ)
        initial_state = Statevector.from_label("0-+0++++0")
        final_state = qetu_sim(QETU_circ_WMI, initial_state)
        ground_state_energy, ground_state_vector = calculate_reference_ground_state(u, t, True)
        success_probability = scipy.linalg.norm(final_state)**2
        overlap = abs(np.vdot(final_state / scipy.linalg.norm(final_state), ground_state_vector))**2
        self.assertGreaterEqual(overlap, 0.95, "The overlap with the ground state is not high enough.")
        print("overlap: " + str(overlap))

if __name__ == "__main__":
    unittest.main()