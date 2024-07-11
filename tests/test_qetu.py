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
        #E_min, E_mu_m, E_mu_p, E_max = calculate_qsp_params(u, t)
        #qsp = QSPPhase()
        #phi_seq_su2 = qsp.cvx_qsp_heaviside(
        #    degree, 
        #    E_min,
        #    E_mu_m, 
        #    E_mu_p, 
        #    E_max
        #)
        #phi_vec = convert_Zrot_to_Xrot(phi_seq_su2)
        #mat_step_func = scipy.io.loadmat('phase_angles/step_function_sigma_d' + str(degree) + '.mat')
        step_function_qsp_angles = [
            0.7854,   -0.0000,   -0.0000,   -0.0000,   -0.0000,   -0.0000,   -0.0000,   -0.0000,
           -0.0000,   -0.0000,   -0.0000,   -0.0000,   -0.0000,   -0.0000,   -0.0001,   -0.0001,
           -0.0001,   -0.0002,   -0.0003,   -0.0004,   -0.0005,   -0.0007,   -0.0009,   -0.0011,
           -0.0014,   -0.0016,   -0.0019,   -0.0022,   -0.0025,   -0.0028,   -0.0030,   -0.0032,
           -0.0033,   -0.0034,   -0.0033,   -0.0032,   -0.0029,   -0.0026,   -0.0022,   -0.0017,
           -0.0013,   -0.0008,   -0.0004,   -0.0000,    0.0002,    0.0004,    0.0006,    0.0007,
            0.0008,    0.0009,    0.0011,    0.0014,    0.0018,    0.0024,    0.0031,    0.0038,
            0.0045,    0.0050,    0.0053,    0.0052,    0.0046,    0.0034,    0.0016,   -0.0008,
           -0.0037,   -0.0070,   -0.0105,   -0.0137,   -0.0164,   -0.0182,   -0.0189,   -0.0181,
           -0.0157,   -0.0117,   -0.0062,    0.0005,    0.0081,    0.0160,    0.0236,    0.0303,
            0.0355,    0.0386,    0.0393,    0.0375,    0.0331,    0.0266,    0.0183,    0.0089,
           -0.0008,   -0.0102,   -0.0184,   -0.0249,   -0.0293,   -0.0312,   -0.0308,   -0.0281,
           -0.0236,   -0.0180,   -0.0118,   -0.0058,   -0.0005,    0.0033,    0.0055,    0.0057,
            0.0041,    0.0006,   -0.0043,   -0.0102,   -0.0167,   -0.0232,   -0.0293,   -0.0346,
           -0.0388,   -0.0417,   -0.0434,   -0.0438,   -0.0431,   -0.0416,   -0.0395,   -0.0371,
           -0.0347,   -0.0324,   -0.0305,   -0.0290,   -0.0281,   -0.0278,   -0.0281,   -0.0290,
           -0.0305,   -0.0324,   -0.0347,   -0.0371,   -0.0395,   -0.0416,   -0.0431,   -0.0438,
           -0.0434,   -0.0417,   -0.0388,   -0.0346,   -0.0293,   -0.0232,   -0.0167,   -0.0102,
           -0.0043,    0.0006,    0.0041,    0.0057,    0.0055,    0.0033,   -0.0005,   -0.0058,
           -0.0118,   -0.0180,   -0.0236,   -0.0281,   -0.0308,   -0.0312,   -0.0293,   -0.0249,
           -0.0184,   -0.0102,   -0.0008,    0.0089,    0.0183,    0.0266,    0.0331,    0.0375,
            0.0393,    0.0386,    0.0355,    0.0303,    0.0236,    0.0160,    0.0081,    0.0005,
           -0.0062,   -0.0117,   -0.0157,   -0.0181,   -0.0189,   -0.0182,   -0.0164,   -0.0137,
           -0.0105,   -0.0070,   -0.0037,   -0.0008,    0.0016,    0.0034,    0.0046,    0.0052,
            0.0053,    0.0050,    0.0045,    0.0038,    0.0031,    0.0024,    0.0018,    0.0014,
            0.0011,    0.0009,    0.0008,    0.0007,    0.0006,    0.0004,    0.0002,   -0.0000,
           -0.0004,   -0.0008,   -0.0013,   -0.0017,   -0.0022,   -0.0026,   -0.0029,   -0.0032,
           -0.0033,   -0.0034,   -0.0033,   -0.0032,   -0.0030,   -0.0028,   -0.0025,   -0.0022,
           -0.0019,   -0.0016,   -0.0014,   -0.0011,   -0.0009,   -0.0007,   -0.0005,   -0.0004,
           -0.0003,   -0.0002,   -0.0001,   -0.0001,   -0.0001,   -0.0000,   -0.0000,   -0.0000,
           -0.0000,   -0.0000,   -0.0000,   -0.0000,   -0.0000,   -0.0000,   -0.0000,   -0.0000,
           -0.0000,   -0.0000,    0.7854]
        phi_vec = convert_Zrot_to_Xrot(step_function_qsp_angles)
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