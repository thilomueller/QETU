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
        degree = 50
        trotter_steps = 1
        step_function_qsp_angles = [
            7.93660392e-01,  1.40384957e-03,  1.77477715e-03,  9.17941620e-04,
            7.82363809e-04, -6.59977854e-04, -1.44149937e-03, -3.63268962e-03,
            -5.16004369e-03, -8.15864994e-03, -1.04737256e-02, -1.42753446e-02,
            -1.73431125e-02, -2.18448613e-02, -2.55030871e-02, -3.04783298e-02,
            -3.44414472e-02, -3.95504607e-02, -4.34228338e-02, -4.82166458e-02,
            -5.15097980e-02, -5.54733361e-02, -5.76793896e-02, -6.03316840e-02,
            -6.10338607e-02, -6.20455411e-02, -6.10338607e-02, -6.03316840e-02,
            -5.76793896e-02, -5.54733361e-02, -5.15097980e-02, -4.82166458e-02,
            -4.34228338e-02, -3.95504607e-02, -3.44414472e-02, -3.04783298e-02,
            -2.55030871e-02, -2.18448613e-02, -1.73431125e-02, -1.42753446e-02,
            -1.04737256e-02, -8.15864994e-03, -5.16004369e-03, -3.63268962e-03,
            -1.44149937e-03, -6.59977854e-04,  7.82363809e-04,  9.17941620e-04,
            1.77477715e-03,  1.40384957e-03,  7.93660392e-01]
        phi_vec = step_function_qsp_angles
        QETU_circ = construct_QETU_circ(u, t, trotter_steps, phi_vec)
        QETU_circ_WMI = transpile_QETU_to_WMI(QETU_circ)
        final_state = qetu_sim(QETU_circ_WMI)
        H_ref = ref_fh_hamiltonian(u=u, t=t, WMI_qubit_layout=True, include_aux=True)
        λ, v = np.linalg.eigh(H_ref)
        ground_state_energy = λ[0]
        ground_state_vector = v[:,0]
        success_probability = scipy.linalg.norm(final_state)**2
        overlap = abs(np.vdot(final_state, ground_state_vector))**2
        self.assertGreaterEqual(overlap, 0.99, "The overlap with the ground state is not high enough.")
        print("overlap: " + str(overlap))

if __name__ == "__main__":
    unittest.main()