import numpy as np
from qiskit import *
from util import *

class TFIM():
    """
        Implements the Transverse Field Ising Model.
    """
    def __init__(
            self,
            nqubits,
            g_coupling) -> None:
        assert nqubits >= 0, "The number of qubits cannot be negative"
        self.nqubits = nqubits
        self.g_coupling = g_coupling

    @property
    def Hamiltonian(self):
        """
            Computes the exact Hamiltonian of the TFIM.
            H_TFIM = -\sum_{j=1}^{n-1} Z_j Z_{j+1} - g * \sum_{j=1}^g X_j
        """
        H_1 = np.zeros((2**self.nqubits, 2**self.nqubits), dtype = 'complex_')
        H_2 = np.zeros((2**self.nqubits, 2**self.nqubits), dtype = 'complex_')

        # H_1 = I_1 ⊗ I_2 ⊗ ... ⊗ I_{j-1} ⊗ Z_j ⊗ Z_{j+1} ⊗ I_{j+2} ⊗ ... ⊗ I_n
        for i in range(self.nqubits-1):
            ZZ_gate = np.kron(Z_gate, Z_gate)
            H_1 -= np.kron(np.kron(np.eye(2**i), ZZ_gate), np.eye(2**(self.nqubits-2-i)))

        # H_2 = I_1 ⊗ I_2 ⊗ ... ⊗ I_{j-1} ⊗ X_j ⊗ I_{j+1} ⊗ ... ⊗ I_n
        for i in range(self.nqubits):
            H_2 -= np.kron(np.kron(np.eye(2**i), X_gate), np.eye(2**(self.nqubits-1-i)))
        H_2 *= self.g_coupling

        H_TFIM = H_1 + H_2
        return H_TFIM
    
    def Trotterization(
            self,
            delta_t,
            n = 2
    ):
        """
            Build the circuit for the second-order Trotter approximation of the TFIM Hamiltonian.

            exp(-i*Δt*H_TFIM) = exp(-i*Δt*(H_1 + H_2))
                              ≈ [exp(-i*Δt*H_1 / (2n)) * exp(-i*Δt*H_2 / n) * exp(-i*Δt*H_1 / (2n))]^n

             ┌─────┐            ┌────┐  ┌─────┐            
            ─┤     ├────────────┤ RX ├──┤     ├────────────
             | RZZ |            └────┘  | RZZ |            
             |     |   ┌─────┐  ┌────┐  |     |   ┌─────┐  
            ─┤     ├───┤     ├──┤ RX ├──┤     ├───┤     ├──
             └─────┘   | RZZ |  └────┘  └─────┘   | RZZ |    
             ┌─────┐   |     |  ┌────┐  ┌─────┐   |     |  
            ─┤     ├───┤     ├──┤ RX ├──┤     ├───┤     ├──
             | RZZ |   └─────┘  └────┘  | RZZ |   └─────┘  
             |     |   ┌─────┐  ┌────┐  |     |   ┌─────┐  
            ─┤     ├───┤     ├──┤ RX ├──┤     ├───┤     ├──
             └─────┘   | RZZ |  └────┘  └─────┘   | RZZ |  
             ┌─────┐   |     |  ┌────┐  ┌─────┐   |     |  
            ─┤     ├───┤     ├──┤ RX ├──┤     ├───┤     ├──
             | RZZ |   └─────┘  └────┘  | RZZ |   └─────┘  
             |     |            ┌────┐  |     |            
            ─┤     ├────────────┤ RX ├──┤     ├────────────
             └─────┘            └────┘  └─────┘            
            ^-----------------^ ^-----^ ^----------------^
                    H_1           H_2          H_1
            ^---------------------------------------------^
                            repeat n times

            Input:
                delta_t
                n
            Output:
                ciruit of the tortterized Hamiltonian
        """
        trotter_order = 2

        def H_1():
            """
                Implementation of the first part of the TFIM Hamiltonian.

                exp(-i*Δt*H_1 / (2n)) = \prod_j exp(-i*Δt/(2n) * Z_j ⊗ Z_{j+1})
                                      = \prod_j RZZ(-Δt/n)
            """
            q = QuantumRegister(self.nqubits, 'q')
            circuit = QuantumCircuit(q)
            for j in range(self.nqubits-1):
                if j % 2 == 0:
                    circuit.rzz(-delta_t/n, q[j], q[j+1])
            for j in range(self.nqubits-1):
                if j % 2 != 0:
                    circuit.rzz(-delta_t/n, q[j], q[j+1])
            return circuit
        
        def H_2():
            """
                Implementation of the second part of the TFIM Hamiltonian.

                exp(-i*Δt*H_2 / n) = \prod_j exp(-i*Δt/n * g* X_j)
                                   = \prod_j RX(-2*Δt*g)
            """
            q = QuantumRegister(self.nqubits, 'q')
            circuit = QuantumCircuit(q)
            for j in range(self.nqubits):
                circuit.rx(-2*self.g_coupling*delta_t/n, q[j])
            return circuit
        

        q = QuantumRegister(self.nqubits, 'q')
        circuit = QuantumCircuit(q)
        for _ in range(n-1):
            circuit.compose(H_1(), q, inplace=True)
            circuit.compose(H_2(), q, inplace=True)
            circuit.compose(H_1(), q, inplace=True)

        return circuit
    

    def Control_free_Trotter(
            self,
            n = 2
    ):
        """
            Build the circuit of the controlled time evolution of the TFIM Hamiltonian

            [ exp(i*Δt*H)         0       ]  =  [ K * exp(-i*Δt*H) * K         0       ]
            [     0          exp(-i*Δt*H) ]     [        0                exp(-i*Δt*H) ]

            ───○────────────────────────○───
             ┌─┴─┐  ┌──────────────┐  ┌─┴─┐ 
            ─┤ K ├──┤ exp(-i*Δt*H) ├──┤ K ├─
             └───┘  └──────────────┘  └───┘ 

            K = Y ⊗ Z ⊗ Y ⊗ Z ⊗ Y ⊗ ...

            Input:
                n
            Output:
            
        """
        def K():
            """"
                Implementation of the Pauli term K that anticommutes with each term in the TFIM Hamiltonian

                             ───o─────o─────o─────o───
                              ┌─┴─┐   │     │     │  
                             ─┤ Z ├───┼─────┼─────┼───
                ───○────      └───┘ ┌─┴─┐   │     │   
                 ┌─┴─┐    =  ───────┤ Y ├───┼─────┼───
                ─┤ K ├──            └───┘ ┌─┴─┐   │  
                 └───┘       ─────────────┤ Z ├───┼───
                                          └───┘ ┌─┴─┐ 
                             ───────────────────┤ Y ├─
                                                └───┘    
            """
            q = QuantumRegister(self.nqubits, 'q')
            anc = QuantumRegister(1, 'ancilla')
            circuit = QuantumCircuit(anc, q)
            for j in range(self.nqubits):
                if j % 2 != 0:
                    circuit.cy(anc, q[j], ctrl_state='0')
                else:
                    circuit.cz(anc, q[j], ctrl_state='0')
            return circuit

        q = QuantumRegister(self.nqubits, 'q')
        anc = QuantumRegister(1, 'ancilla')
        circuit = QuantumCircuit(anc, q)
        circuit.compose(K(), anc[:] + q[:], inplace=True)
        circuit.barrier(q)
        circuit.compose(self.Trotterization(n), q, inplace=True)
        circuit.barrier(q)
        circuit.compose(K(), anc[:] + q[:], inplace=True)
        return circuit
