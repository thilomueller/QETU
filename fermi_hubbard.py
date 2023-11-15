import numpy as np
from qiskit import *
from util import *

class FermiHubbard():
    """
        Implements the Fermi Hubbard model
    """
    def __init__(
        self,
        nqubits,
        μ,      # chemical potential parameter, controlling the average number of particles in the system
        u,      # on-site Couloumb repulsion energy, describing the energy cost associated with two fermions with opposite spins occupying the same site
        t       # hopping energy, describing the amplitude for particles to hop from one site to its neighboring site

    ) -> None:
        assert nqubits >= 0, "The number of qubits cannot be negative"
        self.nqubits = nqubits
        self.μ = μ
        self.u = u
        self.t = t

    @property
    def Hamiltonian(self):
        """
            Computes the exact spin Hamiltonian of the Fermi Hubbard model after the Jordan-Wigner transformation.
            H_FH = 1/2 (1/2 *u - μ) * \sum_{j=1}^n \sum_{σ ∈ [↑,↓]} Z_{j,σ} + 1/4 u * \sum_{j=1}^n Z_{j,↑} Z_{j,↓} - \sum_{j=1}^{n-1} \sum_{σ ∈ [↑,↓]} ()
        """
        # each fermion is emulated by two qubits (spin up and spin down)
        nsysqubits = 2*self.nqubits

        H_1 = np.zeros((2**nsysqubits, 2**nsysqubits), dtype = 'complex_')
        H_2 = np.zeros((2**nsysqubits, 2**nsysqubits), dtype = 'complex_')
        H_3 = np.zeros((2**nsysqubits, 2**nsysqubits), dtype = 'complex_')

        # I_{1,↑} ⊗ I_{2,↑} ⊗ ... ⊗ I_{j-1,↑} ⊗ Z_{j,↑} ⊗ I_{j+1,↑} ⊗ ... ⊗ I_{n,↑} + 
        # I_{1,↓} ⊗ I_{2,↓} ⊗ ... ⊗ I_{j-1,↓} ⊗ Z_{j,↓} ⊗ I_{j+1,↓} ⊗ ... ⊗ I_{n,↓}
        for i in range(nsysqubits):
            H_1 += np.kron(np.kron(np.eye(2**i), Z_gate), np.eye(2**(nsysqubits-1-i)))
        H_1 *= 1/2 * (1/2*self.u - self.μ)

        # I_{1,↑} ⊗ I_{2,↑} ⊗ ... ⊗ Z_{j,↑} ⊗ ... ⊗ I_{n,↑} ⊗ I_{1,↓} ⊗ I_{2,↓} ⊗ ... ⊗ Z_{j,↓} ⊗ ... ⊗ I_{n,↓}
        for i in range(self.nqubits):
            H_2 += np.kron(np.kron(np.kron(np.kron(np.eye(2**i), Z_gate), np.eye(2**(self.nqubits-1))), Z_gate), np.eye(2**(self.nqubits-1-i)))
        H_2 *= 1/4*self.u

        sigma_plus = X_gate + 1j * Y_gate
        sigma_minus = X_gate - 1j * Y_gate
        # I_{1,↑} ⊗ I_{2,↑} ⊗ ... ⊗ I_{j-1,↑} ⊗ Σ+_{j,↑} ⊗ Σ-_{j+1,↑} ⊗ I_{j+2,↑} ⊗ ... ⊗ I_{n,↑} ⊗ I_{1,↓} ⊗ I_{2,↓} ⊗ ... ⊗ I_{n,↓} +    
        # I_{1,↑} ⊗ I_{2,↑} ⊗ ... ⊗ I_{j-1,↑} ⊗ Σ-_{j,↑} ⊗ Σ+_{j+1,↑} ⊗ I_{j+2,↑} ⊗ ... ⊗ I_{n,↑} ⊗ I_{1,↓} ⊗ I_{2,↓} ⊗ ... ⊗ I_{n,↓} +
        # I_{1,↑} ⊗ I_{2,↑} ⊗ ... ⊗ I_{n,↑} ⊗ I_{1,↓} ⊗ ... ⊗ I_{j-1,↓} ⊗ Σ+_{j,↓} ⊗ Σ-_{j+1,↓} ⊗ I_{j+2,↓} ⊗ ... ⊗ I_{n,↓} +  
        # I_{1,↑} ⊗ I_{2,↑} ⊗ ... ⊗ I_{n,↑} ⊗ I_{1,↓} ⊗ ... ⊗ I_{j-1,↓} ⊗ Σ-_{j,↓} ⊗ Σ+_{j+1,↓} ⊗ I_{j+2,↓} ⊗ ... ⊗ I_{n,↓}           
        for i in range(self.nqubits-1):
            H_3 += np.kron(np.kron(np.kron(np.eye(2**i), sigma_plus), sigma_minus), np.eye(2**(2*self.nqubits-2-i)))
            H_3 += np.kron(np.kron(np.kron(np.eye(2**i), sigma_minus), sigma_plus), np.eye(2**(2*self.nqubits-2-i)))
            H_3 += np.kron(np.kron(np.kron(np.eye(2**(self.nqubits+i)), sigma_plus), sigma_minus), np.eye(2**(self.nqubits-2-i)))
            H_3 += np.kron(np.kron(np.kron(np.eye(2**(self.nqubits+i)), sigma_minus), sigma_plus), np.eye(2**(self.nqubits-2-i)))
        H_3 *= -self.t

        H_TFIM = H_1 + H_2 + H_3
        return H_TFIM

    def Trotterization(
            self,
            delta_t = 1,
            n = 2,
            control_free = False
    ):
        """
            Build the circuit for the second-order Trotter approximation of the Fermi Hubbard Hamiltonian.

            exp(-i*Δt*H_FH) = exp(-i*Δt*(H_1 + H_2 * H_3))
                            ≈ [exp(-i*Δt/2 * H_1) * exp(-i*Δt/2 * H_2) * exp(-i*Δt * H_3) * exp(-i*Δt/2 * H_2) * exp(-i*Δt/2 * H_1)]^n

            Input:
                delta_t
                n
                control_free
            Output:
                circuit of the trotterized Fermi Hubbard Hamiltonian
        """
        trotter_order = 2
        def H_1():
            """
                Implementation of the first term of the Fermi Hubbard Hamiltonian.
                This term describes the chemical potential.

                exp(-i*Δt/2 * H_1) = exp(-i*Δt*1/4*(1/2 u - μ) \sum_{j,σ} Z_{j,σ} / n)
                                   = \prod_{j,σ} exp(-i*Δt*1/4*(1/2 u - μ) * Z_k / n)
                                   = \prod_{j,σ} RZ(1/2*Δt*(1/2 u - μ)/n)            
                     ┌────┐  
                ↑ ───┤ RZ ├──
                     └────┘      
                     ┌────┐  
                ↑ ───┤ RZ ├──
                     └────┘      
                     ┌────┐  
                ↓ ───┤ RZ ├──
                     └────┘  
                     ┌────┐  
                ↓ ───┤ RZ ├──
                     └────┘  
            """
            λ = 1/2 * delta_t * (1/2 * self.u - self.μ) / n
            spin_up = QuantumRegister(self.nqubits, '↑')
            spin_down = QuantumRegister(self.nqubits, '↓')
            circuit = QuantumCircuit(spin_down, spin_up)
            for j in range(self.nqubits):
                circuit.rz(λ, spin_up[j])
                circuit.rz(λ, spin_down[j])
            return circuit
        
        def H_2():
            """
                Implementation of the second term of the Fermi Hubbard Hamiltonian.
                This term is the interaction term.

                exp(-i*Δt/2 * H_2 / n) = exp(-i*Δt*1/8*u/n * \sum_j Z_{j,↑} ⊗ Z_{j,↓})
                                       = \prod_j exp(-i*Δt*1/8*u/n * Z_{j,↑} ⊗ Z_{j,↓})
                                       = \prod_j RZZ(1/4*Δt*u/n)
                                                      
                ↑ 0: ─■───────────────────────────────
                      │                               
                ↑ 1: ─┼──────────■────────────────────
                      │          │                    
                ↑ 2: ─┼──────────┼──────────■─────────
                      │RZZ       │          │         
                ↓ 0: ─■──────────┼──────────┼─────────
                                 │RZZ       │         
                ↓ 1: ────────────■──────────┼─────────
                                            │RZZ      
                ↓ 2: ───────────────────────■─────────
            """
            θ = 1/4 * delta_t * self.u / n
            spin_up = QuantumRegister(self.nqubits, '↑')
            spin_down = QuantumRegister(self.nqubits, '↓')
            circuit = QuantumCircuit(spin_down, spin_up)
            for j in reversed(range(self.nqubits)):
                circuit.rzz(θ, spin_up[j], spin_down[j])
            return circuit
        
        def H_3():
            """
                Implementation of the thrid term of the Fermi Hubbard Hamiltonian.
                This term is the kinetic hopping term.

                exp(-i*Δt * H_3 / n) = exp(-i*Δt*(-t)/n * \sum_{j,σ} (Σ+_{j,σ} Σ-_{j+1,σ} - Σ+_{j+1,σ} Σ-_{j,σ}))
                                     = \prod_{j,σ} exp(-i*Δt*(-t)/n * (Σ+_{j,σ} Σ-_{j+1,σ} - Σ+_{j+1,σ} Σ-_{j,σ}))


                 ┌─────┐             
                ─┤     ├────────────
                 |  Σ  |                      
                 |     |   ┌─────┐  
                ─┤     ├───┤     ├──
                 └─────┘   |  Σ  |     
                 ┌─────┐   |     |    
                ─┤     ├───┤     ├──
                 |  Σ  |   └─────┘    
                 |     |   ┌─────┐    
                ─┤     ├───┤     ├──
                 └─────┘   |  Σ  |    
                 ┌─────┐   |     |    
                ─┤     ├───┤     ├──
                 |  Σ  |   └─────┘    
                 |     |                        
                ─┤     ├────────────
                 └─────┘                        
            """

            def sigma_gate():
                """""
                    Implementation of the gate used to 
                    exp(-i*Δt*(-t)/n * (Σ+_{j,σ} Σ-_{j+1,σ} - Σ+_{j+1,σ} Σ-_{j,σ}))
                     = exp(-i*Δt*(-t)/n * 2 * (X_{j,σ} ⊗ X_{j+1,σ} + 2 * Y_{j,σ} ⊗ Y_{j+1,σ}))
                     = exp(-i*Δt*(-t)/n * 2 * X_{j,σ} ⊗ X_{j+1,σ}) * exp(-i*Δt*(-t)/n 2 * Y_{j,σ} ⊗ Y_{j+1,σ})
                     = RXX(-4*Δt*t/n) * RYY(-4*Δt*t/n)

                     ┌─────┐         ┌─────┐   ┌─────┐     
                    ─┤     ├──      ─┤     ├───┤     ├──
                     |  Σ  |    =    | RXX |   | RYY |                 
                     |     |         |     |   |     |
                    ─┤     ├──      ─┤     ├───┤     ├──          
                     └─────┘         └─────┘   └─────┘  
                """
                θ = 4 * delta_t * -self.t / n
                q = QuantumRegister(2, 'q')
                circuit = QuantumCircuit(q)
                circuit.rxx(θ, q[0], q[1])
                circuit.ryy(θ, q[0], q[1])
                gate = circuit.to_gate(label="Σ")
                return gate

            spin_up = QuantumRegister(self.nqubits, '↑')
            spin_down = QuantumRegister(self.nqubits, '↓')
            circuit = QuantumCircuit(spin_down, spin_up)
            for j in range(self.nqubits-1):
                if j % 2 == 0:
                    #circuit.append(sigma_gate(), [spin_up[j], spin_up[j+1]])
                    #circuit.append(sigma_gate(), [spin_down[j], spin_down[j+1]])
                    circuit.append(sigma_gate(), [spin_up[j+1], spin_up[j]])
                    circuit.append(sigma_gate(), [spin_down[j+1], spin_down[j]])
            for j in range(self.nqubits-1):
                if j % 2 != 0:
                    #circuit.append(sigma_gate(), [spin_up[j], spin_up[j+1]])
                    #circuit.append(sigma_gate(), [spin_down[j], spin_down[j+1]])
                    circuit.append(sigma_gate(), [spin_up[j+1], spin_up[j]])
                    circuit.append(sigma_gate(), [spin_down[j+1], spin_down[j]])
            return circuit
        
        def K_1():
            """
                Implementation of the Pauli string K_1 that anticommutes with the first term (chemical potential term) of the Fermi Hubbard Hamiltonian.

                K_1 = ⊗_{j,σ} X_{j,σ} = X_{1,↑} ⊗ X_{1,↓} ⊗ X_{2,↑} ⊗ X_{2,↓} ⊗ ... ⊗ X_{n,↑} ⊗ X_{n,↓}
            """
            anc = QuantumRegister(1, 'ancilla')
            spin_up = QuantumRegister(self.nqubits, '↑')
            spin_down = QuantumRegister(self.nqubits, '↓')
            circuit = QuantumCircuit(spin_down, spin_up, anc)
            for j in range(self.nqubits):
                circuit.cx(anc, spin_up[j], ctrl_state='0')
                circuit.cx(anc, spin_down[j], ctrl_state='0')
            return circuit
        
        def K_2():
            """
                Implementation of the Pauli string K_2 that anicommutes with the second term (interaction term) of the Fermi Hubbard Hamiltonian.

                K_2 = ( ⊗_{j=1}^n X_{j,↑} ) ⊗ ( ⊗_{j=1}^n I_{j,↓} )
                    = X_{1,↑} ⊗ X_{2,↑} ⊗ ... ⊗ X_{n,↑} ⊗ I_{1,↓} ⊗ I_{2,↓} ⊗ ... ⊗ I_{n,↓}
            """
            anc = QuantumRegister(1, 'ancilla')
            spin_up = QuantumRegister(self.nqubits, '↑')
            spin_down = QuantumRegister(self.nqubits, '↓')
            circuit = QuantumCircuit(spin_down, spin_up, anc)
            for j in range(self.nqubits):
                circuit.cx(anc, spin_up[j], ctrl_state='0')
            return circuit
        
        def K_3():
            """
                Implementation of the Pauli string K_3 that anticommutes with the third term (kinetic hoppinig term) of the Fermi Hubbard Hamiltonian.

                K_3 = ( ⊗_{j even} (Z_{j,↑} ⊗ Z_{j,↓}) ) ⊗ ( ⊗_{j odd} (I_{j,↑} ⊗ I_{j,↓}) )
                    = I_{1,↑} ⊗ I_{1,↓} ⊗ Z_{2,↑} ⊗ Z_{2,↓} ⊗ I_{3,↑} ⊗ I_{3,↓} ⊗ ...
            """
            anc = QuantumRegister(1, 'ancilla')
            spin_up = QuantumRegister(self.nqubits, '↑')
            spin_down = QuantumRegister(self.nqubits, '↓')
            circuit = QuantumCircuit(spin_down, spin_up, anc)
            for j in range(self.nqubits):
                if j % 2 == 0:
                    circuit.cz(anc, spin_up[j], ctrl_state='0')
                    circuit.cz(anc, spin_down[j], ctrl_state='0')
            return circuit
        
        spin_up = QuantumRegister(self.nqubits, '↑')
        spin_down = QuantumRegister(self.nqubits, '↓')
        if not control_free:
            circuit = QuantumCircuit(spin_down, spin_up)
            for _ in range(n):
                circuit.compose(H_1(), spin_down[:] + spin_up[:], inplace=True)
                circuit.compose(H_2(), spin_down[:] + spin_up[:], inplace=True)
                circuit.compose(H_3(), spin_down[:] + spin_up[:], inplace=True)
                circuit.compose(H_2(), spin_down[:] + spin_up[:], inplace=True)
                circuit.compose(H_1(), spin_down[:] + spin_up[:], inplace=True)
        else:
            anc = QuantumRegister(1, 'ancilla')
            circuit = QuantumCircuit(spin_down, spin_up, anc)
            for _ in range(n):
                circuit.compose(K_1(), spin_down[:] + spin_up[:] + anc[:], inplace=True)
                circuit.compose(H_1(), spin_down[:] + spin_up[:], inplace=True)
                circuit.compose(K_1(), spin_down[:] + spin_up[:] + anc[:], inplace=True)
                circuit.compose(K_2(), spin_down[:] + spin_up[:] + anc[:], inplace=True)
                circuit.compose(H_2(), spin_down[:] + spin_up[:], inplace=True)
                circuit.compose(K_2(), spin_down[:] + spin_up[:] + anc[:], inplace=True)
                circuit.compose(K_3(), spin_down[:] + spin_up[:] + anc[:], inplace=True)
                circuit.compose(H_3(), spin_down[:] + spin_up[:], inplace=True)
                circuit.compose(K_3(), spin_down[:] + spin_up[:] + anc[:], inplace=True)
                circuit.compose(K_2(), spin_down[:] + spin_up[:] + anc[:], inplace=True)
                circuit.compose(H_2(), spin_down[:] + spin_up[:], inplace=True)
                circuit.compose(K_2(), spin_down[:] + spin_up[:] + anc[:], inplace=True)
                circuit.compose(K_1(), spin_down[:] + spin_up[:] + anc[:], inplace=True)
                circuit.compose(H_1(), spin_down[:] + spin_up[:], inplace=True)
                circuit.compose(K_1(), spin_down[:] + spin_up[:] + anc[:], inplace=True)
        return circuit