from qiskit import *
from qetu_sim.wmi_backend_grid import *
from qetu_sim.wmi_decompositions import *

def add_trotter_steps(trotter_circuit, spin_up, spin_down, aux=None, num_sites=4, u=1, t=-1, delta_t=1, n=1, ctrl_free=False, include_barriers=True):
    """
    Adds n Trotter steps to the given quantum circuit.

    Args:
        trotter_circuit - The Qiskit QuantumCircuit to which the Trotter steps are added.
        spin_up - The qubit register containing the up spins.
        spin_down - The qubit register containing the down spins.
        aux - The qubit register containing the auxilary qubit.
        num_sites - The number of sites of the Fermi Hubbard model.
        u - Coulomb repulsion energy.
        t - kinetic hopping energy.
        delta_t - time difference.
        n - The number of Trotter steps that are added.
        ctrl_free - Specifices wether the it is the controlled or uncontrolled version.
        include_barriers - Specifies wether or not to include barriers.
    """
    def K1():
        #############
        # K1
        ############
        trotter_circuit.cx(aux, spin_up[1], ctrl_state="0")
        trotter_circuit.cx(aux, spin_up[2], ctrl_state="0")
        trotter_circuit.cx(aux, spin_down[0], ctrl_state="0")
        trotter_circuit.cx(aux, spin_down[3], ctrl_state="0")
    
    def K2():
        #############
        # K2
        ############
        trotter_circuit.cz(aux, spin_up[1], ctrl_state="0")
        trotter_circuit.cz(aux, spin_up[2], ctrl_state="0")
        trotter_circuit.cz(aux, spin_down[0], ctrl_state="0")
        trotter_circuit.cz(aux, spin_down[3], ctrl_state="0")

    def H_1():
        #############
        # H_1
        ############
        lam = u*delta_t/(4*n)
        for i in range(num_sites):
            trotter_circuit.rzz(lam, spin_up[i], spin_down[i])

    def H_2():
        #############
        # H_2
        ############
        theta = 1*t*delta_t/n
        eta = 0
        hoppings = [(spin_up[0], spin_up[2]), (spin_up[1], spin_up[3]), (spin_down[0], spin_down[1]), (spin_down[2], spin_down[3])]
        for hop in hoppings:
            trotter_circuit.append(ParamISwap(theta, eta), hop)
    
    def H_3():
        #############
        # H_3
        ############

        swaps = [(spin_up[0], spin_down[0]), (spin_up[1], spin_down[1]), (spin_up[2], spin_down[2]), (spin_up[3], spin_down[3])]
        for swap in swaps:
            trotter_circuit.append(fSwap(), swap)

        #trotter_circuit.barrier(aux, spin_up, spin_down)

        theta = 2*t*delta_t/n
        eta = 0
        hoppings = [(spin_up[0], spin_up[2]), (spin_up[1], spin_up[3]), (spin_down[0], spin_down[1]), (spin_down[2], spin_down[3])]
        for hop in hoppings:
            trotter_circuit.append(ParamISwap(theta, eta), hop)

        #trotter_circuit.barrier(aux, spin_up, spin_down)

        # swap back
        swaps = [(spin_up[0], spin_down[0]), (spin_up[1], spin_down[1]), (spin_up[2], spin_down[2]), (spin_up[3], spin_down[3])]
        for swap in swaps:
            trotter_circuit.append(fSwap(), swap)

        #trotter_circuit.barrier(aux, spin_up, spin_down)

    for _ in range(n):           
        if ctrl_free:
            K1()
            if include_barriers:
                trotter_circuit.barrier(aux, spin_up, spin_down)
        elif include_barriers:
            trotter_circuit.barrier(spin_up, spin_down)

        H_1()

        if ctrl_free:
            if include_barriers:
                trotter_circuit.barrier(aux, spin_up, spin_down)
            K1()
            if include_barriers:
                trotter_circuit.barrier(aux, spin_up, spin_down)
            K2()
            if include_barriers:
                trotter_circuit.barrier(aux, spin_up, spin_down)
        elif include_barriers:
            trotter_circuit.barrier(spin_up, spin_down)

        H_2()

        if include_barriers:
            if ctrl_free:
                trotter_circuit.barrier(aux, spin_up, spin_down)
            else:
                trotter_circuit.barrier(spin_up, spin_down)
        
        H_3()
        
        if include_barriers:
            if ctrl_free:
                trotter_circuit.barrier(aux, spin_up, spin_down)
            else:
                trotter_circuit.barrier(spin_up, spin_down)

        H_2()
    
        if ctrl_free:
            if include_barriers:
                trotter_circuit.barrier(aux, spin_up, spin_down)
            K2()
            if include_barriers:
                trotter_circuit.barrier(aux, spin_up, spin_down)
            K1()
            if include_barriers:
                trotter_circuit.barrier(aux, spin_up, spin_down)
        elif include_barriers:
            trotter_circuit.barrier(spin_up, spin_down)

        H_1()
        
        if ctrl_free:
            if include_barriers:
                trotter_circuit.barrier(aux, spin_up, spin_down)
            K1()
            if include_barriers:
                trotter_circuit.barrier(aux, spin_up, spin_down)
        elif include_barriers:
            trotter_circuit.barrier(spin_up, spin_down)

