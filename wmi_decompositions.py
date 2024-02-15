import numpy as np
from qiskit import *

def add_zcz_wmi(circuit, control, target):
    """
    Add a zero-controlled z-gate decomposed into the gate set of the WMI
    """
    circuit.rz(np.pi, target)
    circuit.cp(np.pi, control, target)  

def add_zcx_wmi(circuit, control, target):
    """
    Add a zero-controlled x-gate decomposed into the gate set of the WMI
    """
    circuit.x(control)
    circuit.sx(target)
    circuit.rz(np.pi, target)
    circuit.sx(target)
    circuit.cp(np.pi, control, target)
    circuit.sx(target)
    circuit.rz(np.pi, target)
    circuit.sx(target)
    circuit.x(control)

def add_rzz_wmi(circuit, theta, control, target):
    """
    Add a RZZ gate decomposed into the gate set of the WMI
    """
    circuit.rz(theta, target)
    circuit.cp(2*theta, control, target)
    circuit.rz(theta, target)     