from qiskit import *

from qiskit.circuit.library import XXPlusYYGate, iSwapGate
from qiskit.transpiler import Layout
from qiskit.quantum_info.operators import Operator


from qiskit.providers import BackendV2 as Backend
from qiskit.transpiler import Target
from qiskit.providers import Options
from qiskit.circuit import Parameter, Measure
from qiskit.circuit.library import PhaseGate, SXGate, XGate, YGate, RZGate, CPhaseGate, IGate, RZZGate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.transpiler import InstructionProperties
from qiskit.quantum_info.operators import Operator
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary


import numpy as np
 
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
 
class SYGate(Gate):
    def __init__(self, label=None):
        super().__init__("sy", 1, [], label=label)
 
    def _define(self):
        qc = QuantumCircuit(1)
        qc.ry(np.pi / 2, 0)
        self.definition = qc

class ParamISwap(Gate):
    def __init__(self, theta: ParameterValueType, eta: ParameterValueType, label=None):
        super().__init__("pswap", 2, [theta, eta], label=label)
 
    def _define(self):
        qc = QuantumCircuit(2)
        theta = float(self.params[0])
        eta = float(self.params[1])
        c = np.cos(theta / 2)
        s1 = 1j * np.exp(1j*eta) * np.sin(theta / 2)
        s2 = 1j * np.exp(-1j*eta) * np.sin(theta / 2)
        pswap = Operator([
            [1, 0,  0, 0],
            [0, c, s1, 0],
            [0, s2, c, 0],
            [0, 0,  0, 1]
        ])
        qc.unitary(pswap, [0, 1], label="pswap")
        self.definition = qc

class fSwap(Gate):
    def __init__(self, label=None):
        super().__init__("fSWAP", 2, [], label=label)
 
    def _define(self):
        qc = QuantumCircuit(2)
        fswap = Operator([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1]
        ])
        qc.unitary(fswap, [0, 1], label="fSWAP")
        self.definition = qc
 
 
class WMIBackendGrid(Backend):
    """A fake backend that simulates the 9-qubit grid-like WMI device."""
 
    def __init__(self):
        super().__init__()
 
        # Create Target
        self._target = Target("Target for WMI Backend")
        # Instead of None for this and below instructions you can define
        # a qiskit.transpiler.InstructionProperties object to define properties
        # for an instruction.

        self.name = "WMI 9 qubit Backend"

        coupling_map = [
            (0, 1), (1, 2), (3, 4),
            (4, 5), (6, 7), (7, 8),
            (0, 3), (1, 4), (2, 5),
            (3, 6), (4, 7), (5, 8),
            (1, 0), (2, 1), (4, 3),
            (5, 4), (7, 6), (8, 7),
            (3, 0), (4, 1), (5, 2),
            (6, 3), (7, 4), (8, 5)
        ]

        theta = Parameter("ϴ")
        lam = Parameter("λ")
        beta = Parameter("β")
        eta = Parameter("η")

        # SX
        sx_props = {(qubit,): None for qubit in range(9)}
        self._target.add_instruction(SXGate(), sx_props)
        # SY
        sy_props = {(qubit,): None for qubit in range(9)}
        self._target.add_instruction(SYGate(), sy_props)
        # X
        x_props = {(qubit,): None for qubit in range(9)}
        self._target.add_instruction(XGate(), x_props)#
        # Y
        y_props = {(qubit,): None for qubit in range(9)}
        self._target.add_instruction(YGate(), y_props)
        # RZ
        rz_props = {(qubit,): None for qubit in range(9)}
        self._target.add_instruction(RZGate(theta), rz_props)
        # CPhase
        cp_props = {(qubit1, qubit2): None for (qubit1, qubit2) in coupling_map}
        self._target.add_instruction(CPhaseGate(lam), cp_props)
        # ParamISwap
        param_iswap_props = {(qubit1, qubit2): None for (qubit1, qubit2) in coupling_map}
        self._target.add_instruction(ParamISwap(theta, eta), param_iswap_props)
        # measurement
        meas_props = {(qubit,): None for qubit in range(9)}
        self._target.add_instruction(Measure(), meas_props)
        # Identity
        id_props = {(qubit,): None for qubit in range(9)}
        self._target.add_instruction(IGate(), id_props)


        # iSWAP => U_{iSWAP}(pi)
        q = QuantumRegister(2, "q")
        def_iswap = QuantumCircuit(q)
        def_iswap.append(ParamISwap(theta=np.pi, eta=0), [q[0], q[1]], [])
        SessionEquivalenceLibrary.add_equivalence(iSwapGate(), def_iswap)

        # fSWAP => U_{iSWAP}(pi,0) RZ(-pi/2) RZ(-pi/2)
        q = QuantumRegister(2, "q")
        def_fswap = QuantumCircuit(q)
        def_fswap.append(ParamISwap(theta=np.pi, eta=0), [q[0], q[1]], [])
        def_fswap.rz(-np.pi/2, 0)
        def_fswap.rz(-np.pi/2, 1)
        # remove global phase
        def_fswap.append(SYGate(), [0])
        def_fswap.rz(-np.pi, 0)
        def_fswap.x(0)
        def_fswap.append(SYGate(), [0])
        SessionEquivalenceLibrary.add_equivalence(fSwap(), def_fswap)

        # RZZ(0) => RZ1(0) CP(20) RZ1(0)
        q = QuantumRegister(2, "q")
        def_rzz = QuantumCircuit(q)
        def_rzz.append(RZGate(theta), [q[1]], [])
        def_rzz.append(CPhaseGate(2*theta), [q[0], q[1]], [])
        def_rzz.append(RZGate(theta), [q[1]], [])
        SessionEquivalenceLibrary.add_equivalence(RZZGate(theta), def_rzz)
 
        # Set option validators
        self.options.set_validator("shots", (1, 4096))
        self.options.set_validator("memory", bool)
 
    @property
    def target(self):
        return self._target
 
    @property
    def max_circuits(self):
        return 1024
 
    @classmethod
    def _default_options(cls):
        return Options(shots=1024, memory=False)
 
    def run(circuits, **kwargs):
        # serialize circuits submit to backend and create a job
        for kwarg in kwargs:
            if not hasattr(kwarg, self.options):
                warnings.warn(
                    "Option %s is not used by this backend" % kwarg,
                    UserWarning, stacklevel=2)
        options = {
            'shots': kwargs.get('shots', self.options.shots),
            'memory': kwargs.get('memory', self.options.shots),
        }
        job_json = convert_to_wire_format(circuit, options)
        job_handle = submit_to_backend(job_jsonb)
        return MyJob(self. job_handle, job_json, circuit)