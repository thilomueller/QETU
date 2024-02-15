from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.visualization import plot_histogram
from qiskit.providers.fake_provider import FakeBackend
from qiskit.providers.models import BackendProperties
from qiskit.providers.models.backendproperties import Nduv, Gate
from qiskit.providers.models.backendconfiguration import QasmBackendConfiguration
#from .library.standard_gates.iswap import iSwapGate
from qiskit.circuit.library import iSwapGate
from datetime import datetime
from qiskit.transpiler import CouplingMap, InstructionProperties
from qiskit.providers import QubitProperties
from qiskit.circuit.library import XXPlusYYGate
import numpy as np

class FakeWMI(FakeBackend):
    """A fake backend that simulates the 6-qubit grid-like WMI device."""

    def __init__(self):
        self.backend_name = "fake_wmi"
        self.ver = "0.0.0"
        self.now = datetime.now()
        self.n_qubits = 9
        # Missing the sy gate because it's not native to Qiskit
        # However, this shouldn't introduce a large overhead
        self.gates_1q = ["sx", "x", "y", "rz"]
        self.gates_2q = ["iswap", "cp", "xx_plus_yy"]
        self.basis_gates = self.gates_1q + self.gates_2q
        self.t1 = (36.0, "µs")
        self.t2 = (8.47, "µs")
        self.duration_1q = (40.0, "ns")
        self.duration_2q = (150.0, "ns")
        self.duration_readout = (1500.0, "ns")
        self.fidelity_1q = 0.998
        self.fidelity_2q = 0.94
        self.fidelity_readout = 0.85
        self.frequency = (5.0, "GHz")

        self.coupling_map = self._generate_cmap()
        configuration = self._build_conf()
        self._configuration = configuration
        self._properties = self._build_probs()
        self.version = 0 # must be an integer for some reason
        super().__init__(configuration)

    def properties(self):
        return self._properties

    def _generate_cmap(self) -> list[list[int]]:
        """Build a closed grid-like coupling map."""
        coupling_map = [(0, 1), (1, 2), (3, 4),
                        (4, 5), (6, 7), (7, 8),
                        (0, 3), (1, 4), (2, 5),
                        (3, 6), (4, 7), (5, 8)]
        return coupling_map

    def _build_probs(self) -> BackendProperties:
        """Build the backend properties object."""
        qubits = []
        for i in range(self.n_qubits):
            qubits.append(
                [
                    Nduv(date=self.now, name="T1", unit=self.t1[1], value=self.t1[0]),
                    Nduv(date=self.now, name="T2", unit=self.t2[1], value=self.t2[0]),
                    Nduv(
                        date=self.now,
                        name="frequency",
                        unit=self.frequency[1],
                        value=self.frequency[0],
                    ),
                    Nduv(
                        date=self.now,
                        name="readout_error",
                        unit="",
                        value=1 - self.fidelity_readout,
                    ),
                    Nduv(
                        date=self.now,
                        name="readout_length",
                        unit=self.duration_readout[1],
                        value=self.duration_readout[0],
                    ),
                ]
            )

        single_gates = []
        for i in range(self.n_qubits):
            for gate in self.gates_1q:
                err = 1 - self.fidelity_1q if gate != "rz" else 0.0
                dur = self.duration_1q[0] if gate != "rz" else 0.0
                unit = self.duration_1q[1]
                single_gates.append(
                    Gate(
                        gate=gate,
                        name=f"{gate}_{i}",
                        qubits=[i],
                        parameters=[
                            Nduv(date=self.now, name="gate_error", unit="", value=err),
                            Nduv(
                                date=self.now,
                                name="gate_length",
                                unit=unit,
                                value=dur,
                            ),
                        ],
                    )
                )

        two_gates = []
        for i, j in self.coupling_map:
            for gate in self.gates_2q:
                err = 1 - self.fidelity_2q
                dur = self.duration_2q[0]
                unit = self.duration_2q[1]
                two_gates.append(
                    Gate(
                        gate=gate,
                        name=f"{gate}{i}_{j}",
                        qubits=[i, j],
                        parameters=[
                            Nduv(date=self.now, name="gate_error", unit="", value=err),
                            Nduv(
                                date=self.now, name="gate_length", unit=unit, value=dur
                            ),
                        ],
                    )
                )

        return BackendProperties(
            backend_name=self.backend_name,
            backend_version=self.ver,
            last_update_date=self.now,
            qubits=qubits,
            gates=single_gates + two_gates,
            general=[],
        )

    def _build_conf(self) -> QasmBackendConfiguration:
        """Build the backend configuration object."""
        return QasmBackendConfiguration(
            backend_name=self.backend_name,
            backend_version=self.ver,
            n_qubits=self.n_qubits,
            basis_gates=self.basis_gates,
            gates=[],
            local=True,
            simulator=True,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=10000,
            coupling_map=self.coupling_map,
        )

def add_iswap_labels(qc : QuantumCircuit) -> QuantumCircuit:
    """
        This function takes a quantum circuit as an imput and replaces every iSwap gate with a custom labeled ISwap gate.
        This is necessary so that the noise model can be applied to that gate.
    """
    import qiskit.quantum_info as qi
    # iSWAP matrix operator
    iswap_op = qi.Operator([[1, 0, 0, 0],
                            [0, 0, 1j, 0],
                            [0, 1j, 0, 0],
                            [0, 0, 0, 1]])
    qc1 = qc.copy()
    for (i, data) in enumerate(qc1.data):
        if data[0].name == "iswap":
            # get the list of indices the quantum gate is operating on
            list_of_indices = [q.index for q in data[1]]
            # create a new labeled iswap gate
            iswap_gate = QuantumCircuit(qc1.num_qubits)
            iswap_gate.unitary(iswap_op, list_of_indices, label='iswap')
            # remove the unlabel gate and replace it with a labeled one
            qc1.data.pop(i)
            qc1.data.insert(i, iswap_gate.data[0])
    return qc1

def add_xx_plus_yy_labels(qc : QuantumCircuit) -> QuantumCircuit:
    """
        This function takes a quantum circuit as an imput and replaces every XXPlusYYGate gate with a custom labeled gate.
        This is necessary so that the noise model can be applied to that gate.
    """
    import qiskit.quantum_info as qi
    qc1 = qc.copy()
    for (i, data) in enumerate(qc1.data):
        if data[0].name == "xx_plus_yy":
            # get parameters
            theta, beta = data[0].params
            # construct operator
            xx_plus_yy_op = qi.Operator([[1, 0,                                   0,                                    0],
                                         [0, np.cos(theta/2),                     -1j*np.sin(theta/2)*np.exp(-1j*beta), 0],
                                         [0, -1j*np.sin(theta/2)*np.exp(1j*beta), np.cos(theta/2),                      0],
                                         [0, 0,                                   0,                                    1]])
            # get the list of indices the quantum gate is operating on
            list_of_indices = [q.index for q in data[1]]
            # create a new labeled iswap gate
            xx_plus_yy_gate = QuantumCircuit(qc1.num_qubits)
            xx_plus_yy_gate.unitary(xx_plus_yy_op, list_of_indices, label='xx_plus_yy')
            # remove the unlabel gate and replace it with a labeled one
            qc1.data.pop(i)
            qc1.data.insert(i, xx_plus_yy_gate.data[0])
    return qc1