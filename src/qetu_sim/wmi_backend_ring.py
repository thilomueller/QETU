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

class WMIBackendRing(FakeBackend):
    """A fake backend that simulates the 6-qubit ring-like WMI device."""

    def __init__(self):
        self.backend_name = "fake_wmi"
        self.ver = "0.0.0"
        self.now = datetime.now()
        self.n_qubits = 6
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
        coupling_map = [(0, 1), (1, 2), (0, 3),
                        (3, 4), (4, 5), (2, 5),
                        
                        (1, 0), (2, 1), (3, 0),
                        (4, 3), (5, 4), (5, 2)
                        ]
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