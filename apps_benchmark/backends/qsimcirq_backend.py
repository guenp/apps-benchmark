"""
qsimcirq backend for apps-benchmark.

Wraps Google's qsim (via qsimcirq) as a synchronous AbstractBackend.
Qiskit circuits are translated to Cirq through OpenQASM 2.0; results
are returned as Qiskit-style {bitstring: count} histograms so they
plug directly into the existing benchmark-runner scoring path.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from qiskit import QuantumCircuit

from apps_benchmark.core.backend import (
    AbstractBackend,
    JobData,
    MeasurementBatch,
)
from apps_benchmark.errors import BackendConnectionError, BackendError

logger = logging.getLogger(__name__)


class QsimcirqBackend(AbstractBackend):
    """Local high-performance simulator backend powered by qsim.

    Parameters
    ----------
    cpu_threads:
        Worker threads for qsim's CPU path. 0 means qsim auto-detect.
    use_gpu:
        Run qsim on CUDA. Requires a qsimcirq build with GPU support.
    gpu_mode:
        qsim GPU mode flag (0 = single-state, 1 = multi-state batching).
    max_fused_gate_size:
        qsim's `f` parameter. 2 is the safe default; 4 is faster on
        dense circuits like QAOA cost layers but uses more memory.
    verbose:
        Forward qsim's internal timing prints.
    """

    def __init__(
        self,
        cpu_threads: int = 0,
        use_gpu: bool = False,
        gpu_mode: int = 1,
        max_fused_gate_size: int = 2,
        verbose: bool = False,
    ):
        try:
            import qsimcirq
        except ImportError as e:
            raise BackendConnectionError(
                "qsimcirq is not installed. Run: pip install qsimcirq"
            ) from e

        options_kwargs = dict(
            max_fused_gate_size=max_fused_gate_size,
            verbosity=1 if verbose else 0,
            use_gpu=use_gpu,
            gpu_mode=gpu_mode,
        )
        if cpu_threads > 0:
            options_kwargs["cpu_threads"] = cpu_threads

        self._qsim_options = qsimcirq.QSimOptions(**options_kwargs)
        self._simulator = qsimcirq.QSimSimulator(qsim_options=self._qsim_options)
        self._device = "gpu" if use_gpu else "cpu"
        self._cpu_threads = cpu_threads
        self._max_fused_gate_size = max_fused_gate_size
        logger.info(
            "Initialized qsimcirq backend (device=%s, threads=%s, fusion=%d)",
            self._device, cpu_threads or "auto", max_fused_gate_size,
        )

    # ------------------------------------------------------------------
    # AbstractBackend interface
    # ------------------------------------------------------------------
    def name(self) -> str:
        return f"qsimcirq.{self._device}"

    def run(
        self,
        circuits: list[QuantumCircuit],
        shots: int = 1000,
        job_name: Optional[str] = None,
    ) -> tuple[MeasurementBatch, str, JobData]:
        if not circuits:
            raise ValueError("No circuits provided to QsimcirqBackend.run()")

        from cirq.contrib.qasm_import import circuit_from_qasm
        from qiskit.qasm2 import dumps

        measurement_batch: MeasurementBatch = []
        for idx, qc in enumerate(circuits):
            try:
                qasm_str = dumps(qc)
                cirq_circuit = circuit_from_qasm(qasm_str)
            except Exception as e:
                raise BackendError(
                    f"Failed to translate circuit {idx} to Cirq via OpenQASM: {e}"
                ) from e

            try:
                result = self._simulator.run(cirq_circuit, repetitions=shots)
            except Exception as e:
                raise BackendError(f"qsim execution failed on circuit {idx}: {e}") from e

            counts = self._cirq_result_to_counts(result)
            measurement_batch.append(counts)

        job_id = str(uuid.uuid4())
        job_data = self.serialize_job_data(
            circuits, shots, job_name or "qsimcirq_job"
        )
        logger.info("qsim ran %d circuit(s), %d shots each (job %s)",
                    len(circuits), shots, job_id)
        return measurement_batch, job_id, job_data

    def validate_connection(self) -> bool:
        try:
            qc = QuantumCircuit(1, 1)
            qc.h(0)
            qc.measure(0, 0)
            self.run([qc], shots=8, job_name="qsimcirq_self_test")
            return True
        except Exception as e:
            logger.warning("qsimcirq self-test failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _cirq_result_to_counts(result) -> dict[str, int]:
        """Convert a cirq.Result to a Qiskit-style {bitstring: count} dict.

        Qiskit's bitstring convention is little-endian within a register
        (qubit 0 == rightmost character). We mirror that here so scoring
        functions written against Aer also work against qsim.
        """
        import numpy as np

        if not result.measurements:
            return {}

        keys = sorted(result.measurements.keys())
        rows = None
        for k in keys:
            arr = result.measurements[k]
            rows = arr if rows is None else np.hstack([rows, arr])
        if rows is None:
            return {}

        counts: dict[str, int] = {}
        for row in rows:
            bitstr = "".join(str(int(b)) for b in row[::-1])
            counts[bitstr] = counts.get(bitstr, 0) + 1
        return counts
