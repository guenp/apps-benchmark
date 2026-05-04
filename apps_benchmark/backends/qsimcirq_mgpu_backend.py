"""
qsimcirq multi-GPU backend for apps-benchmark.

Subclass of QsimcirqBackend that detects whether the installed qsimcirq
is Google's standard build or the NVIDIA cuQuantum Appliance fork, and
constructs QSimOptions accordingly. Everything else (run, scoring, etc.)
is inherited unchanged.
"""

from __future__ import annotations

import inspect
import logging

from apps_benchmark.backends.qsimcirq_backend import QsimcirqBackend
from apps_benchmark.errors import BackendConnectionError

logger = logging.getLogger(__name__)


class QsimcirqMGpuBackend(QsimcirqBackend):
    """qsimcirq backend with multi-GPU / cuQuantum Appliance support.

    Inspects the installed qsimcirq's QSimOptions signature at runtime to
    handle both the upstream Google build (use_gpu=) and the NVIDIA
    cuQuantum Appliance fork (disable_gpu=, gpu_mode accepts sequences).
    """

    def __init__(
        self,
        cpu_threads: int = 0,
        use_gpu: bool = False,
        gpu_mode: int = 0,
        max_fused_gate_size: int = 2,
        verbose: bool = False,
    ):
        try:
            import qsimcirq
        except ImportError as e:
            raise BackendConnectionError(
                "qsimcirq is not installed. Run: pip install qsimcirq"
            ) from e

        # Two qsimcirq variants in the wild:
        #   - Google's: QSimOptions(use_gpu=bool, gpu_mode=int, ...)
        #   - NVIDIA cuQuantum Appliance fork: QSimOptions(disable_gpu=bool,
        #     gpu_mode=Union[int, Sequence[int]], ...)
        # Detect which one we have by inspecting the signature.
        sig = inspect.signature(qsimcirq.QSimOptions)
        params = sig.parameters

        options_kwargs = dict(
            max_fused_gate_size=max_fused_gate_size,
            verbosity=1 if verbose else 0,
        )
        if cpu_threads > 0:
            options_kwargs["cpu_threads"] = cpu_threads

        if "use_gpu" in params:
            # Google variant
            options_kwargs["use_gpu"] = use_gpu
            if use_gpu:
                options_kwargs["gpu_mode"] = gpu_mode
        elif "disable_gpu" in params:
            # NVIDIA cuQuantum Appliance variant
            options_kwargs["disable_gpu"] = not use_gpu
            if use_gpu:
                # NVIDIA gpu_mode: 1 = single-GPU cuStateVec, int>=2 = multi-GPU,
                # tuple = explicit device list. 0 isn't valid here.
                options_kwargs["gpu_mode"] = gpu_mode if gpu_mode >= 1 else 1
        else:
            raise BackendConnectionError(
                f"Unrecognized qsimcirq.QSimOptions signature: {list(params)}"
            )

        self._qsim_options = qsimcirq.QSimOptions(**options_kwargs)
        self._simulator = qsimcirq.QSimSimulator(qsim_options=self._qsim_options)
        self._device = "gpu" if use_gpu else "cpu"
        self._cpu_threads = cpu_threads
        self._max_fused_gate_size = max_fused_gate_size
        logger.info(
            "Initialized qsimcirq backend (device=%s, threads=%s, fusion=%d, variant=%s)",
            self._device, cpu_threads or "auto", max_fused_gate_size,
            "google" if "use_gpu" in params else "nvidia",
        )

    def name(self) -> str:
        return f"qsimcirq_mgpu.{self._device}"
