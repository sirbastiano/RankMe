import time
import torch
import torch.nn as nn
from rankme.base import StatelessMetric
from thop import profile
from typing import Any, Dict, Optional, Tuple


def _get_param_device(model: nn.Module) -> torch.device:
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


class ParamCount(StatelessMetric):
    """Return number of trainable parameters of a model."""

    def forward(self, model: nn.Module, include_non_trainable: bool = False) -> int:
        """
        Args:
            model: PyTorch model
            include_non_trainable: if True, include parameters with requires_grad=False
        Returns:
            int: total number of parameters
        """
        if include_non_trainable:
            total = sum(p.numel() for p in model.parameters())
        else:
            total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # return plain int per user's request
        return int(total)


class ModelSizeMB(StatelessMetric):
    """Return model size in megabytes (MB) computed from parameters (and optionally buffers)."""

    def forward(self, model: nn.Module, include_buffers: bool = False) -> float:
        """
        Args:
            model: PyTorch model
            include_buffers: whether to include buffers (e.g., running_mean/var) in size
        Returns:
            float: model size in megabytes
        """
        total_bytes = 0
        for p in model.parameters():
            total_bytes += p.numel() * p.element_size()
        if include_buffers:
            for b in model.buffers():
                total_bytes += b.numel() * b.element_size()
        mb = total_bytes / (1024 ** 2)
        # return float MB
        return float(mb)


class Flops(StatelessMetric):
    """
    Estimate FLOPs for a single forward pass using module forward hooks.
    """

    def forward(
        self,
        model: nn.Module,
        example_input: Optional[torch.Tensor] = None,
        input_size: Optional[Tuple[int, ...]] = None,
        device: Optional[torch.device] = None,
        custom_ops: Optional[Dict[Any, Any]] = None,
    ) -> float:
        """
        Args:
            model: PyTorch model
            example_input: a real example tensor to run through the model (preferred)
            input_size: if example_input is None, shape tuple used to create a random input, e.g. (1,3,224,224)
            device: device to run the sample forward on (defaults to model device)
        Returns:
            float: estimated FLOPs (or MACs depending on thop version)
        """
        # prepare input
        if example_input is None:
            if input_size is None:
                raise ValueError("Either example_input or input_size must be provided to estimate FLOPs.")
            example_input = torch.rand(*input_size)

        if device is None:
            device = _get_param_device(model)
        example_input = example_input.to(device)

        # Use thop exclusively per user's request. If thop is not installed, raise informative error.
        try:
            from thop import profile  # type: ignore
        except Exception as e:
            raise RuntimeError("thop is required for FLOPs computation; install it with `pip install thop`. ") from e

        # prepare input for thop
        if example_input is None:
            if input_size is None:
                raise ValueError("Either example_input or input_size must be provided to estimate FLOPs.")
            example_input = torch.rand(*input_size)

        if device is None:
            device = _get_param_device(model)
        example_input = example_input.to(device)

        # thop.profile returns MACs in many versions; treat returned value as flops-like numeric
        out = profile(model, inputs=(example_input,), custom_ops=custom_ops or {}, verbose=False)
        if isinstance(out, tuple) and len(out) >= 1:
            macs = out[0]
        else:
            macs = out

        # return plain float per user's request
        return float(macs)


class PrecisionBits(StatelessMetric):
    """Estimate average number of bits used per parameter (weighted by element count).

    Returns the weighted average number of bits across all parameters. Also exposes
    an utility to get per-dtype summary via `get_dtype_summary`.
    """

    _dtype_bits_map = {
        torch.float32: 32,
        torch.float64: 64,
        torch.float16: 16,
        torch.bfloat16: 16,
        torch.int8: 8,
        torch.uint8: 8,
        torch.int32: 32,
        torch.int64: 64,
        torch.bool: 1,
    }

    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        Args:
            model: PyTorch model
        Returns:
            torch.Tensor: scalar float tensor with average bits per parameter
        """
        total_elements = 0
        total_bits = 0.0
        for p in model.parameters():
            n = p.numel()
            total_elements += n
            bits = self._dtype_bits_map.get(p.dtype, None)
            if bits is None:
                # fallback to element_size() * 8 bits
                bits = p.element_size() * 8
            total_bits += n * bits

        if total_elements == 0:
            return 0.0
        avg_bits = float(total_bits) / float(total_elements)
        # return plain float
        return float(avg_bits)

    def get_dtype_summary(self, model: nn.Module) -> Dict[str, int]:
        """Return a dict {dtype_name: number_of_elements} summarizing dtypes."""
        summary: Dict[str, int] = {}
        for p in model.parameters():
            name = str(p.dtype)
            summary[name] = summary.get(name, 0) + p.numel()
        return summary


class InferenceTime(StatelessMetric):
    """Measure inference time (average per forward) on CPU and GPU (if available).

    The forward returns a 1D tensor with two values: [cpu_time_s, gpu_time_s]
    If GPU is not available, the second value will be float('nan').
    """

    def _measure(
        self,
        model: nn.Module,
        inp: torch.Tensor,
        device: torch.device,
        runs: int,
        warmup: int,
    ) -> float:
        # move model and data to device, preserve original model device
        orig_device = _get_param_device(model)
        model.to(device)
        inp = inp.to(device)
        model.eval()
        with torch.no_grad():
            # warmup
            for _ in range(warmup):
                _ = model(inp)
                if device.type == "cuda":
                    torch.cuda.synchronize()
            # timed runs
            start = time.perf_counter()
            for _ in range(runs):
                _ = model(inp)
                if device.type == "cuda":
                    torch.cuda.synchronize()
            end = time.perf_counter()
        # move model back
        try:
            model.to(orig_device)
        except Exception:
            pass
        avg = (end - start) / runs
        return avg

    def forward(
        self,
        model: nn.Module,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        channels: Optional[int] = None,
        input_size: Optional[Tuple[int, ...]] = None,
        runs: int = 50,
        warmup: int = 10,
    ) -> Tuple[float, float]:
        """
        Args:
            model: PyTorch model
            rows, cols, channels: spatial dimensions and channels to synthesize a gaussian input
            input_size: alternatively provide (B,C,H,W) or (C,H,W) to infer rows/cols/channels
            runs: number of timed runs
            warmup: number of warmup runs
        Returns:
            tuple: (cpu_avg_s, gpu_avg_s_or_nan)
        """
        # Build gaussian random input with shape (1, C, H, W).
        # Accept either explicit rows, cols, channels or an input_size tuple.
        if input_size is not None:
            if len(input_size) == 4:
                b, ch, h, w = input_size
                if b != 1:
                    # we always use batch size 1 for timing
                    # keep quiet but enforce batch=1
                    pass
                channels = ch
                rows = h
                cols = w
            elif len(input_size) == 3:
                ch, h, w = input_size
                channels = ch
                rows = h
                cols = w
            else:
                raise ValueError("input_size must be (B,C,H,W) or (C,H,W) when provided")

        # require rows, cols, channels to be set now
        if rows is None or cols is None or channels is None:
            raise ValueError("rows, cols and channels must be provided either via arguments or input_size")

        # create gaussian input with mean=0.0 and std=1.0 with shape (1, C, H, W)
        inp = torch.randn(1, channels, rows, cols)

        # CPU measurement
        cpu_dev = torch.device("cpu")
        try:
            cpu_time = float(self._measure(model, inp, cpu_dev, runs, warmup))
        except Exception:
            cpu_time = float("nan")

        # GPU measurement if available
        if torch.cuda.is_available():
            try:
                gpu_dev = torch.device("cuda")
                gpu_time = float(self._measure(model, inp, gpu_dev, runs, warmup))
            except Exception:
                gpu_time = float("nan")
        else:
            gpu_time = float("nan")

        return (cpu_time, gpu_time)
