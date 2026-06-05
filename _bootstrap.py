"""
Environment shim. Import this FIRST, before creating any optimizer.

The base anaconda torch install in this environment is corrupted: it ships both
an old `torch/onnx/_internal/exporter.py` and a newer `exporter/` package, so
`import torch.onnx` raises ImportError (DiagnosticOptions missing). That import
is triggered lazily the first time `torch._dynamo` loads, which happens when you
construct an optimizer (Optimizer.add_param_group is wrapped by _disable_dynamo).

We never use ONNX or torch.compile here, so we replace the broken `torch.onnx`
(and its `operators` submodule, the only thing _dynamo actually imports) with a
minimal stub. This leaves the rest of torch fully functional.
"""

import importlib
import sys
import types

import torch


def _install_onnx_stub():
    try:
        importlib.import_module("torch.onnx.operators")  # if it imports, env is fine
        return
    except Exception:
        pass  # fall through and install the stub

    onnx = types.ModuleType("torch.onnx")
    ops = types.ModuleType("torch.onnx.operators")

    ops.shape_as_tensor = lambda x: torch._shape_as_tensor(x)
    ops.reshape_from_tensor_shape = lambda x, shape: torch._reshape_from_tensor(x, shape)
    onnx.operators = ops
    onnx.is_in_onnx_export = lambda: False

    sys.modules["torch.onnx"] = onnx
    sys.modules["torch.onnx.operators"] = ops
    torch.onnx = onnx


_install_onnx_stub()
