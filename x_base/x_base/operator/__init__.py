from .attention import attention_manager
from .matmul import matmul_manager, WeightQuantLinearModule
from .rope import rope_manager
from . import rmsnorm  # noqa: F401 — triggers F.rms_norm auto-patch to torch_npu.npu_rms_norm
