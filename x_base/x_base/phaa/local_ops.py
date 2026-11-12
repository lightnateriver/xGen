"""Optional local boundaries; PHAA communication and buffer joins stay in core."""
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PHAALocalOps:
    # Kernels receive shape metadata explicitly. A None callback keeps core math.
    prepare_inputs: Callable | None = None
    allocate_outputs: Callable | None = None
    restore_output: Callable | None = None
    merge_outputs: Callable | None = None
    attention: Callable | None = None
