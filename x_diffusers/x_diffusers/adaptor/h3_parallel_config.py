"""H3 opt-in TP + pure-Ulysses configuration, independent of torch/CUDA imports."""
from dataclasses import dataclass
import os


def add_arguments(parser):
    group = parser.add_argument_group('Experimental CUDA H3 DiT parallelism')
    group.add_argument('--h3_parallel_mode', choices=('sp', 'tp_cp', 'tp_cp_shared', 'tp_sp'), default='sp')
    group.add_argument('--h3_dit_tp', type=int, default=1)
    group.add_argument('--h3_tp_comm', choices=('bf16', 'fp8'), default='bf16')
    group.add_argument('--h3_cp_comm', choices=('auto', 'bf16', 'fp8'), default='auto')
    group.add_argument('--h3_tp_comm_chunk_elements', type=int, default=262144,
                       help='FP8 TP packet elements (128..4194304, multiple of 128); larger packets use more scratch memory. Default preserves the initial implementation.')
    group.add_argument('--h3_parallel_warmup_mib', type=int, default=0,
                       help='Experimental: initialize DiT collectives before weights move to CUDA; CP buffer MiB per rank (0 disables, maximum 1024). Initialization only, requires tp_cp.')
    return parser


def validate_resident_args(initial, requested):
    """A resident DiT cannot silently change its groups, weights or wire dtype."""
    if initial is None:
        return
    defaults = dict(h3_parallel_mode='sp', h3_dit_tp=1, sp=1, h3_tp_comm='bf16',
                    h3_cp_comm='auto', h3_cuda_precision='auto', model=None,
                    pretrained_model_name_or_path=None, h3_tp_comm_chunk_elements=262144,
                    h3_parallel_warmup_mib=0)
    changed = any(getattr(initial,key,default) != getattr(requested,key,default)
                  for key,default in defaults.items())
    changed |= ((getattr(initial,'task_type',None) == 'ref2va') !=
                (getattr(requested,'task_type',None) == 'ref2va'))
    if changed:
        raise ValueError('Restart the manager to change H3 TP/CP layout, transport, precision or workflow')


@dataclass(frozen=True)
class H3ParallelConfig:
    mode: str = 'sp'
    tp: int = 1
    cp: int = 1
    tp_comm: str = 'bf16'
    cp_comm: str = 'auto'
    tp_chunk_elements: int = 262144
    warmup_mib: int = 0

    @classmethod
    def from_args(cls, args, backend, world_size):
        mode = getattr(args, 'h3_parallel_mode', 'sp')
        tp = getattr(args, 'h3_dit_tp', 1)
        cp = getattr(args, 'sp', 1)
        tp_comm = getattr(args, 'h3_tp_comm', 'bf16')
        cp_comm = getattr(args, 'h3_cp_comm', 'auto')
        chunk_elements = getattr(args, 'h3_tp_comm_chunk_elements', 262144)
        warmup_mib = getattr(args, 'h3_parallel_warmup_mib', 0)
        if type(warmup_mib) is not int or not 0 <= warmup_mib <= 1024:
            raise ValueError('h3_parallel_warmup_mib must be an integer in [0, 1024]')
        if type(chunk_elements) is not int or not 128 <= chunk_elements <= 4194304 or chunk_elements % 128:
            raise ValueError('h3_tp_comm_chunk_elements must be a multiple of 128 in [128, 4194304]')
        if mode not in ('sp', 'tp_cp', 'tp_cp_shared', 'tp_sp') or tp_comm not in ('bf16', 'fp8') or cp_comm not in ('auto', 'bf16', 'fp8'):
            raise ValueError('Invalid H3 parallel mode or communication dtype')
        if mode == 'sp':
            if (tp, tp_comm, cp_comm, chunk_elements, warmup_mib) != (1, 'bf16', 'auto', 262144, 0):
                raise ValueError('H3 DiT TP/communication options require --h3_parallel_mode tp_cp')
            return cls()
        if backend != 'cuda' or getattr(args, 'model', None) != 'MiniMax-H3':
            raise ValueError('tp_cp requires CUDA MiniMax-H3')
        if mode == 'tp_cp_shared':
            if type(tp) is not int or type(cp) is not int or (tp, cp, world_size) != (8, 8, 8):
                raise ValueError('tp_cp_shared requires TP8 and CP8 on the same eight ranks')
        elif mode == 'tp_sp':
            if type(tp) is not int or type(cp) is not int or (tp, cp, world_size) != (2, 8, 8):
                raise ValueError('tp_sp requires TP2 projections and SP8 attention on eight ranks')
        elif type(tp) is not int or type(cp) is not int or tp not in (1, 2, 4) or cp < 1 or tp * cp != world_size:
            raise ValueError('tp_cp requires DiT TP=1, 2 or 4 and TP * --sp = world size (DP=CFG=1)')
        if getattr(args, 'ring_degree', None) not in (None, 1) or getattr(args, 'ulysses_degree', None) not in (None, cp):
            raise ValueError('tp_cp supports pure Ulysses only; Ring must be 1')
        if getattr(args, 'task_type', None) not in ('t2va', 'fl2va', 'ref2va'):
            raise ValueError('tp_cp supports T2VA, FL2VA and Ref2VA')
        conflicts = ('h3_refiner', 'joint', 'fsdp', 'inf_vram_blocks_num', 'h3_nvfp4_checkpoint',
                     'h3_nvfp4_resident', 'h3_nvfp4_compile', 'h3_turbo8', 'h3_turbo_sla_lora',
                     'adaln_cache_file', 'transformer_path', 'lora_path_list', 'cache_dit')
        if any(getattr(args, key, None) for key in conflicts):
            raise ValueError('tp_cp does not support Refiner, packed weights, LoRA or external offload/cache recipes')
        if getattr(args, 'h3_cuda_precision', 'none') not in ('fp8-bf16', 'fp8-sage'):
            raise ValueError('tp_cp requires an explicit fp8-bf16 or fp8-sage precision profile')
        if cp_comm == 'auto':
            phaa = getattr(args, 'h3_cuda_phaa_fp8', None)
            phaa = os.environ.get('XGEN_PHAA_FP8') == '1' if phaa is None else phaa
            cp_comm = tp_comm if mode == 'tp_cp_shared' else ('fp8' if phaa else 'bf16')
        if mode == 'tp_cp_shared' and cp_comm != tp_comm:
            raise ValueError('Shared TP/CP fuses layout boundaries and requires matching TP/CP wire dtypes')
        shared_ignored = ('XGEN_PHAA_GATHER_PRIORITY', 'XGEN_PHAA_DUAL_COMM', 'XGEN_PHAA_DEVICE_JOIN',
                          'XGEN_PHAA_ASYNC_EXCHANGE', 'XGEN_PHAA_DIRECT_RETURN', 'XGEN_PHAA_FP8_ROW_TILING',
                          'XGEN_PHAA_FUSED_RETURN_PACK', 'XGEN_PHAA_DIRECT_GATHER', 'XGEN_PHAA_FP8_MIXED_STRIDES')
        if mode == 'tp_cp_shared' and (any((getattr(args, 'h3_cuda_phaa_direct_pack', None)
                                          if key == 'XGEN_PHAA_FP8_MIXED_STRIDES' and getattr(args, 'h3_cuda_phaa_direct_pack', None) is not None
                                          else os.environ.get(key) == '1') for key in shared_ignored)
                                      or os.environ.get('XGEN_PHAA_CHUNK_ORDER', 'natural') != 'natural'):
            raise ValueError('Shared TP/CP fuses QKV redistribution; standalone PHAA transport options are unsupported')
        if cp == 1 and cp_comm == 'fp8':
            raise ValueError('FP8 CP transport requires --sp > 1')
        if tp == 1 and tp_comm == 'fp8':
            raise ValueError('FP8 TP transport requires --h3_dit_tp 2 or 4')
        if chunk_elements != 262144 and (tp == 1 or tp_comm != 'fp8'):
            raise ValueError('Non-default TP packet size requires multi-rank FP8 TP transport')
        # Keep advanced world-spanning PHAA experimental streams out of subgroups.
        if any(os.environ.get(key) == '1' for key in ('XGEN_PHAA_DUAL_COMM', 'XGEN_PHAA_ASYNC_EXCHANGE',
                                                      'XGEN_PHAA_DIRECT_RETURN', 'XGEN_PHAA_DIRECT_GATHER')):
            raise ValueError('tp_cp currently supports the ordinary PHAA transport only')
        return cls(mode, tp, cp, tp_comm, cp_comm, chunk_elements, warmup_mib)
