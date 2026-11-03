import logging
import os

import torch
import torch.nn.functional as F

from x_base.device import device_type
logger = logging.getLogger("attention")

try:
    import cann_ops

    CANN_OPS_AVAILABLE = True
except ImportError:
    logger.error("Not install CANN-OPS")
    CANN_OPS_AVAILABLE = False

try:
    from mindiesd import attention_forward, sparse_attention
    LA_AVAILABLE = True
except ImportError:
    logger.error("Not install mindiesd, not laserattention")
    LA_AVAILABLE = False

SAGE_SEQ_LEN_THRESHOLD = int(os.getenv("SAGE_SEQ_LEN_THRESHOLD", 10000))


def base_attention_infer(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    return F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_mask, dropout_p=dropout_p,
        is_causal=is_causal, scale=scale,
    )


def sage_attention_infer_quant_scale(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    # scale移到quant，性能下降
    B, N, Sq, D = query.shape
    B, N, Sk, D = key.shape
    if D == 128 and Sq > SAGE_SEQ_LEN_THRESHOLD and Sk > SAGE_SEQ_LEN_THRESHOLD:
        q_int8, deq_scale1 = cann_ops.block_quant(query, sm_scale=D ** -0.5, block_size=128, is_query=True)
        k_int8, deq_scalek = cann_ops.block_quant(key, sm_scale=None, block_size=1024, is_query=False)
        return cann_ops.npu_sage_attention(
            q_int8.contiguous(),
            k_int8.contiguous(),
            value.contiguous(),
            deq_scale1=deq_scale1,
            deq_scalek=deq_scalek,
            num_heads=N,
            input_layout="BNSD",
            scale_value=0.0,
            num_key_value_heads=N,
            next_tokens=65535,
            atten_mask=attn_mask)
    else:
        return base_attention_infer(query, key, value)


def sage_attention_infer(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    B, N, Sq, D = query.shape
    B, N, Sk, D = key.shape
    if D == 128 and Sq > SAGE_SEQ_LEN_THRESHOLD and Sk > SAGE_SEQ_LEN_THRESHOLD:
        q_int8, deq_scale1 = cann_ops.block_quant(query, block_size=128, is_query=True)
        k_int8, deq_scalek = cann_ops.block_quant(key, block_size=1024, is_query=False)
        return cann_ops.npu_sage_attention(
            q_int8.contiguous(),
            k_int8.contiguous(),
            value.contiguous(),
            deq_scale1=deq_scale1,
            deq_scalek=deq_scalek,
            num_heads=N,
            input_layout="BNSD",
            scale_value=D ** -0.5,
            num_key_value_heads=N,
            next_tokens=65535,
            atten_mask=attn_mask)
    else:
        return base_attention_infer(query, key, value)


def laser_attention_infer(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    B, N, Sq, D = query.shape
    B, N, Sk, D = key.shape
    scale = D ** -0.5
    MIN_SEQLEN_SELF = 4000
    if Sk >= MIN_SEQLEN_SELF:
        return attention_forward(query, key, value,
                                opt_mode="manual", op_type="ascend_laser_attention", layout="BNSD", head_first=True)
    else:
        return base_attention_infer(query, key, value)


class AttentionManager:
    def __init__(self):
        self.is_sage_attention = False
        self.cuda_sage_attention = None
        self.is_laser_attention = False
        self.is_sparse_attention = False

        # rainfusion
        self.is_rainfusion_attention = False
        self.rf_grid_size = None
        self.rf_skip_timesteps = 0
        self.rf_sparsity = 0.64
        self.rf_used_len = None  # meaningful seq len (without SP padding); None = use Sq
        self.t_idx = -1

    def enable_rainfusion_attention(self):
        self.is_rainfusion_attention = True

    def disable_rainfusion_attention(self):
        self.is_rainfusion_attention = False

    def enable_sage_attention(self):
        if device_type() == "cuda":
            from sageattention import sageattn
            self.cuda_sage_attention = sageattn
            self.is_sage_attention = True
        elif not CANN_OPS_AVAILABLE:
            logger.error("Not install CANN-OPS")
        else:
            self.is_sage_attention = True

    def disable_sage_attention(self):
        self.is_sage_attention = False

    def enable_laser_attention(self):
        if not LA_AVAILABLE:
            logger.error("Not install laserattention")
        else:
            self.is_laser_attention = True

    def disable_laser_attention(self):
        self.is_laser_attention = False

    def enable_sparse_attention(self):
        self.is_sparse_attention = True

    def disable_sparse_attention(self):
        self.is_sparse_attention = False
        
    def set_t_idx(self, t_idx):
        self.t_idx = t_idx

    def update_t_idx(self):
        self.t_idx += 1

    def set_rainfusion_attention(self, latent_shape: list, skip_timesteps: int = 0, sparsity: float = 0.64):
        """
        每次推理前需要根据输入初始化 Rainfusion 参数
        :param latent_shape: latents的THW网格大小
        """
        if not self.is_rainfusion_attention:
            logger.error("Before setting the parameters for rainfusion, rainfusion enable is required！")
        t, h, w = latent_shape[-3:]
        self.rf_grid_size = [t // 1, h // 2, w // 2]
        self.rf_skip_timesteps = skip_timesteps
        self.rf_sparsity = sparsity
        self.rf_used_len = None
        self.t_idx = -1

    def set_rainfusion_seq_len(self, seq_len: int):
        """Set the meaningful packed sequence length (without SP padding).

        MiniMax-H3 packs [text | refs | audio | video] into one sequence.
        Under SP, _split_sequence_for_sp adds zero-padding to make the
        length divisible by sp_size.  After all-to-all, Sq = seq_len + pad.
        The sparse kernel must only see the first seq_len tokens, so the
        caller (transformer forward) publishes seq_len here.
        """
        self.rf_used_len = int(seq_len)

    def rainfusion_attention_infer(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        """RainFusion (rf_v2) sparse attention via mindiesd.sparse_attention.

        MiniMax-H3 packs [text | refs | audio | video] into one sequence.
        The sparse kernel needs:
        - txt_len = prefix rows (everything before target video) kept dense
        - latent_shape = [t, h//2, w//2] grid for the video tail
        - input sliced to used_len (no SP padding) and padded back afterwards

        Falls back to dense attention when:
        - t_idx < skip_timesteps (early denoise steps)
        - Sq != Sk (cross-attention)
        - Sq < video_len (sequence shorter than video grid)
        """
        B, N, Sq, D = query.shape
        _, _, Sk, _ = key.shape
        attn_scale = scale if scale is not None else D ** -0.5

        # 前几步或交叉注意力走 laser
        if self.t_idx < self.rf_skip_timesteps or Sq != Sk:
            return laser_attention_infer(query, key, value, attn_mask=attn_mask,
                                         dropout_p=dropout_p, is_causal=is_causal, scale=scale)

        # rf_v2: video grid must fit within the sequence
        if self.rf_grid_size is not None:
            t, h, w = self.rf_grid_size
            video_len = t * h * w
            if Sq < video_len:
                return laser_attention_infer(query, key, value, attn_mask=attn_mask,
                                             dropout_p=dropout_p, is_causal=is_causal, scale=scale)

            # Meaningful sequence length (without SP padding).
            # rf_used_len is set by the transformer; falls back to Sq (no SP).
            used_len = self.rf_used_len if self.rf_used_len is not None else Sq
            if used_len < video_len:
                return laser_attention_infer(query, key, value, attn_mask=attn_mask,
                                             dropout_p=dropout_p, is_causal=is_causal, scale=scale)

            # prefix = everything before the target video (text + refs + audio)
            prefix_len = used_len - video_len
        else:
            return laser_attention_infer(query, key, value, attn_mask=attn_mask,
                                         dropout_p=dropout_p, is_causal=is_causal, scale=scale)

        orig_dtype = query.dtype
        if orig_dtype == torch.float32:
            query = query.to(torch.bfloat16)
            key = key.to(torch.bfloat16)
            value = value.to(torch.bfloat16)

        # Slice to used_len to strip SP padding; the sparse kernel must not
        # see zero-padding rows mixed into the video tail.
        need_pad_back = used_len < Sq
        if need_pad_back:
            query = query[:, :, :used_len, :]
            key = key[:, :, :used_len, :]
            value = value[:, :, :used_len, :]

        out = sparse_attention(
            query, key, value,
            scale=attn_scale,
            head_num=N,
            input_layout="BNSD",
            sparse_type="rf_v2",
            sparsity=self.rf_sparsity,
            txt_len=prefix_len,
            latent_shape_q=self.rf_grid_size,
            latent_shape_k=self.rf_grid_size,
            block_size=128,
        )

        if orig_dtype == torch.float32:
            out = out.to(orig_dtype)

        if need_pad_back:
            padded = torch.zeros(B, N, Sq, D, dtype=out.dtype, device=out.device)
            padded[:, :, :used_len, :] = out
            out = padded

        return out

    def set_ada_bsa_sparse_flash_attention(self, sparsity: float = 0.7):
       self.ada_sparsity = sparsity

    def ada_bsa_sparse_flash_attention_infer(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        B, N, Sq, D = query.shape
        B, N, Sk, D = key.shape
        scale = D ** -0.5
        if Sq == Sk:
            # Going to sparse attention
            head_num=None
            keep_sink=True
            eep_recent=True
            keep_sink=True
            keep_recent=True
            sparsity=self.ada_sparsity
            cdf_threshold=1.0
            sparse_size=128
            stride=8
            smask, sct = torch.ops.mindiesd.sparse_block_estimate(
                query,
                key,
                actual_seq_lengths=None,
                actual_seq_lengths_kv=None,
                input_layout="BNSD",
                stride=stride,
                sparse_size=sparse_size,
                num_heads=N,
                num_key_value_heads=N,
                scale_value=scale / stride,
                threshold=cdf_threshold,
                causal=is_causal,
                keep_sink=keep_sink,
                keep_recent=keep_recent,
                row_sparse=1.0 - sparsity
            )
            value = value.contiguous()
            out = torch.ops.mindiesd.block_sparse_attention(
                query,
                key,
                value,
                num_heads=N,
                num_key_value_heads=N,
                input_layout="BNSD",
                scale_value=scale,
                causal=is_causal,
                sparse_size=sparse_size,
                sparse_mask=smask,
                sparse_count_table=sct
            )
            return out
        else:
            return base_attention_infer(query, key, value)

    def attention(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        if self.is_sage_attention:
            if query.is_cuda:
                if (attn_mask is not None or dropout_p or query.shape[-1] not in (64, 128)
                        or query.dtype not in (torch.float16, torch.bfloat16)):
                    return base_attention_infer(query, key, value, attn_mask, dropout_p, is_causal, scale)
                return self.cuda_sage_attention(query, key, value, tensor_layout="HND",
                                                is_causal=is_causal, sm_scale=scale)
            return sage_attention_infer(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                                        is_causal=is_causal, scale=scale)
        elif self.is_laser_attention:
            return laser_attention_infer(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                                         is_causal=is_causal, scale=scale)
        elif self.is_rainfusion_attention:
            return self.rainfusion_attention_infer(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                                                   is_causal=is_causal, scale=scale)
        elif self.is_sparse_attention:
            return self.ada_bsa_sparse_flash_attention_infer(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                                         is_causal=is_causal, scale=scale)
        else:
            return base_attention_infer(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                                        is_causal=is_causal, scale=scale)


attention_manager = AttentionManager()
