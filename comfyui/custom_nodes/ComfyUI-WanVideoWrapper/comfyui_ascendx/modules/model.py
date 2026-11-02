import torch
import torch.nn as nn
import torch.distributed as dist

from ascendx_video.base.sequence_parallelism.comm import all_to_all_before_attn, all_to_all_after_attn
from ascendx_video.base.sequence_parallelism.process_group_mesh import ProcessGroupMesh
from .attention import attention
from comfy.ldm.flux.math import apply_rope as apply_rope_comfy
from ascendx_video.base import get_pad
import os

def is_phaa_enabled():
    env_val  = os.getenv("PHAA_ENABLE") # 通算并行开关
    if env_val is None:
        return False
    clean_val = env_val.strip().lower()
    return clean_val in ["1", "true", "on"]

def get_sp_group():
    if not dist.is_initialized():
        return None, 1
    sp_size = dist.get_world_size()
    if sp_size > 1:
        process_group_mesh = ProcessGroupMesh(sp_size)
        return process_group_mesh.get_group_along_axis(0), sp_size
    return None, 1

def get_chunk(tensor, chunk=0, dim=0):
    world_size = dist.get_world_size()
    return torch.narrow(tensor, dim, world_size * chunk, world_size)

def apply_pad(tensor, pad, dim=2):
    if pad > 0:
        pad_size = list(tensor.shape)
        pad_size[dim] = pad
        tensor = torch.cat(
            [tensor, torch.zeros(pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)
    return tensor

def reshape_for_attention(x):
    x = x.flatten(0, 1).permute(1, 0, 2, 3) 
    if get_pad("pad") > 0:
        x = x.narrow(1, 0, x.shape[2] - get_pad("pad"))
    return x

def reshape_after_attention(x):
    x = apply_pad(x, pad=get_pad("pad"), dim=1)
    W_S = dist.get_world_size()
    return x.unflatten(1, (W_S, -1)).permute(1, 2, 0, 3, 4).flatten(-2, -1)


class SPWanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, num_chunks=1):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        use_chunked = num_chunks > 1
        if use_chunked:
            return self.forward_chunked(x, num_chunks)
        else:
            return self._norm(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps).to(x.dtype)

    def forward_chunked(self, x, num_chunks=4):
        output = torch.empty_like(x)

        chunk_sizes = [x.shape[1] // num_chunks + (1 if i < x.shape[1] % num_chunks else 0)
                       for i in range(num_chunks)]

        start_idx = 0
        for size in chunk_sizes:
            end_idx = start_idx + size

            chunk = x[:, start_idx:end_idx, :]

            norm_factor = torch.rsqrt(chunk.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            output[:, start_idx:end_idx, :] = chunk * norm_factor.to(chunk.dtype) * self.weight

            start_idx = end_idx

        return output


class SPWanFusedRMSNorm(nn.RMSNorm):
    def forward(self, x, num_chunks=1):
        use_chunked = num_chunks > 1
        if use_chunked:
            return self.forward_chunked(x, num_chunks)
        else:
            return super().forward(x)

    def forward_chunked(self, x, num_chunks=4):
        output = torch.empty_like(x)

        chunk_sizes = [x.shape[1] // num_chunks + (1 if i < x.shape[1] % num_chunks else 0)
                       for i in range(num_chunks)]

        start_idx = 0
        for size in chunk_sizes:
            end_idx = start_idx + size
            chunk = x[:, start_idx:end_idx, :]
            output[:, start_idx:end_idx, :] = super().forward(chunk)
            start_idx = end_idx

        return output


class SPWanSelfAttention(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 num_heads,
                 qk_norm=True,
                 eps=1e-6,
                 attention_mode="sdpa",
                 rms_norm_function="default",
                 kv_dim=None,
                 sp_group=None,
                 comm_stream=None):
        super().__init__()
        self.comm_stream = comm_stream
        self.dim = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.qk_norm = qk_norm
        self.eps = eps
        self.attention_mode = attention_mode

        #radial attention
        self.mask_map = None
        self.decay_factor = 0.2
        self.cond_size = None

        # layers
        self.q = nn.Linear(in_features, out_features)
        if kv_dim is not None:
            self.k = nn.Linear(kv_dim, out_features)
            self.v = nn.Linear(kv_dim, out_features)
        else:
            self.k = nn.Linear(in_features, out_features)
            self.v = nn.Linear(in_features, out_features)
        self.o = nn.Linear(in_features, out_features)

        if rms_norm_function=="pytorch":
            self.norm_q = SPWanFusedRMSNorm(out_features, eps=eps) if qk_norm else nn.Identity()
            self.norm_k = SPWanFusedRMSNorm(out_features, eps=eps) if qk_norm else nn.Identity()
        else:
            self.norm_q = SPWanRMSNorm(out_features, eps=eps) if qk_norm else nn.Identity()
            self.norm_k = SPWanRMSNorm(out_features, eps=eps) if qk_norm else nn.Identity()

        self.sp_group = sp_group

    def qkv_fn(self, x):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        q, k, v = map(
            lambda x: all_to_all_before_attn(x, self.sp_group, scatter_dim=2, gather_dim=1),
            [q, k, v],
        )
        return q, k, v

    def forward(self, q, k, v, seq_lens, attention_mode_override=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        attention_mode = self.attention_mode
        if attention_mode_override is not None:
            attention_mode = attention_mode_override

        x = attention(q, k, v, k_lens=seq_lens, attention_mode=attention_mode)

        x = all_to_all_after_attn(x, self.sp_group, scatter_dim=1, gather_dim=2)
        return self.o(x.flatten(2))
    
    def forward_phaa(self, query_, key_, value_, query, key, value, query_a2a_buf, key_a2a_buf, value_a2a_buf, seq_lens, fa_is_finished=None, all2all_before_attn_is_finished=None, step=None, fa_split_num=0, hidden_states_buf=None, hidden_states=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        attention_mode = self.attention_mode

        hidden_states_buf[step] = attention(query_, key_, value_, k_lens=seq_lens, attention_mode=attention_mode).type_as(query)

        hidden_states_buf[step] = reshape_after_attention(hidden_states_buf[step])         
        fa_is_finished[step].record()
            
        with torch.npu.stream(self.comm_stream):
            if step + 1 < fa_split_num:
                for source, target in zip([query, key, value], [query_a2a_buf, key_a2a_buf, value_a2a_buf]):
                    torch.distributed.all_to_all_single(get_chunk(target, step + 1),
                                                        get_chunk(source, step + 1),
                                                        group=self.sp_group,
                                                        async_op=False)
                all2all_before_attn_is_finished[step + 1].record(self.comm_stream)
            

            self.comm_stream.wait_event(fa_is_finished[step])
            torch.distributed.all_to_all_single(get_chunk(hidden_states, step),
                                                hidden_states_buf[step],
                                                group=self.sp_group,
                                                async_op=False)  


class SPWanT2VCrossAttention(SPWanSelfAttention):
    def __init__(self, in_features, out_features, num_heads, kv_dim=None, qk_norm=True, eps=1e-6, attention_mode='sdpa', rms_norm_function="default", sp_group=None):
        super().__init__(in_features, out_features, num_heads, qk_norm, eps, attention_mode=attention_mode, kv_dim=kv_dim, rms_norm_function=rms_norm_function, sp_group=sp_group)

    def forward(self, x, context, grid_sizes=None, clip_embed=None, audio_proj=None, audio_scale=1.0,
                num_latent_frames=21, nag_params=None, nag_context=None, is_uncond=False, rope_func="comfy",
                inner_t=None, inner_c=None, cross_freqs=None,
                adapter_proj=None, adapter_attn_mask=None, ip_scale=1.0, orig_seq_len=None, **kwargs):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        # compute query
        q = self.norm_q(self.q(x), num_chunks=2 if rope_func == "comfy_chunked" else 1).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        q, k, v = map(
            lambda x: all_to_all_before_attn(x, self.sp_group, scatter_dim=2, gather_dim=1),
            [q, k, v],
        )

        x_text = attention(q, k, v, attention_mode=self.attention_mode)
        x_text = all_to_all_after_attn(x_text, self.sp_group, scatter_dim=1, gather_dim=2)

        x_text = x_text.flatten(2)

        x = x_text
        return self.o(x)


class SPWanI2VCrossAttention(SPWanSelfAttention):

    def __init__(self, in_features, out_features, num_heads, qk_norm=True, eps=1e-6, attention_mode='sdpa',
                 rms_norm_function="default", sp_group=None):
        super().__init__(in_features, out_features, num_heads, qk_norm, eps, rms_norm_function=rms_norm_function, sp_group=sp_group)
        self.k_img = nn.Linear(in_features, out_features)
        self.v_img = nn.Linear(in_features, out_features)
        self.norm_k_img = SPWanRMSNorm(out_features, eps=eps) if qk_norm else nn.Identity()
        self.attention_mode = attention_mode

    def forward(self, x, context, grid_sizes=None, clip_embed=None, audio_proj=None,
                audio_scale=1.0, num_latent_frames=21, nag_params=None, nag_context=None, is_uncond=False,
                rope_func="comfy",
                adapter_proj=None, adapter_attn_mask=None, ip_scale=1.0, orig_seq_len=None, **kwargs):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
        """
        raise Exception(f"Unsupported now, cant not come here.")

WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': SPWanT2VCrossAttention,
    'i2v_cross_attn': SPWanI2VCrossAttention,
}
class SP_PhaaWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 in_features,
                 out_features,
                 ffn_dim,
                 ffn2_dim,
                 num_heads,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 attention_mode="sdpa",
                 rope_func="comfy",
                 rms_norm_function="default",
                 use_motion_attn=False,
                 use_humo_audio_attn=False,
                 face_fuser_block=False,
                 sp_group=None,
                 comm_stream=None, #通信流
                 ):
        super().__init__()
        self.comm_stream = comm_stream
    
        self.dim = out_features
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.attention_mode = attention_mode
        self.rope_func = rope_func
        #radial attn
        self.dense_timesteps = 10
        self.dense_block = False
        self.dense_attention_mode = "sageattn"

        self.kv_cache = None
        self.use_motion_attn = use_motion_attn
        self.has_face_fuser_block = face_fuser_block

        # layers
        self.norm1 = SPWanLayerNorm(out_features, eps)
        self.sp_group = sp_group
        if self.sp_group is not None:
            self.self_attn = SPWanSelfAttention(in_features, out_features, num_heads, qk_norm, eps, self.attention_mode, rms_norm_function=rms_norm_function, sp_group=self.sp_group, comm_stream=self.comm_stream)
        else:
            self.self_attn = WanSelfAttention(in_features, out_features, num_heads, qk_norm, eps, self.attention_mode, rms_norm_function=rms_norm_function)
        
        # MTV Crafter motion attn
        if self.use_motion_attn:
            self.norm4 = WanLayerNorm(out_features, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
            self.motion_attn = MTVCrafterMotionAttention(in_features, out_features, num_heads, qk_norm, eps, self.attention_mode)
        
        if cross_attn_type != "no_cross_attn":
            self.norm3 = SPWanLayerNorm(out_features, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
            if self.sp_group is not None:
                WAN_CROSSATTENTION_CLASSES['t2v_cross_attn'] = SPWanT2VCrossAttention
                WAN_CROSSATTENTION_CLASSES['i2v_cross_attn'] = SPWanI2VCrossAttention
                self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](in_features, out_features, num_heads, qk_norm, eps, rms_norm_function=rms_norm_function, sp_group=self.sp_group)
            else:
                self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](in_features, out_features, num_heads, qk_norm, epsrms_norm_function=rms_norm_function)

        self.norm2 = SPWanLayerNorm(out_features, eps)
        self.ffn = nn.Sequential(
            nn.Linear(in_features, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn2_dim, out_features))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, out_features) / in_features**0.5)
        self.seg_idx = None

        # HuMo audio cross-attn
        if use_humo_audio_attn:
            self.audio_cross_attn_wrapper = AudioCrossAttentionWrapper(in_features, out_features, num_heads, qk_norm, eps, kv_dim=1536)

        if face_fuser_block:
            from .wananimate.face_blocks import FaceBlock
            self.fuser_block = FaceBlock(self.dim, num_heads)

    #@torch.compiler.disable()
    def get_mod(self, e):
        if e.dim() == 3:
            return (self.modulation  + e).chunk(6, dim=1) # 1, 6, dim
        elif e.dim() == 4:
            e_mod = self.modulation.unsqueeze(2) + e
            return tuple(ei.squeeze(1) for ei in modulated.chunk(6, dim=1))
        else:
            raise ValueError(
                f"输入张量 e 的维度必须是 3 或 4，当前维度为 {e.dim()}"
            )
    
    def modulate(self, x, shift_msa, scale_msa, seg_idx=None):
        """
        Modulate x with shift and scale. If seg_idx is provided, apply segmented modulation.
        """
        norm_x = self.norm1(x)
        if seg_idx is not None:
            parts = []
            for i in range(2):
                part = torch.addcmul(
                    shift_msa[:, i:i + 1],
                    norm_x[:, seg_idx[i]:seg_idx[i + 1]],
                    1 + scale_msa[:, i:i + 1]
                )
                parts.append(part)
            norm_x = torch.cat(parts, dim=1)
            return norm_x
        else:
            return torch.addcmul(shift_msa, norm_x, 1 + scale_msa)
    
    def ffn_chunked(self, x, shift_mlp, scale_mlp, num_chunks=4):
        modulated_input = torch.addcmul(shift_mlp, self.norm2(x), 1 + scale_mlp)
        
        result = torch.empty_like(x)
        seq_len = modulated_input.shape[1]
        
        chunk_sizes = [seq_len // num_chunks + (1 if i < seq_len % num_chunks else 0) 
                    for i in range(num_chunks)]
        
        start_idx = 0
        for size in chunk_sizes:
            end_idx = start_idx + size
            chunk = modulated_input[:, start_idx:end_idx, :]
            result[:, start_idx:end_idx, :] = self.ffn(chunk)
            start_idx = end_idx
        
        return result


    #region attention forward
    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        current_step,
        last_step=False,
        video_attention_split_steps=None,
        clip_embed=None,
        camera_embed=None,
        audio_proj=None,
        audio_scale=1.0,
        num_latent_frames=21,
        original_seq_len=None,
        enhance_enabled=False,
        nag_params=None,
        nag_context=None,
        is_uncond=False,
        multitalk_audio_embedding=None,
        ref_target_masks=None,
        human_num=0,
        inner_t=None,
        inner_c=None,
        cross_freqs=None,
        x_ip=None, e_ip=None,
        freqs_ip=None,
        adapter_proj=None,
        ip_scale=1.0,
        reverse_time=False,
        mtv_motion_tokens=None, mtv_motion_rotary_emb=None, mtv_strength=1.0, mtv_freqs=None,
        humo_audio_input=None, humo_audio_scale=1.0,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        self.original_seq_len = original_seq_len
        self.zero_timestep = len(e) == 2
        if self.zero_timestep: #s2v zero timestep
            self.seg_idx = e[1]
            self.seg_idx = min(max(0, self.seg_idx), x.size(1))
            self.seg_idx = [0, self.seg_idx, x.size(1)]
            e = e[0]

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.get_mod(e.to(x.device))
        del e
        input_x = self.modulate(x, shift_msa, scale_msa, seg_idx=self.seg_idx)
        del shift_msa, scale_msa

        if x_ip is not None:
            shift_msa_ip, scale_msa_ip, gate_msa_ip, shift_mlp_ip, scale_mlp_ip, gate_mlp_ip = self.get_mod(e_ip.to(x.device))
            input_x_ip = self.modulate(self.norm1(x_ip), shift_msa_ip, scale_msa_ip)
            self.cond_size = input_x_ip.shape[1]
            input_x = torch.concat([input_x, input_x_ip], dim=1)
            self.kv_cache = None
        if camera_embed is not None:
            # encode ReCamMaster camera
            camera_embed = self.cam_encoder(camera_embed.to(x))
            camera_embed = camera_embed.repeat(1, 2, 1)
            camera_embed = camera_embed.unsqueeze(2).unsqueeze(3).repeat(1, 1, grid_sizes[0][1], grid_sizes[0][2], 1)
            camera_embed = rearrange(camera_embed, 'b f h w d -> b (f h w) d')
            input_x += camera_embed

        # self-attention
        x_ref_attn_map = None

        # self-attention variables
        q_ip = k_ip = v_ip = None

        # Prepare shapes
        query = self.self_attn.norm_q(self.self_attn.q(input_x))
        key = self.self_attn.norm_k(self.self_attn.k(input_x))
        value = self.self_attn.v(input_x)
        W_S = dist.get_world_size(self.sp_group)
        fa_split_num = 2
        attn_heads_per_split = self.num_heads // W_S // fa_split_num
        B, Sq, H = query.shape
        Skv = key.shape[1]

        # Prepare events for synchronization
        all2all_before_attn_is_finished = [torch.npu.Event() for _ in range(fa_split_num)]
        fa_is_finished = [torch.npu.Event() for _ in range(fa_split_num)]

        # Prepare output tensors
        query_a2a_buf = torch.empty((W_S * fa_split_num, Sq, B, attn_heads_per_split, H // self.num_heads),
                                    device=query.device, dtype=query.dtype)
        key_a2a_buf = torch.empty((W_S * fa_split_num, Skv, B, attn_heads_per_split, H // self.num_heads),
                                  device=key.device, dtype=key.dtype)
        value_a2a_buf = torch.empty((W_S * fa_split_num, Skv, B, attn_heads_per_split, H // self.num_heads),
                                    device=value.device, dtype=value.dtype)
        hidden_states = torch.empty((W_S * fa_split_num, Sq, B, H // W_S // fa_split_num),
                                    device=query.device, dtype=query.dtype)
        hidden_states_buf = [None] * fa_split_num

        query = query.unflatten(-1, (W_S * fa_split_num, attn_heads_per_split, -1)).permute(
            [2, 1, 0, 3, 4]).contiguous()
 
        self.comm_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(self.comm_stream):
            torch.distributed.all_to_all_single(get_chunk(query_a2a_buf, 0), get_chunk(query, 0),
                                                group=self.sp_group, async_op=False)

        key = key.unflatten(-1, (W_S * fa_split_num, attn_heads_per_split, -1)).permute([2, 1, 0, 3, 4]).contiguous()
        self.comm_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(self.comm_stream):
            torch.distributed.all_to_all_single(get_chunk(key_a2a_buf, 0), get_chunk(key, 0),
                                                group=self.sp_group, async_op=False)

        value = value.unflatten(-1, (W_S * fa_split_num, attn_heads_per_split, -1)).permute(
            [2, 1, 0, 3, 4]).contiguous()
        self.comm_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(self.comm_stream):
            torch.distributed.all_to_all_single(get_chunk(value_a2a_buf, 0), get_chunk(value, 0),
                                                group=self.sp_group, async_op=False)
            all2all_before_attn_is_finished[0].record(self.comm_stream)
        
        for step in range(fa_split_num):
            # 2. Compute attention
            torch.npu.current_stream().wait_event(all2all_before_attn_is_finished[step])
            # WS S B N D -> B N WS*S D
            query_, key_, value_ = map(lambda x: reshape_for_attention(get_chunk(x, step)),
                                       [query_a2a_buf, key_a2a_buf, value_a2a_buf])
                                    
            # FETA
            if enhance_enabled:
                feta_scores = get_feta_scores(query_a2a_buf[step], key_a2a_buf[step])
            #RoPE
            query_, key_  = apply_rope_comfy(query_, key_, freqs)

            self.self_attn.forward_phaa(query_, key_, value_, query, key, value, query_a2a_buf, key_a2a_buf, value_a2a_buf, seq_lens, fa_is_finished=fa_is_finished, all2all_before_attn_is_finished=all2all_before_attn_is_finished, step=step, fa_split_num=fa_split_num, hidden_states_buf=hidden_states_buf, hidden_states=hidden_states)
        torch.npu.current_stream().wait_stream(self.comm_stream)

        hidden_states = hidden_states.permute([2, 1, 0, 3]).flatten(-2, -1)
        y = self.self_attn.o(hidden_states)
            
         # FETA
        if enhance_enabled:
            y.mul_(feta_scores)

        #ReCamMaster
        if camera_embed is not None:
            y = self.projector(y)        

        if x_ip is not None:
            y, y_ip = (
                y[:, : -self.cond_size],
                y[:, -self.cond_size :],
            )

        if self.zero_timestep:
            z = []
            for i in range(2):
                z.append(y[:, self.seg_idx[i]:self.seg_idx[i + 1]] * gate_msa[:, i:i + 1])
            y = torch.cat(z, dim=1)
            x = x.add(y)
        else:
            x = x.addcmul(y, gate_msa)
        del y, gate_msa

        # cross-attention & ffn function
        if context is not None:           
            x = self.cross_attn_ffn(x, context, grid_sizes, shift_mlp, scale_mlp, gate_mlp, clip_embed, 
                                    audio_proj, audio_scale, num_latent_frames, nag_params, nag_context, is_uncond, 
                                    multitalk_audio_embedding, x_ref_attn_map, human_num, inner_t, inner_c, cross_freqs,
                                    adapter_proj=adapter_proj, ip_scale=ip_scale, 
                                    mtv_freqs=mtv_freqs, mtv_motion_tokens=mtv_motion_tokens, mtv_motion_rotary_emb=mtv_motion_rotary_emb, mtv_strength=mtv_strength,
                                    humo_audio_input=humo_audio_input, humo_audio_scale=humo_audio_scale
                                    )
        else:
            if self.rope_func == "comfy_chunked":
                y = self.ffn_chunked(x, shift_mlp, scale_mlp)
            else:
                y = self.ffn(torch.addcmul(shift_mlp, self.norm2(x), 1 + scale_mlp))
            x = x.addcmul(y, gate_mlp)
        del gate_mlp

        if x_ip is not None:
            print("10")
            x_ip = x_ip.addcmul(y_ip, gate_msa_ip)
            y_ip = self.ffn(torch.addcmul(shift_mlp_ip, self.norm2(x_ip), 1 + scale_mlp_ip))
            x_ip = x_ip.addcmul(y_ip, gate_mlp_ip)

        return x, x_ip

    
    def cross_attn_ffn(self, x, context, grid_sizes, shift_mlp, scale_mlp, gate_mlp, clip_embed, 
                       audio_proj, audio_scale, num_latent_frames, nag_params, 
                       nag_context, is_uncond, multitalk_audio_embedding, x_ref_attn_map, human_num, 
                       inner_t, inner_c, cross_freqs, adapter_proj, ip_scale, mtv_freqs, mtv_motion_tokens, mtv_motion_rotary_emb, mtv_strength,
                       humo_audio_input, humo_audio_scale):

            x = x + self.cross_attn(self.norm3(x), context, grid_sizes, clip_embed=clip_embed,
                                    audio_proj=audio_proj, audio_scale=audio_scale,
                                    num_latent_frames=num_latent_frames, nag_params=nag_params, nag_context=nag_context, is_uncond=is_uncond,
                                    rope_func=self.rope_func, inner_t=inner_t, inner_c=inner_c, cross_freqs=cross_freqs,
                                    adapter_proj=adapter_proj, ip_scale=ip_scale, orig_seq_len=self.original_seq_len)
            # MultiTalk
            if multitalk_audio_embedding is not None and not isinstance(self, VaceWanAttentionBlock):
                x_audio = self.audio_cross_attn(self.norm_x(x), encoder_hidden_states=multitalk_audio_embedding,
                                            shape=grid_sizes[0], x_ref_attn_map=x_ref_attn_map, human_num=human_num)
                x = x + x_audio * audio_scale

            # MTV-Crafter Motion Attention
            if self.use_motion_attn and mtv_motion_tokens is not None and mtv_motion_rotary_emb is not None:                
                x_motion = self.motion_attn(self.norm4(x), mtv_motion_tokens, mtv_motion_rotary_emb, grid_sizes, mtv_freqs)
                x = x + x_motion * mtv_strength

            # HuMo Audio Cross-Attention
            if humo_audio_input is not None:
                x = self.audio_cross_attn_wrapper(x, humo_audio_input, grid_sizes, humo_audio_scale)

            if self.rope_func == "comfy_chunked" and not self.zero_timestep:
                y = self.ffn_chunked(x, shift_mlp, scale_mlp)
            else:
                norm2_x = self.norm2(x)
                if self.zero_timestep:
                    parts = []
                    for i in range(2):
                        parts.append(norm2_x[:, self.seg_idx[i]:self.seg_idx[i + 1]] *
                                    (1 + scale_mlp[:, i:i + 1]) + shift_mlp[:, i:i + 1])
                    norm2_x = torch.cat(parts, dim=1)
                    y = self.ffn(norm2_x)
                else:
                    input_x = torch.addcmul(shift_mlp, norm2_x, 1 + scale_mlp)
                    del shift_mlp, scale_mlp, norm2_x
                    y = self.ffn(input_x)
            if self.zero_timestep:
                z = []
                for i in range(2):
                    z.append(y[:, self.seg_idx[i]:self.seg_idx[i + 1]] * gate_mlp[:, i:i + 1])
                y = torch.cat(z, dim=1)
                x = x.add(y)
            else:
                x = x.addcmul(y, gate_mlp)
            return x
    
    @torch.compiler.disable()
    def split_cross_attn_ffn(self, x, context, shift_mlp, scale_mlp, gate_mlp, clip_embed=None, grid_sizes=None):
        # Get number of prompts
        num_prompts = context.shape[0]
        num_clip_embeds = 0 if clip_embed is None else clip_embed.shape[0]
        num_segments = max(num_prompts, num_clip_embeds)
        
        # Extract spatial dimensions
        frames, height, width = grid_sizes[0]  # Assuming batch size 1
        tokens_per_frame = height * width
        
        # Distribute frames across prompts
        frames_per_segment = max(1, frames // num_segments)
        
        # Process each prompt segment
        x_combined = torch.zeros_like(x)
        
        for i in range(num_segments):
            # Calculate frame boundaries for this segment
            start_frame = i * frames_per_segment
            end_frame = min((i+1) * frames_per_segment, frames) if i < num_segments-1 else frames
            
            # Convert frame indices to token indices
            start_idx = start_frame * tokens_per_frame
            end_idx = end_frame * tokens_per_frame
            segment_indices = torch.arange(start_idx, end_idx, device=x.device, dtype=torch.long)
            
            # Get prompt segment (cycle through available prompts if needed)
            prompt_idx = i % num_prompts
            segment_context = context[prompt_idx:prompt_idx+1]
            
            # Handle clip_embed for this segment (cycle through available embeddings)
            segment_clip_embed = None
            if clip_embed is not None:
                clip_idx = i % num_clip_embeds
                segment_clip_embed = clip_embed[clip_idx:clip_idx+1]
            
            # Get tensor segment
            x_segment = x[:, segment_indices, :]
            
            # Process segment with its prompt and clip embedding
            processed_segment = self.cross_attn(self.norm3(x_segment), segment_context, clip_embed=segment_clip_embed)
            processed_segment = processed_segment.to(x.dtype)
            
            # Add to combined result
            x_combined[:, segment_indices, :] = processed_segment
        
        # Continue with FFN
        x = x + x_combined
        y = self.ffn_chunked(x, shift_mlp, scale_mlp)
        x = x.addcmul(y, gate_mlp)
        return x

class SPWanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x)