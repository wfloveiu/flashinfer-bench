import math

import torch
import torch.nn.functional as F

from .utils import prepare_chunk_indices, input_guard, autocast_custom_fwd
from .cumsum import chunk_local_cumsum
from .chunk_fwd_intra import chunk_gated_delta_rule_fwd_intra
from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
from .chunk_o import chunk_fwd_o
from .fused_gdn_gating import fused_gdn_gating

def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    transpose_state_layout: bool = False,
    output: torch.Tensor | None = None,
    final_state_buf: torch.Tensor | None = None,
):
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    # fused kkt + solve_tril + recompute_w_u (avoids HBM round-trip for A)
    w, u, _ = chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        transpose_state_layout=transpose_state_layout,
        final_state_buf=final_state_buf,
    )

    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        transpose_state_layout=transpose_state_layout,
        output=output,
    )
    return o, final_state



@torch.compiler.disable
def chunk_gated_delta_rule(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state):
    """
    Chunked gated delta rule with the same interface as baseline_run.

    Inputs:
        q: [total_seq_len, num_q_heads, head_size] bfloat16
        k: [total_seq_len, num_k_heads, head_size] bfloat16
        v: [total_seq_len, num_v_heads, head_size] bfloat16
        state: [num_seqs, num_v_heads, head_size, head_size] float32  (H, V, K layout)
        A_log: [num_v_heads] float32
        a: [total_seq_len, num_v_heads] bfloat16
        dt_bias: [num_v_heads] float32
        b: [total_seq_len, num_v_heads] bfloat16
        cu_seqlens: [num_seqs+1] int64
        scale: scalar float32
    Returns:
        output: [total_seq_len, num_v_heads, head_size] bfloat16
        new_state: [num_seqs, num_v_heads, head_size, head_size] float32
    """
    _, _, head_size = q.shape

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_size)


    g_log, beta = fused_gdn_gating(A_log, a, b, dt_bias) # [1, T, HV]
    
    
    # ---------- reshape 3D -> 4D (B=1 for varlen mode) ----------
    q_4d = q.unsqueeze(0)                                   # [1, T, Hq, K]
    k_4d = k.unsqueeze(0)                                   # [1, T, Hk, K]
    v_4d = v.unsqueeze(0)                                   # [1, T, HV, V]

    chunk_indices = prepare_chunk_indices(cu_seqlens, 64) if cu_seqlens is not None else None
    output_4d = output.unsqueeze(0)  # [T, Hv, D] -> [1, T, Hv, D], zero-copy view
    chunk_gated_delta_rule_fwd(
        q=q_4d,
        k=k_4d,
        v=v_4d,
        g=g_log,
        beta=beta,
        scale=scale,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        transpose_state_layout=True,
        output=output_4d,
        final_state_buf=new_state,
    )

    return output, new_state