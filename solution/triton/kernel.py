import torch
import torch.nn.functional as F
import math
from .chunk_gdn import chunk_gated_delta_rule


def kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state):
    """
    Inputs:
        q: [total_seq_len, num_q_heads, head_size] bfloat16
        k: [total_seq_len, num_k_heads, head_size] bfloat16
        v: [total_seq_len, num_v_heads, head_size] bfloat16
        state: [num_seqs, num_v_heads, head_size, head_size] float32
        A_log: [num_v_heads]  float32
        a: [total_seq_len, num_v_heads] bfloat16
        dt_bias: [num_v_heads]  float32
        b: [total_seq_len, num_v_heads] bfloat16
        cu_seqlens: [num_seqs+1] int64
        scale: scalar float32
    Outputs:
        output: [total_seq_len, num_v_heads, head_size] bfloat16
        new_state: [num_seqs, num_v_heads, head_size, head_size] float32
    """

    return chunk_gated_delta_rule(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state)


