"""
Fused DeltaNet Recurrent State Update - v3: B200 Optimized.

Key optimization over V2/FLA: gate and beta pre-computation is FUSED
into the Triton kernel itself, eliminating ~55μs of host-side PyTorch
kernel launch overhead (3 separate CUDA launches for sigmoid, exp, softplus).

Tile shape: [BV, K_full] — same single-pass strategy as V2.
Grid: (NV * HV, B) where NV = cdiv(V, BV).

In-kernel scalar computation per program:
    x = a[batch, head] + dt_bias[head]
    log_decay = -exp(A_log[head]) * softplus(x)
    decay = exp(log_decay)
    beta = sigmoid(b[batch, head])

DeltaNet linear attention recurrence (per head, per decode step):
    S *= decay                          # gate decay
    residual = v - S^T @ k              # delta rule residual
    delta = beta * residual
    S += outer(k, delta)                # rank-1 state update
    o = S^T @ q                         # output query

State layout: [B, HV, V, K] (k-last, K contiguous).
"""
import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice


@triton.jit
def fast_exp(x):
    return tldevice.fast_expf(x.to(tl.float32))


@triton.jit
def _deltanet_recurrent_v3_kernel(
    # Input vectors (per token, per head)
    Q_ptr,        # [B, 1, H, K]   bf16
    K_ptr,        # [B, 1, H, K]   bf16
    V_ptr,        # [B, 1, HV, V]  bf16
    # Raw gate parameters — NOT pre-computed
    A_log_ptr,    # [HV]           f32
    A_ptr,        # [B, 1, HV]     bf16
    Dt_bias_ptr,  # [HV]           f32
    B_ptr,        # [B, 1, HV]     bf16
    # State and output
    S_ptr,        # State [B, HV, V, K]     f32
    New_S_ptr,    # New State [B, HV, V, K] f32
    O_ptr,        # Output [B, 1, HV, V]    bf16
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    """
    Fused DeltaNet recurrent step - v3 with in-kernel gate computation.

    Grid: (NV * HV, B)  where NV = cdiv(V, BV).
    Each program handles one [BV, K] tile — one V-block of one head.

    Gate/beta computation is done per-program as scalar ops (zero overhead
    compared to 3 separate PyTorch kernel launches on host).
    """
    # Grid: (NV, B * HV) — matches FLA layout
    i_v = tl.program_id(0)            # which V-block
    bh = tl.program_id(1)             # batch_id * HV + V_head_id
    batch_id = bh // HV
    V_head_id = bh % HV
    head_id = V_head_id // (HV // H)  # GQA mapping

    # ===== In-kernel gate/beta computation (replaces 3 PyTorch launches) =====
    # Load raw scalars
    a_val = tl.load(A_ptr + batch_id * HV + V_head_id).to(tl.float32)
    dt_bias_val = tl.load(Dt_bias_ptr + V_head_id).to(tl.float32)
    a_log_val = tl.load(A_log_ptr + V_head_id).to(tl.float32)
    b_val = tl.load(B_ptr + batch_id * HV + V_head_id).to(tl.float32)

    # softplus(x) = log(1 + exp(x)), with numerical stability for large x
    x = a_val + dt_bias_val
    sp = tl.where(x > 20.0, x, tl.log(1.0 + fast_exp(x)))

    # log_decay = -exp(A_log) * softplus(x)
    log_decay = -fast_exp(a_log_val) * sp
    decay = fast_exp(log_decay)

    # beta = sigmoid(b) = 1 / (1 + exp(-b))
    beta = tl.sigmoid(b_val)

    # ===== Standard DeltaNet recurrence (same as V2) =====

    # K offsets — full K dimension
    dk_offs = tl.arange(0, K)  # [K]

    # Pre-load full q and k vectors [K]
    q_base = Q_ptr + batch_id * (H * K) + head_id * K
    k_base = K_ptr + batch_id * (H * K) + head_id * K
    b_q = tl.load(q_base + dk_offs).to(tl.float32) * scale  # [K]
    b_k = tl.load(k_base + dk_offs).to(tl.float32)          # [K]

    # V offsets for this tile
    dv_offs = i_v * BV + tl.arange(0, BV)  # [BV]
    dv_mask = dv_offs < V

    # Load v slice [BV]
    b_v = tl.load(V_ptr + batch_id * (HV * V) + V_head_id * V + dv_offs,
                   mask=dv_mask, other=0.0).to(tl.float32)

    # State base pointers
    s_base = S_ptr + batch_id * (HV * V * K) + V_head_id * (V * K)
    new_s_base = New_S_ptr + batch_id * (HV * V * K) + V_head_id * (V * K)

    # Load state tile [BV, K] — CONTIGUOUS in memory
    s_ptrs = s_base + dv_offs[:, None] * K + dk_offs[None, :]  # [BV, K]
    s_mask = dv_mask[:, None]
    s_tile = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32)

    # ① Gate decay: S *= decay
    s_tile = s_tile * decay

    # ② S^T @ k for this block: dot(S[v,:], k) per v → [BV]
    stk = tl.sum(s_tile * b_k[None, :], axis=1)  # [BV]

    # ③ Delta rule: delta = beta * (v - S^T @ k)
    delta = beta * (b_v - stk)  # [BV]

    # ④ Rank-1 update: S[v,:] += delta[v] * k[:]
    s_tile += delta[:, None] * b_k[None, :]  # [BV, K]

    # Store updated state tile [BV, K]
    new_s_ptrs = new_s_base + dv_offs[:, None] * K + dk_offs[None, :]
    tl.store(new_s_ptrs, s_tile, mask=s_mask)

    # ⑤ Output: o[v] = dot(S_new[v,:], q) per v → [BV]
    b_o = tl.sum(s_tile * b_q[None, :], axis=1)  # [BV]

    # Store output slice [BV]
    o_ptrs = O_ptr + batch_id * (HV * V) + V_head_id * V + dv_offs
    tl.store(o_ptrs, b_o.to(tl.bfloat16), mask=dv_mask)


# =============================================================================
# 官方评测框架的入口包装 (Wrapper)
# =============================================================================
def kernel(
    q: torch.Tensor,        # [B, T, H, K] bf16
    k: torch.Tensor,        # [B, T, H, K] bf16
    v: torch.Tensor,        # [B, T, HV, V] bf16
    state: torch.Tensor,    # [B, HV, V, K] f32 (k-last)
    A_log: torch.Tensor,    # [HV] f32
    a: torch.Tensor,        # [B, T, HV] bf16
    dt_bias: torch.Tensor,  # [HV] f32
    b: torch.Tensor,        # [B, T, HV] bf16
    scale: float,           # float32 scalar
    output: torch.Tensor,   # [B, T, HV, V] bf16 (DPS 预分配输出)
    new_state: torch.Tensor # [B, HV, V, K] f32 (DPS 预分配状态)
):
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]

    if scale is None or scale == 0.0:
        scale = 1.0 / (K ** 0.5)

    # NO host-side precomputation — raw parameters go directly to kernel
    # This eliminates ~55μs of PyTorch kernel launch overhead

    # Grid: V-blocks × heads parallelized
    BV = min(16, triton.next_power_of_2(V))  # fixed BV, same strategy as FLA
    NV = triton.cdiv(V, BV)
    grid = (NV, B * HV)

    _deltanet_recurrent_v3_kernel[grid](
        q, k, v,
        A_log, a, dt_bias, b,
        state, new_state, output, scale,
        T, H, HV,
        K, V,
        BV=BV,
        num_warps=4,
        num_stages=1,
    )

    return output, new_state
