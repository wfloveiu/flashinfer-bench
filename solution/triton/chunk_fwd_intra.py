# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Fused kkt + solve_tril kernel for gated delta rule

import torch
import triton
import triton.language as tl

from .utils import prepare_chunk_indices, exp, autotune_cache_kwargs, IS_TMA_SUPPORTED
from .wy_fast import recompute_w_u_fwd


DOT_PRECISION_LIST = ['tf32'] if IS_TMA_SUPPORTED else ['ieee']


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'DOT_PRECISION': prec}, num_warps=num_warps)
        for BK in [32, 64, 128]
        for num_warps in [1, 2, 4]
        for prec in DOT_PRECISION_LIST
    ],
    key=['H', 'Hk', 'K', 'BC'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_fwd_kkt_solve_kernel(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hk: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Fused kkt + solve_tril: compute beta * K @ K^T and (I+A)^{-1} in one pass."""
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    i_hk = i_h // (H // Hk)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    k_base = k + (bos * Hk + i_hk) * K
    A_base = A + (bos * H + i_h) * BT

    o_i = tl.arange(0, BC)
    m_tc0 = (i_tc0 + o_i) < T
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    # Load beta
    p_b0 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc0,), (BC,), (0,))
    p_b1 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc1,), (BC,), (0,))
    p_b2 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc2,), (BC,), (0,))
    p_b3 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc3,), (BC,), (0,))
    b_b0 = tl.load(p_b0, boundary_check=(0,)).to(tl.float32)
    b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
    b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
    b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)

    # Load gate
    if USE_G:
        p_g0 = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_tc0,), (BC,), (0,))
        p_g1 = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_tc1,), (BC,), (0,))
        p_g2 = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_tc2,), (BC,), (0,))
        p_g3 = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_tc3,), (BC,), (0,))
        b_g0 = tl.load(p_g0, boundary_check=(0,)).to(tl.float32)
        b_g1 = tl.load(p_g1, boundary_check=(0,)).to(tl.float32)
        b_g2 = tl.load(p_g2, boundary_check=(0,)).to(tl.float32)
        b_g3 = tl.load(p_g3, boundary_check=(0,)).to(tl.float32)

    # Step 1: compute all 10 lower-triangular [BC, BC] blocks of K @ K^T
    b_A00 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A11 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A22 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A33 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A32 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_k0 = tl.make_block_ptr(k_base, (T, K), (Hk*K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        b_A00 += tl.dot(b_k0, tl.trans(b_k0))

        p_k1 = tl.make_block_ptr(k_base, (T, K), (Hk*K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))
        b_A11 += tl.dot(b_k1, tl.trans(b_k1))
        b_A10 += tl.dot(b_k1, tl.trans(b_k0))

        p_k2 = tl.make_block_ptr(k_base, (T, K), (Hk*K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
        b_k2 = tl.load(p_k2, boundary_check=(0, 1))
        b_A22 += tl.dot(b_k2, tl.trans(b_k2))
        b_A20 += tl.dot(b_k2, tl.trans(b_k0))
        b_A21 += tl.dot(b_k2, tl.trans(b_k1))

        p_k3 = tl.make_block_ptr(k_base, (T, K), (Hk*K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
        b_k3 = tl.load(p_k3, boundary_check=(0, 1))
        b_A33 += tl.dot(b_k3, tl.trans(b_k3))
        b_A30 += tl.dot(b_k3, tl.trans(b_k0))
        b_A31 += tl.dot(b_k3, tl.trans(b_k1))
        b_A32 += tl.dot(b_k3, tl.trans(b_k2))

    # Step 2: apply gate and beta scaling
    if USE_G:
        b_A00 *= exp(b_g0[:, None] - b_g0[None, :])
        b_A11 *= exp(b_g1[:, None] - b_g1[None, :])
        b_A22 *= exp(b_g2[:, None] - b_g2[None, :])
        b_A33 *= exp(b_g3[:, None] - b_g3[None, :])
        b_A10 *= exp(b_g1[:, None] - b_g0[None, :])
        b_A20 *= exp(b_g2[:, None] - b_g0[None, :])
        b_A21 *= exp(b_g2[:, None] - b_g1[None, :])
        b_A30 *= exp(b_g3[:, None] - b_g0[None, :])
        b_A31 *= exp(b_g3[:, None] - b_g1[None, :])
        b_A32 *= exp(b_g3[:, None] - b_g2[None, :])

    m_d = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_A00 = tl.where(m_d & (m_tc0[:, None] & m_tc0[None, :]), b_A00, 0.) * b_b0[:, None]
    b_A11 = tl.where(m_d & (m_tc1[:, None] & m_tc1[None, :]), b_A11, 0.) * b_b1[:, None]
    b_A22 = tl.where(m_d & (m_tc2[:, None] & m_tc2[None, :]), b_A22, 0.) * b_b2[:, None]
    b_A33 = tl.where(m_d & (m_tc3[:, None] & m_tc3[None, :]), b_A33, 0.) * b_b3[:, None]
    b_A10 = tl.where(m_tc1[:, None] & m_tc0[None, :], b_A10, 0.) * b_b1[:, None]
    b_A20 = tl.where(m_tc2[:, None] & m_tc0[None, :], b_A20, 0.) * b_b2[:, None]
    b_A21 = tl.where(m_tc2[:, None] & m_tc1[None, :], b_A21, 0.) * b_b2[:, None]
    b_A30 = tl.where(m_tc3[:, None] & m_tc0[None, :], b_A30, 0.) * b_b3[:, None]
    b_A31 = tl.where(m_tc3[:, None] & m_tc1[None, :], b_A31, 0.) * b_b3[:, None]
    b_A32 = tl.where(m_tc3[:, None] & m_tc2[None, :], b_A32, 0.) * b_b3[:, None]

    # Step 3: forward substitution on diagonal blocks
    b_Ai00 = -b_A00
    b_Ai11 = -b_A11
    b_Ai22 = -b_A22
    b_Ai33 = -b_A33

    for i in range(2, min(BC, T - i_tc0)):
        b_a = tl.sum(tl.where((o_i == i)[:, None], -b_A00, 0.), 0)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai00, 0)
        b_Ai00 = tl.where((o_i == i)[:, None], b_a, b_Ai00)
    for i in range(2, min(BC, T - i_tc1)):
        b_a = tl.sum(tl.where((o_i == i)[:, None], -b_A11, 0.), 0)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai11, 0)
        b_Ai11 = tl.where((o_i == i)[:, None], b_a, b_Ai11)
    for i in range(2, min(BC, T - i_tc2)):
        b_a = tl.sum(tl.where((o_i == i)[:, None], -b_A22, 0.), 0)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai22, 0)
        b_Ai22 = tl.where((o_i == i)[:, None], b_a, b_Ai22)
    for i in range(2, min(BC, T - i_tc3)):
        b_a = tl.sum(tl.where((o_i == i)[:, None], -b_A33, 0.), 0)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai33, 0)
        b_Ai33 = tl.where((o_i == i)[:, None], b_a, b_Ai33)

    b_Ai00 += m_I
    b_Ai11 += m_I
    b_Ai22 += m_I
    b_Ai33 += m_I

    # Step 4: block merge
    b_Ai10 = -tl.dot(tl.dot(b_Ai11, b_A10, input_precision=DOT_PRECISION), b_Ai00, input_precision=DOT_PRECISION)
    b_Ai21 = -tl.dot(tl.dot(b_Ai22, b_A21, input_precision=DOT_PRECISION), b_Ai11, input_precision=DOT_PRECISION)
    b_Ai32 = -tl.dot(tl.dot(b_Ai33, b_A32, input_precision=DOT_PRECISION), b_Ai22, input_precision=DOT_PRECISION)

    b_Ai20 = -tl.dot(b_Ai22, tl.dot(b_A20, b_Ai00, input_precision=DOT_PRECISION) + tl.dot(b_A21, b_Ai10, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)
    b_Ai31 = -tl.dot(b_Ai33, tl.dot(b_A31, b_Ai11, input_precision=DOT_PRECISION) + tl.dot(b_A32, b_Ai21, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)
    b_Ai30 = -tl.dot(b_Ai33, tl.dot(b_A30, b_Ai00, input_precision=DOT_PRECISION) + tl.dot(b_A31, b_Ai10, input_precision=DOT_PRECISION) + tl.dot(b_A32, b_Ai20, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)

    # Step 5: store
    p_A00 = tl.make_block_ptr(A_base, (T, BT), (H*BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_A10 = tl.make_block_ptr(A_base, (T, BT), (H*BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_A11 = tl.make_block_ptr(A_base, (T, BT), (H*BT, 1), (i_tc1, BC), (BC, BC), (1, 0))
    p_A20 = tl.make_block_ptr(A_base, (T, BT), (H*BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_A21 = tl.make_block_ptr(A_base, (T, BT), (H*BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
    p_A22 = tl.make_block_ptr(A_base, (T, BT), (H*BT, 1), (i_tc2, 2*BC), (BC, BC), (1, 0))
    p_A30 = tl.make_block_ptr(A_base, (T, BT), (H*BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
    p_A31 = tl.make_block_ptr(A_base, (T, BT), (H*BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
    p_A32 = tl.make_block_ptr(A_base, (T, BT), (H*BT, 1), (i_tc3, 2*BC), (BC, BC), (1, 0))
    p_A33 = tl.make_block_ptr(A_base, (T, BT), (H*BT, 1), (i_tc3, 3*BC), (BC, BC), (1, 0))

    tl.store(p_A00, b_Ai00.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A10, b_Ai10.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A11, b_Ai11.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A20, b_Ai20.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A21, b_Ai21.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A22, b_Ai22.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A30, b_Ai30.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A31, b_Ai31.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A32, b_Ai32.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A33, b_Ai33.to(A.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_intra(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused kkt + solve_tril + recompute_w_u."""
    B, T, Hk, K = k.shape
    H = beta.shape[-1] # H = HV
    BT = 64
    BC = 16

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    A = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)
    chunk_gated_delta_rule_fwd_kkt_solve_kernel[(NT, B * H)](
        k=k, g=g, beta=beta, A=A,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        T=T, H=H, Hk=Hk, K=K, BT=BT, BC=BC,
    )

    w, u = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A, g=g,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )
    return w, u, A
