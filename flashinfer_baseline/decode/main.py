import math
import torch
from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose


def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    if isinstance(scale, torch.Tensor):
        scale = float(scale.item())
    else:
        scale = float(scale)
    if scale == 0.0:
        scale = 1.0 / math.sqrt(q.shape[-1])

    B, T, num_v_heads, head_size = v.shape
    output = torch.empty(B, T, num_v_heads, head_size, dtype=q.dtype, device=q.device)

    out, new_state = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=state,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=scale,
        output=output,
        use_qk_l2norm=False,
    )

    return out, new_state
