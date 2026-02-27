import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd import Function


@triton.jit
def _flatten_fwd_kernel(
    points_ptr,
    out_ptr,
    N,
    S: tl.constexpr,
    total,
    BLOCK: tl.constexpr,
):
    """Each thread computes one cubic Bezier sample.

    Index mapping: flat_idx → (m, s) where m = flat_idx // S, s = flat_idx % S.
    Points layout: [anchor, ctrl1, ctrl2, anchor, ctrl1, ctrl2, ...] (N, 2).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total  # total = M * S

    m = offs // S
    s = offs % S

    # t ∈ [0, 1), evenly spaced
    t = s.to(tl.float32) / S
    mt = 1.0 - t

    # Control-point indices (mod N for closed paths)
    i0 = (m * 3) % N
    i1 = (m * 3 + 1) % N
    i2 = (m * 3 + 2) % N
    i3 = ((m + 1) * 3) % N

    # Gather control points — row-major (N, 2)
    p0x = tl.load(points_ptr + i0 * 2, mask=mask)
    p0y = tl.load(points_ptr + i0 * 2 + 1, mask=mask)
    p1x = tl.load(points_ptr + i1 * 2, mask=mask)
    p1y = tl.load(points_ptr + i1 * 2 + 1, mask=mask)
    p2x = tl.load(points_ptr + i2 * 2, mask=mask)
    p2y = tl.load(points_ptr + i2 * 2 + 1, mask=mask)
    p3x = tl.load(points_ptr + i3 * 2, mask=mask)
    p3y = tl.load(points_ptr + i3 * 2 + 1, mask=mask)

    # Bernstein basis polynomials
    c0 = mt * mt * mt
    c1 = 3.0 * mt * mt * t
    c2 = 3.0 * mt * t * t
    c3 = t * t * t

    ox = c0 * p0x + c1 * p1x + c2 * p2x + c3 * p3x
    oy = c0 * p0y + c1 * p1y + c2 * p2y + c3 * p3y

    tl.store(out_ptr + offs * 2, ox, mask=mask)
    tl.store(out_ptr + offs * 2 + 1, oy, mask=mask)


@triton.jit
def _flatten_bwd_kernel(
    grad_out_ptr,
    grad_points_ptr,
    N,
    S: tl.constexpr,
    total,
    BLOCK: tl.constexpr,
):
    """Backward: scatter-add grad_output into grad_points via atomic add."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    m = offs // S
    s = offs % S

    t = s.to(tl.float32) / S
    mt = 1.0 - t

    c0 = mt * mt * mt
    c1 = 3.0 * mt * mt * t
    c2 = 3.0 * mt * t * t
    c3 = t * t * t

    gx = tl.load(grad_out_ptr + offs * 2, mask=mask, other=0.0)
    gy = tl.load(grad_out_ptr + offs * 2 + 1, mask=mask, other=0.0)

    i0 = (m * 3) % N
    i1 = (m * 3 + 1) % N
    i2 = (m * 3 + 2) % N
    i3 = ((m + 1) * 3) % N

    # Atomic accumulate — multiple samples can map to the same point
    tl.atomic_add(grad_points_ptr + i0 * 2, c0 * gx, mask=mask)
    tl.atomic_add(grad_points_ptr + i0 * 2 + 1, c0 * gy, mask=mask)
    tl.atomic_add(grad_points_ptr + i1 * 2, c1 * gx, mask=mask)
    tl.atomic_add(grad_points_ptr + i1 * 2 + 1, c1 * gy, mask=mask)
    tl.atomic_add(grad_points_ptr + i2 * 2, c2 * gx, mask=mask)
    tl.atomic_add(grad_points_ptr + i2 * 2 + 1, c2 * gy, mask=mask)
    tl.atomic_add(grad_points_ptr + i3 * 2, c3 * gx, mask=mask)
    tl.atomic_add(grad_points_ptr + i3 * 2 + 1, c3 * gy, mask=mask)


class FlattenBezier(Function):
    """Flatten cubic Bezier segments into a polyline via Triton kernels."""

    @staticmethod
    def forward(ctx, points: Tensor, M: int, S: int, block: int = 256) -> Tensor:
        points = points.contiguous()
        N = points.shape[0]
        total = M * S
        out = torch.empty(total, 2, device=points.device, dtype=points.dtype)

        grid = ((total + block - 1) // block,)
        _flatten_fwd_kernel[grid](points, out, N, S, total, BLOCK=block)

        ctx.save_for_backward(points)
        ctx.N = N
        ctx.M = M
        ctx.S = S
        ctx.block = block
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, None, None, None]:
        (points,) = ctx.saved_tensors
        N, M, S, block = ctx.N, ctx.M, ctx.S, ctx.block
        total = M * S

        grad_points = torch.zeros_like(points)
        grad_output = grad_output.contiguous()

        grid = ((total + block - 1) // block,)
        _flatten_bwd_kernel[grid](grad_output, grad_points, N, S, total, BLOCK=block)

        return grad_points, None, None, None


def flatten_bezier(
    points: Tensor, M: int, samples_per_seg: int = 16, block: int = 256
) -> Tensor:
    """Flatten cubic Bezier path into a polyline (V, 2).

    Differentiable w.r.t. *points*.  *block* is the Triton block size.
    """
    return FlattenBezier.apply(points, M, samples_per_seg, block)
