import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd import Function

# ---------------------------------------------------------------------------
# Forward: 2-D grid  (P_blocks, V_chunks)
# ---------------------------------------------------------------------------


@triton.jit
def _coverage_fwd_kernel(
    pixels_ptr,
    poly_ptr,
    part_cross_ptr,
    part_dsq_ptr,
    part_near_ptr,
    P,
    V,
    BLOCK_P: tl.constexpr,
    CHUNK_V: tl.constexpr,
):
    """2-D grid ``(ceil(P/BLOCK_P), ceil(V/CHUNK_V))``.

    Each program processes *BLOCK_P* pixels over *CHUNK_V* consecutive edges
    and writes partial results (crossing count, min squared distance, nearest
    edge index) to intermediate buffers laid out as ``(num_v_chunks, P)``.
    """
    pid_p = tl.program_id(0)
    pid_v = tl.program_id(1)

    p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    p_mask = p_offs < P

    px = tl.load(pixels_ptr + p_offs * 2, mask=p_mask, other=0.0)
    py = tl.load(pixels_ptr + p_offs * 2 + 1, mask=p_mask, other=0.0)

    v_start = pid_v * CHUNK_V

    crossings = tl.zeros([BLOCK_P], dtype=tl.int32)
    best_dsq = tl.full([BLOCK_P], 1e12, dtype=tl.float32)
    best_idx = tl.zeros([BLOCK_P], dtype=tl.int32)

    for i in range(CHUNK_V):
        e = v_start + i
        # Modulo keeps loads in-bounds; `e_mask` zeroes out-of-range results.
        e_safe = e % V
        e_next = (e_safe + 1) % V
        e_mask_i = (e < V).to(tl.int32)  # scalar 0/1
        e_mask_f = e_mask_i.to(tl.float32)  # scalar 0.0/1.0

        x1 = tl.load(poly_ptr + e_safe * 2)
        y1 = tl.load(poly_ptr + e_safe * 2 + 1)
        x2 = tl.load(poly_ptr + e_next * 2)
        y2 = tl.load(poly_ptr + e_next * 2 + 1)

        # --- even-odd crossing (horizontal ray to +x) ---
        cond_a = (y1 <= py) & (py < y2)
        cond_b = (y2 <= py) & (py < y1)
        cond = cond_a | cond_b
        t_val = (py - y1) / (y2 - y1 + 1e-10)
        x_int = x1 + t_val * (x2 - x1)
        crossing = cond & (x_int > px)
        crossings += crossing.to(tl.int32) * e_mask_i

        # --- squared distance to edge segment ---
        ex = x2 - x1
        ey = y2 - y1
        edge_len_sq = ex * ex + ey * ey

        to_px_x = px - x1
        to_px_y = py - y1
        proj = (to_px_x * ex + to_px_y * ey) / (edge_len_sq + 1e-8)
        proj = tl.minimum(tl.maximum(proj, 0.0), 1.0)

        closest_x = x1 + proj * ex
        closest_y = y1 + proj * ey
        dx = px - closest_x
        dy = py - closest_y
        dist_sq = dx * dx + dy * dy
        # Push invalid edges to infinity so they never win the min
        dist_sq = dist_sq + (1.0 - e_mask_f) * 1e12

        update = dist_sq < best_dsq
        best_dsq = tl.where(update, dist_sq, best_dsq)
        best_idx = tl.where(update, e_safe, best_idx)

    # Write partials — layout (num_v_chunks, P), row = pid_v
    out_base = pid_v * P
    tl.store(part_cross_ptr + out_base + p_offs, crossings, mask=p_mask)
    tl.store(part_dsq_ptr + out_base + p_offs, best_dsq, mask=p_mask)
    tl.store(part_near_ptr + out_base + p_offs, best_idx, mask=p_mask)


# ---------------------------------------------------------------------------
# Reduce across V-chunks  →  cov / nearest / inside
# ---------------------------------------------------------------------------


@triton.jit
def _coverage_reduce_kernel(
    part_cross_ptr,
    part_dsq_ptr,
    part_near_ptr,
    cov_ptr,
    nearest_ptr,
    inside_ptr,
    P,
    num_v_chunks,
    sigma,
    BLOCK_P: tl.constexpr,
):
    """1-D grid ``(ceil(P/BLOCK_P),)``.

    Reduces partial crossing counts (sum) and partial min-dist (min) across
    V-chunks, then applies sigmoid soft coverage.
    """
    pid = tl.program_id(0)
    p_offs = pid * BLOCK_P + tl.arange(0, BLOCK_P)
    p_mask = p_offs < P

    total_cross = tl.zeros([BLOCK_P], dtype=tl.int32)
    best_dsq = tl.full([BLOCK_P], 1e12, dtype=tl.float32)
    best_idx = tl.zeros([BLOCK_P], dtype=tl.int32)

    for c in range(num_v_chunks):
        offs = c * P + p_offs
        cross_c = tl.load(part_cross_ptr + offs, mask=p_mask, other=0)
        dsq_c = tl.load(part_dsq_ptr + offs, mask=p_mask, other=1e12)
        near_c = tl.load(part_near_ptr + offs, mask=p_mask, other=0)

        total_cross += cross_c
        update = dsq_c < best_dsq
        best_dsq = tl.where(update, dsq_c, best_dsq)
        best_idx = tl.where(update, near_c, best_idx)

    inside = (total_cross % 2) == 1
    dist = tl.sqrt(best_dsq + 1e-8)
    sign = tl.where(inside, 1.0, -1.0)
    cov = tl.sigmoid(sign * dist / sigma)

    tl.store(cov_ptr + p_offs, cov, mask=p_mask)
    tl.store(nearest_ptr + p_offs, best_idx, mask=p_mask)
    tl.store(inside_ptr + p_offs, inside.to(tl.int32), mask=p_mask)


# ---------------------------------------------------------------------------
# Backward: 1-D grid  (P_blocks,)  — only nearest edge per pixel
# ---------------------------------------------------------------------------


@triton.jit
def _coverage_bwd_kernel(
    grad_cov_ptr,
    cov_ptr,
    pixels_ptr,
    poly_ptr,
    nearest_ptr,
    inside_ptr,
    grad_poly_ptr,
    P,
    V,
    sigma,
    BLOCK_P: tl.constexpr,
):
    """Backward: only the nearest edge per pixel receives gradient.

    Gradient formula (from perpendicularity at the closest point)::

        d(dist_sq)/d(v1) = -2 (1 - proj) diff
        d(dist_sq)/d(v2) = -2 proj diff
    """
    pid = tl.program_id(0)
    p_offs = pid * BLOCK_P + tl.arange(0, BLOCK_P)
    p_mask = p_offs < P

    grad_cov = tl.load(grad_cov_ptr + p_offs, mask=p_mask, other=0.0)
    cov = tl.load(cov_ptr + p_offs, mask=p_mask, other=0.0)
    px = tl.load(pixels_ptr + p_offs * 2, mask=p_mask, other=0.0)
    py = tl.load(pixels_ptr + p_offs * 2 + 1, mask=p_mask, other=0.0)
    e_idx = tl.load(nearest_ptr + p_offs, mask=p_mask, other=0)
    inside_flag = tl.load(inside_ptr + p_offs, mask=p_mask, other=0)

    # Gather nearest-edge vertices
    e_next = (e_idx + 1) % V
    x1 = tl.load(poly_ptr + e_idx * 2, mask=p_mask, other=0.0)
    y1 = tl.load(poly_ptr + e_idx * 2 + 1, mask=p_mask, other=0.0)
    x2 = tl.load(poly_ptr + e_next * 2, mask=p_mask, other=0.0)
    y2 = tl.load(poly_ptr + e_next * 2 + 1, mask=p_mask, other=0.0)

    # Recompute projection
    ex = x2 - x1
    ey = y2 - y1
    edge_len_sq = ex * ex + ey * ey
    to_px_x = px - x1
    to_px_y = py - y1
    proj = (to_px_x * ex + to_px_y * ey) / (edge_len_sq + 1e-8)
    proj = tl.minimum(tl.maximum(proj, 0.0), 1.0)

    closest_x = x1 + proj * ex
    closest_y = y1 + proj * ey
    diff_x = px - closest_x
    diff_y = py - closest_y
    dist_sq = diff_x * diff_x + diff_y * diff_y
    dist = tl.sqrt(dist_sq + 1e-8)

    # Chain rule:  cov = sigmoid(sign * dist / sigma)
    sign = tl.where(inside_flag == 1, 1.0, -1.0)
    dcov_ddistsq = cov * (1.0 - cov) * sign / (2.0 * sigma * dist)
    scale = grad_cov * dcov_ddistsq

    gv1x = scale * (-2.0) * (1.0 - proj) * diff_x
    gv1y = scale * (-2.0) * (1.0 - proj) * diff_y
    gv2x = scale * (-2.0) * proj * diff_x
    gv2y = scale * (-2.0) * proj * diff_y

    tl.atomic_add(grad_poly_ptr + e_idx * 2, gv1x, mask=p_mask)
    tl.atomic_add(grad_poly_ptr + e_idx * 2 + 1, gv1y, mask=p_mask)
    tl.atomic_add(grad_poly_ptr + e_next * 2, gv2x, mask=p_mask)
    tl.atomic_add(grad_poly_ptr + e_next * 2 + 1, gv2y, mask=p_mask)


# ---------------------------------------------------------------------------
# autograd.Function wrapper
# ---------------------------------------------------------------------------


class SoftCoverage(Function):
    """Soft coverage mask from pixel positions and a polyline, via Triton.

    Uses a 2-D grid ``(P_blocks, V_chunks)`` in the forward pass for high
    GPU occupancy, followed by a cheap 1-D reduce across V-chunks.
    """

    @staticmethod
    def forward(
        ctx,
        pixels: Tensor,
        poly: Tensor,
        sigma: float,
        chunk_v: int = 32,
        block_p: int = 256,
    ) -> Tensor:
        pixels = pixels.contiguous()
        poly = poly.contiguous()
        P = pixels.shape[0]
        V = poly.shape[0]

        num_v_chunks = (V + chunk_v - 1) // chunk_v
        BLOCK_P = block_p

        # Partial buffers — layout (num_v_chunks, P)
        part_cross = torch.empty(
            num_v_chunks * P, device=pixels.device, dtype=torch.int32
        )
        part_dsq = torch.empty(
            num_v_chunks * P, device=pixels.device, dtype=torch.float32
        )
        part_near = torch.empty(
            num_v_chunks * P, device=pixels.device, dtype=torch.int32
        )

        grid_fwd = ((P + BLOCK_P - 1) // BLOCK_P, num_v_chunks)
        _coverage_fwd_kernel[grid_fwd](
            pixels,
            poly,
            part_cross,
            part_dsq,
            part_near,
            P,
            V,
            BLOCK_P=BLOCK_P,
            CHUNK_V=chunk_v,
        )

        # Final outputs
        cov = torch.empty(P, device=pixels.device, dtype=torch.float32)
        nearest = torch.empty(P, device=pixels.device, dtype=torch.int32)
        inside = torch.empty(P, device=pixels.device, dtype=torch.int32)

        grid_red = ((P + BLOCK_P - 1) // BLOCK_P,)
        _coverage_reduce_kernel[grid_red](
            part_cross,
            part_dsq,
            part_near,
            cov,
            nearest,
            inside,
            P,
            num_v_chunks,
            sigma,
            BLOCK_P=BLOCK_P,
        )
        # Partials no longer needed — let Python GC free them

        ctx.save_for_backward(pixels, poly, cov, nearest, inside)
        ctx.sigma = sigma
        ctx.V = V
        ctx.block_p = block_p
        return cov

    @staticmethod
    def backward(ctx, grad_cov: Tensor) -> tuple[None, Tensor | None, None, None, None]:
        pixels, poly, cov, nearest, inside = ctx.saved_tensors
        P = pixels.shape[0]
        V = ctx.V

        grad_poly = torch.zeros_like(poly)
        grad_cov = grad_cov.contiguous()

        BLOCK_P = ctx.block_p
        grid = ((P + BLOCK_P - 1) // BLOCK_P,)
        _coverage_bwd_kernel[grid](
            grad_cov,
            cov,
            pixels,
            poly,
            nearest,
            inside,
            grad_poly,
            P,
            V,
            ctx.sigma,
            BLOCK_P=BLOCK_P,
        )

        return None, grad_poly, None, None, None


def soft_coverage(
    pixels: Tensor,
    poly: Tensor,
    sigma: float = 1.0,
    chunk_v: int = 32,
    block_p: int = 256,
) -> Tensor:
    """Compute per-pixel soft coverage given pixel coords and a polyline.

    Differentiable w.r.t. *poly*.  *chunk_v* controls how many edges each
    GPU block processes before writing partial results.  *block_p* is the
    Triton block size for the pixel dimension.
    """
    return SoftCoverage.apply(pixels, poly, sigma, chunk_v, block_p)
