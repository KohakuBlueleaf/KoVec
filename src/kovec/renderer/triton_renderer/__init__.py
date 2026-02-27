import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from kovec.renderer.base import VectorRenderer
from kovec.vector.types import VectorPath, VectorScene

from .coverage_kernel import soft_coverage
from .flatten_kernel import flatten_bezier


def _path_coverage(
    path: VectorPath,
    width: int,
    height: int,
    samples_per_seg: int,
    sigma: float,
    chunk_v: int = 32,
    block_p: int = 256,
    block_flatten: int = 256,
) -> Tensor:
    """Soft coverage mask (H, W) for a single path using Triton kernels."""
    M = len(path.num_control_points)
    poly = flatten_bezier(path.points, M, samples_per_seg, block=block_flatten)

    if len(poly) < 3:
        return torch.zeros(height, width, device=path.points.device)

    # Bbox culling — detach to avoid graph through min/max
    pad = sigma * 3.0
    with torch.no_grad():
        xs_min = poly[:, 0].min().item()
        xs_max = poly[:, 0].max().item()
        ys_min = poly[:, 1].min().item()
        ys_max = poly[:, 1].max().item()

    x0 = max(0, int(xs_min - pad))
    x1 = min(width, int(xs_max + pad) + 1)
    y0 = max(0, int(ys_min - pad))
    y1 = min(height, int(ys_max + pad) + 1)

    if x1 <= x0 or y1 <= y0:
        return torch.zeros(height, width, device=path.points.device)

    bh, bw = y1 - y0, x1 - x0
    dev = path.points.device
    xs = torch.arange(x0, x1, device=dev, dtype=torch.float32) + 0.5
    ys = torch.arange(y0, y1, device=dev, dtype=torch.float32) + 0.5
    px = xs.unsqueeze(0).expand(bh, -1)
    py = ys.unsqueeze(1).expand(-1, bw)
    pixels = torch.stack([px, py], dim=-1).reshape(-1, 2)

    cov = soft_coverage(pixels, poly, sigma, chunk_v=chunk_v, block_p=block_p).reshape(
        bh, bw
    )

    return F.pad(cov, (x0, width - x1, y0, height - y1))


def _path_to_svg_d(path: VectorPath) -> str:
    """Convert VectorPath to SVG ``d`` attribute string."""
    pts = path.points.detach().cpu()
    M = len(path.num_control_points)
    N = len(pts)

    p = pts[0]
    d = f"M {p[0]:.3f},{p[1]:.3f}"
    for i in range(M):
        c1 = pts[(i * 3 + 1) % N]
        c2 = pts[(i * 3 + 2) % N]
        a = pts[((i + 1) * 3) % N]
        d += (
            f" C {c1[0]:.3f},{c1[1]:.3f}"
            f" {c2[0]:.3f},{c2[1]:.3f}"
            f" {a[0]:.3f},{a[1]:.3f}"
        )

    if path.is_closed:
        d += " Z"
    return d


class TritonRenderer(VectorRenderer):
    """Triton-accelerated differentiable vector renderer.

    Drop-in replacement for ``TorchRenderer`` — same API, same output
    semantics, but the inner rasterisation loops run as Triton kernels
    (fwd + bwd) instead of pure-PyTorch loops.

    Parameters
    ----------
    device : torch.device
    samples_per_seg : int
        Polyline samples per cubic Bezier segment.
    sigma : float
        Anti-aliasing width in pixels (sigmoid steepness).
    chunk_v : int
        Number of polyline edges each GPU block processes before writing
        partial results.  Tune for occupancy vs. memory trade-off.
    block_p : int
        Triton block size for the pixel dimension in coverage kernels.
    block_flatten : int
        Triton block size for the flatten (Bezier → polyline) kernel.
    """

    def __init__(
        self,
        device: torch.device,
        samples_per_seg: int = 16,
        sigma: float = 1.0,
        chunk_v: int = 32,
        block_p: int = 256,
        block_flatten: int = 256,
    ) -> None:
        self.device = device
        self.samples_per_seg = samples_per_seg
        self.sigma = sigma
        self.chunk_v = chunk_v
        self.block_p = block_p
        self.block_flatten = block_flatten

    def render(self, scene: VectorScene, width: int, height: int) -> Tensor:
        """Render *scene* → (H, W, 4) RGBA float tensor (differentiable)."""
        canvas = torch.zeros(height, width, 4, device=self.device)

        for path, group in zip(scene.paths, scene.groups):
            if group.fill_color[3].item() < 1e-6:
                continue

            cov = _path_coverage(
                path,
                width,
                height,
                self.samples_per_seg,
                self.sigma,
                self.chunk_v,
                self.block_p,
                self.block_flatten,
            )
            color = group.fill_color  # (4,) RGBA
            alpha = cov * color[3]  # (H, W)

            # Porter-Duff "over" compositing
            src_rgb = color[:3].view(1, 1, 3) * alpha.unsqueeze(-1)
            dst_rgb = canvas[:, :, :3]
            dst_a = canvas[:, :, 3]

            out_rgb = src_rgb + dst_rgb * (1.0 - alpha).unsqueeze(-1)
            out_a = alpha + dst_a * (1.0 - alpha)
            canvas = torch.cat([out_rgb, out_a.unsqueeze(-1)], dim=-1)

        return canvas

    def save_svg(self, scene: VectorScene, path: str, width: int, height: int) -> None:
        """Write *scene* to an SVG file."""
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
        ]

        for vp, vg in tqdm(
            zip(scene.paths, scene.groups),
            total=len(scene),
            desc="Saving SVG",
            disable=True,
        ):
            d = _path_to_svg_d(vp)
            r, g, b, a = vg.fill_color.detach().cpu().tolist()
            ri, gi, bi = int(r * 255), int(g * 255), int(b * 255)
            lines.append(f'  <path d="{d}" fill="rgba({ri},{gi},{bi},{a:.4f})" />')

        lines.append("</svg>")
        with open(path, "w") as f:
            f.write("\n".join(lines))
