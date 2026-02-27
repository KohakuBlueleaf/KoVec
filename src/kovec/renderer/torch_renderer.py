import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from kovec.renderer.base import VectorRenderer
from kovec.vector.types import VectorPath, VectorScene


def _flatten_path(path: VectorPath, samples_per_seg: int = 16) -> Tensor:
    """Sample cubic Bezier segments into a polyline. Returns (V, 2).

    Differentiable w.r.t. ``path.points``.
    Layout assumes [anchor, ctrl1, ctrl2, anchor, ctrl1, ctrl2, ...] with
    ``num_control_points`` all equal to 2 (cubic).
    """
    points = path.points  # (N, 2)
    M = len(path.num_control_points)
    N = len(points)

    idx = torch.arange(M, device=points.device)
    p0 = points[(idx * 3) % N]
    p1 = points[(idx * 3 + 1) % N]
    p2 = points[(idx * 3 + 2) % N]
    p3 = points[((idx + 1) * 3) % N]

    t = torch.linspace(0, 1, samples_per_seg + 1, device=points.device)[:-1]
    t = t.reshape(1, -1, 1)  # (1, S, 1)
    mt = 1.0 - t

    sampled = (
        mt**3 * p0[:, None, :]
        + 3.0 * mt**2 * t * p1[:, None, :]
        + 3.0 * mt * t**2 * p2[:, None, :]
        + t**3 * p3[:, None, :]
    )  # (M, S, 2)
    return sampled.reshape(-1, 2)


def _even_odd(pixels: Tensor, poly: Tensor, chunk: int = 512) -> Tensor:
    """Even-odd fill rule. Returns (P,) bool.

    No gradient is needed here — called inside ``torch.no_grad()``.
    """
    V = len(poly)
    px = pixels[:, 0:1]  # (P, 1)
    py = pixels[:, 1:2]

    v1 = poly
    v2 = poly.roll(-1, 0)

    crossings = torch.zeros(len(pixels), device=pixels.device)

    for s in range(0, V, chunk):
        e = min(s + chunk, V)
        y1 = v1[s:e, 1:2].t()  # (1, C)
        y2 = v2[s:e, 1:2].t()
        x1 = v1[s:e, 0:1].t()
        x2 = v2[s:e, 0:1].t()

        cond = ((y1 <= py) & (py < y2)) | ((y2 <= py) & (py < y1))
        t = (py - y1) / (y2 - y1 + 1e-10)
        x_int = x1 + t * (x2 - x1)
        crossings += (cond & (x_int > px)).float().sum(dim=1)

    return (crossings % 2) > 0.5


def _min_edge_dist_sq(pixels: Tensor, poly: Tensor, chunk: int = 512) -> Tensor:
    """Squared distance from each pixel to the nearest polygon edge. (P,).

    Differentiable w.r.t. ``poly``.
    """
    V = len(poly)
    v1 = poly
    v2 = poly.roll(-1, 0)

    best = torch.full((len(pixels),), float("inf"), device=pixels.device)

    for s in range(0, V, chunk):
        e = min(s + chunk, V)
        edge = v2[s:e] - v1[s:e]  # (C, 2)
        edge_len_sq = (edge * edge).sum(-1)  # (C,)

        to_px = pixels[:, None, :] - v1[None, s:e, :]  # (P, C, 2)
        proj = (to_px * edge[None]).sum(-1) / (edge_len_sq[None] + 1e-8)
        proj = proj.clamp(0.0, 1.0)

        closest = v1[None, s:e, :] + proj[:, :, None] * edge[None]  # (P, C, 2)
        dsq = ((pixels[:, None, :] - closest) ** 2).sum(-1)  # (P, C)

        best = torch.minimum(best, dsq.min(dim=1).values)

    return best


def _path_coverage(
    path: VectorPath,
    width: int,
    height: int,
    samples_per_seg: int,
    sigma: float,
) -> Tensor:
    """Soft coverage mask (H, W) for a single path. Differentiable."""
    poly = _flatten_path(path, samples_per_seg)

    if len(poly) < 3:
        return torch.zeros(height, width, device=path.points.device)

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
    pixels = torch.stack([px, py], dim=-1).reshape(-1, 2)  # (P, 2) as (x, y)

    # Inside / outside (no grad — topology is fixed per step)
    with torch.no_grad():
        inside = _even_odd(pixels, poly)

    # Signed distance (differentiable)
    dist = (_min_edge_dist_sq(pixels, poly) + 1e-8).sqrt()
    sign = torch.where(inside, 1.0, -1.0)
    cov = torch.sigmoid(sign * dist / sigma).reshape(bh, bw)

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
        d += f" C {c1[0]:.3f},{c1[1]:.3f} {c2[0]:.3f},{c2[1]:.3f} {a[0]:.3f},{a[1]:.3f}"

    if path.is_closed:
        d += " Z"
    return d


class TorchRenderer(VectorRenderer):
    """Pure-PyTorch differentiable vector renderer.

    Uses bbox-culled soft rasterisation: cubic Bezier → polyline → even-odd
    fill + sigmoid soft edges. Gradients flow through control points and
    fill colours.

    Parameters
    ----------
    device : torch.device
    samples_per_seg : int
        Number of polyline samples per cubic Bezier segment.
    sigma : float
        Anti-aliasing width in pixels (controls sigmoid steepness).
    """

    def __init__(
        self,
        device: torch.device,
        samples_per_seg: int = 16,
        sigma: float = 1.0,
    ) -> None:
        self.device = device
        self.samples_per_seg = samples_per_seg
        self.sigma = sigma

    def render(self, scene: VectorScene, width: int, height: int) -> Tensor:
        """Render *scene* → (H, W, 4) RGBA float tensor (differentiable)."""
        canvas = torch.zeros(height, width, 4, device=self.device)

        for path, group in zip(scene.paths, scene.groups):
            if group.fill_color[3].item() < 1e-6:
                continue

            cov = _path_coverage(path, width, height, self.samples_per_seg, self.sigma)
            color = group.fill_color  # (4,) RGBA
            alpha = cov * color[3]  # (H, W)

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
