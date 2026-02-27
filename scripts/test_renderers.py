"""Renderer benchmark and correctness tests.

Compares TorchRenderer and TritonRenderer (when available) on:
  1. Speed & VRAM across scene complexities (combined 2x2 figure)
  2. Paths count sweep — varying path count at fixed resolution
  3. Canvas resolution sweep — varying resolution at fixed scene
  4. Per-kernel Triton parameter sweeps (CHUNK_V, BLOCK_P, BLOCK_FLATTEN)
  5. Forward accuracy (pixel-wise cross-reference)
  6. Gradient accuracy (cross-reference: triton vs torch autograd)
  7. Gradient verification (cosine-similarity finite-diff + topology note)
  8. Visual sanity check

OOM-safe: catches CUDA OOM, marks results as NaN, extrapolates for plots,
and annotates OOM points with ``x(OOM)`` on figures.

Usage::

    python scripts/test_renderers.py --device cuda --res 256 --save-dir renders/benchmark
"""

import argparse
import math
import time
from pathlib import Path

import torch
from torch import Tensor

from kovec.renderer.torch_renderer import TorchRenderer
from kovec.vector.types import VectorPath, VectorPathGroup, VectorScene

_HAS_TRITON = False
try:
    from kovec.renderer.triton_renderer import TritonRenderer
    from kovec.renderer.triton_renderer.coverage_kernel import soft_coverage
    from kovec.renderer.triton_renderer.flatten_kernel import flatten_bezier

    _HAS_TRITON = True
except Exception:
    pass

_HAS_PIL = False
try:
    from PIL import Image

    _HAS_PIL = True
except Exception:
    pass

_HAS_MPL = False
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    _HAS_MPL = True
except Exception:
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEP = "=" * 72
OOM_SENTINEL = float("nan")
DPI = 300

PATH_COUNTS = [1, 3, 5, 10, 20, 40, 80, 120, 160, 200]
SEG_COUNTS = [2, 4, 6, 8, 12, 16, 20, 24]
# Configs for correctness tests only (small enough to avoid OOM on both renderers)
ACCURACY_CONFIGS = [
    ("tiny", 1, 4),
    ("small", 3, 4),
    ("med", 10, 6),
    ("large", 20, 8),
]
CANVAS_SIZES = [64, 128, 256, 384, 512, 768, 1024]
CHUNK_VS = [8, 16, 32, 64, 128]
BLOCK_PS = [64, 128, 256, 512, 1024]
BLOCK_FLATS = [64, 128, 256, 512, 1024]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _header(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def _fmt(val: float, width: int = 16, fmt: str = ".3f") -> str:
    if math.isnan(val):
        return "OOM".rjust(width)
    return f"{val:{fmt}}".rjust(width)


def _cuda_cleanup() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def timed_safe(fn, device: torch.device, warmup: int = 5, iters: int = 20) -> float:
    """Return **median** execution time in ms, or NaN on OOM."""
    try:
        for _ in range(warmup):
            fn()

        if device.type == "cuda":
            torch.cuda.synchronize()
            times: list[float] = []
            for _ in range(iters):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                fn()
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
        else:
            times = []
            for _ in range(iters):
                t0 = time.perf_counter()
                fn()
                times.append((time.perf_counter() - t0) * 1000.0)

        times.sort()
        return times[len(times) // 2]
    except torch.cuda.OutOfMemoryError:
        _cuda_cleanup()
        return OOM_SENTINEL


def peak_vram_mb_safe(fn, device: torch.device) -> float:
    """Peak GPU memory delta in MiB, or NaN on OOM."""
    if device.type != "cuda":
        return 0.0
    try:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated(device)
        fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(device)
        return (peak - base) / (1024 * 1024)
    except torch.cuda.OutOfMemoryError:
        _cuda_cleanup()
        return OOM_SENTINEL


def extrapolate_nans(values: list[float], xs: list[float]) -> list[float]:
    """Fill trailing NaNs via linear extrapolation from last two valid points."""
    out = list(values)
    valid = [(x, v) for x, v in zip(xs, values) if not math.isnan(v)]
    if len(valid) < 2:
        return out
    for i, v in enumerate(out):
        if math.isnan(v):
            (x0, y0), (x1, y1) = valid[-2], valid[-1]
            slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0
            out[i] = max(0.0, y1 + slope * (xs[i] - x1))
    return out


def tensor_to_pil(img: Tensor) -> "Image.Image":
    arr = (img.detach().clamp(0, 1) * 255).byte().cpu().numpy()
    return Image.fromarray(arr, "RGBA")


def _bench_fwd_bwd_vram(rend, scene, W, H, device, warmup, iters):
    """Measure fwd time, fwd+bwd time, and VRAM for a (renderer, scene) pair.

    Returns ``(fwd_ms, bwd_ms, vram_mb)``, any of which may be NaN on OOM.
    """
    fwd_ms = timed_safe(
        lambda r=rend, sc=scene: r.render(sc, W, H), device, warmup, iters
    )
    if math.isnan(fwd_ms):
        return OOM_SENTINEL, OOM_SENTINEL, OOM_SENTINEL

    sc_b = scene.clone()
    params = sc_b.enable_gradients()
    all_p = [p for g in params.values() for p in g]

    def _fb(r=rend, s=sc_b, ps=all_p):
        for p in ps:
            if p.grad is not None:
                p.grad.zero_()
        r.render(s, W, H).sum().backward()

    bwd_ms = timed_safe(_fb, device, warmup, iters)
    sc_b.disable_gradients()

    sc_v = scene.clone()
    sc_v.enable_gradients()
    vram_mb = peak_vram_mb_safe(
        lambda r=rend, s=sc_v: r.render(s, W, H).sum().backward(), device
    )
    sc_v.disable_gradients()

    return fwd_ms, bwd_ms, vram_mb


# ---------------------------------------------------------------------------
# Scene generators
# ---------------------------------------------------------------------------


def make_random_scene(
    num_paths: int,
    segments_per_path: int,
    width: int,
    height: int,
    device: torch.device,
    seed: int = 42,
) -> VectorScene:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    scene = VectorScene()
    for i in range(num_paths):
        N = segments_per_path * 3
        pts = torch.rand(N, 2, generator=gen) * torch.tensor(
            [float(width), float(height)]
        )
        pts = pts.to(device)
        path = VectorPath(
            points=pts,
            num_control_points=torch.full(
                (segments_per_path,), 2, dtype=torch.int64, device=device
            ),
            stroke_width=torch.tensor(0.0, device=device),
        )
        c = torch.rand(4, generator=gen).to(device)
        c[3] = c[3].clamp(0.3, 1.0)
        group = VectorPathGroup(
            shape_idx=i,
            fill_color=c,
            stroke_color=torch.zeros(4, device=device),
        )
        scene.append(path, group)
    return scene


def make_circle_scene(
    cx: float, cy: float, r: float, device: torch.device
) -> VectorScene:
    k = 4.0 * (math.sqrt(2.0) - 1.0) / 3.0
    pts = torch.tensor(
        [
            [cx, cy - r],
            [cx + k * r, cy - r],
            [cx + r, cy - k * r],
            [cx + r, cy],
            [cx + r, cy + k * r],
            [cx + k * r, cy + r],
            [cx, cy + r],
            [cx - k * r, cy + r],
            [cx - r, cy + k * r],
            [cx - r, cy],
            [cx - r, cy - k * r],
            [cx - k * r, cy - r],
        ],
        device=device,
        dtype=torch.float32,
    )
    path = VectorPath(
        points=pts,
        num_control_points=torch.full((4,), 2, dtype=torch.int64, device=device),
        stroke_width=torch.tensor(0.0, device=device),
    )
    group = VectorPathGroup(
        shape_idx=0,
        fill_color=torch.tensor([1.0, 0.0, 0.0, 1.0], device=device),
        stroke_color=torch.zeros(4, device=device),
    )
    scene = VectorScene()
    scene.append(path, group)
    return scene


def make_multi_scene(device: torch.device, W: int, H: int) -> VectorScene:
    """Three overlapping circles (RGB) for visual sanity check."""
    scene = VectorScene()
    cx, cy = W / 2, H / 2
    r = min(W, H) * 0.25
    off = r * 0.5
    k = 4.0 * (math.sqrt(2.0) - 1.0) / 3.0
    colors = [(1, 0.2, 0.2, 0.8), (0.2, 1, 0.2, 0.8), (0.2, 0.2, 1, 0.8)]
    centres = [
        (cx, cy - off),
        (cx - off * 0.87, cy + off * 0.5),
        (cx + off * 0.87, cy + off * 0.5),
    ]
    for idx, ((ox, oy), col) in enumerate(zip(centres, colors)):
        pts = torch.tensor(
            [
                [ox, oy - r],
                [ox + k * r, oy - r],
                [ox + r, oy - k * r],
                [ox + r, oy],
                [ox + r, oy + k * r],
                [ox + k * r, oy + r],
                [ox, oy + r],
                [ox - k * r, oy + r],
                [ox - r, oy + k * r],
                [ox - r, oy],
                [ox - r, oy - k * r],
                [ox - k * r, oy - r],
            ],
            device=device,
            dtype=torch.float32,
        )
        path = VectorPath(
            points=pts,
            num_control_points=torch.full((4,), 2, dtype=torch.int64, device=device),
            stroke_width=torch.tensor(0.0, device=device),
        )
        group = VectorPathGroup(
            shape_idx=idx,
            fill_color=torch.tensor(col, device=device, dtype=torch.float32),
            stroke_color=torch.zeros(4, device=device),
        )
        scene.append(path, group)
    return scene


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def benchmark_paths_sweep(renderers, W, H, device, args):
    """Sweep path count at fixed resolution."""
    _header(f"PATHS SWEEP (res={W}, segs=6)")

    segs = 6
    path_counts = PATH_COUNTS

    hdr = f"{'#paths':>8}"
    for rn in renderers:
        hdr += f"  {'fwd_'+rn+'(ms)':>14} {'bwd_'+rn+'(ms)':>14} {rn+'(MiB)':>12}"
    print(hdr)
    print("-" * len(hdr))

    data: dict[str, dict[str, list[float]]] = {
        rn: {"fwd": [], "bwd": [], "vram": []} for rn in renderers
    }

    for np_ in path_counts:
        scene = make_random_scene(np_, segs, W, H, device)
        row = f"{np_:>8}"

        for rn, rend in renderers.items():
            f, b, v = _bench_fwd_bwd_vram(
                rend, scene, W, H, device, args.warmup, args.iters
            )
            data[rn]["fwd"].append(f)
            data[rn]["bwd"].append(b)
            data[rn]["vram"].append(v)
            row += f"  {_fmt(f, 14)} {_fmt(b, 14)} {_fmt(v, 12, '.2f')}"

        print(row)

    return data, path_counts


def benchmark_canvas_sweep(renderers, device, args):
    """Sweep canvas resolution at fixed scene complexity."""
    num_paths, segs = 10, 6
    _header(f"CANVAS SWEEP ({num_paths} paths x {segs} segs)")

    canvas_sizes = CANVAS_SIZES

    hdr = f"{'res':>8}"
    for rn in renderers:
        hdr += f"  {'fwd_'+rn+'(ms)':>14} {'bwd_'+rn+'(ms)':>14} {rn+'(MiB)':>12}"
    print(hdr)
    print("-" * len(hdr))

    data: dict[str, dict[str, list[float]]] = {
        rn: {"fwd": [], "bwd": [], "vram": []} for rn in renderers
    }

    for res in canvas_sizes:
        scene = make_random_scene(num_paths, segs, res, res, device)
        row = f"{res:>8}"

        for rn, rend in renderers.items():
            f, b, v = _bench_fwd_bwd_vram(
                rend, scene, res, res, device, args.warmup, args.iters
            )
            data[rn]["fwd"].append(f)
            data[rn]["bwd"].append(b)
            data[rn]["vram"].append(v)
            row += f"  {_fmt(f, 14)} {_fmt(b, 14)} {_fmt(v, 12, '.2f')}"

        print(row)

    return data, canvas_sizes


def benchmark_segs_sweep(renderers, W, H, device, args):
    """Sweep segments-per-path at fixed path count and resolution."""
    num_paths = 10
    _header(f"SEGS SWEEP ({num_paths} paths, res={W})")

    seg_counts = SEG_COUNTS

    hdr = f"{'segs':>8}"
    for rn in renderers:
        hdr += f"  {'fwd_'+rn+'(ms)':>14} {'bwd_'+rn+'(ms)':>14} {rn+'(MiB)':>12}"
    print(hdr)
    print("-" * len(hdr))

    data: dict[str, dict[str, list[float]]] = {
        rn: {"fwd": [], "bwd": [], "vram": []} for rn in renderers
    }

    for segs in seg_counts:
        scene = make_random_scene(num_paths, segs, W, H, device)
        row = f"{segs:>8}"

        for rn, rend in renderers.items():
            f, b, v = _bench_fwd_bwd_vram(
                rend, scene, W, H, device, args.warmup, args.iters
            )
            data[rn]["fwd"].append(f)
            data[rn]["bwd"].append(b)
            data[rn]["vram"].append(v)
            row += f"  {_fmt(f, 14)} {_fmt(b, 14)} {_fmt(v, 12, '.2f')}"

        print(row)

    return data, seg_counts


def benchmark_chunk_v(device, W, H, args):
    """Sweep CHUNK_V for TritonRenderer."""
    if not _HAS_TRITON or device.type != "cuda":
        return {}, []

    _header("CHUNK_V SWEEP (triton, 10 paths x 6 segs)")
    scene = make_random_scene(10, 6, W, H, device)

    print(f"{'chunk_v':>10} {'fwd(ms)':>12} {'bwd(ms)':>12} {'vram(MiB)':>12}")
    print("-" * 48)

    fwd_t: list[float] = []
    bwd_t: list[float] = []
    vram_v: list[float] = []

    for cv in CHUNK_VS:
        rend = TritonRenderer(device, samples_per_seg=16, sigma=1.0, chunk_v=cv)
        f, b, v = _bench_fwd_bwd_vram(
            rend, scene, W, H, device, args.warmup, args.iters
        )
        fwd_t.append(f)
        bwd_t.append(b)
        vram_v.append(v)
        print(f"{cv:>10} {_fmt(f, 12)} {_fmt(b, 12)} {_fmt(v, 12, '.2f')}")

    return {"fwd": fwd_t, "bwd": bwd_t, "vram": vram_v}, CHUNK_VS


# ---------------------------------------------------------------------------
# Per-kernel parameter sweeps
# ---------------------------------------------------------------------------


def _make_kernel_test_data(device, W, H, num_paths=10, segs=6, sps=16):
    """Create polyline + pixel grid for isolated kernel benchmarks."""
    scene = make_random_scene(num_paths, segs, W, H, device)
    path = scene.paths[0]
    M = len(path.num_control_points)
    points = path.points.clone().requires_grad_(True)
    poly = flatten_bezier(points, M, sps)

    xs = torch.arange(0, W, device=device, dtype=torch.float32) + 0.5
    ys = torch.arange(0, H, device=device, dtype=torch.float32) + 0.5
    px = xs.unsqueeze(0).expand(H, -1)
    py = ys.unsqueeze(1).expand(-1, W)
    pixels = torch.stack([px, py], dim=-1).reshape(-1, 2)

    return points, M, poly, pixels


def benchmark_block_p_coverage(device, W, H, args):
    """Sweep BLOCK_P for the coverage kernel."""
    if not _HAS_TRITON or device.type != "cuda":
        return {}, []

    _header("BLOCK_P SWEEP — coverage kernel (10 paths x 6 segs)")
    points, M, poly_base, pixels = _make_kernel_test_data(device, W, H)
    poly = poly_base.detach().requires_grad_(True)

    print(f"{'block_p':>10} {'cov_fwd(ms)':>14} {'cov_bwd(ms)':>14} {'vram(MiB)':>14}")
    print("-" * 54)

    fwd_t: list[float] = []
    bwd_t: list[float] = []
    vram_v: list[float] = []

    for bp in BLOCK_PS:
        fwd_ms = timed_safe(
            lambda b=bp: soft_coverage(
                pixels, poly.detach(), 1.0, chunk_v=32, block_p=b
            ),
            device,
            args.warmup,
            args.iters,
        )

        if math.isnan(fwd_ms):
            fwd_t.append(OOM_SENTINEL)
            bwd_t.append(OOM_SENTINEL)
            vram_v.append(OOM_SENTINEL)
            print(f"{bp:>10} {'OOM':>14} {'OOM':>14} {'OOM':>14}")
            continue

        def _fb(b=bp):
            p = poly.detach().requires_grad_(True)
            soft_coverage(pixels, p, 1.0, chunk_v=32, block_p=b).sum().backward()

        bwd_ms = timed_safe(_fb, device, args.warmup, args.iters)
        mb = peak_vram_mb_safe(lambda b=bp: _fb(b), device)

        fwd_t.append(fwd_ms)
        bwd_t.append(bwd_ms)
        vram_v.append(mb)
        print(f"{bp:>10} {_fmt(fwd_ms, 14)} {_fmt(bwd_ms, 14)} {_fmt(mb, 14, '.2f')}")

    return {"fwd": fwd_t, "bwd": bwd_t, "vram": vram_v}, BLOCK_PS


def benchmark_block_flatten(device, W, H, args):
    """Sweep BLOCK for the flatten kernel."""
    if not _HAS_TRITON or device.type != "cuda":
        return {}, []

    _header("BLOCK SWEEP — flatten kernel (10 paths x 6 segs)")
    scene = make_random_scene(10, 6, W, H, device)
    path = scene.paths[0]
    M = len(path.num_control_points)

    print(f"{'block':>10} {'flat_fwd(ms)':>14} {'flat_bwd(ms)':>14} {'vram(MiB)':>14}")
    print("-" * 54)

    fwd_t: list[float] = []
    bwd_t: list[float] = []
    vram_v: list[float] = []

    for blk in BLOCK_FLATS:
        pts_det = path.points.detach()
        fwd_ms = timed_safe(
            lambda b=blk: flatten_bezier(pts_det, M, 16, block=b),
            device,
            args.warmup,
            args.iters,
        )

        if math.isnan(fwd_ms):
            fwd_t.append(OOM_SENTINEL)
            bwd_t.append(OOM_SENTINEL)
            vram_v.append(OOM_SENTINEL)
            print(f"{blk:>10} {'OOM':>14} {'OOM':>14} {'OOM':>14}")
            continue

        def _fb(b=blk):
            p = path.points.detach().requires_grad_(True)
            flatten_bezier(p, M, 16, block=b).sum().backward()

        bwd_ms = timed_safe(_fb, device, args.warmup, args.iters)
        mb = peak_vram_mb_safe(lambda b=blk: _fb(b), device)

        fwd_t.append(fwd_ms)
        bwd_t.append(bwd_ms)
        vram_v.append(mb)
        print(f"{blk:>10} {_fmt(fwd_ms, 14)} {_fmt(bwd_ms, 14)} {_fmt(mb, 14, '.2f')}")

    return {"fwd": fwd_t, "bwd": bwd_t, "vram": vram_v}, BLOCK_FLATS


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


def test_accuracy(renderers, configs, W, H, device):
    if len(renderers) < 2:
        return
    names = list(renderers.keys())
    _header(f"FORWARD ACCURACY: {names[0]} vs {names[1]}")

    print(f"{'Config':<10} {'max_abs':>14} {'mean_abs':>14} {'rmse':>14}")
    print("-" * 54)

    ra, rb = renderers[names[0]], renderers[names[1]]
    for label, num_paths, segs in configs:
        scene = make_random_scene(num_paths, segs, W, H, device)
        try:
            with torch.no_grad():
                oa = ra.render(scene, W, H)
                ob = rb.render(scene, W, H)
            d = (oa - ob).abs()
            print(
                f"{label:<10}"
                f"{d.max().item():>14.6e}"
                f"{d.mean().item():>14.6e}"
                f"{d.pow(2).mean().sqrt().item():>14.6e}"
            )
        except torch.cuda.OutOfMemoryError:
            _cuda_cleanup()
            print(f"{label:<10}{'OOM':>14}{'OOM':>14}{'OOM':>14}")


def test_gradient_accuracy(renderers, configs, W, H, device):
    """Cross-reference gradient: triton autograd vs torch autograd.

    This is the PRIMARY gradient validation — torch autograd is the reference.
    """
    if len(renderers) < 2:
        return
    names = list(renderers.keys())
    _header(f"GRADIENT ACCURACY: {names[0]} vs {names[1]} (torch autograd = reference)")

    print(
        f"{'Config':<10} {'pt_max':>14} {'pt_mean':>14}"
        f" {'col_max':>14} {'col_mean':>14}"
    )
    print("-" * 68)

    ra, rb = renderers[names[0]], renderers[names[1]]
    for label, num_paths, segs in configs:
        scene = make_random_scene(num_paths, segs, W, H, device)
        try:
            sa = scene.clone()
            sa.enable_gradients()
            ra.render(sa, W, H).sum().backward()

            sb = scene.clone()
            sb.enable_gradients()
            rb.render(sb, W, H).sum().backward()

            pt_d = [
                (a.points.grad - b.points.grad).abs()
                for a, b in zip(sa.paths, sb.paths)
                if a.points.grad is not None and b.points.grad is not None
            ]
            col_d = [
                (a.fill_color.grad - b.fill_color.grad).abs()
                for a, b in zip(sa.groups, sb.groups)
                if a.fill_color.grad is not None and b.fill_color.grad is not None
            ]

            if pt_d:
                ap = torch.cat([x.flatten() for x in pt_d])
                pm, pmn = ap.max().item(), ap.mean().item()
            else:
                pm = pmn = float("nan")
            if col_d:
                ac = torch.cat([x.flatten() for x in col_d])
                cm, cmn = ac.max().item(), ac.mean().item()
            else:
                cm = cmn = float("nan")

            print(f"{label:<10}{pm:>14.6e}{pmn:>14.6e}{cm:>14.6e}{cmn:>14.6e}")
        except torch.cuda.OutOfMemoryError:
            _cuda_cleanup()
            print(f"{label:<10}{'OOM':>14}{'OOM':>14}{'OOM':>14}{'OOM':>14}")


def test_gradient_check(renderers, W, H, device):
    """Finite-difference sanity check with cosine similarity.

    The even-odd fill rule runs under ``torch.no_grad()``, so topology
    changes (inside/outside flips) are invisible to the analytical gradient.
    Finite perturbations can trigger these flips, causing disagreement.
    We therefore use **cosine similarity** (robust to sporadic outliers) as
    the primary metric and report mean relative error only on significant
    entries.
    """
    _header("GRADIENT VERIFICATION (finite-diff, cosine similarity)")
    print(
        "  Note: even-odd topology is detached — finite-diff can disagree near\n"
        "        the boundary. Cosine similarity is the robust metric here.\n"
    )

    scene = make_circle_scene(W / 2, H / 2, W / 4, device)
    eps = 5e-3  # small enough to minimize topology flips

    for rname, rend in renderers.items():
        try:
            # Analytical gradient
            sa = scene.clone()
            sa.enable_gradients(points=True, colors=False)
            rend.render(sa, W, H).sum().backward()
            ag = sa.paths[0].points.grad.clone().flatten()

            # Finite-difference gradient (no_grad — graphs discarded immediately)
            n_pts = sa.paths[0].points.shape[0]
            fd_grads: list[float] = []
            with torch.no_grad():
                for i in range(n_pts):
                    for j in range(2):
                        sp = scene.clone()
                        sp.paths[0].points[i, j] += eps
                        lp = rend.render(sp, W, H).sum().item()

                        sm = scene.clone()
                        sm.paths[0].points[i, j] -= eps
                        lm = rend.render(sm, W, H).sum().item()

                        fd_grads.append((lp - lm) / (2.0 * eps))

            fd = torch.tensor(fd_grads, device=ag.device)

            # Cosine similarity (immune to scaling / near-zero outliers)
            cos_sim = torch.nn.functional.cosine_similarity(
                ag.unsqueeze(0).float(), fd.unsqueeze(0).float()
            ).item()

            # Mean relative error only on entries with meaningful gradient
            significant = (ag.abs() > 0.01) | (fd.abs() > 0.01)
            if significant.any():
                diff = (ag[significant] - fd[significant]).abs()
                scale = torch.maximum(ag[significant].abs(), fd[significant].abs())
                mean_rel = (diff / scale.clamp(min=1e-6)).mean().item()
            else:
                mean_rel = 0.0

            status = "PASS" if cos_sim > 0.85 else "WARN"
            print(
                f"  {rname:<10} cos_sim={cos_sim:.4f}  mean_rel={mean_rel:.4e}"
                f"  ({n_pts * 2} params, {significant.sum().item()} significant)"
                f"  [{status}]"
            )
        except torch.cuda.OutOfMemoryError:
            _cuda_cleanup()
            print(f"  {rname:<10} OOM")


def test_visual(renderers, W, H, device, save_dir):
    _header("VISUAL SANITY CHECK")
    scene = make_multi_scene(device, W, H)

    for rname, rend in renderers.items():
        try:
            with torch.no_grad():
                out = rend.render(scene, W, H)
            rgb, alpha = out[:, :, :3], out[:, :, 3]
            fill = (alpha > 0.01).float().mean().item() * 100
            print(
                f"  {rname:<10} "
                f"rgb=[{rgb.min().item():.3f}, {rgb.max().item():.3f}]  "
                f"alpha=[{alpha.min().item():.3f}, {alpha.max().item():.3f}]  "
                f"fill={fill:.1f}%"
            )
            if _HAS_PIL and save_dir:
                d = Path(save_dir)
                d.mkdir(parents=True, exist_ok=True)
                tensor_to_pil(out).save(str(d / f"visual_{rname}.png"))
                print(f"           -> saved {d / f'visual_{rname}.png'}")
        except torch.cuda.OutOfMemoryError:
            _cuda_cleanup()
            print(f"  {rname:<10} OOM")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_line(ax, x, y, xs_float, label, marker, linestyle="-"):
    """Plot valid points solid; OOM points as dashed extrapolation with x(OOM)."""
    valid_x = [xi for xi, yi in zip(x, y) if not math.isnan(yi)]
    valid_y = [yi for yi in y if not math.isnan(yi)]
    if not valid_x:
        return
    line = ax.plot(valid_x, valid_y, marker=marker, label=label, linestyle=linestyle)
    color = line[0].get_color()

    extrap = extrapolate_nans(y, xs_float)
    oom_x = [xi for xi, yi in zip(x, y) if math.isnan(yi)]
    oom_y = [ei for yi, ei in zip(y, extrap) if math.isnan(yi)]
    if oom_x and valid_x:
        bridge_x = [valid_x[-1]] + oom_x
        bridge_y = [valid_y[-1]] + oom_y
        ax.plot(bridge_x, bridge_y, marker="x", linestyle="--", alpha=0.4, color=color)
        for xi, yi in zip(oom_x, oom_y):
            ax.annotate(
                "x(OOM)",
                (xi, yi),
                fontsize=7,
                alpha=0.7,
                ha="center",
                va="bottom",
                color=color,
                fontweight="bold",
            )


def _explicit_xticks(ax, values):
    """Set explicit numeric tick labels (no 2^k scientific notation)."""
    ax.set_xticks(values)
    ax.set_xticklabels([str(v) for v in values])
    ax.minorticks_off()


def _plot_speedup_bars(ax, labels, data_a, data_b, phase="fwd"):
    """Bar chart of speedup = data_a[phase] / data_b[phase]."""
    su = []
    valid_labels = []
    for i, (a, b) in enumerate(zip(data_a[phase], data_b[phase])):
        if not math.isnan(a) and not math.isnan(b) and b > 0:
            su.append(a / b)
            valid_labels.append(
                labels[i] if isinstance(labels[i], str) else str(labels[i])
            )
    if not su:
        ax.text(
            0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes
        )
        return
    colors = ["#00b894" if s > 1 else "#d63031" for s in su]
    ax.bar(range(len(su)), su, color=colors)
    ax.set_xticks(range(len(su)), valid_labels, rotation=30, fontsize=7)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Speedup (torch / triton)")
    ax.legend(
        handles=[
            Patch(color="#00b894", label="triton faster"),
            Patch(color="#d63031", label="torch faster"),
        ],
        fontsize=7,
    )


def _plot_diff_bar(ax, labels, vals_a, vals_b, name_a, name_b, ylabel):
    """Bar chart: vals_a - vals_b  (positive = A worse)."""
    diffs = []
    valid_labels = []
    for i, (a, b) in enumerate(zip(vals_a, vals_b)):
        if not math.isnan(a) and not math.isnan(b):
            diffs.append(a - b)
            lbl = labels[i] if isinstance(labels[i], str) else str(labels[i])
            valid_labels.append(lbl)
    if not diffs:
        return
    colors = ["#d63031" if d > 0 else "#00b894" for d in diffs]
    ax.bar(range(len(diffs)), diffs, color=colors)
    ax.set_xticks(range(len(diffs)), valid_labels, rotation=30, fontsize=7)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel(ylabel)
    ax.legend(
        handles=[
            Patch(color="#d63031", label=f"{name_a} worse"),
            Patch(color="#00b894", label=f"{name_b} worse"),
        ],
        fontsize=7,
    )
    ax.grid(True, alpha=0.3)


def _plot_sweep_quad(fig_path, x_vals, data, x_label, title_prefix):
    """2x2 figure: fwd time, bwd time, speedup, VRAM — both renderers."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    xs = [float(v) for v in x_vals]
    str_labels = [str(v) for v in x_vals]
    has_both = "torch" in data and "triton" in data

    # (0,0) Forward time
    ax = axes[0, 0]
    for rn, rd in data.items():
        _plot_line(ax, x_vals, rd["fwd"], xs, rn, "o")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Forward time (ms)")
    ax.set_title(f"{title_prefix} — Forward Speed")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) Backward time
    ax = axes[0, 1]
    for rn, rd in data.items():
        _plot_line(ax, x_vals, rd["bwd"], xs, rn, "s")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Fwd+Bwd time (ms)")
    ax.set_title(f"{title_prefix} — Fwd+Bwd Speed")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0) Speedup
    ax = axes[1, 0]
    if has_both:
        _plot_speedup_bars(ax, x_vals, data["torch"], data["triton"], "bwd")
        ax.set_title(f"{title_prefix} — Fwd+Bwd Speedup")
    else:
        ax.text(
            0.5,
            0.5,
            "Single renderer",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.grid(True, alpha=0.3)

    # (1,1) VRAM
    ax = axes[1, 1]
    for rn, rd in data.items():
        _plot_line(ax, x_vals, rd["vram"], xs, rn, "^")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Peak VRAM delta (MiB)")
    ax.set_title(f"{title_prefix} — VRAM")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title_prefix, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(fig_path), dpi=DPI)
    plt.close(fig)


def _plot_kernel_sweep(ax_time, ax_vram, param_vals, data, param_name, kernel_name):
    """Plot kernel sweep: time (fwd/bwd) and VRAM vs parameter."""
    xs = [float(v) for v in param_vals]
    _plot_line(ax_time, param_vals, data["fwd"], xs, "forward", "o")
    _plot_line(ax_time, param_vals, data["bwd"], xs, "fwd+bwd", "s")
    ax_time.set_xlabel(param_name)
    ax_time.set_ylabel("Time (ms)")
    ax_time.set_title(f"{kernel_name}: {param_name} vs Speed")
    _explicit_xticks(ax_time, param_vals)
    ax_time.legend(fontsize=8)
    ax_time.grid(True, alpha=0.3)

    _plot_line(ax_vram, param_vals, data["vram"], xs, "vram", "^")
    ax_vram.set_xlabel(param_name)
    ax_vram.set_ylabel("Peak VRAM delta (MiB)")
    ax_vram.set_title(f"{kernel_name}: {param_name} vs VRAM")
    _explicit_xticks(ax_vram, param_vals)
    ax_vram.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# save_plots
# ---------------------------------------------------------------------------


def save_plots(
    paths_data,
    path_counts,
    canvas_data,
    canvas_sizes,
    segs_data,
    seg_counts,
    chunk_data,
    chunk_vs,
    block_p_data,
    block_ps,
    block_flat_data,
    block_flats,
    save_dir,
):
    if not _HAS_MPL:
        print("\n  (matplotlib not installed -- skipping plots)")
        return
    if not save_dir:
        return

    d = Path(save_dir)
    d.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # 1. paths_sweep.png — 2x2: fwd, bwd, speedup, VRAM
    # ---------------------------------------------------------------
    if paths_data and path_counts:
        _plot_sweep_quad(
            d / "paths_sweep.png",
            path_counts,
            paths_data,
            "# paths",
            "Paths Count Sweep",
        )

    # ---------------------------------------------------------------
    # 2. canvas_sweep.png — 2x2
    # ---------------------------------------------------------------
    if canvas_data and canvas_sizes:
        _plot_sweep_quad(
            d / "canvas_sweep.png",
            canvas_sizes,
            canvas_data,
            "Resolution",
            "Canvas Size Sweep",
        )

    # ---------------------------------------------------------------
    # 3. segs_sweep.png — 2x2
    # ---------------------------------------------------------------
    if segs_data and seg_counts:
        _plot_sweep_quad(
            d / "segs_sweep.png",
            seg_counts,
            segs_data,
            "Segments per path",
            "Segments Sweep",
        )

    # ---------------------------------------------------------------
    # 4. diff_comparison.png — time + VRAM diffs from paths sweep
    # ---------------------------------------------------------------
    if paths_data and path_counts and "torch" in paths_data and "triton" in paths_data:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
        _plot_diff_bar(
            ax1,
            path_counts,
            paths_data["torch"]["fwd"],
            paths_data["triton"]["fwd"],
            "torch",
            "triton",
            "Fwd time diff (ms)",
        )
        ax1.set_title("Forward Time Diff")

        _plot_diff_bar(
            ax2,
            path_counts,
            paths_data["torch"]["bwd"],
            paths_data["triton"]["bwd"],
            "torch",
            "triton",
            "Bwd time diff (ms)",
        )
        ax2.set_title("Fwd+Bwd Time Diff")

        _plot_diff_bar(
            ax3,
            path_counts,
            paths_data["torch"]["vram"],
            paths_data["triton"]["vram"],
            "torch",
            "triton",
            "VRAM diff (MiB)",
        )
        ax3.set_title("VRAM Diff")

        fig.suptitle(
            "Torch vs Triton Differences (positive = torch worse)",
            fontsize=12,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(str(d / "diff_comparison.png"), dpi=DPI)
        plt.close(fig)

    # ---------------------------------------------------------------
    # 4. triton_sweeps.png — 3x2: chunk_v, block_p, block_flatten
    # ---------------------------------------------------------------
    has_chunk = bool(chunk_data) and bool(chunk_vs)
    has_bp = bool(block_p_data) and bool(block_ps)
    has_bf = bool(block_flat_data) and bool(block_flats)
    n_rows = sum([has_chunk, has_bp, has_bf])

    if n_rows > 0:
        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4.5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, 2)

        row = 0
        if has_chunk:
            _plot_kernel_sweep(
                axes[row, 0], axes[row, 1], chunk_vs, chunk_data, "CHUNK_V", "Coverage"
            )
            row += 1
        if has_bp:
            _plot_kernel_sweep(
                axes[row, 0],
                axes[row, 1],
                block_ps,
                block_p_data,
                "BLOCK_P",
                "Coverage",
            )
            row += 1
        if has_bf:
            _plot_kernel_sweep(
                axes[row, 0],
                axes[row, 1],
                block_flats,
                block_flat_data,
                "BLOCK",
                "Flatten",
            )
            row += 1

        fig.suptitle("Triton Kernel Parameter Sweeps", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(str(d / "triton_sweeps.png"), dpi=DPI)
        plt.close(fig)

    print(f"\n  Plots saved to {d}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Renderer benchmark & tests")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--res", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="renders/benchmark")
    parser.add_argument(
        "--vram-fraction",
        type=float,
        default=0.9,
        help="Fraction of dedicated VRAM PyTorch may use (0.0-1.0). "
        "Prevents spilling into Windows shared GPU memory so OOM "
        "triggers at the real VRAM limit. Set to 1.0 to disable.",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("! CUDA unavailable, falling back to CPU (triton skipped)")
        args.device = "cpu"

    device = torch.device(args.device)
    W = H = args.res

    print(f"Device : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        dedicated_mb = props.total_memory / (1024 * 1024)
        print(f"GPU    : {props.name}")
        print(f"VRAM   : {dedicated_mb:.0f} MiB dedicated")

        # Cap PyTorch allocator to dedicated VRAM only — prevents Windows
        # shared GPU memory (system RAM) from silently absorbing overflows.
        if 0.0 < args.vram_fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(
                args.vram_fraction, torch.cuda.current_device()
            )
            usable_mb = dedicated_mb * args.vram_fraction
            print(
                f"Limit  : {usable_mb:.0f} MiB ({args.vram_fraction:.0%} of dedicated)"
            )
        elif args.vram_fraction >= 1.0:
            print("Limit  : none (shared GPU memory may be used)")
    print(f"Canvas : {W}x{H}  |  Warmup: {args.warmup}  Iters: {args.iters}")

    # --- Renderers ---
    renderers: dict = {"torch": TorchRenderer(device, 16, 1.0)}
    if _HAS_TRITON and device.type == "cuda":
        renderers["triton"] = TritonRenderer(device, 16, 1.0, chunk_v=32)
    print(f"Renderers: {', '.join(renderers)}")

    # --- Benchmarks ---
    paths_data, path_counts = benchmark_paths_sweep(renderers, W, H, device, args)
    canvas_data, canvas_sizes = benchmark_canvas_sweep(renderers, device, args)
    segs_data, seg_counts = benchmark_segs_sweep(renderers, W, H, device, args)
    chunk_data, chunk_vs = benchmark_chunk_v(device, W, H, args)
    block_p_data, block_ps = benchmark_block_p_coverage(device, W, H, args)
    block_flat_data, block_flats = benchmark_block_flatten(device, W, H, args)

    # --- Correctness ---
    test_accuracy(renderers, ACCURACY_CONFIGS, W, H, device)
    test_gradient_accuracy(renderers, ACCURACY_CONFIGS, W, H, device)
    test_gradient_check(renderers, W, H, device)
    test_visual(renderers, W, H, device, args.save_dir)

    # --- Plots ---
    save_plots(
        paths_data,
        path_counts,
        canvas_data,
        canvas_sizes,
        segs_data,
        seg_counts,
        chunk_data,
        chunk_vs,
        block_p_data,
        block_ps,
        block_flat_data,
        block_flats,
        args.save_dir,
    )

    print(f"\n{SEP}\n  ALL DONE\n{SEP}")


if __name__ == "__main__":
    main()
