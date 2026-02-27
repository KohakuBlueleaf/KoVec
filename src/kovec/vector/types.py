import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor


class VectorPath(BaseModel):
    """Renderer-agnostic closed Bezier path."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    points: Tensor  # (N, 2) float32
    num_control_points: Tensor  # (M,) int — 2 for cubic segments
    stroke_width: Tensor  # scalar
    is_closed: bool = True

    def clone(self) -> "VectorPath":
        return VectorPath(
            points=self.points.clone(),
            num_control_points=self.num_control_points.clone(),
            stroke_width=self.stroke_width.clone(),
            is_closed=self.is_closed,
        )

    def to(self, device: torch.device) -> "VectorPath":
        self.points = self.points.to(device)
        self.num_control_points = self.num_control_points.to(device)
        self.stroke_width = self.stroke_width.to(device)
        return self


class VectorPathGroup(BaseModel):
    """Associates a path index with fill / stroke colors (RGBA)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    shape_idx: int
    fill_color: Tensor  # (4,) float32 RGBA
    stroke_color: Tensor  # (4,) float32 RGBA

    def clone(self) -> "VectorPathGroup":
        return VectorPathGroup(
            shape_idx=self.shape_idx,
            fill_color=self.fill_color.clone(),
            stroke_color=self.stroke_color.clone(),
        )

    def to(self, device: torch.device) -> "VectorPathGroup":
        self.fill_color = self.fill_color.to(device)
        self.stroke_color = self.stroke_color.to(device)
        return self


class VectorScene(BaseModel):
    """Collection of paths + groups — the renderer-agnostic SVG graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    paths: list[VectorPath] = []
    groups: list[VectorPathGroup] = []

    def enable_gradients(
        self,
        points: bool = True,
        colors: bool = True,
        opt_mask: list[int] | None = None,
    ) -> dict[str, list[Tensor]]:
        """Enable requires_grad and return param groups for the optimizer."""
        if opt_mask is None:
            opt_mask = [1] * len(self.paths)

        point_vars: list[Tensor] = []
        color_vars: list[Tensor] = []

        for i, (path, group) in enumerate(zip(self.paths, self.groups)):
            if not opt_mask[i]:
                continue
            if points:
                path.points.requires_grad_(True)
                point_vars.append(path.points)
            if colors:
                group.fill_color.requires_grad_(True)
                color_vars.append(group.fill_color)

        params: dict[str, list[Tensor]] = {}
        if point_vars:
            params["point"] = point_vars
        if color_vars:
            params["color"] = color_vars
        return params

    def disable_gradients(self) -> None:
        for path in self.paths:
            path.points.requires_grad_(False)
        for group in self.groups:
            group.fill_color.requires_grad_(False)

    def __len__(self) -> int:
        return len(self.paths)

    def append(self, path: VectorPath, group: VectorPathGroup) -> None:
        group.shape_idx = len(self.paths)
        self.paths.append(path)
        self.groups.append(group)

    def insert_at(self, index: int, path: VectorPath, group: VectorPathGroup) -> None:
        self.paths.insert(index, path)
        self.groups.insert(index, group)
        self.reindex_groups()

    def remove(self, indices: set[int]) -> None:
        self.paths = [p for i, p in enumerate(self.paths) if i not in indices]
        self.groups = [g for i, g in enumerate(self.groups) if i not in indices]
        self.reindex_groups()

    def reindex_groups(self) -> None:
        for i, g in enumerate(self.groups):
            g.shape_idx = i

    def to(self, device: torch.device) -> "VectorScene":
        for p in self.paths:
            p.to(device)
        for g in self.groups:
            g.to(device)
        return self

    def clone(self) -> "VectorScene":
        scene = VectorScene()
        for p, g in zip(self.paths, self.groups):
            scene.paths.append(p.clone())
            scene.groups.append(g.clone())
        scene.reindex_groups()
        return scene
