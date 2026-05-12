from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def load_ply(path: str | Path) -> tuple[PlyData, np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path), mmap=True)
    vertex = ply["vertex"].data
    field_names = vertex.dtype.names or ()
    required = {"x", "y", "z"}
    missing = required.difference(field_names)
    if missing:
        raise ValueError(f"PLY is missing required vertex fields: {sorted(missing)}")
    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32, copy=False)
    return ply, vertex, xyz


def inspect_ply(path: str | Path) -> dict[str, object]:
    ply, vertex, xyz = load_ply(path)
    return {
        "path": str(path),
        "point_count": int(vertex.shape[0]),
        "fields": list(vertex.dtype.names or ()),
        "bbox_min": xyz.min(axis=0).tolist(),
        "bbox_max": xyz.max(axis=0).tolist(),
        "is_binary": not ply.text,
    }


def write_ply_with_label(
    ply: PlyData,
    labels: np.ndarray,
    out_path: str | Path,
    label_field: str = "label",
) -> None:
    vertex = ply["vertex"].data
    if labels.shape[0] != vertex.shape[0]:
        raise ValueError("label length must match vertex count")

    labels = labels.astype(np.uint8, copy=False)
    base_descr = list(vertex.dtype.descr)
    label_descr = (label_field, "u1")
    if label_field in (vertex.dtype.names or ()):
        base_descr = [item for item in base_descr if item[0] != label_field]
    new_dtype = np.dtype(base_descr + [label_descr])
    merged = np.empty(vertex.shape[0], dtype=new_dtype)

    for name in vertex.dtype.names or ():
        if name == label_field:
            continue
        merged[name] = vertex[name]
    merged[label_field] = labels

    elements = []
    for element in ply.elements:
        if element.name == "vertex":
            elements.append(PlyElement.describe(merged, "vertex"))
        else:
            elements.append(element)

    out_ply = PlyData(
        elements,
        text=ply.text,
        byte_order=ply.byte_order,
        comments=ply.comments,
        obj_info=ply.obj_info,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_ply.write(str(out_path))

