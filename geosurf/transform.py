#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geodesic / Euclidean Distance Transform on a surface mesh.

Algorithms supported
-------------------
- dijkstra : graph shortest-path distance along mesh edges (approximate geodesic)
- heat     : heat method via potpourri3d (fast, good approximation on many meshes)
- euclid   : straight-line (3D Euclidean) distance to the nearest seed

I/O
---
- Reads .vtk (legacy) or .vtp (XML) PolyData
- Writes .vtk or .vtp depending on output suffix
- The resulting scalar array (default name: "Distance") is added to point data
  and also set as the active scalars for easy coloring.

Usage
-----
python geodesic_transform.py \
  --vtk lh.white.vtk --seed-id 123 --algo heat --out lh.white_dist.vtk

python geodesic_transform.py \
  --vtk surf.vtp --seeds 10,42,77 --algo dijkstra --combine min --out out.vtp

Requirements
------------
- vtk (pip install vtk)
- potpourri3d (only if using --algo heat)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

# --- Third-party imports ------------------------------------------------------

try:
    import vtk  # type: ignore
except Exception:
    print("[ERROR] Missing vtk. Install via `pip install vtk`.", file=sys.stderr)
    raise

try:
    # heat method (optional)
    import potpourri3d as pp3d  # type: ignore
    import numpy as np
except ImportError:
    pp3d = None  # heat method unavailable
    import numpy as np  # numpy is still needed for dijkstra/euclid

# --- Mesh I/O and utilities ---------------------------------------------------

def read_polydata(path: str) -> vtk.vtkPolyData:
    """Load a VTK/VTP polydata mesh from disk."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == ".vtk":
        reader = vtk.vtkPolyDataReader()
    else:
        # Fallback for some XML variants; will still require PolyData inside.
        reader = vtk.vtkXMLGenericDataObjectReader()

    reader.SetFileName(path)
    reader.Update()
    dataobj = reader.GetOutput()

    poly = vtk.vtkPolyData.SafeDownCast(dataobj)
    if poly is None:
        raise ValueError(f"Unsupported or non-PolyData file: {path}")
    return poly


def ensure_triangulated(poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """
    Return a triangulated copy of the input mesh.
    Most geodesic methods assume triangles.
    """
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.PassLinesOff()
    tri.PassVertsOff()
    tri.Update()
    return tri.GetOutput()


def extract_VF(poly: vtk.vtkPolyData) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert vtkPolyData to (V, F) numpy arrays:

    V : (n_pts, 3) float64 vertex positions
    F : (n_faces, 3) int32  triangle indices

    Assumes the mesh is already triangulated.
    """
    n = poly.GetNumberOfPoints()
    V = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        V[i, :] = poly.GetPoint(i)

    faces: List[Tuple[int, int, int]] = []
    polys = poly.GetPolys()
    polys.InitTraversal()
    ids = vtk.vtkIdList()
    while polys.GetNextCell(ids):
        if ids.GetNumberOfIds() == 3:
            faces.append((ids.GetId(0), ids.GetId(1), ids.GetId(2)))
        # Non-tri cells should not appear after triangulation

    if not faces:
        raise ValueError("No triangle faces found. Triangulation may have failed.")
    F = np.asarray(faces, dtype=np.int32)
    return V, F


def write_polydata_vtk(poly: vtk.vtkPolyData, path: str) -> None:
    """Write legacy .vtk PolyData (binary)."""
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(poly)
    writer.SetFileTypeToBinary()
    if writer.Write() == 0:
        raise IOError(f"Failed to write: {path}")


def write_polydata_vtp(poly: vtk.vtkPolyData, path: str) -> None:
    """Write XML .vtp PolyData (binary)."""
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(poly)
    writer.SetDataModeToBinary()
    if writer.Write() == 0:
        raise IOError(f"Failed to write: {path}")

# --- Algorithms ---------------------------------------------------------------

def compute_dijkstra_multi(
    poly: vtk.vtkPolyData,
    seeds: List[int],
    array_name: str = "DijkstraDistance",
    combine: str = "min",
    stack: bool = False,
) -> vtk.vtkPolyData:
    """
    Dijkstra on the vertex-edge graph. For multiple seeds, combine per-seed fields:

    combine in {"min","mean","sum"}
      - "min" (default): distance to the nearest seed
      - "mean"/"sum"   : reductions over per-seed distances (rarely needed)

    Notes
    -----
    vtkDijkstraGraphGeodesicPath exposes cumulative weights via an OUT parameter.
    """
    import numpy as _np

    npts = poly.GetNumberOfPoints()
    per_seed: List[_np.ndarray] = []

    for s in seeds:
        dijk = vtk.vtkDijkstraGraphGeodesicPath()
        dijk.SetInputData(poly)
        dijk.SetStartVertex(s)
        dijk.SetStopWhenEndReached(False)  # fill cumulative weights to all vertices
        dijk.Update()

        weights = vtk.vtkDoubleArray()
        dijk.GetCumulativeWeights(weights)  # OUT parameter
        arr_np = _np.array([weights.GetValue(i) for i in range(npts)], dtype=float)
        per_seed.append(arr_np)

    if len(per_seed) == 1:
        combined = per_seed[0]
    else:
        stack_np = _np.vstack(per_seed)  # shape: (n_seeds, npts)
        if combine == "mean":
            combined = stack_np.mean(axis=0)
        elif combine == "sum":
            combined = stack_np.sum(axis=0)
        else:
            combined = stack_np.min(axis=0)  # "min"

    out = vtk.vtkPolyData()
    out.DeepCopy(poly)

    def to_vtk_array(name: str, arr: _np.ndarray) -> vtk.vtkDoubleArray:
        v = vtk.vtkDoubleArray()
        v.SetName(name)
        v.SetNumberOfValues(npts)
        for i, val in enumerate(arr):
            v.SetValue(i, float(val))
        return v

    arr_comb = to_vtk_array(array_name, combined)
    out.GetPointData().AddArray(arr_comb)
    # out.GetPointData().SetScalars(arr_comb)

    if stack and len(per_seed) > 1:
        for idx, arr_np in enumerate(per_seed):
            out.GetPointData().AddArray(to_vtk_array(f"{array_name}_seed{idx}", arr_np))

    return out


def compute_heat_potpourri(
    poly: vtk.vtkPolyData,
    seeds: List[int],
    array_name: str = "HeatDistance",
    combine: str = "min",
    stack: bool = False,
) -> vtk.vtkPolyData:
    """
    Compute geodesic distances using the Heat Method via potpourri3d.

    Parameters
    ----------
    poly : vtk.vtkPolyData
        Input triangulated mesh.
    seeds : List[int]
        List of seed vertex indices.
    array_name : str
        Name of the scalar field to store distances.
    combine : {"min", "mean", "sum"}
        How to combine distances when multiple seeds are provided:
          - "min": distance to the nearest seed (fast, default)
          - "mean": average distance across all seeds
          - "sum": sum of distances to all seeds
    stack : bool
        If True, write per-seed distance arrays as well.

    Returns
    -------
    vtk.vtkPolyData
        The same mesh with a new scalar array containing geodesic distances.
    """
    if pp3d is None:
        raise RuntimeError("potpourri3d is not installed; cannot use --algo heat")

    # Convert VTK PolyData to (V, F) arrays
    V, F = extract_VF(poly)

    # Initialize solver (factorization is reused for multiple seeds)
    solver = pp3d.MeshHeatMethodDistanceSolver(V, F)

    if combine == "min":
        # Multi-source computation: computes distance to the nearest seed efficiently
        dist = solver.compute_distance_multisource(seeds)
        per_seed = None
    else:
        # Compute distance from each seed separately, then combine
        per_seed = [solver.compute_distance(s) for s in seeds]  # list of (n_pts,)
        stack_np = np.vstack(per_seed)  # shape = (n_seeds, n_pts)
        if combine == "mean":
            dist = stack_np.mean(axis=0)
        elif combine == "sum":
            dist = stack_np.sum(axis=0)
        else:
            dist = stack_np.min(axis=0)

    # Create output mesh and attach the distance array
    out = vtk.vtkPolyData()
    out.DeepCopy(poly)

    arr = vtk.vtkDoubleArray()
    arr.SetName(array_name)
    arr.SetNumberOfValues(len(dist))
    for i, d in enumerate(dist):
        arr.SetValue(i, float(d))
    out.GetPointData().AddArray(arr)
    # out.GetPointData().SetScalars(arr)

    # Optionally, write per-seed distance fields
    if stack and per_seed is not None:
        for idx, vec in enumerate(per_seed):
            a = vtk.vtkDoubleArray()
            a.SetName(f"{array_name}_seed{idx}")
            a.SetNumberOfValues(len(vec))
            for i, d in enumerate(vec):
                a.SetValue(i, float(d))
            out.GetPointData().AddArray(a)

    return out


def compute_euclid_distance(
    poly: vtk.vtkPolyData,
    seeds: List[int],
    array_name: str = "EuclidDistance",
) -> vtk.vtkPolyData:
    """
    Straight-line 3D distance from each vertex to the nearest seed.
    Useful as a baseline; does NOT respect surface connectivity.
    """
    n = poly.GetNumberOfPoints()
    P = np.zeros((n, 3), dtype=float)
    for i in range(n):
        P[i, :] = poly.GetPoint(i)

    S = np.array([P[s] for s in seeds], dtype=float)  # (n_seeds, 3)

    # Compute min over seeds of ||P - s||_2
    # Simple loop to avoid excessive memory on huge meshes; still vectorized per seed.
    minsq = None
    for s in S:
        diff = P - s
        sq = np.einsum("ij,ij->i", diff, diff)
        minsq = sq if minsq is None else np.minimum(minsq, sq)
    dist = np.sqrt(minsq)

    out = vtk.vtkPolyData()
    out.DeepCopy(poly)

    arr = vtk.vtkDoubleArray()
    arr.SetName(array_name)
    arr.SetNumberOfValues(n)
    for i, d in enumerate(dist):
        arr.SetValue(i, float(d))
    out.GetPointData().AddArray(arr)
    # out.GetPointData().SetScalars(arr)
    return out

# --- CLI ----------------------------------------------------------------------

def positive_int(val: str) -> int:
    """Argparse helper for non-negative integers."""
    try:
        iv = int(val)
    except Exception:
        raise argparse.ArgumentTypeError(f"'{val}' is not an integer")
    if iv < 0:
        raise argparse.ArgumentTypeError("seed id must be non-negative")
    return iv


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Geodesic / Euclidean distance transform (heat, dijkstra, euclid)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--vtk", required=True, help="Input surface (.vtk or .vtp)")

    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--seed-id", type=positive_int, help="Single seed id (0-based)")
    g.add_argument("--seeds", type=str, help="Comma-separated seed ids, e.g. 1,2,3")
    g.add_argument("--seed-file", type=str, help="Text file with one seed id per line")

    p.add_argument("--ids-one-based", action="store_true",
                   help="Treat provided seed id(s) as 1-based (convert to 0-based)")

    p.add_argument("--algo", choices=["auto", "dijkstra", "heat", "euclid"], default="auto",
                   help="Which distance algorithm to use")

    p.add_argument("--array-name", default="Geo_Transform",
                   help="Name of the output point-data scalar array")

    p.add_argument("--combine", choices=["min", "mean", "sum"], default="min",
                   help="Multi-seed reduction for Dijkstra")

    p.add_argument("--stack", action="store_true",
                   help="Also write per-seed arrays in Dijkstra mode")

    p.add_argument("--out", default=None,
                   help="Output path (.vtk or .vtp). If omitted, appends _dist.ext")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # --- Load mesh
    in_path = args.vtk
    if not os.path.exists(in_path):
        print(f"[ERROR] Input file not found: {in_path}", file=sys.stderr)
        return 2
    poly = read_polydata(in_path)
    npts = poly.GetNumberOfPoints()
    if npts == 0:
        print("[ERROR] Mesh has zero points.", file=sys.stderr)
        return 3

    # --- Parse seeds
    seeds: List[int] = []
    if args.seed_id is not None:
        seeds = [args.seed_id]
    elif args.seeds is not None:
        try:
            seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        except Exception:
            print("[ERROR] Failed to parse --seeds (use comma-separated integers).", file=sys.stderr)
            return 2
    elif args.seed_file is not None:
        if not os.path.exists(args.seed_file):
            print(f"[ERROR] Seed file not found: {args.seed_file}", file=sys.stderr)
            return 2
        with open(args.seed_file, "r") as f:
            for line in f:
                ln = line.strip()
                if ln and not ln.startswith("#"):
                    seeds.append(int(ln))
    else:
        print("[ERROR] Provide --seed-id or --seeds or --seed-file.", file=sys.stderr)
        return 2

    if args.ids_one_based:
        seeds = [s - 1 for s in seeds]

    for s in seeds:
        if s < 0 or s >= npts:
            print(f"[ERROR] Seed {s} out of range [0, {npts-1}].", file=sys.stderr)
            return 4

    # --- Prepare mesh
    poly_tri = ensure_triangulated(poly)

    # --- Choose algorithm
    algo = args.algo
    if algo == "auto":
        # Favor heat when available, otherwise fall back to Dijkstra.
        algo = "heat" if pp3d is not None else "dijkstra"

    if algo == "heat" and pp3d is None:
        print("[WARN] potpourri3d not installed; falling back to dijkstra.", file=sys.stderr)
        algo = "dijkstra"

    # --- Run
    if algo == "heat":
        out_poly = compute_heat_potpourri(
    poly_tri, seeds, args.array_name, combine=args.combine, stack=args.stack
        )

    elif algo == "dijkstra":
        out_poly = compute_dijkstra_multi(
            poly_tri, seeds, args.array_name, combine=args.combine, stack=args.stack
        )
    else:  # euclid
        out_poly = compute_euclid_distance(poly_tri, seeds, args.array_name)

    # --- Write
    out_path = args.out
    if out_path is None:
        base, ext = os.path.splitext(in_path)
        out_path = base + ("_dist.vtk" if ext.lower() == ".vtk" else "_dist.vtp")

    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".vtk":
        write_polydata_vtk(out_poly, out_path)
    else:
        write_polydata_vtp(out_poly, out_path)

    # --- Report
    arr = out_poly.GetPointData().GetArray(args.array_name)
    rng = arr.GetRange() if arr is not None else (0.0, 0.0)
    print("[OK] Wrote:", out_path)
    print(f"[INFO] Points: {npts}, Scalar: '{args.array_name}', Range: {rng[0]:.6f} ~ {rng[1]:.6f}")
    print(f"[INFO] Algorithm: {algo}, Seeds (0-based): {seeds}")
    return 0
