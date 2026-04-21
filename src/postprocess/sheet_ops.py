"""
Winner-aligned post-processing chain for Vesuvius surface masks.

Pipeline (applied in order on a binary surface mask):
  1) remove_small_components   : drop any connected component below min_voxels
  2) plug_voxel_holes          : 2x2x2 LUT that makes each neighborhood 6-connected watertight
  3) heightmap_patch_sheet     : per-sheet, project to a height map and linearly fill gaps
  4) binary_close              : spherical radius-3 closing
  5) fill_cavities             : scipy.ndimage.binary_fill_holes on the whole mask

All functions operate on numpy bool arrays of shape (D, H, W).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


# ---------------------------------------------------------------------------
# Step 1: remove small connected components
# ---------------------------------------------------------------------------
def remove_small_components(mask: np.ndarray, min_voxels: int = 20_000) -> np.ndarray:
    if not mask.any():
        return mask
    lbl, n = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    if n == 0:
        return mask
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    keep = sizes >= min_voxels
    return keep[lbl]


# ---------------------------------------------------------------------------
# Step 2: 1-voxel hole plugger via 2x2x2 LUT
# ---------------------------------------------------------------------------
# For each of the 256 possible 2x2x2 binary neighborhoods we precompute the set
# of voxels that must be turned ON so the neighborhood is 6-connected watertight.
# "Watertight" here means: every background voxel in the 2^3 cube is 6-connected
# to a background voxel outside the cube, OR we fill the voxel to avoid a
# dangling 1-voxel pocket. The rule we actually enforce (matching the winner's
# description): fill background voxels whose only 6-neighbors *inside the cube*
# are foreground, i.e. voxels completely surrounded on their cube-interior
# faces by foreground.

def _build_plug_lut() -> np.ndarray:
    """Return shape (256, 2, 2, 2) uint8 array: for each of 256 neighborhoods,
    the neighborhood AFTER plugging 1-voxel holes."""
    lut = np.zeros((256, 2, 2, 2), dtype=np.uint8)
    # Cube coordinates
    coords = [(i, j, k) for i in range(2) for j in range(2) for k in range(2)]
    # 6-neighbor offsets
    offsets = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    for code in range(256):
        cube = np.zeros((2, 2, 2), dtype=np.uint8)
        for idx, (i, j, k) in enumerate(coords):
            if (code >> idx) & 1:
                cube[i, j, k] = 1
        out = cube.copy()
        # Iterate until no change (a 2x2x2 cube converges in at most a few passes)
        changed = True
        while changed:
            changed = False
            for i, j, k in coords:
                if out[i, j, k] == 1:
                    continue
                # Count 6-neighbors that are IN-CUBE and foreground.
                in_cube_neighbors = 0
                in_cube_fg_neighbors = 0
                for di, dj, dk in offsets:
                    ni, nj, nk = i + di, j + dj, k + dk
                    if 0 <= ni < 2 and 0 <= nj < 2 and 0 <= nk < 2:
                        in_cube_neighbors += 1
                        if out[ni, nj, nk] == 1:
                            in_cube_fg_neighbors += 1
                # A 2x2x2 corner has 3 in-cube 6-neighbors. If all 3 are FG, the
                # voxel is a 1-voxel hole inside the cube.
                if in_cube_neighbors > 0 and in_cube_fg_neighbors == in_cube_neighbors:
                    out[i, j, k] = 1
                    changed = True
        lut[code] = out
    return lut


_PLUG_LUT = _build_plug_lut()


def plug_voxel_holes(mask: np.ndarray, passes: int = 2) -> np.ndarray:
    """Plug 1-voxel holes using the 2x2x2 neighborhood LUT.

    We scan every 2x2x2 window (stride-1) and OR the LUT's plugged cube back in.
    A couple of passes catch holes that straddle multiple windows.
    """
    out = mask.astype(np.uint8, copy=True)
    for _ in range(passes):
        prev = out.copy()
        # Pad by 0 so we can index edges uniformly
        D, H, W = out.shape
        # Unfold into 2x2x2 cubes centered at every origin corner of a (2,2,2) window
        # Encode each window into its 8-bit code.
        c000 = out[0:D - 1, 0:H - 1, 0:W - 1]
        c100 = out[1:D,     0:H - 1, 0:W - 1]
        c010 = out[0:D - 1, 1:H,     0:W - 1]
        c110 = out[1:D,     1:H,     0:W - 1]
        c001 = out[0:D - 1, 0:H - 1, 1:W]
        c101 = out[1:D,     0:H - 1, 1:W]
        c011 = out[0:D - 1, 1:H,     1:W]
        c111 = out[1:D,     1:H,     1:W]
        # Bit order must match _build_plug_lut: idx = 4*i + 2*j + k (i=rowmajor over (i,j,k))
        # In the LUT builder, the list comprehension iterates i, j, k so idx0=(0,0,0), idx1=(0,0,1), ...
        code = (c000 * 1 + c001 * 2 + c010 * 4 + c011 * 8
                + c100 * 16 + c101 * 32 + c110 * 64 + c111 * 128).astype(np.int64)
        plugged = _PLUG_LUT[code]           # (D-1, H-1, W-1, 2, 2, 2)
        # Scatter-OR back. For each origin (d,h,w) and each cube offset (i,j,k),
        # accumulate plugged[..., i, j, k] into out[d+i, h+j, w+k].
        acc = np.zeros_like(out)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    acc[i:i + (D - 1), j:j + (H - 1), k:k + (W - 1)] |= plugged[..., i, j, k]
        out = np.maximum(out, acc)
        if np.array_equal(out, prev):
            break
    return out.astype(bool)


# ---------------------------------------------------------------------------
# Step 3: height-map patching of large holes (per sheet)
# ---------------------------------------------------------------------------
def _interp_1d_nan(a: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs along each row of a 2D array. Rows with no
    finite values are left as NaN. Rows where NaNs are at the ends are filled
    by clamping (edge extension)."""
    out = a.copy()
    for r in range(out.shape[0]):
        row = out[r]
        valid = np.isfinite(row)
        if valid.sum() < 2:
            continue
        idx = np.arange(row.size)
        out[r] = np.interp(idx, idx[valid], row[valid])
    return out


def _distance_to_edge_1d(valid_mask: np.ndarray) -> np.ndarray:
    """For a 1D boolean mask of 'originally valid' cells (True) and 'gap' cells
    (False), return distance-to-nearest-True for each cell along each row."""
    out = np.empty_like(valid_mask, dtype=np.float32)
    for r in range(valid_mask.shape[0]):
        row = valid_mask[r]
        if not row.any():
            out[r] = 0.0
            continue
        # Distance transform on 1D: simple left/right pass
        n = row.size
        d = np.full(n, np.inf, dtype=np.float32)
        last = -np.inf
        for i in range(n):
            if row[i]:
                last = i
            d[i] = i - last
        last = np.inf
        for i in range(n - 1, -1, -1):
            if row[i]:
                last = i
            d[i] = min(d[i], last - i)
        out[r] = d
    return out


def _patch_sheet_one_axis(sheet: np.ndarray, axis: int) -> np.ndarray:
    """Build a height map along `axis`, interpolate along X and Y, rebuild the
    volume. Returns a bool mask of the patched sheet."""
    # Project sheet to 2D by collapsing `axis`. Heights = (z_min, z_max) at each (u,v).
    order = [a for a in range(3) if a != axis] + [axis]   # project so axis is last
    sh = np.transpose(sheet, order)                        # (U, V, Z)
    present = sh.any(axis=-1)
    if not present.any():
        return np.zeros_like(sheet)

    # Height map: min and max index where sh is True (else NaN)
    Z = sh.shape[-1]
    zgrid = np.arange(Z)[None, None, :]
    mask_f = sh.astype(np.float32)
    z_any = np.where(sh, zgrid, np.nan)
    with np.errstate(invalid="ignore"):
        z_min = np.nanmin(z_any, axis=-1)
        z_max = np.nanmax(z_any, axis=-1)

    # Interp along U (rows), then along V (columns), then average weighted by
    # distance-to-original-support.
    valid = present.copy()
    zmin_u = _interp_1d_nan(z_min)
    zmax_u = _interp_1d_nan(z_max)
    zmin_v = _interp_1d_nan(z_min.T).T
    zmax_v = _interp_1d_nan(z_max.T).T

    du = _distance_to_edge_1d(valid)
    dv = _distance_to_edge_1d(valid.T).T
    w_u = np.where(np.isfinite(zmin_u) & np.isfinite(zmax_u),
                   1.0 / (1.0 + du), 0.0)
    w_v = np.where(np.isfinite(zmin_v) & np.isfinite(zmax_v),
                   1.0 / (1.0 + dv), 0.0)
    w_sum = w_u + w_v
    zmin_hat = np.where(w_sum > 0, (np.nan_to_num(zmin_u) * w_u + np.nan_to_num(zmin_v) * w_v) / np.maximum(w_sum, 1e-6), np.nan)
    zmax_hat = np.where(w_sum > 0, (np.nan_to_num(zmax_u) * w_u + np.nan_to_num(zmax_v) * w_v) / np.maximum(w_sum, 1e-6), np.nan)

    # Rebuild the volume: for each (u,v) with finite z_min_hat/z_max_hat, fill z in [zmin_hat..zmax_hat]
    U, V, _ = sh.shape
    out = np.zeros_like(sh)
    uu, vv = np.where(np.isfinite(zmin_hat) & np.isfinite(zmax_hat))
    zl = np.round(zmin_hat[uu, vv]).astype(int).clip(0, Z - 1)
    zh = np.round(zmax_hat[uu, vv]).astype(int).clip(0, Z - 1)
    for u, v, a, b in zip(uu, vv, zl, zh):
        if a <= b:
            out[u, v, a:b + 1] = True

    # Un-transpose
    inv = np.argsort(order)
    return np.transpose(out, inv)


def heightmap_patch_sheet(sheet: np.ndarray, hole_count_gate: bool = True) -> np.ndarray:
    """Given a bool volume containing a single connected sheet, return a
    patched version that fills large holes via height-map interpolation along
    the projection axis with maximal projected area. If hole_count_gate is
    True, discard the patch if it increases the number of holes (per winner)."""
    if not sheet.any():
        return sheet
    # Choose projection axis = the one with largest projected 2D area
    areas = [sheet.any(axis=a).sum() for a in range(3)]
    axis = int(np.argmax(areas))
    patched = _patch_sheet_one_axis(sheet, axis) | sheet

    if hole_count_gate:
        def _count_holes(vol):
            # holes = connected components of background *inside* the sheet's bbox minus outer background
            inv = ~vol
            lbl, _ = ndi.label(inv)
            outer = lbl[0, 0, 0] if lbl.size else 0
            return len(np.unique(lbl)) - (2 if outer != 0 else 1)
        if _count_holes(patched) > _count_holes(sheet):
            return sheet
    return patched


def patch_all_sheets(mask: np.ndarray) -> np.ndarray:
    """Apply heightmap_patch_sheet to every connected component independently."""
    if not mask.any():
        return mask
    lbl, n = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    out = np.zeros_like(mask)
    for i in range(1, n + 1):
        sheet = lbl == i
        out |= heightmap_patch_sheet(sheet)
    return out


# ---------------------------------------------------------------------------
# Step 4 & 5: closing + fill_holes
# ---------------------------------------------------------------------------
def _spherical_footprint(radius: int) -> np.ndarray:
    r = radius
    d = 2 * r + 1
    z, y, x = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (x * x + y * y + z * z) <= r * r


def binary_close(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    return ndi.binary_closing(mask, structure=_spherical_footprint(radius))


def fill_cavities(mask: np.ndarray) -> np.ndarray:
    return ndi.binary_fill_holes(mask)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def full_postprocess(
    surface_prob: np.ndarray,
    threshold: float = 0.23,
    min_component_voxels: int = 20_000,
    do_plug: bool = True,
    do_patch: bool = True,
    do_close: bool = True,
    do_fill: bool = True,
    closing_radius: int = 3,
) -> np.ndarray:
    """Run the full 5-step chain starting from a probability volume. Returns
    a bool mask."""
    mask = surface_prob > threshold
    mask = remove_small_components(mask, min_voxels=min_component_voxels)
    if do_plug:
        mask = plug_voxel_holes(mask)
    if do_patch:
        mask = patch_all_sheets(mask)
    if do_close:
        mask = binary_close(mask, radius=closing_radius)
    if do_fill:
        mask = fill_cavities(mask)
    return mask
