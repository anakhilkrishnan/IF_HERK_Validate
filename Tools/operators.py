"""
Semi-discrete operators of the scalar tests, in NumPy.

These must be the SAME stencils as Src/ADV_core/ScalarOperators.H, applied on
a periodic grid (np.roll). reference.py and ifrk_prototype.py both use them,
so the two cannot drift apart.

Convention, as in the C++:  dphi/dt = -N(phi) + nu * lap(phi)
"""
import json
import os

import numpy as np


def _shift(f, ax, s):
    """f at index +s along axis ax (periodic)."""
    return np.roll(f, -s, axis=ax)


def N_linear(phi, c, dx):
    """sum_d c_d (phi_{+1} - phi_{-1}) / (2 dx_d)"""
    out = np.zeros_like(phi)
    for ax in range(phi.ndim):
        out += c[ax] * (_shift(phi, ax, 1) - _shift(phi, ax, -1)) / (2.0 * dx[ax])
    return out


def N_scalar_burgers(phi, c, dx):
    """sum_d c_d (phi_{+1}^2 - phi_{-1}^2) / (4 dx_d)"""
    sq = phi * phi
    out = np.zeros_like(phi)
    for ax in range(phi.ndim):
        out += c[ax] * (_shift(sq, ax, 1) - _shift(sq, ax, -1)) / (4.0 * dx[ax])
    return out


def laplacian(phi, dx):
    """(2*SPACEDIM+1)-point Laplacian"""
    out = np.zeros_like(phi)
    for ax in range(phi.ndim):
        out += (_shift(phi, ax, 1) - 2.0 * phi + _shift(phi, ax, -1)) / dx[ax] ** 2
    return out


N_BY_PROBLEM = {"linear_advection": N_linear, "scalar_burgers": N_scalar_burgers}


def load_run(run_dir):
    """run_info.json plus the initial fields, as a dict {name: array}."""
    with open(os.path.join(run_dir, "run_info.json")) as f:
        info = json.load(f)
    init = {name: np.load(os.path.join(run_dir, fn)) for name, fn in info["init"].items()}
    return info, init


def load_fields(run_dir, run):
    """The fields of one entry of info['runs'] (or info['reference_run'])."""
    return {name: np.load(os.path.join(run_dir, fn)) for name, fn in run["files"].items()}


def drop_duplicate_faces(arr, n_cell):
    """A face-centred field in a periodic direction carries n+1 faces, the
    last duplicating the first. Drop it so every point is counted once."""
    sl = tuple(slice(0, n) for n in n_cell)
    return arr[sl]
