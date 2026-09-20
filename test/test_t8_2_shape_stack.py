#!/usr/bin/env python3
"""T8.2 dynamics shape-stack tests: build, node reproduction, weights, roundtrip."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from binary_masses import hierarchical_metallicity as hm  # noqa: E402
from binary_masses.hierarchical_metallicity import (  # noqa: E402
    DynamicsLikelihoodShapeStack,
    shape_trilinear_weights_numpy,
)

AXES = {
    "log_b": np.log(np.array([1.0e-3, 1.7e-3, 3.0e-3])),
    "log_uc": np.log(np.array([33.0, 38.0, 44.0])),
    "log_c": np.log(np.array([2.5, 6.0, 13.0])),
}


def build_tiny_stack():
    return DynamicsLikelihoodShapeStack.build(
        row_indices=np.array([2, 9]),
        u=np.array([10.0, 12.0]),
        u_sigma=np.array([1.0, 1.5]),
        axes=AXES,
        sqrt_mtot_min=0.5,
        sqrt_mtot_max=2.0,
        sqrt_mtot_points=32,
        velocity_quadrature_nodes=64,
        system_chunk=2,
        progress=False,
    )


def test_stack_build_shape_and_constant_restore():
    stack = build_tiny_stack()
    assert stack.log_good_stack.shape == (27, 2, 32)
    assert np.all(np.isfinite(stack.log_good_stack))
    # ambient constants restored after the build loop
    assert np.isclose(hm.RICE_GOOD_B, 2.544e-3, rtol=0, atol=0)
    assert np.isclose(hm.RICE_GOOD_BASIS_RAW_INTEGRAL, 0.9978600946, rtol=0, atol=1e-12)


def test_node_reproduction_and_weight_properties():
    # at a node the trilinear mix must return exactly that member
    values = (AXES["log_b"][1], AXES["log_uc"][2], AXES["log_c"][0])
    idx, w = shape_trilinear_weights_numpy(values, AXES)
    assert idx[np.argmax(w)] == 9 * 1 + 3 * 2 + 0
    assert abs(w.sum() - 1.0) < 1e-12
    assert np.all(w >= 0)
    # exactly at the middle nodes all weight lands on that single member
    middle = tuple(float(a[1]) for a in AXES.values())
    idx_m, wm = shape_trilinear_weights_numpy(middle, AXES)
    assert idx_m[np.argmax(wm)] == 9 * 1 + 3 * 1 + 1
    assert abs(wm.max() - 1.0) < 1e-12
    # random interior points: weights valid
    rng = np.random.default_rng(0)
    for _ in range(50):
        point = tuple(rng.uniform(a[0], a[2]) for a in AXES.values())
        _, wr = shape_trilinear_weights_numpy(point, AXES)
        assert abs(wr.sum() - 1.0) < 1e-12 and np.all(wr >= 0)


def test_effective_table_at_node_matches_member():
    stack = build_tiny_stack()
    values = (AXES["log_b"][0], AXES["log_uc"][1], AXES["log_c"][2])
    idx, w = shape_trilinear_weights_numpy(values, AXES)
    member = 9 * 0 + 3 * 1 + 2
    effective = np.logaddexp.reduce(
        np.log(w)[:, None, None] + stack.log_good_stack[idx], axis=0
    )
    assert np.allclose(effective, stack.log_good_stack[member], atol=1e-5)


def test_save_load_roundtrip(tmp_path=None):
    stack = build_tiny_stack()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "stack.npz"
        stack.save(path)
        loaded = DynamicsLikelihoodShapeStack.load(path)
    assert np.array_equal(loaded.row_indices, stack.row_indices)
    assert np.allclose(loaded.sqrt_mtot_grid, stack.sqrt_mtot_grid, rtol=0, atol=0)
    assert np.array_equal(loaded.log_good_stack, stack.log_good_stack)
    assert loaded.metadata["schema"] == "t8d2-rice-shapestack-v1"
    assert len(loaded.node_constants) == 27


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: OK")
    print("All T8.2 shape-stack tests passed.")
