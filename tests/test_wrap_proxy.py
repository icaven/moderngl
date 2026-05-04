"""
Tests for the Python-side WrapProxy that backs the `wrap` property on
Sampler, Texture, Texture3D, and TextureArray.
"""
import gc
import sys

import moderngl


def test_wrap_proxy_len(ctx):
    """`len(obj.wrap)` should report the number of wrap axes."""
    sampler = ctx.sampler()
    assert 3 == len(sampler.wrap)

    tex2d = ctx.texture((4, 4), 4)
    assert 2 == len(tex2d.wrap)

    tex3d = ctx.texture3d((4, 4, 4), 4)
    assert 3 == len(tex3d.wrap)

    tex_array = ctx.texture_array((4, 4, 2), 4)
    assert 2 == len(tex_array.wrap)


def test_set_repeat_value_no_leak_on_invalid_value(ctx):
    """Regression: assigning a non-bool to `repeat_*` used to leak a dict
    per call. set_repeat_value allocated `dict` and `key` up front and
    returned -1 on the invalid-value path without DECREFing them.

    `sys.getallocatedblocks()` exposes the leak even though the empty
    dict isn't gc-tracked (CPython skips tracking dicts that can't form
    cycles)."""
    tex = ctx.texture((4, 4), 4)

    # Warm-up to settle any one-time allocations.
    for _ in range(10):
        try:
            tex.repeat_x = "warm-up"
        except moderngl.Error:
            pass
    gc.collect()

    n_before = sys.getallocatedblocks()

    iterations = 1000
    for _ in range(iterations):
        try:
            tex.repeat_x = "not a bool"
        except moderngl.Error:
            pass

    gc.collect()
    leaked = sys.getallocatedblocks() - n_before

    assert leaked < 100, (
        f"leaked {leaked} allocated blocks over {iterations} invalid assignments"
    )