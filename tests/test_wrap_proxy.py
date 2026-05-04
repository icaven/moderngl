"""
Tests for the Python-side WrapProxy that backs the `wrap` property on
Sampler, Texture, Texture3D, and TextureArray.
"""


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