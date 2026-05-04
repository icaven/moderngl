"""
Tests for bindless texture handle functionality.

Tests the new ability to assign lists of texture handles to uniform arrays,
which was added to support bindless texture arrays in OpenGL.
"""
import pytest
import moderngl


def test_single_handle_backward_compatibility(ctx):
    """Tests that single handle assignment still works (backward compatibility)."""
    if not ctx.supports_bindless:
        pytest.skip("Bindless textures not supported")

    prog = ctx.program(
        vertex_shader="""
            #version 440
            void main() {
                gl_Position = vec4(0.0);
            }
        """,
        fragment_shader="""
            #version 440
            #extension GL_ARB_bindless_texture : require
            layout (bindless_sampler) uniform sampler2D Texture;
            out vec4 fragColor;
            void main() {
                fragColor = texture(Texture, vec2(0.5));
            }
        """,
    )

    texture = ctx.texture((4, 4), 4)
    handle = texture.get_handle()

    # Should work without raising an exception
    prog["Texture"].handle = handle

    texture.release()


def test_handle_list_correct_length(ctx):
    """Tests assigning a list of handles with correct array length."""
    if not ctx.supports_bindless:
        pytest.skip("Bindless textures not supported")

    prog = ctx.program(
        vertex_shader="""
            #version 440
            void main() {
                gl_Position = vec4(0.0);
            }
        """,
        fragment_shader="""
            #version 440
            #extension GL_ARB_bindless_texture : require
            layout (bindless_sampler) uniform sampler2D Textures[3];
            out vec4 fragColor;
            void main() {
                fragColor = texture(Textures[0], vec2(0.5)) +
                            texture(Textures[1], vec2(0.5)) +
                            texture(Textures[2], vec2(0.5));
            }
        """,
    )

    # Create 3 textures and get their handles
    textures = [ctx.texture((4, 4), 4) for _ in range(3)]
    handles = [tex.get_handle() for tex in textures]

    # Should work - list length matches array_length
    prog["Textures"].handle = handles

    # Cleanup
    for tex in textures:
        tex.release()


def test_handle_list_wrong_length_too_many(ctx):
    """Tests that assigning too many handles raises ValueError."""
    if not ctx.supports_bindless:
        pytest.skip("Bindless textures not supported")

    prog = ctx.program(
        vertex_shader="""
            #version 440
            void main() {
                gl_Position = vec4(0.0);
            }
        """,
        fragment_shader="""
            #version 440
            #extension GL_ARB_bindless_texture : require
            layout (bindless_sampler) uniform sampler2D Textures[3];
            out vec4 fragColor;
            void main() {
                fragColor = texture(Textures[0], vec2(0.5)) +
                            texture(Textures[1], vec2(0.5)) +
                            texture(Textures[2], vec2(0.5));
            }
        """,
    )

    # Create 5 textures but uniform array is size 3
    textures = [ctx.texture((4, 4), 4) for _ in range(5)]
    handles = [tex.get_handle() for tex in textures]

    # Should raise ValueError about length mismatch
    with pytest.raises(ValueError, match="has 5 elements but uniform array requires exactly 3"):
        prog["Textures"].handle = handles

    # Cleanup
    for tex in textures:
        tex.release()


def test_handle_list_wrong_length_too_few(ctx):
    """Tests that assigning too few handles raises ValueError."""
    if not ctx.supports_bindless:
        pytest.skip("Bindless textures not supported")

    prog = ctx.program(
        vertex_shader="""
            #version 440
            void main() {
                gl_Position = vec4(0.0);
            }
        """,
        fragment_shader="""
            #version 440
            #extension GL_ARB_bindless_texture : require
            layout (bindless_sampler) uniform sampler2D Textures[3];
            out vec4 fragColor;
            void main() {
                fragColor = texture(Textures[0], vec2(0.5)) +
                            texture(Textures[1], vec2(0.5)) +
                            texture(Textures[2], vec2(0.5));
            }
        """,
    )

    # Create 2 textures but uniform array is size 3
    textures = [ctx.texture((4, 4), 4) for _ in range(2)]
    handles = [tex.get_handle() for tex in textures]

    # Should raise ValueError about length mismatch
    with pytest.raises(ValueError, match="has 2 elements but uniform array requires exactly 3"):
        prog["Textures"].handle = handles

    # Cleanup
    for tex in textures:
        tex.release()


def test_handle_list_empty(ctx):
    """Tests that assigning an empty list is rejected."""
    if not ctx.supports_bindless:
        pytest.skip("Bindless textures not supported")

    prog = ctx.program(
        vertex_shader="""
            #version 440
            void main() {
                gl_Position = vec4(0.0);
            }
        """,
        fragment_shader="""
            #version 440
            #extension GL_ARB_bindless_texture : require
            layout (bindless_sampler) uniform sampler2D Textures[3];
            out vec4 fragColor;
            void main() {
                fragColor = texture(Textures[0], vec2(0.5)) +
                            texture(Textures[1], vec2(0.5)) +
                            texture(Textures[2], vec2(0.5));
            }
        """,
    )

    # Should raise ValueError about length mismatch
    with pytest.raises(ValueError, match="has 0 elements but uniform array requires exactly 3"):
        prog["Textures"].handle = []


def test_handle_scalar_to_array_uniform(ctx):
    """Tests that assigning a scalar handle to an array uniform is rejected.

    Without this check, the scalar would silently be written to slot 0 of
    the array via ProgramUniformHandleui64ARB, leaving slots 1..N-1
    untouched. That's almost always a bug (forgot to wrap in a list).
    """
    if not ctx.supports_bindless:
        pytest.skip("Bindless textures not supported")

    prog = ctx.program(
        vertex_shader="""
            #version 440
            void main() {
                gl_Position = vec4(0.0);
            }
        """,
        fragment_shader="""
            #version 440
            #extension GL_ARB_bindless_texture : require
            layout (bindless_sampler) uniform sampler2D Textures[3];
            out vec4 fragColor;
            void main() {
                fragColor = texture(Textures[0], vec2(0.5)) +
                            texture(Textures[1], vec2(0.5)) +
                            texture(Textures[2], vec2(0.5));
            }
        """,
    )

    texture = ctx.texture((4, 4), 4)
    handle = texture.get_handle()

    with pytest.raises(ValueError, match="uniform array of length 3 .* list"):
        prog["Textures"].handle = handle

    texture.release()


def test_handle_list_non_integer_elements(ctx):
    """Tests that non-integer elements in handle list raise TypeError."""
    if not ctx.supports_bindless:
        pytest.skip("Bindless textures not supported")

    prog = ctx.program(
        vertex_shader="""
            #version 440
            void main() {
                gl_Position = vec4(0.0);
            }
        """,
        fragment_shader="""
            #version 440
            #extension GL_ARB_bindless_texture : require
            layout (bindless_sampler) uniform sampler2D Textures[3];
            out vec4 fragColor;
            void main() {
                fragColor = texture(Textures[0], vec2(0.5)) +
                            texture(Textures[1], vec2(0.5)) +
                            texture(Textures[2], vec2(0.5));
            }
        """,
    )

    # Try with strings
    with pytest.raises(TypeError, match="All items in handle list must be integers"):
        prog["Textures"].handle = ["not", "an", "integer"]


def test_handle_list_mixed_types(ctx):
    """Tests that mixing valid and invalid types raises TypeError."""
    if not ctx.supports_bindless:
        pytest.skip("Bindless textures not supported")

    prog = ctx.program(
        vertex_shader="""
            #version 440
            void main() {
                gl_Position = vec4(0.0);
            }
        """,
        fragment_shader="""
            #version 440
            #extension GL_ARB_bindless_texture : require
            layout (bindless_sampler) uniform sampler2D Textures[3];
            out vec4 fragColor;
            void main() {
                fragColor = texture(Textures[0], vec2(0.5)) +
                            texture(Textures[1], vec2(0.5)) +
                            texture(Textures[2], vec2(0.5));
            }
        """,
    )

    texture = ctx.texture((4, 4), 4)
    handle = texture.get_handle()

    # Mix integer and string
    with pytest.raises(TypeError, match="All items in handle list must be integers"):
        prog["Textures"].handle = [handle, "bad", handle]

    texture.release()


def test_handle_large_array(ctx):
    """Tests assigning handles to a larger array (performance/stress test)."""
    if not ctx.supports_bindless:
        pytest.skip("Bindless textures not supported")

    prog = ctx.program(
        vertex_shader="""
            #version 440
            void main() {
                gl_Position = vec4(0.0);
            }
        """,
        fragment_shader="""
            #version 440
            #extension GL_ARB_bindless_texture : require
            layout (bindless_sampler) uniform sampler2D Textures[16];
            out vec4 fragColor;
            void main() {
                vec4 color = vec4(0.0);
                for (int i = 0; i < 16; i++) {
                    color += texture(Textures[i], vec2(0.5));
                }
                fragColor = color;
            }
        """,
    )

    # Create 16 textures
    textures = [ctx.texture((4, 4), 4) for _ in range(16)]
    handles = [tex.get_handle() for tex in textures]

    # Should work with larger arrays
    prog["Textures"].handle = handles

    # Cleanup
    for tex in textures:
        tex.release()


def test_handle_uniform_not_array(ctx):
    """Tests that single uniform (not array) still works with list of size 1."""
    if not ctx.supports_bindless:
        pytest.skip("Bindless textures not supported")

    prog = ctx.program(
        vertex_shader="""
            #version 440
            void main() {
                gl_Position = vec4(0.0);
            }
        """,
        fragment_shader="""
            #version 440
            #extension GL_ARB_bindless_texture : require
            layout (bindless_sampler) uniform sampler2D Texture;
            out vec4 fragColor;
            void main() {
                fragColor = texture(Texture, vec2(0.5));
            }
        """,
    )

    texture = ctx.texture((4, 4), 4)
    handle = texture.get_handle()

    # For a non-array uniform, array_length should be 1
    # So a list with 1 element should work
    if prog["Texture"].array_length == 1:
        prog["Texture"].handle = [handle]

    texture.release()


def test_handle_negative_location(ctx):
    """Tests that invalid uniform location is caught."""
    if not ctx.supports_bindless:
        pytest.skip("Bindless textures not supported")

    prog = ctx.program(
        vertex_shader="""
            #version 440
            void main() {
                gl_Position = vec4(0.0);
            }
        """,
        fragment_shader="""
            #version 440
            out vec4 fragColor;
            void main() {
                fragColor = vec4(1.0);
            }
        """,
    )

    # Try to access a uniform that doesn't exist
    # This should either not exist in prog dict or handle gracefully
    assert "NonExistentUniform" not in prog


def test_handle_integer_overflow(ctx):
    """Tests that integer overflow in handle values is caught."""
    if not ctx.supports_bindless:
        pytest.skip("Bindless textures not supported")

    prog = ctx.program(
        vertex_shader="""
            #version 440
            void main() {
                gl_Position = vec4(0.0);
            }
        """,
        fragment_shader="""
            #version 440
            #extension GL_ARB_bindless_texture : require
            layout (bindless_sampler) uniform sampler2D Textures[2];
            out vec4 fragColor;
            void main() {
                fragColor = texture(Textures[0], vec2(0.5)) +
                            texture(Textures[1], vec2(0.5));
            }
        """,
    )

    # Try with an absurdly large number that would overflow uint64
    # Python integers can be arbitrarily large, so this tests C++ overflow handling
    huge_number = 2**128  # Way beyond uint64 max

    with pytest.raises((OverflowError, ValueError)):
        prog["Textures"].handle = [huge_number, huge_number]


def test_get_single_handle(ctx):
    """Tests that getting a single handle works."""
    if not ctx.supports_bindless:
        pytest.skip("Bindless textures not supported")

    prog = ctx.program(
        vertex_shader="""
            #version 440
            void main() {
                gl_Position = vec4(0.0);
            }
        """,
        fragment_shader="""
            #version 440
            #extension GL_ARB_bindless_texture : require
            layout (bindless_sampler) uniform sampler2D Texture;
            out vec4 fragColor;
            void main() {
                // Actually use the texture to prevent optimization
                fragColor = texture(Texture, vec2(0.5, 0.5));
            }
        """,
    )

    texture = ctx.texture((4, 4), 4)
    handle = texture.get_handle()

    # Set the handle
    prog["Texture"].handle = handle

    # Get the handle back and verify it matches
    retrieved_handle = prog["Texture"].handle
    assert retrieved_handle == handle, f"Expected {handle}, got {retrieved_handle}"

    texture.release()


def test_get_handle_array(ctx):
    """Tests that getting a list of handles works."""
    if not ctx.supports_bindless:
        pytest.skip("Bindless textures not supported")

    prog = ctx.program(
        vertex_shader="""
            #version 440
            void main() {
                gl_Position = vec4(0.0);
            }
        """,
        fragment_shader="""
            #version 440
            #extension GL_ARB_bindless_texture : require
            layout (bindless_sampler) uniform sampler2D Textures[3];
            out vec4 fragColor;
            void main() {
                // Actually use the textures to prevent optimization
                fragColor = texture(Textures[0], vec2(0.5, 0.5)) +
                            texture(Textures[1], vec2(0.5, 0.5)) +
                            texture(Textures[2], vec2(0.5, 0.5));
            }
        """,
    )

    # Create 3 textures and get their handles
    textures = [ctx.texture((4, 4), 4) for _ in range(3)]
    handles = [tex.get_handle() for tex in textures]

    # Set the handles
    prog["Textures"].handle = handles

    # Get the handles back and verify they match
    retrieved_handles = prog["Textures"].handle
    assert isinstance(retrieved_handles, list), f"Expected list, got {type(retrieved_handles)}"
    assert len(retrieved_handles) == 3, f"Expected 3 handles, got {len(retrieved_handles)}"
    assert retrieved_handles == handles, f"Expected {handles}, got {retrieved_handles}"

    # Cleanup
    for tex in textures:
        tex.release()


def test_bindless_texture_array_integration(bindless_ctx, bindless_textures, ndc_quad):
    """Integration test: actually samples from bindless texture arrays and verifies output."""
    import numpy as np

    # Use first 3 textures from the fixture
    test_textures = bindless_textures[:3]

    # Get handles for the textures
    handles = [tex.get_handle() for tex in test_textures]

    # Create a shader that samples from a texture array
    prog = bindless_ctx.program(
        vertex_shader="""
            #version 440
            in vec2 in_vert;
            void main() {
                gl_Position = vec4(in_vert, 0.0, 1.0);
            }
        """,
        fragment_shader="""
            #version 440
            #extension GL_ARB_bindless_texture : require
            layout (bindless_sampler) uniform sampler2D Textures[3];
            uniform int texIndex;
            out vec4 fragColor;
            void main() {
                fragColor = texture(Textures[texIndex], vec2(0.5, 0.5));
            }
        """,
    )

    # Set the texture handles
    prog["Textures"].handle = handles

    # Create framebuffer and VAO
    fbo = bindless_ctx.simple_framebuffer((4, 4))
    fbo.use()
    vao = bindless_ctx.simple_vertex_array(prog, ndc_quad, "in_vert")

    # Test sampling from each texture in the array
    for i in range(3):
        prog["texIndex"].value = i
        fbo.clear(0.0, 0.0, 0.0, 1.0)
        vao.render(mode=moderngl.TRIANGLE_STRIP)

        result = np.frombuffer(fbo.read(components=4), dtype=np.uint8)
        expected_value = i * 40

        # Check that we got approximately the right color
        actual_r = result[0]
        assert abs(actual_r - expected_value) < 5, \
            f"Texture {i}: expected ~{expected_value}, got {actual_r}"

    # Cleanup
    vao.release()
    fbo.release()
