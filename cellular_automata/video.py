import logging
from typing import Optional

import cv2
import jax
import jax.numpy as jnp
import moderngl
import numpy as np
import pygame

logger = logging.getLogger(__name__)


@jax.jit
def default_to_color_fn(frames: jnp.ndarray) -> np.ndarray:
    """
    Simple grayscale mapping using just channel 0, assumed ~ [-1..1].
    Optimized version that processes the entire batch at once.
    JIT-compiled for better performance.
    """
    channel0 = frames[..., 0]
    normalized = ((jnp.clip(channel0, -1.0, 1.0) + 1.0) * 0.5 * 255.0).astype(jnp.uint8)
    return jnp.stack([normalized, normalized, normalized], axis=-1)


def export_video(
    states: jnp.ndarray,
    simulation_dt: float,
    video_dt: float,
    video_fps: int,
    to_color_fn=default_to_color_fn,
    out_filename: Optional[str] = None,
    play: bool = False,
):
    # sample frames based on video_dt, simulation_dt and video_fps
    sample_rate = int(video_dt / simulation_dt) // video_fps
    sampled_states = states[::sample_rate]

    logger.info("Converting frames to color")
    color_frames_jax = to_color_fn(sampled_states)
    logger.info("Frames converted to color")

    logger.info("Converting frames to numpy array")
    color_frames = jax.device_get(color_frames_jax)
    # color_frames = color_frames_jax
    logger.info("Frames converted to numpy array")

    if out_filename is not None:
        logger.info("Saving video")
        save_video(color_frames=color_frames, fps=video_fps, out_filename=out_filename)
        logger.info("Video saved")

    if play:
        play_video(color_frames=color_frames, fps=video_fps)


def play_video(color_frames: np.ndarray, fps: int):
    pygame.init()
    height, width = color_frames.shape[1:3]
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
    )
    screen = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)

    # Initialize ModernGL
    ctx = moderngl.create_context()

    # Create texture and program for rendering
    prog = ctx.program(
        vertex_shader="""
            #version 330
            in vec2 in_vert;
            in vec2 in_texcoord;
            out vec2 v_texcoord;
            void main() {
                gl_Position = vec4(in_vert, 0.0, 1.0);
                v_texcoord = in_texcoord;
            }
        """,
        fragment_shader="""
            #version 330
            uniform sampler2D texture0;
            in vec2 v_texcoord;
            out vec4 f_color;
            void main() {
                f_color = texture(texture0, v_texcoord);
            }
        """,
    )

    # Create vertex buffer and texture coordinates
    vertices = np.array([-1, -1, -1, 1, 1, -1, 1, 1], dtype="f4")
    texcoords = np.array([0, 1, 0, 0, 1, 1, 1, 0], dtype="f4")

    vbo = ctx.buffer(vertices.tobytes())
    tbo = ctx.buffer(texcoords.tobytes())
    vao = ctx.vertex_array(
        prog,
        [
            (vbo, "2f", "in_vert"),
            (tbo, "2f", "in_texcoord"),
        ],
    )

    # Create texture
    texture = ctx.texture((width, height), 3)
    texture.use(0)
    prog["texture0"].value = 0

    frame_idx = 0
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get((pygame.QUIT, pygame.KEYDOWN)):
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_q
            ):
                running = False
                break

        # Update texture with new frame
        texture.write(color_frames[frame_idx].tobytes())

        # Render
        ctx.clear()
        vao.render(mode=moderngl.TRIANGLE_STRIP)
        pygame.display.flip()

        frame_idx = (frame_idx + 1) % len(color_frames)
        clock.tick_busy_loop(fps)

    # Cleanup
    pygame.quit()


def save_video(
    color_frames: jnp.ndarray,
    fps: int,
    out_filename: str,
):
    frames, height, width, channels = color_frames.shape

    writer = cv2.VideoWriter(
        filename=out_filename,
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=fps,
        frameSize=(width, height),
        isColor=True,
    )

    # Use numpy's vectorize with correct signature format
    np.vectorize(writer.write, signature="(m,n,c)->()")(color_frames)
    # Or just use a simple loop since VideoWriter.write is inherently sequential
    # for frame in color_frames:
    #     writer.write(frame)

    writer.release()
    print(f"Saved {frames} frames to {out_filename}")
