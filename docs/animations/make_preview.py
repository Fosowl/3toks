"""Make a small looping GIF from the decision-tree video.

The README shows this GIF because GitHub does not play MP4 files that
sit inside a repository. Use it after you render a new video.
It needs only PyAV and Pillow, which Manim already installs.

Run: python3 docs/animations/make_preview.py [video.mp4] [preview.gif]
"""
import sys
from fractions import Fraction
from pathlib import Path

import av
from PIL import Image

HERE = Path(__file__).resolve().parent
PREVIEW_WIDTH = 800
PREVIEW_FPS = 12
PALETTE_COLORS = 255
PALETTE_SAMPLES = 24
LAST_FRAME_HOLD_MS = 2500


def sample_frames(video_path: Path, width: int, fps: int) -> list:
    """Read the video and keep about `fps` frames for each second.

    The GIF writer calls this once. It gives every kept frame as a
    small RGB picture.

    Args:
        video_path (Path): Video file to read.
        width (int): Width of each kept frame in pixels.
        fps (int): Number of frames to keep for each second.

    Returns:
        list[PIL.Image.Image]: Resized frames in play order.
    """
    frames = []
    step = Fraction(1, fps)
    next_time = Fraction(0)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts * stream.time_base < next_time:
                continue
            frames.append(_shrink(frame.to_image(), width))
            next_time += step
    return frames


def _shrink(image: Image.Image, width: int) -> Image.Image:
    """Resize one picture to `width` pixels and keep its shape.

    Args:
        image (PIL.Image.Image): Picture to resize.
        width (int): New width in pixels.

    Returns:
        PIL.Image.Image: The smaller picture.
    """
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.LANCZOS)


def build_palette(frames: list) -> Image.Image:
    """Make one shared color palette from frames spread over the video.

    One palette for all frames stops colors from flickering.

    Args:
        frames (list[PIL.Image.Image]): All GIF frames.

    Returns:
        PIL.Image.Image: A palette picture for `Image.quantize`.
    """
    stride = max(1, len(frames) // PALETTE_SAMPLES)
    picks = frames[::stride]
    sheet = Image.new("RGB", (picks[0].width, picks[0].height * len(picks)))
    for index, pick in enumerate(picks):
        sheet.paste(pick, (0, index * pick.height))
    return sheet.quantize(PALETTE_COLORS, method=Image.MEDIANCUT)


def save_gif(frames: list, palette: Image.Image, out_path: Path) -> None:
    """Write the frames as a looping GIF.

    Args:
        frames (list[PIL.Image.Image]): Frames in play order.
        palette (PIL.Image.Image): Shared palette from `build_palette`.
        out_path (Path): GIF file to write.

    Returns:
        None
    """
    indexed = [f.quantize(palette=palette, dither=Image.NONE) for f in frames]
    delay = round(1000 / PREVIEW_FPS)
    durations = [delay] * (len(indexed) - 1) + [LAST_FRAME_HOLD_MS]
    indexed[0].save(out_path, save_all=True, append_images=indexed[1:],
                    duration=durations, loop=0, optimize=True)


def main(argv: list) -> int:
    """Build the preview GIF from the command line.

    Args:
        argv (list[str]): Optional video path, then optional GIF path.

    Returns:
        int: Exit code, 0 when the GIF is written.
    """
    video = Path(argv[0]) if argv else HERE / "decision_tree.mp4"
    target = Path(argv[1]) if len(argv) > 1 else HERE / "decision_tree.gif"
    frames = sample_frames(video, PREVIEW_WIDTH, PREVIEW_FPS)
    save_gif(frames, build_palette(frames), target)
    print(f"{target.name}: {len(frames)} frames, "
          f"{target.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
