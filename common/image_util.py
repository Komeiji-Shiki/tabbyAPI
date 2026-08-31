import asyncio
import aiohttp
import base64
import binascii
import io
import re
from dataclasses import dataclass

from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

from common.networking import handle_request_error
from common.tabby_config import config


DEFAULT_VIDEO_FPS = 2.0
DEFAULT_MAX_VIDEO_FRAMES = 64
MIN_VIDEO_FRAMES = 4

_DATA_URL_RE = re.compile(
    r"^data:(?P<mime>[^;,]+);base64,(?P<payload>.*)$",
    flags=re.DOTALL,
)


@dataclass
class VideoFrameBatch:
    frames: list[Image.Image]
    timestamps: list[float]
    source_fps: float | None
    frame_indices: list[int]


class VideoDecoderUnavailableError(RuntimeError):
    pass


def _http_error(message: str, status_code: int = 400) -> HTTPException:
    error_message = handle_request_error(message, exc_info=False).error.message
    return HTTPException(status_code, error_message)


async def _read_media_bytes(
    url: str,
    allowed_data_mime_prefixes: tuple[str, ...],
    media_name: str,
) -> bytes:
    if url.startswith("data:"):
        match = _DATA_URL_RE.match(url)
        if not match or not match.group("mime").startswith(allowed_data_mime_prefixes):
            raise _http_error(f"Failed to read base64 {media_name} input.")

        try:
            return base64.b64decode(match.group("payload"), validate=True)
        except (binascii.Error, ValueError):
            raise _http_error(f"Failed to read base64 {media_name} input.") from None

    if config.network.disable_fetch_requests:
        raise _http_error(
            f"Failed to fetch {media_name} from {url} as fetch requests are disabled."
        )

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.read()
    except aiohttp.ClientError:
        pass

    raise _http_error(f"Failed to fetch {media_name} from {url}.")


async def get_image(url: str) -> Image.Image:
    bytes_image = await _read_media_bytes(url, ("image/",), "image")
    try:
        return Image.open(io.BytesIO(bytes_image))
    except (UnidentifiedImageError, OSError):
        raise _http_error("Failed to decode image input.") from None


def is_animated_image(image: Image.Image) -> bool:
    return bool(getattr(image, "is_animated", False) and getattr(image, "n_frames", 1) > 1)


def _desired_frame_count(
    total_frames: int,
    source_fps: float | None,
    fps: float | None,
    num_frames: int | None,
    max_frames: int,
) -> int:
    if total_frames < 1:
        raise ValueError("Video contains no frames")
    if fps is not None and num_frames is not None:
        raise ValueError("Only one of fps and num_frames can be specified")
    if max_frames < 1:
        raise ValueError("max_frames must be positive")

    if num_frames is None:
        target_fps = fps or DEFAULT_VIDEO_FPS
        if source_fps and source_fps > 0:
            desired = int(total_frames / source_fps * target_fps)
        else:
            desired = total_frames
        desired = max(desired, min(MIN_VIDEO_FRAMES, total_frames))
    else:
        desired = num_frames

    return min(max(desired, 1), max_frames, total_frames)


def _linspace_indices(total_frames: int, count: int) -> list[int]:
    if count == 1:
        return [0]
    return [round(i * (total_frames - 1) / (count - 1)) for i in range(count)]


def _decode_pillow_frames(
    image: Image.Image,
    fps: float | None,
    num_frames: int | None,
    max_frames: int,
) -> VideoFrameBatch:
    total_frames = getattr(image, "n_frames", 1)
    frame_timestamps = []
    elapsed = 0.0

    for index in range(total_frames):
        image.seek(index)
        frame_timestamps.append(elapsed)
        duration_ms = image.info.get("duration", 100)
        if not isinstance(duration_ms, (int, float)) or duration_ms <= 0:
            duration_ms = 100
        elapsed += duration_ms / 1000

    source_fps = total_frames / elapsed if elapsed > 0 else None
    desired = _desired_frame_count(
        total_frames,
        source_fps,
        fps,
        num_frames,
        max_frames,
    )
    indices = _linspace_indices(total_frames, desired)

    frames = []
    for index in indices:
        image.seek(index)
        frames.append(image.convert("RGB").copy())

    return VideoFrameBatch(
        frames=frames,
        timestamps=[frame_timestamps[index] for index in indices],
        source_fps=source_fps,
        frame_indices=indices,
    )


def _fraction_to_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return result if result > 0 else None


def _decode_video_bytes(
    bytes_video: bytes,
    fps: float | None,
    num_frames: int | None,
    max_frames: int,
) -> VideoFrameBatch:
    try:
        import av
    except ImportError as exc:
        raise VideoDecoderUnavailableError from exc

    def open_video():
        return av.open(io.BytesIO(bytes_video))

    with open_video() as container:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise ValueError("Input contains no video stream")

        source_fps = _fraction_to_float(stream.average_rate)
        duration = None
        if stream.duration is not None and stream.time_base is not None:
            duration = _fraction_to_float(stream.duration * stream.time_base)

        total_frames = int(stream.frames or 0)
        if total_frames < 1 and duration and source_fps:
            total_frames = max(1, round(duration * source_fps))

        if total_frames < 1:
            total_frames = sum(1 for _ in container.decode(video=stream.index))

    desired = _desired_frame_count(
        total_frames,
        source_fps,
        fps,
        num_frames,
        max_frames,
    )
    indices = _linspace_indices(total_frames, desired)
    wanted = set(indices)
    frames = []
    timestamps = []
    decoded_indices = []

    with open_video() as container:
        stream = next(item for item in container.streams if item.type == "video")
        for index, frame in enumerate(container.decode(video=stream.index)):
            if index in wanted:
                frames.append(frame.to_image().convert("RGB"))
                frame_time = _fraction_to_float(frame.time)
                if frame_time is None:
                    frame_time = index / source_fps if source_fps else float(index)
                timestamps.append(frame_time)
                decoded_indices.append(index)
            if index >= indices[-1]:
                break

    if not frames:
        raise ValueError("Video decoder returned no frames")

    return VideoFrameBatch(
        frames=frames,
        timestamps=timestamps,
        source_fps=source_fps,
        frame_indices=decoded_indices,
    )


async def get_animated_image_frames(
    image: Image.Image,
    fps: float | None = None,
    num_frames: int | None = None,
    max_frames: int = DEFAULT_MAX_VIDEO_FRAMES,
) -> VideoFrameBatch:
    try:
        return await asyncio.to_thread(
            _decode_pillow_frames,
            image,
            fps,
            num_frames,
            max_frames,
        )
    except (EOFError, OSError, ValueError) as exc:
        raise _http_error(f"Failed to decode animated image input: {exc}") from None


async def get_video_frames(
    url: str,
    fps: float | None = None,
    num_frames: int | None = None,
    max_frames: int = DEFAULT_MAX_VIDEO_FRAMES,
) -> VideoFrameBatch:
    bytes_video = await _read_media_bytes(url, ("video/", "image/"), "video")

    try:
        image = Image.open(io.BytesIO(bytes_video))
    except (UnidentifiedImageError, OSError):
        image = None

    if image is not None:
        return await get_animated_image_frames(image, fps, num_frames, max_frames)

    try:
        return await asyncio.to_thread(
            _decode_video_bytes,
            bytes_video,
            fps,
            num_frames,
            max_frames,
        )
    except VideoDecoderUnavailableError:
        raise _http_error(
            "Video input requires the PyAV package to be installed.",
            status_code=501,
        ) from None
    except (EOFError, OSError, ValueError) as exc:
        raise _http_error(f"Failed to decode video input: {exc}") from None
