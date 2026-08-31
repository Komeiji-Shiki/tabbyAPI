"""Vision utilities for ExLlamaV3."""

from typing import TYPE_CHECKING

from common.optional_dependencies import dependencies
from common.image_util import (
    DEFAULT_MAX_VIDEO_FRAMES,
    VideoFrameBatch,
    get_animated_image_frames,
    get_image,
    get_video_frames,
    is_animated_image,
)
from common.logger import xlogger
from fastapi import HTTPException

# Since this is used outside the Exl3 backend, the dependency
# may be optional
if dependencies.exllamav3:
    from exllamav3.tokenizer import MMEmbedding

if TYPE_CHECKING:
    from backends.exllamav3.model import ExllamaV3Container

from collections import OrderedDict
from hashlib import blake2b
from typing import OrderedDict as OrderedDictType

_EMBEDDING_CACHE_CAPACITY = 32
_embedding_cache: OrderedDictType[bytes, tuple[str, object]] = OrderedDict()


def _image_key_128(s: str) -> bytes:
    return blake2b(s.encode("utf-8"), digest_size=16).digest()


def _get_cached_embedding(identity: str):
    key = _image_key_128(identity)

    cached = _embedding_cache.get(key)
    if cached is not None:
        cached_identity, embedding = cached
        if cached_identity == identity:
            _embedding_cache.move_to_end(key)
            return embedding
    return None


def _store_cached_embedding(identity: str, embedding):
    key = _image_key_128(identity)
    _embedding_cache[key] = (identity, embedding)
    _embedding_cache.move_to_end(key)

    if len(_embedding_cache) > _EMBEDDING_CACHE_CAPACITY:
        _embedding_cache.popitem(last=False)

    embeddings = embedding if isinstance(embedding, list) else [embedding]
    xlogger.debug(
        "Stored MMEmbedding",
        {
            "embedding_count": len(embeddings),
            "metadata": [item.metadata for item in embeddings],
            "token_length": sum(item.mm_length for item in embeddings),
            "cache_size": len(_embedding_cache),
        },
    )


def _encode_video_frames(
    container: "ExllamaV3Container",
    frame_batch: VideoFrameBatch,
) -> list["MMEmbedding"]:
    get_video_embeddings = getattr(container.vision_model, "get_video_embeddings", None)
    if get_video_embeddings is None:
        raise HTTPException(400, "The loaded vision model does not support video input.")

    return get_video_embeddings(
        tokenizer=container.tokenizer,
        frames=frame_batch.frames,
        timestamps=frame_batch.timestamps,
        text_alias=None,
    )


async def get_image_embedding_exl3(
    container: "ExllamaV3Container",
    url: str,
) -> "MMEmbedding | list[MMEmbedding]":
    identity = f"image:{url}"
    cached = _get_cached_embedding(identity)
    if cached is not None:
        return cached

    image = await get_image(url)
    if is_animated_image(image):
        frame_batch = await get_animated_image_frames(image)
        embedding = _encode_video_frames(container, frame_batch)
    else:
        embedding = container.vision_model.get_image_embeddings(
            tokenizer=container.tokenizer,
            image=image,
            text_alias=None,
        )

    _store_cached_embedding(identity, embedding)

    return embedding


async def get_video_embedding_exl3(
    container: "ExllamaV3Container",
    url: str,
    fps: float | None = None,
    num_frames: int | None = None,
    max_frames: int = DEFAULT_MAX_VIDEO_FRAMES,
) -> list["MMEmbedding"]:
    identity = f"video:{fps}:{num_frames}:{max_frames}:{url}"
    cached = _get_cached_embedding(identity)
    if cached is not None:
        return cached

    frame_batch = await get_video_frames(
        url,
        fps=fps,
        num_frames=num_frames,
        max_frames=max_frames,
    )
    embeddings = _encode_video_frames(container, frame_batch)
    _store_cached_embedding(identity, embeddings)
    return embeddings


def clear_image_embedding_cache():
    _embedding_cache.clear()
