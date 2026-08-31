from typing import List, TYPE_CHECKING

from pydantic import BaseModel, Field

from backends.exllamav3.vision import (
    get_image_embedding_exl3,
    get_video_embedding_exl3,
)

if TYPE_CHECKING:
    from backends.exllamav3.model import ExllamaV3Container


class MultimodalEmbeddingWrapper(BaseModel):
    """Common multimodal embedding wrapper"""

    content: list = Field(default_factory=list)
    text_alias: List[str] = Field(default_factory=list)

    def _append(self, embedding):
        embeddings = embedding if isinstance(embedding, list) else [embedding]
        self.content.extend(embeddings)
        self.text_alias.extend(item.text_alias for item in embeddings)

    async def add(self, container: "ExllamaV3Container", url: str):
        embedding = await get_image_embedding_exl3(container, url)
        self._append(embedding)

    async def add_video(
        self,
        container: "ExllamaV3Container",
        url: str,
        fps: float | None = None,
        num_frames: int | None = None,
        max_frames: int = 64,
    ):
        embeddings = await get_video_embedding_exl3(
            container,
            url,
            fps=fps,
            num_frames=num_frames,
            max_frames=max_frames,
        )
        self._append(embeddings)
