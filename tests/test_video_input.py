import base64
import io
import unittest

from PIL import Image
from pydantic import ValidationError

from common.image_util import (
    _desired_frame_count,
    get_image,
    get_video_frames,
    is_animated_image,
)
from endpoints.OAI.types.chat_completion import ChatCompletionMessagePart


def animated_gif_data_url() -> str:
    frames = [
        Image.new("RGB", (16, 16), color)
        for color in ("red", "green", "blue", "white")
    ]
    output = io.BytesIO()
    frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    payload = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/gif;base64,{payload}"


class VideoInputSchemaTests(unittest.TestCase):
    def test_video_url_part(self):
        part = ChatCompletionMessagePart.model_validate(
            {
                "type": "video_url",
                "video_url": {"url": "https://example.test/video.mp4", "fps": 1.5},
            }
        )
        self.assertEqual(part.video_url.fps, 1.5)
        self.assertEqual(part.video_url.max_frames, 64)

    def test_fps_and_num_frames_are_mutually_exclusive(self):
        with self.assertRaises(ValidationError):
            ChatCompletionMessagePart.model_validate(
                {
                    "type": "video_url",
                    "video_url": {
                        "url": "https://example.test/video.mp4",
                        "fps": 2,
                        "num_frames": 8,
                    },
                }
            )

    def test_num_frames_cannot_exceed_cap(self):
        with self.assertRaises(ValidationError):
            ChatCompletionMessagePart.model_validate(
                {
                    "type": "video_url",
                    "video_url": {
                        "url": "https://example.test/video.mp4",
                        "num_frames": 16,
                        "max_frames": 8,
                    },
                }
            )


class AnimatedImageTests(unittest.IsolatedAsyncioTestCase):
    async def test_image_loader_preserves_animation(self):
        image = await get_image(animated_gif_data_url())
        self.assertTrue(is_animated_image(image))
        self.assertEqual(image.n_frames, 4)

    async def test_gif_is_sampled_as_video(self):
        batch = await get_video_frames(
            animated_gif_data_url(),
            num_frames=3,
            max_frames=8,
        )
        self.assertEqual(len(batch.frames), 3)
        self.assertEqual(batch.frame_indices, [0, 2, 3])
        for actual, expected in zip(batch.timestamps, [0.0, 0.2, 0.3]):
            self.assertAlmostEqual(actual, expected)
        self.assertTrue(all(frame.mode == "RGB" for frame in batch.frames))

    def test_default_sampling_uses_qwen_minimum(self):
        self.assertEqual(_desired_frame_count(30, 30.0, None, None, 64), 4)

    async def test_mp4_is_decoded_and_sampled(self):
        import av

        output = io.BytesIO()
        with av.open(output, mode="w", format="mp4") as container:
            stream = container.add_stream("mpeg4", rate=10)
            stream.width = 64
            stream.height = 48
            stream.pix_fmt = "yuv420p"
            for index in range(10):
                image = Image.new("RGB", (64, 48), (index * 20, 0, 0))
                for packet in stream.encode(av.VideoFrame.from_image(image)):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

        payload = base64.b64encode(output.getvalue()).decode("ascii")
        batch = await get_video_frames(
            f"data:video/mp4;base64,{payload}",
            num_frames=4,
        )
        self.assertEqual(len(batch.frames), 4)
        self.assertEqual(batch.frame_indices, [0, 3, 6, 9])
        self.assertTrue(all(frame.size == (64, 48) for frame in batch.frames))


if __name__ == "__main__":
    unittest.main()
