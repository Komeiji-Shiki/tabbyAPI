import pathlib
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from common import model
from endpoints.core.types.model import ModelLoadRequest
from endpoints.core.utils.model import persist_load_params, stream_model_load


class _FailingContainer:
    use_draft_model = False
    use_vision = False

    def __init__(self):
        self.unload = AsyncMock()

    async def load_gen(self, *_args, **_kwargs):
        if False:
            yield None
        raise RuntimeError("synthetic load failure")


class ModelLoadCleanupTests(unittest.IsolatedAsyncioTestCase):
    def test_config_save_only_persists_explicit_request_fields(self):
        data = ModelLoadRequest(
            model_name="test",
            max_seq_len=4096,
            cache_size=4100,
            draft_model={"draft_mode": "disabled"},
        )

        with patch(
            "endpoints.core.utils.model.update_config_file_and_memory"
        ) as update_config:
            updates = persist_load_params(data)

        self.assertEqual(
            updates,
            {
                "model": {
                    "model_name": "test",
                    "max_seq_len": 4096,
                    "cache_size": 4352,
                },
                "draft_model": {"draft_mode": "disabled"},
            },
        )
        update_config.assert_called_once_with(updates)

    def test_cache_size_rounds_up_instead_of_being_rejected(self):
        data = ModelLoadRequest(model_name="test", cache_size=4097)
        self.assertEqual(data.cache_size, 4097)
        container = model.ExllamaV3Container()
        self.assertEqual(container.adjust_cache_size(data.cache_size), 4352)

    async def test_failed_load_unloads_partial_container(self):
        failed = _FailingContainer()
        hf_model = Mock()
        hf_model.hf_config.get_max_position_embeddings.return_value = 4096
        progress = Mock()

        old_container = model.container
        model.container = None
        try:
            with (
                patch.object(model, "apply_load_defaults", AsyncMock(return_value={})),
                patch.object(model.HFModel, "from_directory", AsyncMock(return_value=hf_model)),
                patch.object(model, "validate_backend"),
                patch.object(
                    model.ExllamaV3Container,
                    "create",
                    AsyncMock(return_value=failed),
                ),
                patch.object(model, "get_loading_progress_bar", return_value=progress),
                patch.object(model, "_release_cuda_cache") as release_cache,
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic load failure"):
                    async for _ in model.load_model_gen(pathlib.Path("models/test")):
                        pass

            failed.unload.assert_awaited_once_with(skip_wait=True)
            release_cache.assert_called_once()
            self.assertIsNone(model.container)
            progress.stop.assert_called_once()
        finally:
            model.container = old_container

    async def test_transport_only_options_do_not_reload_same_model(self):
        model_path = pathlib.Path("models/test")
        old_container = model.container
        model.container = SimpleNamespace(
            model=object(),
            model_dir=model_path,
            loaded=True,
        )
        try:
            events = [
                event
                async for event in model.load_model_gen(
                    model_path,
                    skip_wait=False,
                    draft_model={"draft_model_dir": "draft_models"},
                )
            ]
            self.assertEqual(events, [(1, 1, model.ModelType.MODEL.value)])
            self.assertIsNotNone(model.container)
        finally:
            model.container = old_container

    async def test_explicit_runtime_option_reloads_same_model(self):
        model_path = pathlib.Path("models/test")
        old_container = model.container
        model.container = SimpleNamespace(
            model=object(),
            model_dir=model_path,
            loaded=True,
        )
        unload = AsyncMock()
        try:
            with (
                patch.object(model, "unload_model", unload),
                patch.object(
                    model,
                    "apply_load_defaults",
                    AsyncMock(side_effect=RuntimeError("stopped after unload")),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "stopped after unload"):
                    async for _ in model.load_model_gen(
                        model_path,
                        skip_wait=False,
                        draft_model={
                            "draft_model_dir": "draft_models",
                            "draft_mode": "disabled",
                        },
                    ):
                        pass

            unload.assert_awaited_once_with()
            self.assertIsNone(model.container)
        finally:
            model.container = old_container

    async def test_stream_load_does_not_forward_unsupplied_request_defaults(self):
        captured = {}

        async def fake_load_gen(_model_path, **kwargs):
            captured.update(kwargs)
            yield 1, 1, model.ModelType.MODEL.value

        data = ModelLoadRequest(model_name="test", cache_size=4100)
        with patch.object(model, "load_model_gen", new=fake_load_gen):
            chunks = [
                chunk
                async for chunk in stream_model_load(data, pathlib.Path("models/test"))
            ]

        self.assertEqual(len(chunks), 2)
        self.assertEqual(captured["skip_wait"], False)
        self.assertEqual(captured["cache_size"], 4352)
        self.assertEqual(set(captured), {"skip_wait", "draft_model", "cache_size"})
        self.assertEqual(
            set(captured["draft_model"]),
            {"draft_model_dir"},
        )

    async def test_load_defaults_keep_reasoning_config_without_use_as_default(self):
        configured_values = {
            "reasoning": True,
            "reasoning_start_token": "<think>",
            "reasoning_end_token": "</think>",
            "start_in_reasoning": "always",
            "tool_calls_in_reasoning": False,
            "reasoning_budget_tokens": 128,
            "reasoning_budget_message": "finish thinking",
            "template_vars_default": {"enable_thinking": True},
            "template_vars_force": {"enable_thinking": True},
            "force_enable_thinking": False,
            "tool_format": "qwen3",
            "harmony": False,
            "muse_glimmer": False,
            "vision": True,
            "vision_offload": True,
            "unrelated_runtime_setting": "must not leak",
        }
        model_config = SimpleNamespace(
            use_as_default=[],
            model_dump=Mock(return_value=configured_values),
        )
        fake_config = SimpleNamespace(
            model=model_config,
            model_defaults={"max_seq_len": 8192},
            draft_model_defaults={"draft_mode": "disabled"},
        )

        with patch.object(model, "config", fake_config):
            resolved = await model.apply_load_defaults(pathlib.Path("__missing_model__"))

        for key, value in configured_values.items():
            if key != "unrelated_runtime_setting":
                self.assertEqual(resolved[key], value)
        self.assertNotIn("unrelated_runtime_setting", resolved)
        self.assertEqual(resolved["max_seq_len"], 8192)
        self.assertEqual(resolved["draft_model"], {"draft_mode": "disabled"})

        with patch.object(model, "config", fake_config):
            overridden = await model.apply_load_defaults(
                pathlib.Path("__missing_model__"),
                reasoning=False,
            )
        self.assertFalse(overridden["reasoning"])


if __name__ == "__main__":
    unittest.main()
