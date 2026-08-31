"""Manages the lifecycle of the global model container and embeddings container."""

import aiofiles
import asyncio
import gc
import pathlib
from enum import Enum
from fastapi import HTTPException
from common.logger import xlogger
from ruamel.yaml import YAML
from typing import Optional

from common.errors import ContextLengthExceededError, ContextLengthHTTPException
from common.logger import get_loading_progress_bar
from common.multimodal import MultimodalEmbeddingWrapper
from common.networking import handle_request_error
from common.sampling import BaseSamplerRequest
from common.tabby_config import config
from common.optional_dependencies import dependencies
from common.transformers_utils import HFModel
from common.utils import deep_merge_dicts, unwrap

if dependencies.exllamav3:
    from backends.exllamav3.model import ExllamaV3Container

# Global variables for model container
container: Optional["ExllamaV3Container"] = None
embeddings_container = None

# Serializes model loads and swaps. The container's load_lock is per-instance
# and can't order operations that span two containers.
load_lock = asyncio.Lock()


def _release_cuda_cache():
    """Best-effort cleanup for failed loads, including partially built containers."""

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        # Cleanup must never hide the original model-load error.
        xlogger.debug(f"Unable to empty the CUDA cache after a failed load: {exc}")


# A broken infinity-emb install (e.g. missing transitive dependencies) only
# disables embeddings instead of preventing server startup
_extras_import_error: Optional[str] = None
if dependencies.extras:
    try:
        from backends.infinity.model import InfinityContainer

        embeddings_container: Optional[InfinityContainer] = None
    except Exception as exc:
        _extras_import_error = str(exc)
        xlogger.warning(f"Embeddings are disabled because infinity-emb failed to import: {exc}")


class ModelType(Enum):
    MODEL = "model"
    DRAFT = "draft"
    VISION = "vision"


# These settings affect how generated text is parsed after a model reload.
# They must follow the live model config even when a UI load request only sends
# hardware/runtime settings and does not list them in `use_as_default`.
REASONING_FIELDS = frozenset(
    {
        "reasoning",
        "reasoning_start_token",
        "reasoning_end_token",
        "start_in_reasoning",
        "tool_calls_in_reasoning",
        "reasoning_budget_tokens",
        "reasoning_budget_message",
        "template_vars_default",
        "template_vars_force",
        "force_enable_thinking",
        "tool_format",
        "harmony",
        "muse_glimmer",
        "vision",
        "vision_offload",
    }
)


def load_progress(module, modules):
    """Wrapper callback for load progress."""
    yield module, modules


def validate_backend(backend: Optional[str], hf_model: HFModel):
    """Check that the requested model can be loaded with the exllamav3 backend."""

    if backend == "exllamav2":
        raise ValueError("The exllamav2 backend is no longer supported. Please use exllamav3.")
    elif backend and backend != "exllamav3":
        raise ValueError(f"Invalid backend '{backend}'. Available backends: ['exllamav3']")

    quant_method = hf_model.quant_method()
    if quant_method in {"exl2", "gptq"}:
        raise ValueError(
            f"Models quantized with '{quant_method}' require the exllamav2 backend, "
            "which is no longer supported. Please use an exl3 or unquantized model."
        )

    if not dependencies.exllamav3:
        raise ValueError(
            "The exllamav3 backend is selected, but required dependencies are not installed."
        )


async def apply_load_defaults(model_path: pathlib.Path, **kwargs):
    """
    Applies model load overrides.

    Priority, lowest to highest: use_as_default keys, the load kwargs
    (startup config or API request), then the model folder's
    tabby_config.yml, which always wins when present.
    """

    override_config_path = model_path / "tabby_config.yml"

    # Inline overrides from the model folder's tabby_config.yml
    inline_overrides = {"draft_model": {}}

    if override_config_path.exists():
        async with aiofiles.open(
            override_config_path, "r", encoding="utf8"
        ) as override_config_file:
            contents = await override_config_file.read()

            # Create a temporary YAML parser
            yaml = YAML(typ="safe")
            inline_config = unwrap(yaml.load(contents), {})

            model_inline_config = unwrap(inline_config.get("model"), {})
            if model_inline_config:
                inline_overrides = {**inline_overrides, **model_inline_config}
            else:
                xlogger.warning(
                    "Cannot find inline model overrides. "
                    'Make sure they are nested under a "model:" key'
                )

            draft_inline_config = unwrap(inline_config.get("draft_model"), {})
            if draft_inline_config:
                inline_overrides["draft_model"] = draft_inline_config

            xlogger.info(f"Applying model overrides from {override_config_path}")

    # use_as_default keys plus parser settings that must survive UI reloads.
    # The cached model_defaults only contains use_as_default keys, so read the
    # live Pydantic model for the reasoning-related fields as well.
    model_default_fields = set(config.model.use_as_default) | REASONING_FIELDS
    configured_model_defaults = {
        key: value
        for key, value in config.model.model_dump().items()
        if key in model_default_fields
    }
    defaults = {
        **config.model_defaults,
        **configured_model_defaults,
        "draft_model": {**config.draft_model_defaults},
    }

    merged_kwargs = deep_merge_dicts(defaults, kwargs, inline_overrides)

    xlogger.debug(
        "Applying load defaults",
        {
            "kwargs": kwargs,
            "defaults": defaults,
            "inline_overrides": inline_overrides,
            "merged_kwargs": merged_kwargs,
        },
    )
    return merged_kwargs


async def unload_model(skip_wait: bool = False, shutdown: bool = False):
    """Unloads a model"""
    global container

    await container.unload(skip_wait=skip_wait, shutdown=shutdown)
    container = None


async def load_model_gen(model_path: pathlib.Path, **kwargs):
    """Generator to load a model"""
    global container

    async with load_lock:
        # 判断本次请求是否携带配置更新
        # 排除掉标志位和路径信息，如果有其他的键，说明要应用新参数，不能走“同模型跳过加载”的捷径
        draft_args = kwargs.get("draft_model")
        config_keys = [
            key
            for key in kwargs
            if key
            not in (
                "model_name",
                "skip_queue",
                "skip_wait",
                "persist",
                "draft_model",
            )
        ]
        has_config_args = bool(config_keys)
        if isinstance(draft_args, dict):
            # draft_model_dir is injected for every request and is not itself a
            # runtime setting. Any other nested field requires a real reload.
            meaningful_draft_args = {
                key: value
                for key, value in draft_args.items()
                if key != "draft_model_dir"
            }
            has_config_args = has_config_args or bool(meaningful_draft_args)

        # Check if the model is already loaded. Compare full resolved paths:
        # quant-style layouts give unrelated models the same directory
        # basename (e.g. <model_a>/exl3/4.00bpw vs <model_b>/exl3/4.00bpw),
        # and a name comparison would silently skip the swap.
        if container and container.model:
            loaded_model_dir = container.model_dir

            if loaded_model_dir.resolve() == model_path.resolve() and container.loaded:
                if not has_config_args:
                    xlogger.info(f'Model "{loaded_model_dir.name}" is already loaded')

                    # Emit a terminal progress event so API clients always
                    # see a "finished" status even when no load was needed
                    yield 1, 1, ModelType.MODEL.value
                    return
                else:
                    xlogger.info(
                        f'Reloading model "{loaded_model_dir.name}" with updated config.'
                    )

            if container.loaded:
                xlogger.info("Unloading existing model.")
                await unload_model()

        # Reset to prepare for a new container
        container = None

        # Model_dir is already provided
        if "model_dir" in kwargs:
            kwargs.pop("model_dir")

        # Merge with config and inline defaults
        # TODO: Figure out a way to do this with Pydantic validation
        # and ModelLoadRequest. Pydantic doesn't have async validators
        kwargs = await apply_load_defaults(model_path, **kwargs)

        # Fetch the extra HF configuration options
        hf_model = await HFModel.from_directory(model_path)

        # Override the max sequence length based on user
        max_seq_len = kwargs.get("max_seq_len")
        if max_seq_len == -1:
            kwargs["max_seq_len"] = hf_model.hf_config.get_max_position_embeddings()

        # Check model compatibility and dependencies before creating a container
        validate_backend(kwargs.get("backend"), hf_model)

        new_container = None
        progress = None
        try:
            new_container = await ExllamaV3Container.create(
                model_path.resolve(), hf_model, **kwargs
            )

            # Add possible types of models that can be loaded
            model_type = [ModelType.MODEL]

            if new_container.use_draft_model:
                model_type.insert(0, ModelType.DRAFT)

            if new_container.use_vision:
                model_type.insert(0, ModelType.VISION)

            load_status = new_container.load_gen(load_progress, **kwargs)

            progress = get_loading_progress_bar()
            progress.start()

            index = 0
            async for module, modules in load_status:
                current_model_type = model_type[index].value
                if module == 0:
                    loading_task = progress.add_task(
                        f"[cyan]Loading {current_model_type} modules", total=modules
                    )
                else:
                    progress.advance(loading_task)

                yield module, modules, current_model_type

                if module == modules:
                    # Switch to model progress if the draft model is loaded
                    if index == len(model_type):
                        progress.stop()
                    else:
                        index += 1

            container = new_container
        except Exception:
            # The new container is intentionally not published globally until loading
            # completes. It still owns every partially allocated CUDA tensor, so it
            # must be explicitly unloaded before the error is returned to the client.
            container = None
            if new_container is not None:
                xlogger.warning("Model load failed. Releasing partially loaded resources.")
                try:
                    await new_container.unload(skip_wait=True)
                except Exception as cleanup_exc:
                    xlogger.error(f"Failed to unload the partial model: {cleanup_exc}")
            _release_cuda_cache()
            raise
        finally:
            if progress is not None:
                progress.stop()


async def load_model(model_path: pathlib.Path, **kwargs):
    async for _ in load_model_gen(model_path, **kwargs):
        pass


async def load_loras(lora_dir, **kwargs):
    """Wrapper to load loras."""
    if len(container.get_loras()) > 0:
        await unload_loras()

    return await container.load_loras(lora_dir, **kwargs)


async def unload_loras():
    """Wrapper to unload loras"""
    await container.unload(loras_only=True)


async def load_embedding_model(model_path: pathlib.Path, **kwargs):
    global embeddings_container

    # Break out if infinity isn't installed
    if not dependencies.extras:
        raise ImportError(
            "Skipping embeddings because infinity-emb is not installed.\n"
            "Please run the following command in your environment "
            "to install extra packages:\n"
            "pip install -U .[extras]"
        )

    # Break out if infinity is installed but not importable
    if _extras_import_error is not None:
        raise ImportError(
            f"Embeddings are disabled because infinity-emb failed to import: {_extras_import_error}"
        )

    # Check if the model is already loaded (by full path, same as load_model_gen)
    if embeddings_container and embeddings_container.engine:
        loaded_model_dir = embeddings_container.model_dir

        if loaded_model_dir.resolve() == model_path.resolve() and embeddings_container.loaded:
            raise ValueError(
                f'Embeddings model "{loaded_model_dir.name}" is already loaded! Aborting.'
            )

        xlogger.info("Unloading existing embeddings model.")
        await unload_embedding_model()

    # Reset to prepare for a new container
    embeddings_container = None

    new_embeddings_container = InfinityContainer(model_path)
    await new_embeddings_container.load(**kwargs)

    embeddings_container = new_embeddings_container


async def unload_embedding_model():
    global embeddings_container

    await embeddings_container.unload()
    embeddings_container = None


async def check_model_container():
    """FastAPI depends that checks if a model isn't loaded or currently loading."""

    if container is None:
        error_message = handle_request_error(
            "No models are currently loaded.",
            exc_info=False,
        ).error.message

        raise HTTPException(503, error_message)


async def check_embeddings_container():
    """
    FastAPI depends that checks if an embeddings model is loaded.

    This is the same as the model container check, but with embeddings instead.
    """

    if embeddings_container is None:
        error_message = handle_request_error(
            "No embedding models are currently loaded.",
            exc_info=False,
        ).error.message

        raise HTTPException(503, error_message)


def check_context_length(
    prompts: str | list[str],
    params: BaseSamplerRequest,
    mm_embeddings: Optional[MultimodalEmbeddingWrapper] = None,
):
    """Reject oversized prompts before a streaming response commits HTTP 200."""

    if isinstance(prompts, str):
        prompts = [prompts]

    try:
        for prompt in prompts:
            container.validate_context_length(prompt, params, mm_embeddings)
    except ContextLengthExceededError as exc:
        error_message = handle_request_error(str(exc), exc_info=False).error.message
        raise ContextLengthHTTPException(error_message) from exc
