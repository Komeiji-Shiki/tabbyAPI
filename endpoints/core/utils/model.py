import asyncio
import pathlib
from asyncio import CancelledError
from typing import Optional

from common import model
from common.networking import get_generator_error, handle_request_disconnect
from common.logger import xlogger
from common.tabby_config import config, update_config_file_and_memory
from endpoints.core.types.model import (
    ModelCard,
    ModelList,
    ModelLoadRequest,
    ModelLoadResponse,
)


def get_model_list(model_path: pathlib.Path, draft_model_path: Optional[str] = None):
    """Get the list of models from the provided path."""

    # Convert the provided draft model path to a pathlib path for
    # equality comparisons
    if draft_model_path:
        draft_model_path = pathlib.Path(draft_model_path).resolve()

    model_card_list = ModelList()
    for path in model_path.iterdir():
        # Don't include the draft models path
        if path.is_dir() and path != draft_model_path:
            model_card = ModelCard(id=path.name)
            model_card_list.data.append(model_card)  # pylint: disable=no-member

    return model_card_list


async def get_current_model_list(model_type: str = "model"):
    """
    Gets the current model in list format and with path only.

    Unified for fetching both models and embedding models.
    """

    current_models = []
    model_path = None

    # Make sure the model container exists
    match model_type:
        case "model":
            if model.container:
                model_path = model.container.model_dir
        case "draft":
            if model.container:
                model_path = model.container.draft_model_dir
        case "embedding":
            if model.embeddings_container:
                model_path = model.embeddings_container.model_dir

    if model_path:
        current_models.append(ModelCard(id=model_path.name))

    return ModelList(data=current_models)


def get_current_model():
    """Gets the current model with all parameters."""

    model_card = model.container.model_info()

    return model_card


def get_dummy_models():
    if config.model.dummy_model_names:
        return [ModelCard(id=dummy_id) for dummy_id in config.model.dummy_model_names]
    else:
        return [ModelCard(id="gpt-3.5-turbo")]


# Keep strong references to detached load tasks; asyncio only holds weak ones
_load_tasks: set = set()


def _normalise_cache_size(value: int) -> int:
    """Match exllamav3's page size without rejecting a convenient user value."""

    return ((value + 255) // 256) * 256


def persist_load_params(data: ModelLoadRequest) -> dict:
    """
    Map the flat ModelLoadRequest payload back into the nested config.yml structure
    and persist it (preserving comments via ruamel) and sync the live config.
    """
    # Only persist fields the caller actually sent. Pydantic request defaults
    # (for example output_chunking=True) must not silently overwrite config.yml.
    load_data = data.model_dump(exclude_none=True, exclude_unset=True)

    model_updates = {}
    draft_updates = {}

    # Top-level request fields map to the config.model section
    for key, value in load_data.items():
        if key in ("draft_model",):
            continue
        if hasattr(config.model, key):
            model_updates[key] = value

    if "cache_size" in model_updates:
        model_updates["cache_size"] = _normalise_cache_size(model_updates["cache_size"])

    # Nested draft_model request maps to the config.draft_model section
    draft_data = load_data.get("draft_model")
    if isinstance(draft_data, dict):
        for key, value in draft_data.items():
            if hasattr(config.draft_model, key):
                draft_updates[key] = value

    updates = {}
    if model_updates:
        updates["model"] = model_updates
    if draft_updates:
        updates["draft_model"] = draft_updates

    if updates:
        update_config_file_and_memory(updates)
        xlogger.info(f"Persisted runtime config: {list(updates.keys())}")

    return updates


async def stream_model_load(
    data: ModelLoadRequest,
    model_path: pathlib.Path,
):
    """Request generation wrapper for the loading process."""

    # Only forward settings that the caller actually supplied. Pydantic defaults
    # and transport flags must not make a same-model no-op look like a config reload.
    load_data = data.model_dump(
        exclude_none=True,
        exclude_unset=True,
        exclude={"model_name", "skip_queue", "persist"},
    )
    if "cache_size" in load_data:
        load_data["cache_size"] = _normalise_cache_size(load_data["cache_size"])

    # Set the draft model directory
    load_data.setdefault("draft_model", {})["draft_model_dir"] = config.draft_model.draft_model_dir

    # Drive the load in a detached task and observe it through a queue,
    # so a client disconnect doesn't cancel a load in progress
    progress_queue: asyncio.Queue = asyncio.Queue()

    async def run_load():
        try:
            load_status = model.load_model_gen(model_path, skip_wait=data.skip_queue, **load_data)
            async for progress in load_status:
                progress_queue.put_nowait(progress)

            progress_queue.put_nowait(None)
        except Exception as exc:
            # Do not put the original exception in the queue: its traceback retains
            # the failed container and CUDA tensors until the SSE request is collected.
            message = str(exc)
            exc.__traceback__ = None
            progress_queue.put_nowait(RuntimeError(message))

    load_task = asyncio.create_task(run_load())
    _load_tasks.add(load_task)
    load_task.add_done_callback(_load_tasks.discard)

    try:
        while True:
            progress = await progress_queue.get()

            if progress is None:
                break

            if isinstance(progress, Exception):
                yield get_generator_error(str(progress))
                break

            module, modules, model_type = progress
            if module != 0:
                response = ModelLoadResponse(
                    model_type=model_type,
                    module=module,
                    modules=modules,
                    status="processing",
                )

                yield response.model_dump_json()

            if module == modules:
                response = ModelLoadResponse(
                    model_type=model_type,
                    module=module,
                    modules=modules,
                    status="finished",
                )

                yield response.model_dump_json()

            # If main model loading completed and persist was requested, update the config file
            # (Avoid triggering it from draft/vision load events)
            if module == modules and model_type == "model" and getattr(data, "persist", False):
                try:
                    persist_load_params(data)
                except Exception as exc:
                    # Loading has already succeeded at this point. Keep the model available,
                    # but report the persistence failure in the server log.
                    xlogger.error(f"Failed to persist load parameters: {exc}")

    except CancelledError:
        # The client disconnected, but the load task keeps running.
        # A repeated request for the same model returns once this load finishes.

        handle_request_disconnect("Model load request disconnected. The load will continue.")
