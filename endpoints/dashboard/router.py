"""
Web dashboard endpoints: a single-page monitor UI plus the JSON APIs it polls.

The page itself is served without authentication (it only contains layout and
JavaScript); every data endpoint requires a normal API key, and the mutating
endpoints require the admin key. Frontend actions reuse the existing core
APIs (/v1/model, /v1/templates, /v1/sampling/...) rather than duplicating
their logic here.
"""

import pathlib
from inspect import getdoc
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from common import metrics, model
from common.auth import check_admin_key, check_api_key
from common.config_models import BaseConfigModel
from common.health import HealthManager
from common.logger import xlogger
from common.tabby_config import config
from endpoints.core.utils.model import get_current_model

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

DASHBOARD_FILE = pathlib.Path(__file__).parent / "dashboard.html"
_page_cache: Optional[str] = None


class CancelJobRequest(BaseModel):
    """Request body for cancelling one active generation job."""

    request_id: str


def _load_page() -> str:
    global _page_cache

    if _page_cache is None:
        _page_cache = DASHBOARD_FILE.read_text(encoding="utf8")

    return _page_cache


def _config_snapshot() -> dict:
    """
    Serialize the live config with field descriptions, so the dashboard can
    show what each option means without duplicating the help text.
    """

    from common.config_models import TabbyConfigModel

    sections = {}
    for section_name, section_field in TabbyConfigModel.model_fields.items():
        section_model = getattr(config, section_name, None)
        if not isinstance(section_model, BaseConfigModel):
            continue

        section_meta = getattr(section_model, "_metadata", None)
        if section_meta is not None and not section_meta.include_in_config:
            continue

        dumped = section_model.model_dump(mode="json")
        fields = type(section_model).model_fields

        entries = []
        for key, field in fields.items():
            entries.append(
                {
                    "key": key,
                    "value": dumped.get(key),
                    "description": field.description,
                }
            )

        sections[section_name] = {
            "doc": getdoc(type(section_model)),
            "fields": entries,
        }

    return sections


def _active_jobs(container) -> List[str]:
    if container is None:
        return []

    return list(getattr(container, "active_job_ids", {}).keys())


@router.get("", include_in_schema=False, response_class=HTMLResponse)
@router.get("/", include_in_schema=False, response_class=HTMLResponse)
async def dashboard_page():
    """Serve the dashboard single-page app."""

    if not DASHBOARD_FILE.exists():
        raise HTTPException(500, "dashboard.html is missing from the install.")

    return HTMLResponse(content=_load_page())


@router.get("/api/overview", dependencies=[Depends(check_api_key)])
async def dashboard_overview():
    """
    Everything the dashboard polls in one call: runtime stats, cache state,
    backend/KV info, system resources, active jobs, health and the config tree.
    """

    container = model.container

    healthy, issues = await HealthManager.is_service_healthy()

    model_info = None
    if container is not None and getattr(container, "loaded", False):
        try:
            card = get_current_model().model_dump(mode="json")
            # The raw chat template is several KB and the dashboard never shows it
            card.get("parameters", {}).pop("prompt_template_content", None)
            model_info = card
        except Exception as exc:
            xlogger.debug(f"Dashboard: failed to read model card: {exc}")

    model_dir = getattr(container, "model_dir", None) if container is not None else None

    return {
        "server": {
            "version": metrics.server_version(),
            "model_name": model_dir.name if model_dir else None,
            "backend": "exllamav3" if container is not None else None,
            "host": config.network.host,
            "port": config.network.port,
            "auth_disabled": bool(config.network.disable_auth),
            "api_servers": config.network.api_servers,
        },
        "health": {
            "status": "healthy" if healthy else "unhealthy",
            "issues": issues,
        },
        "runtime": metrics.collector.overview(),
        "backend": metrics.backend_snapshot(container),
        "system": metrics.system_snapshot(),
        "jobs": _active_jobs(container),
        "model_info": model_info,
        "config": _config_snapshot(),
    }


@router.post("/api/reset", dependencies=[Depends(check_admin_key)])
async def reset_stats():
    """Clear all collected dashboard stats."""

    metrics.collector.reset()
    return {"success": True}


@router.post("/api/jobs/cancel", dependencies=[Depends(check_admin_key)])
async def cancel_job(data: CancelJobRequest):
    """Cancel one in-flight generation by request id."""

    container = model.container
    if container is None:
        raise HTTPException(503, "No model is currently loaded.")

    job = getattr(container, "active_job_ids", {}).get(data.request_id)
    if job is None:
        raise HTTPException(404, f"No active job with request id {data.request_id}.")

    await job.cancel()
    return {"success": True, "request_id": data.request_id}
