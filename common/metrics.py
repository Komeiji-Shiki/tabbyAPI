"""
In-memory runtime metrics, consumed by the web dashboard.

Tracks per-request generation stats (generation / prefill speed, prefix-cache
hits), rolling one-second token buckets for live charts, and system/backend
snapshots. Everything here runs on the event loop thread, so no locking is
needed. Backend internals are read through defensive getattr chains: when a
model isn't loaded (or exllamav3 changes shape) fields simply report None
instead of breaking the endpoint.
"""

import time
from collections import deque
from typing import Any, Dict, List, Optional

from common.logger import xlogger

try:
    import psutil
except ImportError:  # psutil is a hard dependency; degrade quietly if absent
    psutil = None

# Length of the rolling token-rate history, in seconds
SERIES_WINDOW = 180
# Hard cap on how many buckets are kept in memory
BUCKET_RETENTION = 300
# Finished generations kept for the recent-generations table
RECENT_MAX = 100
# Window used for the "live" TPS readouts, in seconds
LIVE_WINDOW = 5.0
# Per-request token-rate window used by the live in-flight readouts, in seconds
TPS_WINDOW = 5.0
# Chart windows offered by the dashboard, in seconds
SERIES_WINDOWS = (30, 60, 180, 600)


def _num(value: Any, default: float = 0.0) -> float:
    """Coerce a metric that may be None or 'Indeterminate' into a float."""

    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


class MetricsCollector:
    """Collects generation stats for the dashboard."""

    def __init__(self):
        self.started_at = time.time()
        self.recent: deque = deque(maxlen=RECENT_MAX)
        self._reset_totals()

        # second-resolution buckets: epoch second -> [generated, prefilled] tokens
        self._buckets: Dict[int, List[float]] = {}
        # request id -> live state of an in-flight generation
        self._inflight: Dict[str, Dict[str, Any]] = {}

        self._process: Optional[Any] = None
        if psutil is not None:
            self._process = psutil.Process()
            self._process.cpu_percent(None)  # prime the counter, first read is 0

    def _reset_totals(self):
        self.totals = {
            "requests": 0,
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "new_prompt_tokens": 0,
            "gen_tokens": 0,
            "prompt_time": 0.0,
            "gen_time": 0.0,
            "queue_time": 0.0,
            "total_time": 0.0,
            "draft_accept": 0,
            "draft_reject": 0,
        }

    # ------------------------------------------------------------------ #
    # Live hooks, called from the generation loop                        #
    # ------------------------------------------------------------------ #

    def note_request_started(self, request_id: str, prompt_tokens: int = 0):
        """Register an in-flight request so the dashboard can track it live."""

        self._inflight[request_id] = {
            "request_id": request_id,
            "prompt_tokens": _int(prompt_tokens),
            "gen_tokens": 0,
            "prefill_curr": 0,
            "prefill_max": 0,
            "prefill_done": False,
            "started": time.time(),
            # (timestamp, cumulative tokens) samples bounding the rate window
            "samples": deque(),
        }

    def note_stream_tokens(self, request_id: str, count: int):
        """Attribute streamed tokens to the one-second bucket and their request."""

        if count <= 0:
            return

        self._current_bucket()[0] += count

        state = self._inflight.get(request_id)
        if state is None:
            return

        state["gen_tokens"] += count
        now = time.time()
        samples: deque = state["samples"]
        samples.append((now, state["gen_tokens"]))
        while samples and now - samples[0][0] > TPS_WINDOW:
            samples.popleft()

    def note_prefill_progress(self, request_id: str, curr: int, maximum: int):
        """Track chunk-by-chunk prefill progress reported by the backend."""

        state = self._inflight.get(request_id)
        if state is None:
            return

        delta = curr - state["prefill_curr"]
        state["prefill_curr"] = curr
        state["prefill_max"] = maximum

        if delta > 0:
            self._current_bucket()[1] += delta

    def note_prefill_finished(self, request_id: str):
        """Mark prefill complete. The request stays in flight while it decodes."""

        state = self._inflight.get(request_id)
        if state is None:
            return

        state["prefill_done"] = True
        if state["prefill_max"]:
            state["prefill_curr"] = state["prefill_max"]

    def note_request_finished(self, request_id: str):
        """Drop a request from the in-flight table once it is fully done."""

        self._inflight.pop(request_id, None)

    def live_requests(self) -> List[dict]:
        """Per-request live view: exact token counts plus a short-rate estimate."""

        now = time.time()
        requests: List[dict] = []

        for state in self._inflight.values():
            samples: deque = state["samples"]
            tps = None
            if len(samples) >= 2:
                first_ts, first_count = samples[0]
                last_ts, last_count = samples[-1]
                span = last_ts - first_ts
                if span > 0.2:
                    tps = round((last_count - first_count) / span, 2)

            requests.append(
                {
                    "request_id": state["request_id"],
                    "prompt_tokens": state["prompt_tokens"],
                    "gen_tokens": state["gen_tokens"],
                    "phase": "prefill" if not state["prefill_done"] else "generate",
                    "prefill_curr": state["prefill_curr"],
                    "prefill_max": state["prefill_max"],
                    "elapsed": round(now - state["started"], 2),
                    "gen_tokens_per_sec": tps,
                }
            )

        requests.sort(key=lambda item: item["elapsed"], reverse=True)
        return requests

    def _current_bucket(self) -> List[float]:
        sec = int(time.time())
        bucket = self._buckets.get(sec)
        if bucket is None:
            self._prune_buckets(sec)
            bucket = [0.0, 0.0]
            self._buckets[sec] = bucket
        return bucket

    def _prune_buckets(self, now_sec: int):
        cutoff = now_sec - BUCKET_RETENTION
        stale = [key for key in self._buckets if key < cutoff]
        for key in stale:
            del self._buckets[key]

    # ------------------------------------------------------------------ #
    # Completion hook, called from gen_logging.log_metrics               #
    # ------------------------------------------------------------------ #

    def record_generation(self, metrics: dict):
        """Store the final per-request metrics and update cumulative totals."""

        prompt_tokens = _int(metrics.get("prompt_tokens"))
        cached_tokens = _num(metrics.get("cached_tokens"))
        new_tokens = max(prompt_tokens - cached_tokens, 0.0)
        gen_tokens = _int(metrics.get("gen_tokens"))
        prompt_time = _num(metrics.get("prompt_time"))
        gen_time = _num(metrics.get("gen_time"))
        queue_time = _num(metrics.get("queue_time"))
        total_time = _num(metrics.get("total_time"))

        entry = {
            "request_id": metrics.get("request_id"),
            "ts": time.time(),
            "prompt_tokens": prompt_tokens,
            "cached_tokens": round(cached_tokens, 1),
            "new_tokens": round(new_tokens, 1),
            "gen_tokens": gen_tokens,
            "prompt_tps": metrics.get("prompt_tokens_per_sec"),
            "gen_tps": metrics.get("gen_tokens_per_sec"),
            "prompt_time": round(prompt_time, 2),
            "gen_time": round(gen_time, 2),
            "queue_time": round(queue_time, 2),
            "total_time": round(total_time, 2),
            "finish_reason": metrics.get("finish_reason"),
            "eos_reason": metrics.get("eos_reason"),
        }

        if "draft_accept" in metrics:
            entry["draft_accept"] = _int(metrics.get("draft_accept"))
            entry["draft_reject"] = _int(metrics.get("draft_reject"))

        self.recent.appendleft(entry)

        totals = self.totals
        totals["requests"] += 1
        totals["prompt_tokens"] += prompt_tokens
        totals["cached_tokens"] += cached_tokens
        totals["new_prompt_tokens"] += new_tokens
        totals["gen_tokens"] += gen_tokens
        totals["prompt_time"] += prompt_time
        totals["gen_time"] += gen_time
        totals["queue_time"] += queue_time
        totals["total_time"] += total_time
        totals["draft_accept"] += _int(metrics.get("draft_accept"))
        totals["draft_reject"] += _int(metrics.get("draft_reject"))

    def reset(self):
        """Clear all collected stats (uptime is preserved)."""

        self.recent.clear()
        self._buckets.clear()
        self._inflight.clear()
        self._reset_totals()

    # ------------------------------------------------------------------ #
    # Derived views                                                      #
    # ------------------------------------------------------------------ #

    def _rate_over(self, seconds: float) -> tuple:
        """Average generated/prefilled tokens per second over a window."""

        now = time.time()
        cutoff = now - seconds
        gen = 0.0
        prefill = 0.0
        for sec, bucket in self._buckets.items():
            if sec >= int(cutoff):
                gen += bucket[0]
                prefill += bucket[1]

        # Never divide by a span that starts before the collector existed
        span = max(min(seconds, now - self.started_at), 1e-3)
        return gen / span, prefill / span

    def series(self, window: int = SERIES_WINDOW) -> dict:
        """One-second token-rate buckets for the live chart."""

        window = max(int(window), 1)
        now = time.time()
        last_sec = int(now)
        first_sec = last_sec - window + 1

        gen_series: List[Optional[float]] = []
        prefill_series: List[Optional[float]] = []

        for sec in range(first_sec, last_sec + 1):
            bucket = self._buckets.get(sec)
            if bucket is None:
                # Before the collector started there is no data at all;
                # inside the window an empty bucket is simply 0 t/s
                if sec < int(self.started_at):
                    gen_series.append(None)
                    prefill_series.append(None)
                else:
                    gen_series.append(0.0)
                    prefill_series.append(0.0)
                continue

            # The trailing bucket covers less than a full second; scale it up
            # so the chart shows a rate instead of a partial count
            scale = 1.0
            if sec == last_sec:
                elapsed = min(now - sec, 1.0)
                scale = 1.0 / max(elapsed, 0.2)

            gen_series.append(round(bucket[0] * scale, 2))
            prefill_series.append(round(bucket[1] * scale, 2))

        return {
            "seconds_per_point": 1,
            "window": window,
            "windows": list(SERIES_WINDOWS),
            "generated": gen_series,
            "prefilled": prefill_series,
        }

    def cache_hit_stats(self) -> dict:
        """Prefix-cache hit ratios, cumulative and over recent requests."""

        totals = self.totals
        prompt = totals["prompt_tokens"]
        cached = totals["cached_tokens"]

        recent_prompt = 0.0
        recent_cached = 0.0
        for entry in self.recent:
            recent_prompt += entry["prompt_tokens"]
            recent_cached += entry["cached_tokens"]

        return {
            "total_tokens": prompt,
            "cached_tokens": round(cached, 1),
            "hit_ratio": round(cached / prompt, 4) if prompt > 0 else None,
            "recent_tokens": round(recent_prompt, 1),
            "recent_cached": round(recent_cached, 1),
            "recent_hit_ratio": (
                round(recent_cached / recent_prompt, 4) if recent_prompt > 0 else None
            ),
        }

    def overview(self, chart_window: int = SERIES_WINDOW) -> dict:
        """Everything the dashboard needs from the collector itself."""

        now = time.time()
        live_gen, live_prefill = self._rate_over(LIVE_WINDOW)
        totals = dict(self.totals)
        for key in ("prompt_time", "gen_time", "queue_time", "total_time"):
            totals[key] = round(totals[key], 2)

        prompt = totals["prompt_tokens"]
        gen_time = totals["gen_time"]
        prefill_time = totals["prompt_time"]
        new_prompt = totals["new_prompt_tokens"]

        draft_total = totals["draft_accept"] + totals["draft_reject"]

        return {
            "uptime": round(now - self.started_at, 1),
            "totals": totals,
            "live": {
                "gen_tokens_per_sec": round(live_gen, 2),
                "prefill_tokens_per_sec": round(live_prefill, 2),
                "requests": self.live_requests(),
            },
            "averages": {
                "gen_tokens_per_sec": round(totals["gen_tokens"] / gen_time, 2) if gen_time > 0 else None,
                "prefill_tokens_per_sec": (
                    round(new_prompt / prefill_time, 2) if prefill_time > 0 else None
                ),
                "queue_time": round(totals["queue_time"] / totals["requests"], 3)
                if totals["requests"] > 0
                else None,
                "draft_accept_ratio": round(totals["draft_accept"] / draft_total, 4)
                if draft_total > 0
                else None,
            },
            "cache": self.cache_hit_stats(),
            "series": self.series(chart_window),
            "recent": [dict(entry) for entry in list(self.recent)[:30]],
        }


# Global collector instance
collector = MetricsCollector()


def system_snapshot() -> dict:
    """CPU / RAM / VRAM readings for the dashboard."""

    snapshot: Dict[str, Any] = {"cpu_count": None, "process": {}, "system": {}, "gpus": []}

    if psutil is not None:
        try:
            snapshot["cpu_count"] = psutil.cpu_count()
            sys_mem = psutil.virtual_memory()
            snapshot["system"] = {
                "cpu_percent": psutil.cpu_percent(None),
                "ram_used": sys_mem.used,
                "ram_total": sys_mem.total,
            }
            if collector._process is not None:
                snapshot["process"] = {
                    "cpu_percent": collector._process.cpu_percent(None),
                    "ram_used": collector._process.memory_info().rss,
                }
        except Exception as exc:  # psutil can race with the OS on rare calls
            xlogger.debug(f"Failed to collect psutil stats: {exc}")

    try:
        import torch

        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                snapshot["gpus"].append(
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "mem_used": total_bytes - free_bytes,
                        "mem_total": total_bytes,
                    }
                )
    except Exception as exc:
        xlogger.debug(f"Failed to collect CUDA stats: {exc}")

    return snapshot


def backend_snapshot(container: Any) -> dict:
    """KV cache, page table and job-queue state from the exllamav3 backend."""

    snapshot: Dict[str, Any] = {
        "model_loaded": container is not None and getattr(container, "loaded", False),
        "max_seq_len": None,
        "cache_size": None,
        "cache_mode": None,
        "chunk_size": None,
        "max_batch_size": None,
        "draft_mode": None,
        "active_jobs": 0,
        "pending_jobs": None,
        "queue": None,
        "kv": None,
        "pagetable_metrics": None,
    }

    if container is None:
        return snapshot

    snapshot["max_seq_len"] = getattr(container, "max_seq_len", None)
    snapshot["cache_size"] = getattr(container, "cache_size", None)
    snapshot["cache_mode"] = getattr(container, "cache_mode", None)
    snapshot["chunk_size"] = getattr(container, "chunk_size", None)
    snapshot["max_batch_size"] = getattr(container, "max_batch_size", None)
    snapshot["active_jobs"] = len(getattr(container, "active_job_ids", {}))

    # The backend records the resolved mode, so "mtp" is reported accurately
    # instead of collapsing into the draft-model branch.
    snapshot["draft_mode"] = getattr(container, "draft_mode", None)
    snapshot["draft_num_tokens"] = getattr(container, "draft_num_tokens", None)

    generator = getattr(container, "generator", None)
    inner = getattr(generator, "generator", None) if generator is not None else None
    if inner is None:
        return snapshot

    try:
        snapshot["pending_jobs"] = inner.num_pending_jobs()
        snapshot["queue"] = inner.num_remaining_jobs()
    except Exception:
        pass

    pagetable = getattr(inner, "pagetable", None)
    if pagetable is not None:
        try:
            page_size = _page_size()
            referenced = len(pagetable.referenced_pages)
            unreferenced = len(pagetable.unreferenced_pages)
            max_pages = pagetable.max_pages
            snapshot["kv"] = {
                "page_size": page_size,
                "max_pages": max_pages,
                "used_pages": referenced,
                "idle_pages": unreferenced,
                "used_tokens": referenced * page_size,
                "total_tokens": max_pages * page_size,
                "usage": round(referenced / max_pages, 4) if max_pages > 0 else None,
            }
            snapshot["pagetable_metrics"] = dict(pagetable.metrics)
        except Exception as exc:
            xlogger.debug(f"Failed to read exllamav3 page table: {exc}")

    return snapshot


def _page_size() -> int:
    try:
        from exllamav3.constants import PAGE_SIZE

        return int(PAGE_SIZE)
    except Exception:
        return 256


def server_version() -> str:
    try:
        from importlib.metadata import version

        return version("tabbyapi")
    except Exception:
        return "unknown"
