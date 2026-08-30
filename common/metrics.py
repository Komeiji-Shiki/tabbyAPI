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
        # request id -> [curr_progress, max_progress] for in-flight prefills
        self._prefill: Dict[str, List[int]] = {}

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

    def note_generated_tokens(self, count: int):
        """Attribute streamed tokens to the current one-second bucket."""

        if count > 0:
            self._current_bucket()[0] += count

    def note_prefill_progress(self, request_id: str, curr: int, maximum: int):
        """Track chunk-by-chunk prefill progress reported by the backend."""

        state = self._prefill.get(request_id)
        if state is None:
            state = [0, 0]
            self._prefill[request_id] = state

        delta = curr - state[0]
        state[0] = curr
        state[1] = maximum

        if delta > 0:
            self._current_bucket()[1] += delta

    def note_prefill_finished(self, request_id: str):
        self._prefill.pop(request_id, None)

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
        self._prefill.clear()
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

    def series(self) -> dict:
        """One-second token-rate buckets for the live chart."""

        now = time.time()
        last_sec = int(now)
        first_sec = last_sec - SERIES_WINDOW + 1

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
            "window": SERIES_WINDOW,
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

    def overview(self) -> dict:
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
                "prefill_progress": [
                    {"request_id": rid, "curr": state[0], "max": state[1]}
                    for rid, state in self._prefill.items()
                    if state[1] > 0
                ],
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
            "series": self.series(),
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

    if getattr(container, "use_draft_model", False):
        snapshot["draft_mode"] = "model"
    elif getattr(container, "ngram_match_min", 0):
        snapshot["draft_mode"] = "ngram"

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
