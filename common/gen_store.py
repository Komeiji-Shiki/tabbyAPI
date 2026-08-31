"""
Durable dashboard stats.

The collector keeps its numbers in memory, so a restart wipes the
recent-generations table and every cumulative total. This module appends each
finished generation to a JSONL file and replays that file at startup.

The design is deliberately one-directional: metrics knows nothing about files
and just hands records to a sink, while everything here is written so that a
failing disk can never interfere with inference. The generation path only does
a non-blocking queue put; actual IO happens in a background task, and every
error is downgraded to a debug message.
"""

import asyncio
import json
import pathlib
from typing import List, Optional

import aiofiles

from common.logger import xlogger
from common.metrics import collector

# Bumped if the record layout changes incompatibly
SCHEMA_VERSION = 2
# How many finished records may sit in the write queue before new ones are
# dropped. Backing up the queue is pointless: stats are cheap to lose, latency
# is not.
QUEUE_MAX = 1000

__all__ = ["GenerationStore", "store", "start_from_config"]


class GenerationStore:
    """Append-only persistence for dashboard generation records."""

    def __init__(self, path: pathlib.Path, max_entries: int = 10000):
        self.path = path
        self.max_entries = max_entries
        self.enabled = False
        self.restored = 0
        # Records currently held on disk, tracked in memory so the dashboard can
        # report it every poll without re-reading the file
        self.stored = 0
        self._queue: Optional[asyncio.Queue] = None
        self._task: Optional[asyncio.Task] = None
        self._written = 0

    # ------------------------------------------------------------------ #
    # Reading                                                            #
    # ------------------------------------------------------------------ #

    def read_records(self, limit: Optional[int] = None) -> List[dict]:
        """
        Load persisted records, oldest first.

        A half-written line from a crash is skipped instead of aborting the
        replay, and an unreadable file degrades to "no history" rather than
        raising during startup.
        """

        if not self.path.is_file():
            return []

        records: List[dict] = []
        broken = 0

        try:
            with self.path.open("r", encoding="utf8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        broken += 1
                        continue
                    if isinstance(record, dict) and record.get("request_id"):
                        records.append(record)
        except OSError as exc:
            xlogger.warning(f"Cannot read the persisted dashboard stats: {exc}")
            return []

        if broken:
            xlogger.debug(f"Skipped {broken} unreadable lines in {self.path}")

        if limit is not None and limit > 0:
            records = records[-limit:]

        return records

    def restore(self) -> int:
        """Fold persisted history back into the live collector."""

        records = self.read_records()
        if not records:
            return 0

        # Oldest first: absorb() pushes to the front, so the newest record ends
        # up on top and the deque trims the oldest by itself.
        for record in records:
            collector.absorb(record)

        self.restored = len(records)
        self.stored = len(records)
        xlogger.info(
            f"Restored {len(records)} dashboard generation record(s) from {self.path}"
        )
        return len(records)

    # ------------------------------------------------------------------ #
    # Writing                                                            #
    # ------------------------------------------------------------------ #

    def sink(self, entry: dict):
        """
        Collector callback. Runs on the event loop between generated tokens, so
        it must never block, touch the disk or raise.
        """

        if not self.enabled or self._queue is None:
            return

        try:
            self._queue.put_nowait(entry)
        except asyncio.QueueFull:
            xlogger.debug("Dropping a dashboard stat: the write queue is full")

    async def start(self) -> int:
        """Restore history, subscribe to the collector and spawn the writer."""

        if self.enabled:
            return self.restored

        restored = self.restore()
        collector.sink = self.sink
        self._queue = asyncio.Queue(maxsize=QUEUE_MAX)
        self._task = asyncio.create_task(self._writer(), name="gen_stats_writer")
        self.enabled = True

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            xlogger.warning(f"Cannot create the folder for dashboard stats: {exc}")

        return restored

    async def _writer(self):
        while True:
            entry = await self._queue.get()
            try:
                await self._append(entry)
            except Exception as exc:
                # Losing one stat row is fine; anything louder is not warranted
                xlogger.debug(f"Failed to persist a dashboard record: {exc}")
            finally:
                self._queue.task_done()

    async def _append(self, entry: dict):
        record = dict(entry)
        record["_v"] = SCHEMA_VERSION
        line = json.dumps(record, ensure_ascii=False) + "\n"

        async with aiofiles.open(self.path, "a", encoding="utf8") as handle:
            await handle.write(line)

        self._written += 1
        self.stored += 1
        if self.max_entries and self._written >= self.max_entries:
            self._written = 0
            await self._trim()

    async def _trim(self):
        """Rewrite the file down to the newest max_entries records."""

        def _rewrite() -> int:
            records = self.read_records()
            keep = records[-self.max_entries :]
            temp = self.path.with_name(self.path.name + ".trim.tmp")

            with temp.open("w", encoding="utf8") as handle:
                for record in keep:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            temp.replace(self.path)
            return len(keep)

        try:
            kept = await asyncio.to_thread(_rewrite)
            self.stored = kept
            xlogger.debug(f"Trimmed dashboard stats to {kept} record(s)")
        except Exception as exc:
            xlogger.debug(f"Failed to trim the dashboard stats file: {exc}")

    async def flush(self):
        """Wait for queued writes to land, then stop accepting more (shutdown)."""

        self.enabled = False

        if self._queue is not None:
            try:
                await asyncio.wait_for(self._queue.join(), timeout=5.0)
            except (asyncio.TimeoutError, RuntimeError):
                xlogger.debug("Timed out flushing dashboard stats")

        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def clear(self):
        """
        Drop the persisted history.

        The dashboard's "clear stats" button resets the collector, so the file
        has to go too - otherwise a restart would resurrect numbers the user
        deliberately erased.
        """

        self._written = 0
        self.restored = 0
        self.stored = 0

        if not self.path.is_file():
            return

        try:
            self.path.unlink()
        except OSError as exc:
            xlogger.warning(f"Cannot clear the persisted dashboard stats: {exc}")


def _resolve_path(raw: str) -> pathlib.Path:
    path = pathlib.Path(raw)
    if not path.is_absolute():
        # Relative to the tabbyAPI install, like model_dir and friends
        path = pathlib.Path(__file__).resolve().parent.parent / path
    return path


# Created at import time; configured by bootstrap() once the config is parsed
store = GenerationStore(pathlib.Path("logs/gen_stats.jsonl"))


async def start_from_config() -> int:
    """
    Configure the store from the loaded config, restore history and start the
    writer. Must be awaited from a running event loop.

    Returns how many records were folded back in, or 0 when persistence is off.
    """

    from common.tabby_config import config

    options = config.logging

    if not getattr(options, "persist_generation_stats", True):
        return 0

    store.path = _resolve_path(
        getattr(options, "generation_stats_path", None) or "logs/gen_stats.jsonl"
    )
    store.max_entries = getattr(options, "generation_stats_max_entries", 10000) or 0

    if store.enabled:
        return store.restored

    return await store.start()
