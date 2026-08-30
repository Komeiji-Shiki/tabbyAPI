import json
import pathlib
import tempfile
import unittest

from common import metrics
from common.gen_store import GenerationStore


def completed(
    request_id,
    prompt_tokens,
    cached_tokens,
    gen_tokens,
    prompt_time=1.0,
    gen_time=1.0,
    queue_time=0.0,
):
    """A metrics dict shaped like the backend's finish chunk."""

    return {
        "request_id": request_id,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "gen_tokens": gen_tokens,
        "prompt_time": prompt_time,
        "gen_time": gen_time,
        "queue_time": queue_time,
        "total_time": prompt_time + gen_time + queue_time,
        "prompt_tokens_per_sec": 100.0,
        "gen_tokens_per_sec": gen_tokens / gen_time if gen_time else 0.0,
        "finish_reason": "stop",
        "eos_reason": "stop_token",
    }


class LiveRequestTrackingTests(unittest.TestCase):
    def setUp(self):
        self.collector = metrics.MetricsCollector()

    def test_tracks_exact_token_counts_and_phase(self):
        self.collector.note_request_started("r9", 123)
        self.collector.note_prefill_progress("r9", 60, 123)

        live = self.collector.live_requests()[0]
        self.assertEqual(live["phase"], "prefill")
        self.assertEqual(live["prompt_tokens"], 123)
        self.assertEqual(live["prefill_curr"], 60)
        self.assertEqual(live["prefill_max"], 123)
        self.assertEqual(live["gen_tokens"], 0)

        self.collector.note_prefill_finished("r9")
        for _ in range(3):
            self.collector.note_stream_tokens("r9", 7)

        live = self.collector.live_requests()[0]
        self.assertEqual(live["phase"], "generate")
        self.assertEqual(live["gen_tokens"], 21)
        # prefill must land on its full length instead of hanging below it
        self.assertEqual(live["prefill_curr"], 123)

        self.collector.note_request_finished("r9")
        self.assertEqual(self.collector.live_requests(), [])

    def test_bucket_only_helpers_need_no_registration(self):
        self.collector.note_stream_tokens("untracked", 5)
        self.assertEqual(self.collector.live_requests(), [])
        series = self.collector.series(30)
        self.assertTrue(any(v for v in series["generated"]))


class AbortedGenerationTests(unittest.TestCase):
    def setUp(self):
        self.collector = metrics.MetricsCollector()

    def _abort(self, request_id, prompt_tokens, gen_tokens):
        self.collector.note_request_started(request_id, prompt_tokens)
        for _ in range(gen_tokens):
            self.collector.note_stream_tokens(request_id, 1)
        return self.collector.record_aborted(request_id)

    def test_cancelled_request_is_counted(self):
        self.assertTrue(self._abort("r1", 500, 40))

        totals = self.collector.totals
        self.assertEqual(totals["requests"], 1)
        self.assertEqual(totals["prompt_tokens"], 500)
        self.assertEqual(totals["gen_tokens"], 40)
        self.assertEqual(totals["cancelled_requests"], 1)
        self.assertEqual(totals["cancelled_prompt_tokens"], 500)
        self.assertEqual(totals["cancelled_gen_tokens"], 40)

        entry = self.collector.recent[0]
        self.assertEqual(entry["finish_reason"], "cancelled")
        self.assertEqual(entry["gen_tokens"], 40)
        # the backend never got to report these, so they stay unknown
        self.assertIsNone(entry["cached_tokens"])
        self.assertIsNone(entry["gen_tps"])

    def test_abort_reason_is_preserved(self):
        self.collector.note_request_started("e1", 10)
        self.collector.record_aborted("e1", "error")
        self.assertEqual(self.collector.recent[0]["finish_reason"], "error")

    def test_abort_of_untracked_request_is_a_no_op(self):
        self.assertFalse(self.collector.record_aborted("ghost"))
        self.assertEqual(self.collector.totals["requests"], 0)

    def test_double_abort_is_ignored(self):
        self._abort("r1", 10, 1)
        self.assertFalse(self.collector.record_aborted("r1"))
        self.assertEqual(self.collector.totals["requests"], 1)

    def test_hit_ratio_excludes_aborted_prompts(self):
        self.collector.record_generation(completed("ok", 1000, 256.0, 50))
        self._abort("bad", 200, 5)

        stats = self.collector.cache_hit_stats()
        # 256 cached out of the 1000 tokens that could report a cache figure,
        # not out of 1200 which would fake the ratio down
        self.assertAlmostEqual(stats["hit_ratio"], 0.256, places=4)
        self.assertEqual(stats["recent_tokens"], 1000)
        # absolute totals still contain everything
        self.assertEqual(stats["total_tokens"], 1200)

    def test_rate_averages_exclude_aborted_work(self):
        self.collector.record_generation(completed("ok", 1000, 256.0, 50, gen_time=1.0))
        self._abort("bad", 200, 5)

        averages = self.collector.overview()["averages"]
        # 50 scored tokens over 1.0s of measured decode, not 55/1.0
        self.assertEqual(averages["gen_tokens_per_sec"], 50.0)
        # queue average divides by the 1 request that reported queue time
        self.assertEqual(averages["queue_time"], 0.0)

    def test_aborted_only_stats_have_no_rates(self):
        self._abort("only", 10, 1)
        overview = self.collector.overview()
        self.assertIsNone(overview["averages"]["gen_tokens_per_sec"])
        self.assertIsNone(overview["cache"]["hit_ratio"])
        self.assertEqual(overview["totals"]["gen_tokens"], 1)


class GenerationStoreReplayTests(unittest.TestCase):
    def setUp(self):
        self.dir = pathlib.Path(tempfile.mkdtemp())
        metrics.collector.reset()
        metrics.collector.sink = None

    def tearDown(self):
        metrics.collector.reset()
        metrics.collector.sink = None

    def test_replay_skips_broken_lines_and_keeps_order(self):
        path = self.dir / "gen.jsonl"
        with path.open("w", encoding="utf8") as handle:
            handle.write(
                json.dumps(
                    {
                        "request_id": "old",
                        "ts": 1.0,
                        "prompt_tokens": 100,
                        "cached_tokens": 25.0,
                        "gen_tokens": 5,
                        "prompt_time": 1.0,
                        "gen_time": 1.0,
                        "queue_time": 0.0,
                        "total_time": 2.0,
                        "finish_reason": "stop",
                    }
                )
                + "\n"
            )
            # what a crash mid-write leaves behind
            handle.write('{"request_id": "half-writ' + "\n")
            handle.write(
                json.dumps(
                    {
                        "request_id": "aborted",
                        "ts": 2.0,
                        "prompt_tokens": 80,
                        "cached_tokens": None,
                        "gen_tokens": 3,
                        "total_time": 0.9,
                        "finish_reason": "cancelled",
                    }
                )
                + "\n"
            )

        store = GenerationStore(path, max_entries=100)
        self.assertEqual(len(store.read_records()), 2)
        self.assertEqual(store.restore(), 2)

        self.assertEqual(
            [entry["request_id"] for entry in metrics.collector.recent],
            ["aborted", "old"],
        )
        totals = metrics.collector.totals
        self.assertEqual(totals["requests"], 2)
        self.assertEqual(totals["cancelled_requests"], 1)
        self.assertEqual(totals["prompt_tokens"], 180)
        self.assertEqual(totals["gen_tokens"], 8)
        self.assertEqual(store.stored, 2)

    def test_missing_file_restores_nothing(self):
        store = GenerationStore(self.dir / "nope.jsonl", max_entries=10)
        self.assertEqual(store.read_records(), [])
        self.assertEqual(store.restore(), 0)

    def test_unreadable_file_degrades_to_no_history(self):
        path = self.dir / "gen.jsonl"
        path.write_text("not json at all\nalso not json\n", encoding="utf8")
        store = GenerationStore(path, max_entries=10)
        self.assertEqual(store.restore(), 0)
        self.assertEqual(metrics.collector.totals["requests"], 0)

    def test_sink_is_inert_before_start(self):
        store = GenerationStore(self.dir / "off.jsonl", max_entries=10)
        metrics.collector.sink = store.sink
        metrics.collector.record_generation(completed("z", 10, 0.0, 2))
        self.assertFalse((self.dir / "off.jsonl").exists())
        self.assertEqual(metrics.collector.totals["requests"], 1)


class GenerationStoreWriteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.dir = pathlib.Path(tempfile.mkdtemp())
        metrics.collector.reset()
        metrics.collector.sink = None

    def tearDown(self):
        metrics.collector.reset()
        metrics.collector.sink = None

    def _lines(self, path):
        return [json.loads(x) for x in path.read_text(encoding="utf8").splitlines()]

    async def test_records_are_written_and_survive_restart(self):
        path = self.dir / "live.jsonl"
        store = GenerationStore(path, max_entries=1000)
        await store.start()

        metrics.collector.record_generation(completed("a", 100, 20.0, 7))
        metrics.collector.note_request_started("b", 50)
        metrics.collector.note_stream_tokens("b", 4)
        metrics.collector.record_aborted("b")

        await store.flush()

        records = self._lines(path)
        self.assertEqual([r["request_id"] for r in records], ["a", "b"])
        self.assertEqual(records[1]["finish_reason"], "cancelled")
        self.assertIsNone(records[1]["cached_tokens"])
        self.assertEqual(store.stored, 2)

        # pretend the process died: a fresh store must rebuild the same numbers
        metrics.collector.reset()
        again = GenerationStore(path, max_entries=1000)
        self.assertEqual(again.restore(), 2)
        self.assertEqual(metrics.collector.totals["requests"], 2)
        self.assertEqual(metrics.collector.totals["gen_tokens"], 11)
        self.assertEqual(metrics.collector.totals["cancelled_gen_tokens"], 4)

    async def test_trim_bounds_the_file_and_keeps_the_newest(self):
        path = self.dir / "trim.jsonl"
        store = GenerationStore(path, max_entries=4)
        await store.start()

        for index in range(9):
            metrics.collector.record_generation(completed(f"t{index}", 10, 0.0, 2))
        await store.flush()

        records = self._lines(path)
        ids = [r["request_id"] for r in records]
        self.assertLessEqual(len(records), 8)
        self.assertEqual(ids[-1], "t8")
        self.assertNotIn("t0", ids)
        self.assertEqual(store.stored, len(records))

    async def test_clear_removes_persisted_history(self):
        path = self.dir / "clear.jsonl"
        store = GenerationStore(path, max_entries=100)
        await store.start()
        metrics.collector.record_generation(completed("c1", 10, 0.0, 2))
        await store.flush()

        self.assertTrue(path.is_file())
        store.clear()
        self.assertFalse(path.is_file())
        self.assertEqual(store.stored, 0)
        # clearing twice must not raise
        store.clear()

    async def test_double_start_is_idempotent(self):
        store = GenerationStore(self.dir / "twice.jsonl", max_entries=100)
        self.assertEqual(await store.start(), 0)
        self.assertEqual(await store.start(), 0)
        self.assertTrue(store.enabled)
        await store.flush()


if __name__ == "__main__":
    unittest.main()
