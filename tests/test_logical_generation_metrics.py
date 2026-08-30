import unittest

from backends.exllamav3.model import _LogicalGenerationMetrics, _token_ids_to_list


class FakeClock:
    def __init__(self, value=100.0):
        self.value = value

    def __call__(self):
        return self.value


class FakeJob:
    def __init__(self):
        self.sequences = [object()]
        self.cached_pages = 0
        self.cached_tokens = 0
        self.time_enqueued = 0.0
        self.time_prefill = 0.0
        self.time_generate = 0.0
        self.time_enqueue = None
        self.time_first_prefill = None
        self.time_first_token = None
        self.accepted_draft_tokens = 0
        self.rejected_draft_tokens = 0

    def prepare_for_requeue(self):
        # Enough of exllamav3's in-place Job.__init__ behavior for the tracker:
        # the same object is reset for the next physical segment.
        self.cached_pages = 0
        self.cached_tokens = 0
        self.accepted_draft_tokens = 0
        self.rejected_draft_tokens = 0
        self.time_enqueue = None
        self.time_first_prefill = None
        self.time_first_token = None
        return self


class FakeSequence:
    def __init__(self, length):
        self.sequence_ids = [0] * length


class LogicalGenerationMetricsTests(unittest.TestCase):
    def test_requeues_do_not_become_prompt_or_cache_usage(self):
        clock = FakeClock()
        tracker = _LogicalGenerationMetrics(1586, clock=clock)
        job = FakeJob()
        job.cached_pages = 3
        job.cached_tokens = 4
        job.time_enqueued = 0.2
        job.time_prefill = 2.0
        job.time_generate = 10.0
        job.accepted_draft_tokens = 10
        job.rejected_draft_tokens = 5
        tracker.attach(job)

        tracker.note_generated_tokens(512, {"job": job})
        job.prepare_for_requeue()
        tracker.note_generated_tokens(600, {"job": job})

        result = {
            "job": job,
            # These are the broken physical-job values that used to leak out.
            "prompt_tokens": 28669,
            "cached_tokens": 26624,
            "new_tokens": 3258,
            "time_enqueued": 0.8,
            "time_prefill": 8.0,
            "time_generate": 20.0,
            "accepted_draft_tokens": 7,
            "rejected_draft_tokens": 3,
        }

        stats = tracker.finish(result)

        self.assertEqual(stats["prompt_tokens"], 1586)
        self.assertEqual(stats["cached_tokens"], 772)
        self.assertEqual(stats["prefill_tokens"], 1586)
        self.assertEqual(stats["gen_tokens"], 1112)
        self.assertEqual(stats["prompt_tokens_per_sec"], 793.0)
        self.assertEqual(stats["gen_tokens_per_sec"], 55.6)
        self.assertEqual(stats["requeue_count"], 1)
        self.assertEqual(stats["draft_accept"], 17)
        self.assertEqual(stats["draft_reject"], 8)

    def test_cancelled_generation_keeps_first_cache_and_live_speed(self):
        clock = FakeClock()
        tracker = _LogicalGenerationMetrics(1000, clock=clock)
        job = FakeJob()
        job.cached_pages = 2
        job.cached_tokens = 8
        job.time_enqueued = 0.1
        job.time_prefill = 1.5
        job.time_generate = 4.0
        tracker.attach(job)

        tracker.note_generated_tokens(400, {"job": job})
        job.prepare_for_requeue()
        tracker.note_generated_tokens(200, {"job": job})

        # The current physical segment is interrupted while decoding. Its base
        # counters contain earlier segments and the timestamps describe the live one.
        job.time_enqueued = 0.3
        job.time_prefill = 2.0
        job.time_generate = 4.0
        job.time_enqueue = 106.0
        job.time_first_prefill = 106.2
        job.time_first_token = 106.5
        clock.value = 110.0

        stats = tracker.abort(job, "r1", "cancelled", "partial")

        self.assertEqual(stats["finish_reason"], "cancelled")
        self.assertEqual(stats["prompt_tokens"], 1000)
        self.assertEqual(stats["cached_tokens"], 520)
        self.assertEqual(stats["gen_tokens"], 600)
        self.assertEqual(stats["gen_time"], 7.5)
        self.assertEqual(stats["gen_tokens_per_sec"], 80.0)
        self.assertEqual(stats["queue_time"], 0.1)
        self.assertEqual(stats["requeue_count"], 1)

    def test_empty_text_tokens_are_still_countable(self):
        self.assertEqual(_token_ids_to_list([1, 2, 3]), [1, 2, 3])

    def test_live_prefill_progress_includes_first_cache_hit(self):
        tracker = _LogicalGenerationMetrics(1000, clock=FakeClock())
        job = FakeJob()
        job.cached_pages = 2
        job.cached_tokens = 8

        progress = tracker.note_prefill(
            {"stage": "prefill", "curr_progress": 776, "job": job}
        )

        self.assertEqual(progress, (776, 1000))

    def test_cancelled_before_prefill_keeps_cache_unknown(self):
        clock = FakeClock()
        tracker = _LogicalGenerationMetrics(1000, clock=clock)
        job = FakeJob()
        job.time_enqueue = 100.0
        clock.value = 101.0

        stats = tracker.abort(job, "queued", "cancelled", "")

        self.assertIsNone(stats["cached_tokens"])
        self.assertEqual(stats["prefill_tokens"], 0)
        self.assertEqual(stats["prompt_tokens_per_sec"], "Indeterminate")
        self.assertEqual(stats["queue_time"], 1.0)

    def test_sequence_length_includes_non_streamed_stop_token(self):
        tracker = _LogicalGenerationMetrics(1000, clock=FakeClock())
        job = FakeJob()
        job.sequences = [FakeSequence(1003)]
        job.time_prefill = 1.0
        job.time_generate = 1.0
        tracker.note_generated_tokens(2, {"job": job})

        stats = tracker.finish(
            {
                "job": job,
                "time_enqueued": 0.0,
                "time_prefill": 1.0,
                "time_generate": 1.0,
            }
        )

        self.assertEqual(stats["gen_tokens"], 3)
        self.assertEqual(stats["gen_tokens_per_sec"], 3.0)


if __name__ == "__main__":
    unittest.main()
