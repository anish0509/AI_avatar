import time

from app.core.timing import StepTimer, step_timer


def test_marks_are_recorded_in_completion_order_with_ms_suffix():
    timer = StepTimer("test setup", "test")
    timer.mark("first")
    timer.mark("second")
    assert list(timer.steps) == ["first_ms", "second_ms"]


def test_each_mark_measures_only_since_the_previous_one():
    timer = StepTimer("test setup", "test")
    time.sleep(0.05)
    first = timer.mark("first")
    second = timer.mark("second")
    # The second step did no work, so it must not inherit the first's 50ms --
    # that would make every later step look progressively slower.
    assert first >= 40
    assert second < 40


def test_total_covers_all_steps():
    timer = StepTimer("test setup", "test")
    time.sleep(0.02)
    timer.mark("first")
    time.sleep(0.02)
    timer.mark("second")
    total = timer.total_ms()
    assert total >= timer.steps["first_ms"] + timer.steps["second_ms"]


def test_log_returns_the_total_so_it_is_assertable_without_reading_logs():
    timer = StepTimer("test setup", "test")
    timer.mark("only")
    assert timer.log(session_id="abc") >= 0


def test_log_with_no_marks_does_not_crash():
    # A setup that raises before its first mark still gets logged by some
    # callers; an empty breakdown must not blow up the error path.
    assert StepTimer("test setup", "test").log() >= 0


def test_step_timer_context_manager_yields_a_usable_timer():
    with step_timer("test setup", "test") as timer:
        timer.mark("first")
        assert timer.log() >= 0
