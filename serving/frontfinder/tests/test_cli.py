from datetime import datetime, timezone

from frontfinder.scheduler.cli import most_recent_completed_cycle


def test_most_recent_completed_cycle_picks_previous_synoptic_hour():
    # 14:00 UTC minus 7h publish lag = 07:00 -> most recent synoptic hour <= 7 is 06Z
    now = datetime(2026, 8, 19, 14, 0, tzinfo=timezone.utc)
    cycle = most_recent_completed_cycle(now, publish_lag_hours=7)
    assert cycle.date == "2026-08-19"
    assert cycle.run_hour == 6


def test_most_recent_completed_cycle_rolls_back_across_midnight():
    # 02:00 UTC minus 7h = 19:00 the previous day -> most recent synoptic hour is 18Z
    now = datetime(2026, 8, 19, 2, 0, tzinfo=timezone.utc)
    cycle = most_recent_completed_cycle(now, publish_lag_hours=7)
    assert cycle.date == "2026-08-18"
    assert cycle.run_hour == 18


def test_most_recent_completed_cycle_exact_synoptic_boundary():
    # 19:00 UTC minus 7h publish lag = 12:00 exactly -> most recent synoptic hour is 12Z
    now = datetime(2026, 8, 19, 19, 0, tzinfo=timezone.utc)
    cycle = most_recent_completed_cycle(now, publish_lag_hours=7)
    assert cycle.date == "2026-08-19"
    assert cycle.run_hour == 12
