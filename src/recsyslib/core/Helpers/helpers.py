from datetime import datetime, date, time, timedelta


def get_nearest_time(time_string: str) -> datetime:
    day = date.today()
    t = time.fromisoformat(time_string)
    first_scheduled_run = datetime.combine(day, t)
    if datetime.now() > first_scheduled_run:
        first_scheduled_run += timedelta(days=1)
    return first_scheduled_run
