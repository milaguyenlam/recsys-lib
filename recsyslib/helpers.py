from datetime import datetime, date, time, timedelta


def get_nearest_time(t: time) -> datetime:
    day = date.today()
    first_scheduled_run = datetime.combine(day, t)
    if datetime.now() > first_scheduled_run:
        first_scheduled_run += timedelta(days=1)
    return first_scheduled_run
