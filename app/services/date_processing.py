def process_dates(scheduled_day, appointment_day):

    waiting_days = (
        appointment_day.date() - scheduled_day.date()
    ).days

    return {
        "Waiting_Days": waiting_days,
        "Scheduled_Month": scheduled_day.month,
        "Scheduled_Day_of_the_Week": scheduled_day.weekday(),
        "Appointment_Month": appointment_day.month,
        "Appointment_Day_of_the_Week": appointment_day.weekday()
    }