import logging
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
from utils.database import get_active_reminders, delete_reminder_from_db

sent_reminders = {}

async def send_medicine_reminders(bot):
    logging.info("Запуск отправки напоминаний...")
    try:
        active_reminders = get_active_reminders()
        if not active_reminders:
            logging.info("Нет активных напоминаний.")
            return

        days_of_week = {
            "Monday": "Понедельник",
            "Tuesday": "Вторник",
            "Wednesday": "Среда",
            "Thursday": "Четверг",
            "Friday": "Пятница",
            "Saturday": "Суббота",
            "Sunday": "Воскресенье"
        }

        current_date = datetime.now()
        current_time = current_date.time()
        current_day = days_of_week[current_date.strftime('%A')]

        for reminder in active_reminders:
            user_id = reminder['user_id']
            name = reminder['name']
            notes = reminder['notes']
            times = reminder['times'].split(',')
            start_date = datetime.strptime(reminder['start_date'], '%Y-%m-%d')
            end_date = datetime.strptime(reminder['end_date'], '%Y-%m-%d')
            days = reminder['days'].split(',')
            duration = reminder['duration']
            dose_count = reminder['dose_count']
            dosage = reminder['dosage']

            # Проверка, находится ли текущая дата в диапазоне
            if start_date <= current_date <= end_date:
                if current_day in days:
                    for time in times:
                        reminder_time = datetime.strptime(time, '%H:%M').time()

                        # Проверка точного времени
                        if (current_time.hour == reminder_time.hour and
                            current_time.minute == reminder_time.minute):

                            # Уникальный ключ для напоминания
                            reminder_key = f"{user_id}_{name}_{time}"

                            # Проверка, было ли уже отправлено напоминание
                            if reminder_key not in sent_reminders:
                                try:
                                    await bot.send_message(user_id, f"Напоминание: Вам нужно выпить {dosage} таб. {name} {notes}")
                                    logging.info(f"Напоминание отправлено пользователю {user_id}: {name}.")
                                    sent_reminders[reminder_key] = True  # Отметить как отправленное
                                except Exception as e:
                                    logging.error(f"Ошибка при отправке сообщения пользователю {user_id}: {e}", exc_info=True)
            else:
                # Удаление напоминания, если текущая дата выходит за диапазон
                if current_date > end_date:
                    await asyncio.to_thread(delete_reminder_from_db, user_id, name, notes, duration, days, dose_count, times, dosage)
                    logging.info(f"Напоминание для пользователя {user_id} удалено, так как вышло за диапазон дат.")
    except Exception as e:
        logging.error(f"Ошибка при отправке напоминаний: {e}", exc_info=True)

def register_handlers_reminder(dp, bot, scheduler):
    if not scheduler.get_job("medicine_reminder_job"):
        scheduler.add_job(
            send_medicine_reminders,  # Pass the coroutine directly
            CronTrigger(second='*/30'), 
            args=[bot],  # Pass the bot object as an argument
            id="medicine_reminder_job"
        )
        logging.info("Задача напоминания успешно добавлена в планировщик.")