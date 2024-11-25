import logging
from aiogram import Bot, Dispatcher, executor
from config import BOT_TOKEN
from handlers import medicine_reminder, start_handler, stock
from utils.database import init_db, get_active_reminders, delete_reminder_from_db
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from datetime import datetime

storage = MemoryStorage()

# Инициализация логирования
logging.basicConfig(level=logging.INFO)

# Инициализация бота и диспетчера
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot, storage=storage)

# Инициализация базы данных
init_db()
logging.info("База данных успешно инициализирована.")

# Инициализация планировщика
scheduler = AsyncIOScheduler()

async def send_medicine_reminders():
    logging.info("Запуск отправки напоминаний...")
    try:
        active_reminders = get_active_reminders()
        if not active_reminders:
            logging.info("Нет активных напоминаний.")
        for reminder in active_reminders:
            user_id = reminder['user_id']
            name = reminder['name']
            notes = reminder['notes']
            times = reminder['times'].split(',')
            start_date = datetime.strptime(reminder['start_date'], '%Y-%m-%d')
            end_date = datetime.strptime(reminder['end_date'], '%Y-%m-%d')
            current_date = datetime.now()
            days = reminder['days'].split(',')

            # Проверка, находится ли текущая дата в диапазоне
            if start_date <= current_date <= end_date:
                current_day = current_date.strftime('%A')
                if current_day in days:
                    for time in times:
                        reminder_time = datetime.strptime(time, '%H:%M').time()
                        current_time = datetime.now().time()
                        if current_time.hour == reminder_time.hour and current_time.minute == reminder_time.minute:
                            try:
                                await bot.send_message(user_id, f"Напоминание: {name}. {notes}")
                                logging.info(f"Напоминание отправлено пользователю {user_id}: {name}.")
                            except Exception as e:
                                logging.error(f"Ошибка при отправке сообщения пользователю {user_id}: {e}")
            else:
                # Удаление напоминания, если текущая дата выходит за диапазон
                delete_reminder_from_db(reminder['id'])
                logging.info(f"Напоминание для пользователя {user_id} удалено, так как вышло за диапазон дат.")
    except Exception as e:
        logging.error(f"Ошибка при отправке напоминаний: {e}")

# Добавление задачи в планировщик
if not scheduler.get_job("medicine_reminder_job"):
    scheduler.add_job(send_medicine_reminders, CronTrigger(minute='*'), id="medicine_reminder_job")
    logging.info("Задача напоминания успешно добавлена в планировщик.")

# Регистрация обработчиков
medicine_reminder.register_handlers(dp)
stock.register_handlers(dp)
start_handler.register_handlers_start(dp)  # Регистрация обработчика /start

# Запуск бота
if __name__ == "__main__":
    try:
        scheduler.start()
        logging.info("Планировщик запущен.")
        executor.start_polling(dp, skip_updates=True)
    except (KeyboardInterrupt, SystemExit):
        logging.info("Остановка бота...")
        scheduler.shutdown(wait=False)
        logging.info("Бот остановлен.")
    except Exception as e:
        logging.error(f"Ошибка при работе бота: {e}")
