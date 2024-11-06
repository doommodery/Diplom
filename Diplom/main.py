import logging
from aiogram import Bot, Dispatcher, executor
from config import BOT_TOKEN
from handlers import medicine_reminder, start_handler, stock
from utils.database import init_db, get_active_reminders
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from aiogram.contrib.fsm_storage.memory import MemoryStorage

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
        active_reminders = get_active_reminders()  # Получаем активные напоминания
        if not active_reminders:
            logging.info("Нет активных напоминаний.")
        for reminder in active_reminders:
            user_id = reminder['user_id']
            name = reminder['name']
            notes = reminder['notes']

            # Отправка сообщения с напоминанием
            try:
                await bot.send_message(user_id, f"Напоминание: {name}. {notes}")
                logging.info(f"Напоминание отправлено пользователю {user_id}: {name}.")
            except Exception as e:
                logging.error(f"Ошибка при отправке сообщения пользователю {user_id}: {e}")
    except Exception as e:
        logging.error(f"Ошибка при отправке напоминаний: {e}")

# Добавление задачи в планировщик
if not scheduler.get_job("medicine_reminder_job"):
    scheduler.add_job(send_medicine_reminders, CronTrigger(hour=8, minute=0), id="medicine_reminder_job")
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
