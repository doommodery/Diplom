import logging
from aiogram import Bot, Dispatcher, executor
from config import BOT_TOKEN
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from datetime import datetime

storage = MemoryStorage()

# Инициализация логирования
logging.basicConfig(level=logging.INFO)

# Инициализация бота и диспетчера
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot, storage=storage)

# Инициализация планировщика
scheduler = AsyncIOScheduler()

async def send_test_reminder(user_id, message):
    try:
        await bot.send_message(user_id, message)
        logging.info(f"Тестовое напоминание отправлено пользователю {user_id}: {message}.")
    except Exception as e:
        logging.error(f"Ошибка при отправке тестового напоминания пользователю {user_id}: {e}")

# Функция для добавления тестового напоминания
def add_test_reminder(user_id, message, date_time):
    trigger = DateTrigger(run_date=date_time)
    scheduler.add_job(send_test_reminder, trigger, args=[user_id, message], id=f"test_reminder_{user_id}")
    logging.info(f"Тестовое напоминание добавлено для пользователя {user_id} на {date_time}.")

# Запуск бота
if __name__ == "__main__":
    try:
        scheduler.start()
        logging.info("Планировщик запущен.")

        # Пример добавления тестового напоминания
        test_user_id = 1977126086  # Замените на реальный ID пользователя
        test_message = "Это тестовое напоминание!"
        test_date_time = datetime(2024, 12, 3, 19, 18)  # Замените на нужное время

        add_test_reminder(test_user_id, test_message, test_date_time)

        executor.start_polling(dp, skip_updates=True)
    except (KeyboardInterrupt, SystemExit):
        logging.info("Остановка бота...")
        scheduler.shutdown(wait=False)
        logging.info("Бот остановлен.")
    except Exception as e:
        logging.error(f"Ошибка при работе бота: {e}")
