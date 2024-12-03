import logging
from aiogram import Bot, Dispatcher, executor
from config import BOT_TOKEN
from handlers import start_handler, stock, reminder_handler
from utils.database import init_db
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from apscheduler.schedulers.asyncio import AsyncIOScheduler

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

# Регистрация обработчиков
start_handler.register_handlers_start(dp)
reminder_handler.register_handlers_reminder(dp, bot, scheduler)

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
