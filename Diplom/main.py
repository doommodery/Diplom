import logging, sqlite3,datetime,asyncio
from aiogram import Bot, Dispatcher, executor
from config import BOT_TOKEN
from handlers import start_handler, stock, reminder_handler, health_condition_handler
from utils.database import init_db
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from handlers.health_condition_handler import delete_old_records

# Инициализация хранилища состояний
storage = MemoryStorage()

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Инициализация бота и диспетчера
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot, storage=storage)

# Инициализация базы данных
init_db()
logging.info("База данных успешно инициализирована.")

# Инициализация планировщика
scheduler = AsyncIOScheduler()

# Регистрация обработчиков (без scheduler.start())
start_handler.register_handlers_start(dp)
reminder_handler.register_handlers_reminder(dp, bot, scheduler)
health_condition_handler.register_handlers_health(dp)

# Функция для запуска планировщика
async def on_startup(dp):
    scheduler.start()  # Теперь планировщик запускается внутри асинхронного контекста
    logging.info("Планировщик запущен.")
    asyncio.create_task(delete_old_records())

# Функция для остановки планировщика
async def on_shutdown(dp):
    scheduler.shutdown(wait=False)
    logging.info("Планировщик остановлен.")

# Запуск бота
if __name__ == "__main__":
    try:
        executor.start_polling(
            dp,
            skip_updates=True,
            on_startup=on_startup,
            on_shutdown=on_shutdown
        )
    except (KeyboardInterrupt, SystemExit):
        logging.info("Остановка бота...")
    except Exception as e:
        logging.error(f"Ошибка при работе бота: {e}")
