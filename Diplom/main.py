import logging
import asyncio
from aiogram import Bot, Dispatcher, executor
from config import BOT_TOKEN
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from core.health_condition import delete_old_records
import torch
from utils import reminder_run
from utils.model_utils import AIModel

# Инициализация хранилища состояний
storage = MemoryStorage()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Инициализация бота и диспетчера
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot, storage=storage)

async def on_startup(dp):
    """Действия при запуске бота"""
    try:
        # Инициализация БД (без await, так как sqlite3 не асинхронный)
        from utils.database import init_db
        init_db()  # Убрали await
        logger.info("База данных инициализирована")

        # Предварительная загрузка модели
        try:
            ai_model = await AIModel.get_instance()
            if ai_model.is_ready:
                logger.info("Model components loaded")
            else:
                logger.warning("Model not loaded correctly")
        except Exception as e:
            logger.error(f"Model load error: {e}")

        # Инициализация планировщика
        scheduler = AsyncIOScheduler()
        scheduler.add_job(delete_old_records, 'interval', days=1)
        scheduler.start()
        logger.info("Планировщик запущен")

        # Регистрация обработчиков
        from handlers import (
            start_handler
        )
        
        start_handler.register_handlers_start(dp)
        reminder_run.register_handlers_reminder(dp, bot, scheduler)
        
        logger.info("Все обработчики зарегистрированы")

    except Exception as e:
        logger.error(f"Ошибка при запуске: {e}", exc_info=True)
        raise

async def on_shutdown(dp):
    """Действия при остановке бота"""
    try:
        # Остановка планировщика
        scheduler = AsyncIOScheduler()
        scheduler.shutdown(wait=False)
        logger.info("Планировщик остановлен")

        # Очистка ресурсов модели
        if AIModel._instance:
            AIModel._instance.model = None
            AIModel._instance.tokenizer = None
            torch.cuda.empty_cache()
        logger.info("Ресурсы модели освобождены")
        
        # Закрытие соединения с БД
        if hasattr(dp, 'db_conn') and dp.db_conn:
            dp.db_conn.close()
            logger.info("Соединение с БД закрыто")
            
    except Exception as e:
        logger.error(f"Ошибка при завершении работы: {e}")
    finally:
        logger.info("Бот завершает работу")

if __name__ == "__main__":
    try:
        logger.info("Запуск бота...")
        executor.start_polling(
            dp,
            skip_updates=True,
            on_startup=on_startup,
            on_shutdown=on_shutdown,
            timeout=60
        )
    except (KeyboardInterrupt, SystemExit):
        logger.info("Остановка бота по запросу пользователя")
    except Exception as e:
        logger.error(f"Критическая ошибка при работе бота: {e}", exc_info=True)
    finally:
        # Гарантированное завершение работы
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.run_until_complete(on_shutdown(dp))
        logger.info("Бот завершил работу")