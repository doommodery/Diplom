from aiogram import types, Dispatcher

# Обработчик команды /start
async def start(message: types.Message):
    await message.answer("Бот запущен и готов к работе!")

# Регистрация обработчика
def register_handlers_start(dp: Dispatcher):
    dp.register_message_handler(start, commands=["start"])

