import geopy
from timezonefinder import TimezoneFinder
import datetime
import pytz
import logging
from aiogram import types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import InlineKeyboardButton
from utils.database import save_user_info
from handlers.start_handler import get_start_keyboard

# Инициализация логирования
logging.basicConfig(level=logging.INFO)

class UserInfo(StatesGroup):
    city = State()  # Состояние для ввода города
    health_info = State()  # Состояние для ввода данных о здоровье

async def handle_personal_data(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.answer()  # Подтверждаем нажатие кнопки
    await callback_query.message.answer("Пожалуйста, введите ваш город проживания:")
    await UserInfo.city.set()  # Переходим в состояние ввода города

async def handle_city(message: types.Message, state: FSMContext):
    city = message.text
    if not city:
        await message.answer("Пожалуйста, введите название города.")
        return

    timezone = await get_timezone(message.bot, message.chat.id, city)  # Добавлен await
    if timezone:
        async with state.proxy() as data:
            data['city'] = city
            data['timezone'] = timezone  # Сохраняем часовой пояс
        await message.answer("Часовой пояс успешно установлен. Теперь введите данные о вашем здоровье:")
        await UserInfo.health_info.set()  # Переходим в состояние ввода данных о здоровье
    else:
        await message.answer("Не удалось определить часовой пояс. Попробуйте еще раз.")

async def handle_health_info(message: types.Message, state: FSMContext):
    health_info = message.text
    if not health_info:
        await message.answer("Пожалуйста, введите данные о вашем здоровье.")
        return

    async with state.proxy() as data:
        city = data.get('city')
        timezone = data.get('timezone')
        user_id = message.from_user.id

    if not city or not timezone:
        await message.answer("Ошибка: не удалось определить город или часовой пояс. Попробуйте снова.")
        await state.finish()
        return

    try:
        # Сохраняем данные в базу данных
        await save_user_info(user_id, timezone, health_info)
        await message.answer("Ваши данные успешно сохранены!")
        await message.answer("Что вы хотите сделать?", reply_markup=get_start_keyboard())  
    except Exception as e:
        logging.error(f"Ошибка при сохранении данных: {e}")
        await message.answer("Произошла ошибка при сохранении данных. Попробуйте позже.")
    finally:
        await state.finish()  # Завершаем состояние

async def get_timezone(bot, chat_id, city):
    try:
        if not city:
            await bot.send_message(chat_id, "Пожалуйста, введите название города.")
            return None

        geo = geopy.geocoders.Nominatim(user_agent="SuperMon_Bot")
        location = geo.geocode(city)  # Преобразуем название города в координаты
        if location is None:
            await bot.send_message(chat_id, "Не удалось найти такой город. Попробуйте написать его название латиницей или указать более крупный город поблизости.")
            return None

        # Логируем координаты
        logging.info(f"Координаты для города {city}: широта={location.latitude}, долгота={location.longitude}")

        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lat=location.latitude, lng=location.longitude)  # Получаем название часового пояса
        if not timezone_str:
            await bot.send_message(chat_id, "Не удалось определить часовой пояс для этого города.")
            return None

        tz = pytz.timezone(timezone_str)
        tz_info = datetime.datetime.now(tz=tz).strftime("%z")  # Получаем смещение часового пояса
        tz_info = tz_info[0:3] + ":" + tz_info[3:]  # Приводим к формату ±ЧЧ:ММ
        await bot.send_message(chat_id, f"Часовой пояс установлен в {timezone_str} ({tz_info} от GMT).")
        return timezone_str

    except Exception as e:
        logging.error(f"Ошибка при определении часового пояса: {e}")
        await bot.send_message(chat_id, "Произошла ошибка при определении часового пояса. Попробуйте позже.")
        return None