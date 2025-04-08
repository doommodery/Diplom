import geopy
from timezonefinder import TimezoneFinder
import datetime
import pytz
import logging
from aiogram import types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from utils.database import save_user_info, get_user_info
from keyboards.keyboards import get_start_keyboard

logging.basicConfig(level=logging.INFO)

class UserInfo(StatesGroup):
    waiting_for_city = State()
    waiting_for_year = State()
    waiting_for_month = State()
    waiting_for_day = State()
    waiting_for_conditions = State()

# Главный обработчик персональных данных
async def handle_personal_data(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.answer()
    user_id = callback_query.from_user.id
    
    old_data = await get_user_info(user_id)
    if old_data:
        birth_date = old_data.get('birth_date', 'не указана')
        if birth_date != 'не указана':
            try:
                birth_date_obj = datetime.datetime.strptime(birth_date, '%d.%m.%Y').date()
                today = datetime.date.today()
                age = today.year - birth_date_obj.year - ((today.month, today.day) < (birth_date_obj.month, birth_date_obj.day))
                age_text = f" ({age} лет)"
            except Exception:
                age_text = ""
        else:
            age_text = ""
        timezone=old_data.get('timezone')    
        message_text = (
            "Ваши текущие данные:\n"
            f"⏰ Часовой пояс: {timezone}\n"
            f"🎂Возраст: {birth_date}{age_text}\n"
            f"💊Хронические состояния: {old_data.get('chronic_conditions', 'не указаны')}\n\n"
            "Хотите изменить их? Введите ваш город:"
        )
    else:
        message_text = "Пожалуйста, введите ваш город проживания:"
    
    await callback_query.message.answer(message_text)
    await UserInfo.waiting_for_city.set()

# Обработчик города
async def handle_city(message: types.Message, state: FSMContext):
    city = message.text.strip()
    if not city:
        await message.answer("Пожалуйста, введите название города.")
        return

    timezone = await get_timezone(message.bot, message.chat.id, city)
    if not timezone:
        return

    async with state.proxy() as data:
        data['timezone'] = timezone

    await show_year_selection(message)
    await UserInfo.waiting_for_year.set()

# Функции для выбора даты рождения
async def show_year_selection(message: types.Message):
    current_year = datetime.datetime.now().year
    years = list(range(current_year - 100, current_year - 10))
    
    keyboard = types.InlineKeyboardMarkup(row_width=5)
    for year in years[-20:]:  # Показываем последние 20 лет
        keyboard.insert(types.InlineKeyboardButton(str(year), callback_data=f"year_{year}"))
    keyboard.row(types.InlineKeyboardButton("Другие года...", callback_data="show_all_years"))
    
    await message.answer("Выберите год рождения:", reply_markup=keyboard)

async def handle_birth_year(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.answer()
    
    if callback_query.data == "show_all_years":
        await show_all_years(callback_query.message)
        return
    
    year = int(callback_query.data.split("_")[1])
    
    async with state.proxy() as data:
        data['birth_year'] = year
    
    await show_month_selection(callback_query.message)
    await UserInfo.waiting_for_month.set()

async def show_all_years(message: types.Message):
    current_year = datetime.datetime.now().year
    years = list(range(current_year - 100, current_year - 10))
    
    keyboard = types.InlineKeyboardMarkup(row_width=5)
    for year in years:
        keyboard.insert(types.InlineKeyboardButton(str(year), callback_data=f"year_{year}"))
    
    await message.answer("Выберите год рождения:", reply_markup=keyboard)

async def show_month_selection(message: types.Message):
    months = ["Январь", "Февраль", "Март", "Апрель", "Май", "Июнь", 
              "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"]
    
    keyboard = types.InlineKeyboardMarkup(row_width=4)
    for i, month in enumerate(months, 1):
        keyboard.insert(types.InlineKeyboardButton(month, callback_data=f"month_{i}"))
    
    await message.answer("Выберите месяц рождения:", reply_markup=keyboard)

async def handle_birth_month(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.answer()
    
    month_num = int(callback_query.data.split("_")[1])
    
    async with state.proxy() as data:
        data['birth_month'] = month_num
    
    await show_day_selection(callback_query.message, data['birth_year'], month_num)
    await UserInfo.waiting_for_day.set()

async def show_day_selection(message: types.Message, year: int, month: int):
    if month == 2:
        days_in_month = 29 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 28
    elif month in [4, 6, 9, 11]:
        days_in_month = 30
    else:
        days_in_month = 31
    
    keyboard = types.InlineKeyboardMarkup(row_width=7)
    for day in range(1, days_in_month + 1):
        keyboard.insert(types.InlineKeyboardButton(str(day), callback_data=f"day_{day}"))
    
    await message.answer("Выберите день рождения:", reply_markup=keyboard)

async def handle_birth_day(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.answer()
    
    day = int(callback_query.data.split("_")[1])
    
    async with state.proxy() as data:
        # Вычисляем возраст
        birth_date = datetime.date(
            data['birth_year'],
            data['birth_month'],
            day
        )
        today = datetime.date.today()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        
        # Записываем возраст в birth_date как строку
        data['birth_date'] = str(age)  # Просто число "25" вместо "01.01.1999"
    
    await callback_query.message.answer(
        "Теперь введите ваши хронические состояния:",
        reply_markup=types.ReplyKeyboardRemove()
    )
    await UserInfo.waiting_for_conditions.set()
    
    await callback_query.message.answer(
        "Теперь введите ваши хронические состояния (если есть), разделяя их запятыми:",
        reply_markup=types.ReplyKeyboardRemove()
    )
    await UserInfo.waiting_for_conditions.set()

# Обработчик хронических состояний
async def handle_chronic_conditions(message: types.Message, state: FSMContext):
    chronic_conditions = message.text.strip() or "Не указаны"
    user_id = message.from_user.id

    async with state.proxy() as data:
        if not all(k in data for k in ['timezone', 'birth_date']):
            await message.answer("Ошибка: недостаточно данных. Пожалуйста, начните заново.")
            await state.finish()
            return

        try:
            await save_user_info(
                user_id=user_id,
                timezone=data['timezone'],
                birth_date=data['birth_date'],
                chronic_conditions=chronic_conditions
            )
            
            age_text = f" ({data.get('age_display', '')} лет)" if data.get('age_display') else ""
            await message.answer(
                "✅ Ваши данные успешно сохранены!\n"
                f"⏰ Часовой пояс: {data['timezone']}\n"
                f"🎂 Возраст: {data['birth_date']}{age_text}\n"
                f"💊 Хронические состояния: {chronic_conditions}"
            )
            
            await message.answer("Что вы хотите сделать?", reply_markup=get_start_keyboard())
        except Exception as e:
            logging.error(f"Ошибка при сохранении данных: {e}")
            await message.answer("Произошла ошибка при сохранении данных. Пожалуйста, попробуйте позже.")
        finally:
            await state.finish()

# Вспомогательная функция для определения часового пояса
async def get_timezone(bot, chat_id, city):
    try:
        geo = geopy.geocoders.Nominatim(user_agent="MedicineReminderBot")
        location = geo.geocode(city)
        if not location:
            await bot.send_message(chat_id, "Не удалось найти такой город. Попробуйте указать более крупный город поблизости.")
            return None

        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lat=location.latitude, lng=location.longitude)
        if not timezone_str:
            await bot.send_message(chat_id, "Не удалось определить часовой пояс для этого города.")
            return None

        tz = pytz.timezone(timezone_str)
        tz_info = datetime.datetime.now(tz=tz).strftime("%z")
        tz_info = f"{tz_info[:3]}:{tz_info[3:]}"
        
        await bot.send_message(chat_id, f"Часовой пояс установлен: {timezone_str} (UTC{tz_info})")
        return timezone_str

    except Exception as e:
        logging.error(f"Ошибка при определении часового пояса: {e}")
        await bot.send_message(chat_id, "Произошла ошибка при определении часового пояса. Пожалуйста, попробуйте позже.")
        return None