from aiogram import types, Dispatcher
from aiogram.types import ReplyKeyboardRemove, InputFile
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
import sqlite3, datetime, asyncio, logging, requests, torch
import pandas as pd
from keyboards.keyboards import get_start_keyboard, get_health_keyboard, get_report_days_keyboard
from config import DB_PATH

from transformers import AutoModelForCausalLM, AutoTokenizer

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Состояние для записи здоровья
class HealthConditionState(StatesGroup):
    waiting_for_condition = State()  # Для записи состояния здоровья
    waiting_for_analysis = State()   # Для анализа состояния здоровья

# Состояние для отчётов
class ReportState(StatesGroup):
    waiting_for_days = State()

# Функция удаления старых записей (запускать при старте)
async def delete_old_records():
    while True:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        two_months_ago = (datetime.datetime.now() - datetime.timedelta(days=60)).strftime("%Y-%m-%d")
        cursor.execute("DELETE FROM user_condition_analysis WHERE date_an <= ?", (two_months_ago,))
        conn.commit()
        conn.close()
        await asyncio.sleep(86400)  # Запускать раз в день

# Функция для получения города по временной зоне
def get_city_by_timezone(timezone: str) -> str:
    """
    Возвращает город по временной зоне.
    Пример: Europe/Moscow -> Moscow
    """
    return timezone.split("/")[-1]

# Функция для получения погодных данных через OpenWeatherMap API
def get_weather_data(city: str, api_key: str) -> str:
    """
    Получает погодные данные для указанного города.
    Возвращает строку в формате: "легкий дождь, 15°C, 80%, 1013 hPa"
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=ru"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        weather_description = data["weather"][0]["description"]  # Описание погоды
        temperature = data["main"]["temp"]  # Температура
        humidity = data["main"]["humidity"]  # Влажность
        pressure = data["main"]["pressure"]  # Атмосферное давление
        
        weather_info = f"{weather_description}, {temperature}°C, {humidity}%, {pressure} hPa"
        return weather_info
    else:
        return "Не удалось получить данные о погоде."

# Функция для получения данных о магнитных бурях через NOAA API
def get_magnetic_storm_data() -> str:
    """
    Получает данные о магнитных бурях.
    Возвращает строку в формате: "Kp-индекс 4"
    """
    url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        # Пример обработки данных (последняя запись)
        last_entry = data[-1]
        kp_index = last_entry[1]  # Kp-индекс (уровень геомагнитной активности)
        
        storm_info = f"Kp-индекс {kp_index}"
        return storm_info
    else:
        return "Не удалось получить данные о магнитных бурях."

# Функция для формирования отчёта
def generate_report(records, user_id: int, start_date: str, end_date: str):
    """
    Формирует отчёт в виде CSV и текстового файла.
    В названия файлов добавляются user_id и промежуток времени.
    """
    # Создаем DataFrame из данных
    df = pd.DataFrame(records, columns=["Состояние здоровья", "Погодные условия", "Дата", "Результат анализа", "Информация о здоровье"])
    
    # Формируем названия файлов
    csv_filename = f"health_report_user_{user_id}_from_{start_date}_to_{end_date}.csv"
    txt_filename = f"health_report_user_{user_id}_from_{start_date}_to_{end_date}.txt"
    
    # Сохраняем в CSV
    df.to_csv(csv_filename, index=False, encoding="utf-8")
    
    # Создаем текстовый файл
    with open(txt_filename, "w", encoding="utf-8") as txt_file:
        for record in records:
            user_condition, weather_condition, date_an, analysis_results, health_info = record
            txt_file.write(f"{date_an}:[Состояние здоровья:\"{user_condition}\",Погодные условия:\"{weather_condition}\",Результат анализа:\"{analysis_results}\",Информация о здоровье:\"{health_info}\"]\n")
    
    # Возвращаем оба файла
    return InputFile(csv_filename), InputFile(txt_filename)

# Обработчик для кнопки "Сделать запись о состоянии здоровья"
async def start_health_entry(callback_query: types.CallbackQuery):
    logging.info("Обработчик для 'Сделать запись о состоянии здоровья' сработал.")
    await callback_query.message.answer("Введите данные о своем состоянии здоровья:", reply_markup=ReplyKeyboardRemove())
    await HealthConditionState.waiting_for_condition.set()

# Обработчик ввода состояния здоровья
async def process_health_entry(message: types.Message, state: FSMContext):
    user_id = message.from_user.id  # user_id из Telegram
    health_data = message.text  # Данные из текстового сообщения
    date_now = datetime.datetime.now().strftime("%Y-%m-%d")

    # Получаем временную зону и health_info из базы данных
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT timezone, health_info FROM user_info WHERE user_id = ?", (user_id,))
    user_info_result = cursor.fetchone()
    
    if user_info_result:
        timezone, health_info = user_info_result
        city = get_city_by_timezone(timezone)  # Получаем город по временной зоне
    else:
        city = "Moscow"  # По умолчанию, если временная зона не указана
        health_info = "Нет данных о здоровье"  # По умолчанию, если health_info не указан

    # Получаем погодные данные
    api_key = "b7e1b3d26225eccd68d57bf5d25a4cdc"  # Замените на ваш API-ключ OpenWeatherMap
    weather_data = get_weather_data(city, api_key)

    # Получаем данные о магнитных бурях
    magnetic_storm_data = get_magnetic_storm_data()

    # Объединяем данные в нужном формате
    weather_condition = f"{weather_data}, {magnetic_storm_data}"

    # Сохраняем в базу данных
    cursor.execute("""
        INSERT INTO user_condition_analysis (user_id, user_condition, weather_condition, date_an, health_info)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, health_data, weather_condition, date_now, health_info))
    conn.commit()
    conn.close()

    await message.answer("Запись сохранена.", reply_markup=get_health_keyboard())  # Исправлено
    await state.finish()

# Обработчик для выбора временного промежутка
async def process_report_days(update: types.Message | types.CallbackQuery, state: FSMContext):
    """
    Обрабатывает выбор временного промежутка и формирует отчёт.
    Поддерживает как Message, так и CallbackQuery.
    """
    # Получаем количество дней из состояния
    data = await state.get_data()
    days = data.get("days", 1)  # По умолчанию 1 день, если значение не задано

    # Определяем user_id и объект для ответа
    if isinstance(update, types.CallbackQuery):
        user_id = update.from_user.id
        message = update.message
    else:
        user_id = update.from_user.id
        message = update

    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)

    # Логирование для отладки
    logging.info(f"Запрос отчёта для user_id={user_id} с {start_date} по {end_date}")

    # Получаем данные из базы данных
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Логирование SQL-запроса
    query = """
        SELECT user_condition, weather_condition, date_an, analysis_results, health_info 
        FROM user_condition_analysis 
        WHERE user_id = ? AND date_an BETWEEN ? AND ?
    """
    params = (user_id, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    logging.info(f"Выполняем SQL-запрос: {query} с параметрами {params}")

    cursor.execute(query, params)
    records = cursor.fetchall()
    conn.close()

    # Логирование результатов запроса
    logging.info(f"Найдено записей: {len(records)}")
    for record in records:
        logging.info(f"Запись: {record}")

    if not records:
        await message.answer("За выбранный период записей не найдено.")
        await state.finish()
        return

    # Формируем отчёты
    csv_report, txt_report = generate_report(
        records,
        user_id=user_id,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )

    # Отправляем оба файла
    await message.answer_document(document=csv_report)
    await message.answer_document(document=txt_report)
    await state.finish()

async def start_health_analysis(callback_query: types.CallbackQuery):
    logging.info("Обработчик для 'Проанализировать состояние здоровья' сработал.")
    await callback_query.message.answer("Введите данные о своем состоянии здоровья для анализа:", reply_markup=ReplyKeyboardRemove())
    await HealthConditionState.waiting_for_analysis.set()

from openai import OpenAI

client = OpenAI(
    api_key="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"  # Замените на ваш реальный API ключ
)

async def analyze_health_with_ai(health_data: str) -> str:
    """
    Анализирует данные о состоянии здоровья с использованием API ChatGPT.
    """
    if not health_data.strip():
        return "Текст для анализа отсутствует."

    try:
        # Формируем входной текст для модели
        input_text = (
            "Ты врач, анализирующий здоровье. Проанализируй текст и определи, является ли состояние здоровья нормальным или аномальным. "
            f"Если состояние аномальное, порекомендуй, к какому врачу обратиться.\n\nТекст: {health_data}"
        )

        # Вызов API ChatGPT
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Используем модель GPT-4
            messages=[
                {"role": "system", "content": "Ты врач, анализирующий здоровье."},
                {"role": "user", "content": input_text}
            ]
        )

        # Получаем ответ от модели
        generated_text = completion.choices[0].message.content
        return generated_text

    except Exception as e:
        return f"Произошла ошибка: {str(e)}"

def remove_repetitions(text):
    """
    Удаляет повторяющиеся фрагменты из текста.
    """
    words = text.split()
    unique_words = []
    for word in words:
        if word not in unique_words[-5:]:  # Проверяем последние 5 слов
            unique_words.append(word)
    return " ".join(unique_words)

async def process_health_analysis(message: types.Message, state: FSMContext):
    user_id = message.from_user.id  # user_id из Telegram
    health_data = message.text  # Данные из текстового сообщения
    date_now = datetime.datetime.now().strftime("%Y-%m-%d")

    # Получаем временную зону и health_info из базы данных
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT timezone, health_info FROM user_info WHERE user_id = ?", (user_id,))
    user_info_result = cursor.fetchone()
    
    if user_info_result:
        timezone, health_info = user_info_result
        city = get_city_by_timezone(timezone)  # Получаем город по временной зоне
    else:
        city = "Moscow"  # По умолчанию, если временная зона не указана
        health_info = "Нет данных о здоровье"  # По умолчанию, если health_info не указан

    # Получаем погодные данные
    api_key = "b7e1b3d26225eccd68d57bf5d25a4cdc"  # Замените на ваш API-ключ OpenWeatherMap
    weather_data = get_weather_data(city, api_key)

    # Получаем данные о магнитных бурях
    magnetic_storm_data = get_magnetic_storm_data()

    # Объединяем данные в нужном формате
    weather_condition = f"{weather_data}, {magnetic_storm_data}"

    # Анализируем состояние здоровья с помощью нейросети
    analysis_results = await analyze_health_with_ai(health_data)

    # Сохраняем в базу данных
    cursor.execute("""
        INSERT INTO user_condition_analysis (user_id, user_condition, weather_condition, date_an, health_info, analysis_results)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, health_data, weather_condition, date_now, health_info, analysis_results))
    conn.commit()
    conn.close()

    # Отправляем результат анализа пользователю
    await message.answer(f"Результат анализа:\n{analysis_results}", reply_markup=get_health_keyboard())
    await state.finish()

# Обработчик для ввода количества дней вручную
async def process_days_input(message: types.Message, state: FSMContext):
    try:
        days = int(message.text)  # Пытаемся преобразовать введенный текст в число
        if days <= 0:
            await message.answer("Введите положительное число дней.")
            return
        
        await state.update_data(days=days)  # Сохраняем количество дней в состояние
        await process_report_days(message, state)  # Вызываем функцию формирования отчёта
    except ValueError:
        await message.answer("Пожалуйста, введите число дней.")

# Обработчик для кнопки "Отправить отчёт"
async def send_report(callback_query: types.CallbackQuery):
    await callback_query.message.answer("Функция отправки отчёта пока в разработке.")

# Обработчик для кнопки "Назад" в разделе отчётов
async def back_to_health(callback_query: types.CallbackQuery):
    await callback_query.message.answer("Возвращаемся в меню 'Здоровье'.", reply_markup=get_health_keyboard())

# Обработчики для кнопок с днями отчётов
async def report_1_day(callback_query: types.CallbackQuery, state: FSMContext):
    await process_report_days_with_days(callback_query, state, days=1)

async def report_7_days(callback_query: types.CallbackQuery, state: FSMContext):
    await process_report_days_with_days(callback_query, state, days=7)

async def report_14_days(callback_query: types.CallbackQuery, state: FSMContext):
    await process_report_days_with_days(callback_query, state, days=14)

async def report_30_days(callback_query: types.CallbackQuery, state: FSMContext):
    await process_report_days_with_days(callback_query, state, days=30)

# Вспомогательная функция для обработки выбора дней
async def process_report_days_with_days(callback_query: types.CallbackQuery, state: FSMContext, days: int):
    await state.update_data(days=days)  # Сохраняем выбранное количество дней
    await process_report_days(callback_query, state)

# Обработчик для кнопки "Здоровье"
async def health_keyboard(callback_query: types.CallbackQuery):
    logging.info("Обработчик для 'Здоровье' сработал.")
    await callback_query.message.answer("Выберите действие:", reply_markup=get_health_keyboard())

# Хэндлер для кнопки "Назад"
async def cancel_health_entry(callback_query: types.CallbackQuery, state: FSMContext):
    await state.finish()
    await callback_query.message.answer("Запись отменена.", reply_markup=get_start_keyboard())

# Обработчик для кнопки "Загрузить отчёт"
async def load_report(callback_query: types.CallbackQuery):
    await callback_query.message.answer("Выберите временной промежуток для отчёта:", reply_markup=get_report_days_keyboard())  # Исправлено
    await ReportState.waiting_for_days.set()

# Регистрация хэндлеров
def register_handlers_health(dp: Dispatcher):
    dp.register_callback_query_handler(start_health_entry, text="record_health")
    dp.register_callback_query_handler(start_health_analysis, text="analyze_health")  # Новая кнопка
    dp.register_message_handler(process_health_entry, state=HealthConditionState.waiting_for_condition)
    dp.register_message_handler(process_health_analysis, state=HealthConditionState.waiting_for_analysis)  # Новый обработчик
    dp.register_callback_query_handler(cancel_health_entry, text="cancel_health_entry", state="*")
    dp.register_callback_query_handler(health_keyboard, text="menu_health")
    dp.register_callback_query_handler(health_keyboard, text="back_to_health")
    
    # Регистрация обработчиков для отчётов
    dp.register_callback_query_handler(load_report, text="report_health")
    dp.register_callback_query_handler(report_1_day, text="report_1_day", state=ReportState.waiting_for_days)
    dp.register_callback_query_handler(report_7_days, text="report_7_days", state=ReportState.waiting_for_days)
    dp.register_callback_query_handler(report_14_days, text="report_14_days", state=ReportState.waiting_for_days)
    dp.register_callback_query_handler(report_30_days, text="report_30_days", state=ReportState.waiting_for_days)
    dp.register_message_handler(process_days_input, state=ReportState.waiting_for_days)
    dp.register_callback_query_handler(send_report, text="send_report")
    dp.register_callback_query_handler(back_to_health, text="back_to_health")
