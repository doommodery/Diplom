from aiogram import types, Dispatcher
from aiogram.types import InputFile
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
import sqlite3, datetime, asyncio, logging, requests, torch
import pandas as pd
from keyboards.keyboards import get_start_keyboard, get_health_keyboard, get_report_days_keyboard
from config import DB_PATH, WEATHER_API_KEY
from datetime import date
import os
from utils.model_utils import AIModel
import re
from generation_sentetic_data import SYMPTOMS,CHRONIC_CONDITIONS

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Состояния
class HealthConditionState(StatesGroup):
    waiting_for_condition = State()
    waiting_for_analysis = State()
    waiting_for_weather = State()

class ReportState(StatesGroup):
    waiting_for_days = State()

import difflib
from typing import Dict, Optional, Tuple

async def process_health_analysis(message: types.Message, state: FSMContext):
    """Обработка анализа состояния здоровья с сохранением состояния"""
    try:
        user_id = message.from_user.id
        health_text = message.text
        
        # Проверяем, было ли предыдущее сообщение с просьбой уточнить
        async with state.proxy() as data:
            if data.get('awaiting_clarification'):
                # Это ответ на просьбу уточнить
                data['awaiting_clarification'] = False
                health_text = message.text  # Используем новый текст
            else:
                # Первое сообщение - валидируем
                validated = validate_symptoms(health_text)
                if not validated:
                    await suggest_better_description(message, state)
                    return
        # Получаем модель и проверяем доступность
        ai_model = await AIModel.get_instance()
        if not ai_model.is_ready:
            await message.answer("⚠️ Система анализа временно недоступна.")
            return

        user_id = message.from_user.id
        health_text = message.text
        
        # Получаем данные пользователя
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT timezone, birth_date, chronic_conditions FROM user_info WHERE user_id = ?", 
                (user_id,)
            )
            user_info = cursor.fetchone()
            
            if not user_info:
                await message.answer("Сначала заполните информацию в профиле.")
                await state.finish()
                return

            timezone, birth_date, chronic_conditions = user_info
            age = birth_date if birth_date else None
            
            # Валидация симптомов
            validated_symptoms = validate_symptoms(health_text)
            if not validated_symptoms:
                await suggest_better_description(message)
                return
            
            # Валидация хронических состояний
            chronic = chronic_conditions if chronic_conditions else "-"
            validated_chronic = validate_chronic_conditions(chronic)
            
            # Получаем погодные данные
            city = get_city_by_timezone(timezone)
            weather = await get_weather_data(city)
            weather_cond = weather.split(',')[0] if weather else "нормальные"

            # Подготовка данных для модели
            inputs = ai_model.tokenizer(
                health_text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(ai_model.device)

            # Подготовка дополнительных признаков
            input_df = pd.DataFrame({
                'Возраст': [age] if age else [0],
                'Хронические состояния': [validated_chronic],
                'Погодные условия': [weather_cond]
            })

            # Преобразование признаков
            try:
                additional_features = ai_model.preprocessor.transform(input_df)
                if hasattr(additional_features, "toarray"):
                    additional_features = additional_features.toarray()
                features_tensor = torch.tensor(additional_features, dtype=torch.float).to(ai_model.device)
            except Exception as e:
                logger.error(f"Ошибка преобразования признаков: {e}")
                await message.answer("⚠️ Ошибка обработки данных.")
                return

            # Предсказание
            with torch.no_grad():
                diagnosis_logits, doctor_logits = ai_model.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    additional_features=features_tensor
                )

            # Обработка результатов
            diagnosis = ai_model.diagnosis_encoder.inverse_transform([torch.argmax(diagnosis_logits).item()])[0]
            doctor = ai_model.doctor_encoder.inverse_transform([torch.argmax(doctor_logits).item()])[0]

            # Формирование ответа
            response = format_response(
                diagnosis=diagnosis,
                doctor=doctor,
                age=age,
                chronic=validated_chronic,
                weather=weather_cond
            )

            # Сохранение в БД
            cursor.execute('''
                INSERT INTO user_condition_analysis 
                (user_id, user_condition, weather_condition, doctor, diagnosis, date_an, chronic_conditions, age)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, health_text, weather, doctor, diagnosis, 
                datetime.datetime.now().strftime("%Y-%m-%d"), 
                chronic_conditions, age
            ))
            conn.commit()

        await message.answer(response, reply_markup=get_health_keyboard())

    except Exception as e:
        logger.error(f"Ошибка анализа: {e}", exc_info=True)
        await message.answer("⚠️ Произошла ошибка при анализе.")
    finally:
        if not (await state.get_data()).get('awaiting_clarification'):
            await state.finish()

def validate_symptoms(text: str, threshold: float = 0.7) -> Optional[Dict]:
    """Валидация симптомов против эталонного списка"""
    normalized_text = normalize_text(text)
    
    # Проверка точных совпадений
    for symptom, data in SYMPTOMS.items():
        if symptom.lower() in normalized_text:
            return data
    
    # Проверка схожести по расстоянию Левенштейна
    best_match = None
    highest_ratio = 0
    
    for symptom in SYMPTOMS.keys():
        ratio = difflib.SequenceMatcher(
            None, 
            normalize_text(symptom), 
            normalized_text
        ).ratio()
        
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = symptom
    
    if highest_ratio >= threshold:
        return SYMPTOMS[best_match]
    
    return None

def validate_chronic_conditions(conditions: str) -> str:
    """Валидация хронических состояний"""
    if not conditions or conditions == "-":
        return "-"
    
    # Разделяем условия, если их несколько
    conditions_list = [c.strip() for c in conditions.split(",")]
    validated = []
    
    for condition in conditions_list:
        # Проверка точных совпадений
        if condition in CHRONIC_CONDITIONS:
            validated.append(condition)
            continue
        
        # Поиск похожих
        best_match = None
        highest_ratio = 0
        
        for chronic_condition in CHRONIC_CONDITIONS.keys():
            ratio = difflib.SequenceMatcher(
                None, 
                normalize_text(chronic_condition), 
                normalize_text(condition)
            ).ratio()
            
            if ratio > highest_ratio:
                highest_ratio = ratio
                best_match = chronic_condition
        
        if highest_ratio > 0.6:  # Более низкий порог для медицинских терминов
            validated.append(best_match)
    
    return ", ".join(validated) if validated else "-"

def normalize_text(text: str) -> str:
    """Нормализация текста для сравнения"""
    text = text.lower().strip()
    # Удаление пунктуации и лишних пробелов
    text = " ".join(re.sub(r"[^а-яё\s]", "", text).split())
    return text

async def suggest_better_description(message: types.Message, state: FSMContext):
    """Предложение пользователю уточнить описание с сохранением состояния"""
    examples = [
        "• Головная боль и тошнота",
        "• Чувствую себя хорошо",
        "• Шум в ушах",
        "• Двоение в глазах,Боль в глазах"
    ]
    
    await state.update_data(awaiting_clarification=True)
    
    await message.answer(
        "Пожалуйста, опишите симптомы более конкретно:\n"
        "1. Что именно беспокоит (например, боль, температура)\n"
        "2. Дополнительные симптомы\n\n"
        "Примеры:\n" + "\n".join(examples),
        reply_markup=types.ReplyKeyboardRemove()
    )

def format_response(
    diagnosis: str, 
    doctor: str, 
    age: Optional[int],
    chronic: str,
    weather: str
) -> str:
    """Форматирование ответа пользователю"""
    response = [
        "🔍 Результаты анализа:\n",
        f"🩺 Предварительный диагноз: {diagnosis}",
        f"👨‍⚕️ Рекомендуемый специалист: {doctor}",
        "\n📌 Дополнительная информация:"
    ]
    
    if age:
        response.append(f"• Возраст: {age}")
    
    response.append(f"• Хронические состояния: {chronic if chronic != '-' else 'нет'}")
    response.append(f"• Погодные условия: {weather}")
    
    return "\n".join(response)

async def delete_old_records():
    """Удаление старых записей (старше 60 дней)"""
    while True:
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            two_months_ago = (datetime.datetime.now() - datetime.timedelta(days=60)).strftime("%Y-%m-%d")
            cursor.execute("DELETE FROM user_condition_analysis WHERE date_an <= ?", (two_months_ago,))
            conn.commit()
            conn.close()
            await asyncio.sleep(86400)  # Проверка раз в день
        except Exception as e:
            logger.error(f"Ошибка при удалении старых записей: {e}")
            await asyncio.sleep(3600)  # Повтор через час при ошибке

def get_city_by_timezone(timezone: str) -> str:
    """Получение города из временной зоны"""
    return timezone.split("/")[-1].replace("_", " ") if timezone else "Moscow"

async def get_weather_data(city: str) -> str:
    """Асинхронное получение данных о погоде"""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric&lang=ru"
    try:
        response = await asyncio.to_thread(requests.get, url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            weather = data["weather"][0]["description"]
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            return f"{weather}, {temp}°C, {humidity}%"
    except Exception as e:
        logger.error(f"Ошибка получения погоды: {e}")
    return "Не удалось получить данные о погоде."

async def generate_report(records, user_id: int, start_date: str, end_date: str):
    """Генерация отчетов в CSV и TXT форматах"""
    try:
        df = pd.DataFrame(records, columns=[
        "Состояние здоровья", "Погодные условия", "Дата", 
        "Диагноз", "Хронические заболевания", "Возраст", "Рекомендованный врач"
        ])
        
        csv_filename = f"reports/health_report_{user_id}_{start_date}_to_{end_date}.csv"
        txt_filename = f"reports/health_report_{user_id}_{start_date}_to_{end_date}.txt"
        
        os.makedirs("reports", exist_ok=True)
        df.to_csv(csv_filename, index=False, encoding="utf-8")
        
        with open(txt_filename, "w", encoding="utf-8") as f:
            for record in records:
                f.write(
                    f"{record[2]}: Состояние: {record[0]}\n"
                    f"Погода: {record[1]}\n"
                    f"Диагноз: {record[3]}\n"
                    f"Врач: {record[6]}\n\n"
                )
        
        return InputFile(csv_filename), InputFile(txt_filename)
    except Exception as e:
        logger.error(f"Ошибка генерации отчета: {e}")
        raise

async def start_health_entry(callback_query: types.CallbackQuery):
    """Начало записи о состоянии здоровья"""
    try:
        await callback_query.message.answer("Опишите ваше текущее состояние здоровья:")
        await HealthConditionState.waiting_for_condition.set()
    except Exception as e:
        logger.error(f"Ошибка в start_health_entry: {e}")
        await callback_query.message.answer("Произошла ошибка. Попробуйте позже.")

async def process_health_entry(message: types.Message, state: FSMContext):
    """Обработка записи о состоянии здоровья"""
    try:
        user_id = message.from_user.id
        health_data = message.text
        date_now = datetime.datetime.now().strftime("%Y-%m-%d")

        async with state.proxy() as data:
            data['health_condition'] = health_data

        # Получаем данные пользователя из БД
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT timezone, birth_date, chronic_conditions FROM user_info WHERE user_id = ?", 
            (user_id,)
        )
        user_info = cursor.fetchone()
        
        if not user_info:
            await message.answer("Сначала заполните информацию о себе в профиле.")
            await state.finish()
            return

        timezone, birth_date, chronic_conditions = user_info
        age = birth_date
        city = get_city_by_timezone(timezone)
        
        # Получаем данные о погоде асинхронно
        weather_data = await get_weather_data(city)

        # Сохраняем запись в БД
        cursor.execute('''
            INSERT INTO user_condition_analysis 
            (user_id, user_condition, weather_condition, date_an, chronic_conditions, age, diagnosis, doctor)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, health_data, weather_data, date_now, chronic_conditions, age,"-","-"))
        
        conn.commit()
        conn.close()
        
        await message.answer(
            "✅ Запись сохранена.\n"
            f"❤️Самочувствие: {health_data}\n"
            f"📍 Местоположение: {city}\n"
            f"🌤 Погода: {weather_data}",
            reply_markup=get_health_keyboard()
        )
    except Exception as e:
        logger.error(f"Ошибка в process_health_entry: {e}")
        await message.answer("Произошла ошибка при сохранении записи.")
    finally:
        await state.finish()

async def start_health_analysis(callback_query: types.CallbackQuery):
    """Начало анализа"""
    try:
        await callback_query.message.answer("Опишите ваше состояние для анализа:")
        await HealthConditionState.waiting_for_analysis.set()
    except Exception as e:
        logger.error(f"Ошибка в start_health_analysis: {str(e)}")
        await callback_query.message.answer("Произошла ошибка.")

# Обработчики отчетов
async def process_report_days(update: types.Message | types.CallbackQuery, state: FSMContext):
    """Формирование отчета за выбранный период"""
    try:
        data = await state.get_data()
        days = data.get("days", 1)
        user_id = update.from_user.id
        message = update.message if isinstance(update, types.CallbackQuery) else update

        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT user_condition, weather_condition, date_an, diagnosis, chronic_conditions, age, doctor
            FROM user_condition_analysis 
            WHERE user_id = ? AND date_an BETWEEN ? AND ?
            ORDER BY date_an DESC
            ''', (user_id, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))
        
        records = cursor.fetchall()
        conn.close()

        if not records:
            await message.answer(f"За последние {days} дней записей не найдено.")
            await state.finish()
            return

        csv_report, txt_report = await generate_report(
            records, user_id, 
            start_date.strftime("%Y-%m-%d"), 
            end_date.strftime("%Y-%m-%d")
        )

        await message.answer_document(csv_report, caption=f"Отчет за {days} дней (CSV)")
        await message.answer_document(txt_report, caption=f"Отчет за {days} дней (TXT)")
        
    except Exception as e:
        logger.error(f"Ошибка формирования отчета: {e}")
        await message.answer("⚠️ Не удалось сформировать отчет. Попробуйте позже.")
    finally:
        await state.finish()

async def process_days_input(message: types.Message, state: FSMContext):
    """Обработка ввода количества дней для отчета"""
    try:
        days = int(message.text)
        if days <= 0:
            await message.answer("Введите положительное число.")
            return
        if days > 365:
            await message.answer("Максимальный период - 365 дней.")
            return
            
        await state.update_data(days=days)
        await process_report_days(message, state)
    except ValueError:
        await message.answer("Пожалуйста, введите число дней.")

# Обработчики кнопок
async def report_1_day(callback_query: types.CallbackQuery, state: FSMContext):
    await state.update_data(days=1)
    await process_report_days(callback_query, state)

async def report_7_days(callback_query: types.CallbackQuery, state: FSMContext):
    await state.update_data(days=7)
    await process_report_days(callback_query, state)

async def report_14_days(callback_query: types.CallbackQuery, state: FSMContext):
    await state.update_data(days=14)
    await process_report_days(callback_query, state)

async def report_30_days(callback_query: types.CallbackQuery, state: FSMContext):
    await state.update_data(days=30)
    await process_report_days(callback_query, state)

async def health_keyboard(callback_query: types.CallbackQuery):
    await callback_query.message.answer("Выберите действие:", reply_markup=get_health_keyboard())

async def cancel_health_entry(callback_query: types.CallbackQuery, state: FSMContext):
    await state.finish()
    await callback_query.message.answer("Действие отменено.", reply_markup=get_start_keyboard())

async def load_report(callback_query: types.CallbackQuery):
    await callback_query.message.answer("Выберите период:", reply_markup=get_report_days_keyboard())
    await ReportState.waiting_for_days.set()

# Регистрация обработчиков
def register_handlers_health(dp: Dispatcher):
    dp.register_callback_query_handler(start_health_entry, text="record_health")
    dp.register_callback_query_handler(start_health_analysis, text="analyze_health")
    dp.register_message_handler(process_health_entry, state=HealthConditionState.waiting_for_condition)
    dp.register_message_handler(process_health_analysis, state=HealthConditionState.waiting_for_analysis)
    dp.register_callback_query_handler(cancel_health_entry, text="cancel_health_entry", state="*")
    
    dp.register_callback_query_handler(load_report, text="report_health")
    dp.register_callback_query_handler(report_1_day, text="report_1_day", state=ReportState.waiting_for_days)
    dp.register_callback_query_handler(report_7_days, text="report_7_days", state=ReportState.waiting_for_days)
    dp.register_callback_query_handler(report_14_days, text="report_14_days", state=ReportState.waiting_for_days)
    dp.register_callback_query_handler(report_30_days, text="report_30_days", state=ReportState.waiting_for_days)
    dp.register_message_handler(process_days_input, state=ReportState.waiting_for_days)
    
    dp.register_callback_query_handler(health_keyboard, text="menu_health")
    dp.register_callback_query_handler(health_keyboard, text="back_to_health")