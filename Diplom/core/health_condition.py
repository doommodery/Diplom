# ================== base imports ==================
import sqlite3, datetime, asyncio, logging, requests, torch
from typing import Dict, List, Tuple
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
# ================== enhanced imports ==================
from aiogram import types
from aiogram.types import InputFile
from aiogram.dispatcher import FSMContext
from core.states import HealthConditionState, ReportState
from utils.model_utils import AIModel
from io import BytesIO
# ================== From other files ==================
from utils.validation import ImprovedSymptomAnalyzer
from config import DB_PATH, WEATHER_API_KEY
from scripts.generation_sentetic_data import SYMPTOMS
from keyboards.keyboards import get_start_keyboard, get_health_keyboard, get_report_days_keyboard

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportGenerationError(Exception):
    """Исключение для ошибок генерации отчётов"""
    def __init__(self, message: str, original_error: Exception = None):
        self.message = message
        self.original_error = original_error
        super().__init__(message)

    def __str__(self):
        if self.original_error:
            return f"{self.message} (Original: {str(self.original_error)})"
        return self.message

async def process_health_analysis(message: types.Message, state: FSMContext):
    user_id = message.from_user.id
    user_input = message.text
    logger.info(f"Начало анализа для пользователя {user_id}, текст: '{user_input}'")

    try:
        # Инициализация анализатора симптомов
        analyzer = ImprovedSymptomAnalyzer(
            use_llm=True,
            symptom_threshold=0.65,
            condition_threshold=0.8
        )
        
        # Анализ симптомов
        symptoms = analyzer.analyze_symptoms(user_input)
        chronic = analyzer.analyze_conditions(user_input)
        
        # Очистка результатов
        symptoms = [s for s in symptoms if s != "Не обнаружено"]
        chronic = [c for c in chronic if c != "Не обнаружено"]
        
        logger.info(f"Результаты анализа - симптомы: {symptoms}, хронические: {chronic}")

        # Получаем данные пользователя из БД
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT timezone, birth_date, chronic_conditions FROM user_info WHERE user_id = ?", 
                (user_id,)
            )
            user_info = cursor.fetchone()
            
            if not user_info:
                logger.error("Данные пользователя не найдены в БД")
                await message.answer("❌ Сначала заполните информацию в профиле.", reply_markup=get_start_keyboard())
                await state.finish()
                return

            timezone, birth_date, existing_chronic = user_info
            
            # Обработка хронических заболеваний
            final_chronic = existing_chronic if existing_chronic and existing_chronic != "-" else "-"
            if chronic:
                chronic_str = ", ".join(chronic)
                final_chronic = f"{existing_chronic}, {chronic_str}" if existing_chronic and existing_chronic != "-" else chronic_str

            # Получение температуры
            temperature = None
            if timezone:
                try:
                    city = timezone.split("/")[-1].replace("_", " ")
                    weather = await get_weather_data(city)
                    if weather and "Не удалось" not in weather:
                        temp_part = weather.split(',')[1].strip()
                        temperature = float(temp_part.split('°')[0].strip())
                except Exception as e:
                    logger.error(f"Ошибка получения погоды: {str(e)}")

        # Подготовка данных для AI модели
        symptoms_str = ", ".join(symptoms) if symptoms else "Чувствую себя хорошо"
        
        ai_model = await AIModel.get_instance()
        if not ai_model.is_ready:
            logger.error("AI модель не готова к работе")
            await message.answer("⚠️ Система анализа временно недоступна.")
            return

        try:
            # Создаем входные данные для модели
            input_data = {
                'Состояние здоровья': symptoms_str,
                'Возраст': int(birth_date) if birth_date else 30,  # дефолтное значение если возраст не указан
                'Температура (°C)': temperature if temperature is not None else 20.0  # дефолтное значение
            }

            # Получаем предсказания
            encodings, additional_features = ai_model.preprocess_input(input_data)
            features_tensor = torch.tensor(additional_features).to(ai_model.device)
            
            inputs = {
                'input_ids': encodings['input_ids'].to(ai_model.device),
                'attention_mask': encodings['attention_mask'].to(ai_model.device),
                'additional_features': features_tensor
            }

            with torch.no_grad():
                diagnosis_logits, doctor_logits = ai_model.model(**inputs)

            diagnosis = ai_model.diagnosis_encoder.inverse_transform([torch.argmax(diagnosis_logits).item()])[0]
            doctor = ai_model.doctor_encoder.inverse_transform([torch.argmax(doctor_logits).item()])[0]

            # Формирование ответа
            response = (
                f"🔍 Результаты анализа:\n\n"
                f"📋 Симптомы: {symptoms_str}\n\n"
                f"🩺 Предварительное заключение: {diagnosis}\n\n"
                f"👨‍⚕️ Рекомендуется консультация: {doctor}\n\n"
                f"📌 Дополнительная информация:\n"
                f"• Возраст: {birth_date if birth_date else 'не указан'}\n"
                f"• Хронические состояния: {existing_chronic if existing_chronic != '-' else 'не указаны'}\n"
                f"• Погодные условия: {temperature if temperature is not None else 'нет данных'}°C\n"
            )

            # Сохранение результатов в БД
            cursor.execute('''
                INSERT INTO user_condition_analysis 
                (user_id, user_condition, weather_condition, doctor, diagnosis, date_an, chronic_conditions, age)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, symptoms_str, str(temperature), doctor, diagnosis, 
                datetime.datetime.now().strftime("%Y-%m-%d"), 
                existing_chronic, birth_date
            ))
            conn.commit()

        except Exception as e:
            logger.error(f"Ошибка AI модели: {str(e)}")
            await message.answer("⚠️ Ошибка анализа данных.")
            return

        await message.answer(response, reply_markup=get_health_keyboard())

    except Exception as e:
        logger.error(f"Критическая ошибка в процессе анализа: {str(e)}", exc_info=True)
        await message.answer("⚠️ Произошла непредвиденная ошибка.")
    finally:
        await state.finish()
        conn.close()
        
async def get_weather_data(city: str) -> str:
    """Получение данных о погоде"""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric&lang=ru"
    try:
        response = await asyncio.to_thread(requests.get, url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return f"{data['weather'][0]['description']}, {data['main']['temp']}°C, {data['main']['humidity']}%"
    except Exception:
        return "Не удалось получить данные о погоде."

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

async def generate_report(
    records: List[Tuple], 
    user_id: int, 
    start_date: str, 
    end_date: str
) -> Tuple[InputFile, ...]:
    """
    Генерирует медицинский отчет в форматах CSV, TXT и PNG графиков.
    
    Args:
        records: Список записей из БД в формате (симптомы, погода, дата, диагноз, ...)
        user_id: Идентификатор пользователя для именования файлов
        start_date: Начальная дата периода отчета (YYYY-MM-DD)
        end_date: Конечная дата периода отчета (YYYY-MM-DD)
    
    Returns:
        Кортеж из InputFile (CSV, TXT, PNG-графики...)
    
    Raises:
        ReportGenerationError: При критических ошибках обработки данных
    """
    try:
        # 1. Валидация входных данных
        if not records:
            return _generate_empty_report(start_date, end_date)

        # 2. Подготовка DataFrame
        df = pd.DataFrame(records, columns=[
            "Состояние здоровья", "Погодные условия", "Дата", 
            "Диагноз", "Хронические заболевания", "Возраст", "Рекомендованный врач"
        ])
        
        # 3. Создание директории для отчетов
        os.makedirs("reports", exist_ok=True)
        base_filename = f"reports/health_report_{user_id}_{start_date}_to_{end_date}"
        
        # 4. Текстовый отчет
        txt_file = _generate_text_report(df, base_filename)
        
        # 5. Визуализации (только если есть симптомы)
        image_files = []
        symptom_details = _extract_symptom_details(df)
        
        if not symptom_details:
            image_files.append(_create_empty_plot("Нет данных о симптомах"))
        else:
            try:
                plot1 = _generate_symptoms_plot(symptom_details, start_date, end_date)
                plot2 = _generate_heatmap(symptom_details)
                image_files.extend([plot1, plot2])
            except Exception as e:
                logger.error(f"Ошибка генерации графиков: {e}")
                image_files.append(_create_error_plot(e))

        # 6. CSV отчет
        csv_file = f"{base_filename}.csv"
        df.to_csv(csv_file, index=False, encoding="utf-8")
        
        return (
            InputFile(csv_file),
            InputFile(txt_file),
            *[InputFile(buf, filename=name) for name, buf in image_files]
        )
        
    except Exception as e:
        logger.critical(f"Критическая ошибка генерации отчета: {e}")
        raise ReportGenerationError(f"Ошибка создания отчета: {e}")

# --- Вспомогательные функции ---

def _generate_empty_report(start_date: str, end_date: str) -> Tuple[InputFile]:
    """Создает заглушку для пустого отчета"""
    content = f"Отчёт за период {start_date} — {end_date}\n\nДанных не найдено."
    buf = BytesIO(content.encode("utf-8"))
    return (InputFile(buf, filename="empty_report.txt"),)

def _extract_symptom_details(df: pd.DataFrame) -> List[Dict]:
    """Извлекает и структурирует данные о симптомах"""
    if df.empty:
        return []

    analyzer = ImprovedSymptomAnalyzer()
    symptom_details = []
    
    for _, row in df.iterrows():
        symptoms = analyzer.analyze_symptoms(row["Состояние здоровья"])
        for symptom in symptoms:
            symptom_details.append({
                'Дата': row['Дата'],
                'Симптом': symptom,
                'Категория': next(
                    (cat for cat in SYMPTOMS.keys() if symptom in SYMPTOMS[cat]),
                    'Другое'
                ),
                'Хронические заболевания': row['Хронические заболевания']
            })
    
    return symptom_details

def _generate_text_report(df: pd.DataFrame, base_path: str) -> str:
    """Генерирует текстовую версию отчета"""
    txt_filename = f"{base_path}.txt"
    
    with open(txt_filename, "w", encoding="utf-8") as f:
        # Статистика по симптомам
        all_symptoms = [
            s for symptoms in df["Состояние здоровья"].apply(
                lambda x: ImprovedSymptomAnalyzer().analyze_symptoms(x)
            ) 
            for s in symptoms
        ]
        
        if all_symptoms:
            symptom_counts = pd.Series(all_symptoms).value_counts()
            f.write("📊 Топ 5 самых частых симптомов:\n")
            for symptom, count in symptom_counts.head(5).items():
                f.write(f"- {symptom}: {count} раз(а)\n")
            f.write("\n")
        
        # Детализация записей
        for record in df.itertuples():
            f.write(
                f"{record.Дата}:\n"
                f"• Состояние: {record.Состояние_здоровья}\n"
                f"• Погода: {record.Погодные_условия}\n"
                f"• Хронические: {record.Хронические_заболевания or 'нет'}\n"
                f"• Рекомендация: {record.Рекомендованный_врач or 'не указана'}\n\n"
            )
    
    return txt_filename

def _generate_symptoms_plot(symptom_details: List[Dict], start_date: str, end_date: str) -> Tuple[str, BytesIO]:
    """График динамики симптомов по дням"""
    symptom_df = pd.DataFrame(symptom_details)
    daily_symptoms = symptom_df.groupby(['Дата', 'Симптом']).size().unstack().fillna(0)
    
    plt.figure(figsize=(12, 6))
    daily_symptoms.plot(kind='bar', stacked=True, ax=plt.gca(), width=0.8)
    plt.title(f'Динамика симптомов\n{start_date} — {end_date}')
    plt.ylabel('Количество упоминаний')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Симптомы', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return ('symptoms_by_day.png', buf)

def _generate_heatmap(symptom_details: List[Dict]) -> Tuple[str, BytesIO]:
    """Тепловая карта распределения симптомов"""
    symptom_df = pd.DataFrame(symptom_details)
    heatmap_data = symptom_df.groupby(['Дата', 'Категория']).size().unstack().fillna(0)
    heatmap_data = heatmap_data.div(heatmap_data.sum(axis=1), axis=0)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap_data.T,
        cmap='YlOrRd',
        annot=True, 
        fmt='.1%',
        linewidths=.5,
        cbar_kws={'label': 'Доля симптомов'}
    )
    plt.title('Распределение симптомов по категориям')
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    plt.close()
    
    return ('symptoms_heatmap.png', buf)

def _create_empty_plot(message: str) -> Tuple[str, BytesIO]:
    """Создает заглушку для графиков при отсутствии данных"""
    plt.figure(figsize=(10, 2))
    plt.text(0.5, 0.5, message, ha='center', va='center')
    plt.axis('off')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return ('empty_plot.png', buf)

def _create_error_plot(error: Exception) -> Tuple[str, BytesIO]:
    """Создает график с сообщением об ошибке"""
    plt.figure(figsize=(10, 3))
    plt.text(
        0.5, 0.5, 
        f"Ошибка визуализации:\n{str(error)}", 
        ha='center', 
        va='center',
        color='red'
    )
    plt.axis('off')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return ('error_plot.png', buf)

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

        # Генерация отчетов (теперь возвращает до 4 файлов)
        report_files = await generate_report(
            records, user_id, 
            start_date.strftime("%Y-%m-%d"), 
            end_date.strftime("%Y-%m-%d")
        )
        
        # Отправка файлов (сохраняем старую логику для txt)
        await message.answer_document(report_files[1], caption=f"Отчет за {days} дней (TXT)")
        
        # Если есть графики - отправляем их
        if len(report_files) > 2:
            await message.answer("📊 Дополнительные визуализации:")
            await message.answer_document(report_files[2], caption="График частоты симптомов")
            await message.answer_document(report_files[3], caption="Тепловая карта симптомов")
        
    except Exception as e:
        logger.error(f"Ошибка формирования отчета: {e}")
        await message.answer("⚠️ Не удалось сформировать отчет. Попробуйте позже.", reply_markup=get_health_keyboard())
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

def get_city_by_timezone(timezone: str) -> str:
    """Получение города из временной зоны"""
    return timezone.split("/")[-1].replace("_", " ") if timezone else "Moscow"

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
