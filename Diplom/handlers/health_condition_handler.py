# ================== base imports ==================
import sqlite3, datetime, asyncio, logging, requests, torch
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
# ================== enhanced imports ==================
from aiogram import types, Dispatcher
from aiogram.types import InputFile
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from utils.model_utils import AIModel
from io import BytesIO
# ================== From other files ==================
from validation import ImprovedSymptomAnalyzer
from config import DB_PATH, WEATHER_API_KEY
from generation_sentetic_data import SYMPTOMS
from keyboards.keyboards import get_start_keyboard, get_health_keyboard, get_report_days_keyboard

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Состояния
class HealthConditionState(StatesGroup):
    waiting_for_condition = State()
    waiting_for_analysis = State()

class ReportState(StatesGroup):
    waiting_for_days = State()

async def process_health_analysis(message: types.Message, state: FSMContext):
    user_id = message.from_user.id
    user_input = message.text
    logger.info(f"Начало анализа для пользователя {user_id}, текст: '{user_input}'")

    try:
        # Инициализация анализатора с логированием
        logger.info("Инициализация анализатора симптомов...")
        analyzer = ImprovedSymptomAnalyzer(
            use_llm=True,
            symptom_threshold=0.65,
            condition_threshold=0.8
        )
        
        # Анализ симптомов с логированием метода
        logger.info("Анализ симптомов...")
        symptoms = analyzer.analyze_symptoms(user_input)
        if analyzer.use_llm:
            logger.info("Анализ симптомов: использована LLM модель")
        else:
            logger.warning("Анализ симптомов: использован локальный метод (ключевые слова)")
        
        # Анализ хронических состояний с логированием метода
        logger.info("Анализ хронических состояний...")
        chronic = analyzer.analyze_conditions(user_input)
        if analyzer.use_llm:
            logger.info("Анализ хронических состояний: использована LLM модель")
        else:
            logger.warning("Анализ хронических состояний: использован локальный метод (ключевые слова)")
        
        # Очистка результатов
        symptoms = [s for s in symptoms if s != "Не обнаружено"]
        chronic = [c for c in chronic if c != "Не обнаружено"]
        
        logger.info(f"Результаты анализа - симптомы: {symptoms}, хронические: {chronic}")

        # Получаем данные пользователя из БД
        logger.debug("Получение данных пользователя из БД...")
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT timezone, birth_date, chronic_conditions FROM user_info WHERE user_id = ?", 
                (user_id,)
            )
            user_info = cursor.fetchone()
            
            if not user_info:
                logger.error("Данные пользователя не найдены в БД")
                await message.answer("❌ Сначала заполните информацию в профиле.")
                await state.finish()
                return

            timezone, birth_date, existing_chronic = user_info
            
            # Обработка хронических заболеваний с логированием
            logger.debug("Обработка хронических заболеваний...")
            final_chronic = existing_chronic if existing_chronic and existing_chronic != "-" else "-"
            if chronic:
                chronic_str = ", ".join(chronic)
                final_chronic = f"{existing_chronic}, {chronic_str}" if existing_chronic and existing_chronic != "-" else chronic_str
                logger.info(f"Обновленные хронические заболевания: {final_chronic}")

            # Получение данных о погоде с логированием
            weather_cond = "не определено"
            if timezone:
                try:
                    city = timezone.split("/")[-1].replace("_", " ")
                    logger.info(f"Запрос погоды для города: {city}")
                    weather = await get_weather_data(city)
                    weather_cond = weather.split(',')[0] if weather else "не удалось получить"
                    logger.info(f"Получены погодные условия: {weather_cond}")
                except Exception as e:
                    logger.error(f"Ошибка получения погоды: {str(e)}")

            # Подготовка данных для AI модели
            symptoms_str = ", ".join(symptoms) if symptoms else "Чувствую себя хорошо"
            logger.debug(f"Подготовленные симптомы для AI модели: {symptoms_str}")
            
            ai_model = await AIModel.get_instance()
            if not ai_model.is_ready:
                logger.error("AI модель не готова к работе")
                await message.answer("⚠️ Система анализа временно недоступна.")
                return

            try:
                logger.info("Запуск AI модели для диагностики...")
                # Создаем DataFrame с дополнительными признаками
                input_df = pd.DataFrame({
                    'Возраст': [birth_date] if birth_date else [0],
                    'Хронические состояния': [final_chronic],
                    'Погодные условия': [weather_cond]
                })

                # Преобразуем признаки
                additional_features = ai_model.preprocessor.transform(input_df)
                features_tensor = torch.tensor(
                    additional_features.toarray() if hasattr(additional_features, "toarray") else additional_features,
                    dtype=torch.float
                ).to(ai_model.device)

                # Токенизация текста симптомов
                inputs = ai_model.tokenizer(
                    symptoms_str,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(ai_model.device)

                # Получение предсказаний
                with torch.no_grad():
                    diagnosis_logits, doctor_logits = ai_model.model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        additional_features=features_tensor
                    )

                diagnosis = ai_model.diagnosis_encoder.inverse_transform([torch.argmax(diagnosis_logits).item()])[0]
                doctor = ai_model.doctor_encoder.inverse_transform([torch.argmax(doctor_logits).item()])[0]
                logger.info(f"Результаты AI модели - диагноз: {diagnosis}, врач: {doctor}")

            except Exception as e:
                logger.error(f"Ошибка AI модели: {str(e)}")
                await message.answer("⚠️ Ошибка анализа данных.")
                return

            # Формирование ответа
            response = (
                f"🔍 Результаты анализа:\n\n"
                f"📋 Симптомы: {symptoms_str}\n\n"
                f"🩺 Предварительное заключение: {diagnosis}\n\n"
                f"👨‍⚕️ Рекомендуется консультация: {doctor}\n\n"
                f"📌 Дополнительная информация:\n"
                f"• Возраст: {birth_date if birth_date else 'не указан'}\n"
                f"• Хронические состояния: {final_chronic if final_chronic != '-' else 'не указаны'}\n"
                f"• Погодные условия: {weather_cond}\n"
            )

            # Сохранение результатов в БД
            try:
                logger.debug("Сохранение результатов в БД...")
                cursor.execute('''
                    INSERT INTO user_condition_analysis 
                    (user_id, user_condition, weather_condition, doctor, diagnosis, date_an, chronic_conditions, age)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id, symptoms_str, weather_cond, doctor, diagnosis, 
                    datetime.datetime.now().strftime("%Y-%m-%d"), 
                    final_chronic, birth_date
                ))
                conn.commit()
                logger.info("Результаты успешно сохранены в БД")
            except sqlite3.Error as e:
                logger.error(f"Ошибка сохранения в БД: {str(e)}")

        await message.answer(response, reply_markup=get_health_keyboard())
        logger.info("Анализ успешно завершен")

    except Exception as e:
        logger.error(f"Критическая ошибка в процессе анализа: {str(e)}", exc_info=True)
        await message.answer(
            "⚠️ Произошла непредвиденная ошибка.\n"
            "Попробуйте описать симптомы более подробно или повторите позже."
        )
    finally:
        await state.finish()
        logger.info("Состояние FSM завершено")
        
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

async def generate_report(records, user_id: int, start_date: str, end_date: str):
    """Генерация отчетов с улучшенной визуализацией"""
    try:
        # 1. Подготовка данных
        df = pd.DataFrame(records, columns=[
            "Состояние здоровья", "Погодные условия", "Дата", 
            "Диагноз", "Хронические заболевания", "Возраст", "Рекомендованный врач"
        ])
        
        # Анализ симптомов
        analyzer = ImprovedSymptomAnalyzer()
        all_symptoms = []
        symptom_details = []
        
        for _, row in df.iterrows():
            symptoms = analyzer.analyze_symptoms(row['Состояние здоровья'])
            for symptom in symptoms:
                all_symptoms.append(symptom)
                symptom_details.append({
                    'Дата': row['Дата'],
                    'Симптом': symptom,
                    'Категория': next((cat for cat in SYMPTOMS.keys() if symptom in SYMPTOMS[cat]), 'Другое')
                })
        
        # Создаем директорию для отчетов
        os.makedirs("reports", exist_ok=True)
        base_filename = f"reports/health_report_{user_id}_{start_date}_to_{end_date}"
        
        # 2. Текстовый отчет с топом симптомов
        txt_filename = f"{base_filename}.txt"
        with open(txt_filename, "w", encoding="utf-8") as f:
            # Статистика по симптомам
            if all_symptoms:
                symptom_counts = pd.Series(all_symptoms).value_counts()
                f.write("📊 Топ 5 самых частых симптомов:\n")
                for symptom, count in symptom_counts.head(5).items():
                    f.write(f"- {symptom}: {count} раз(а)\n")
                f.write("\n")
            
            # Оригинальные записи
            for record in records:
                f.write(
                    f"{record[2]}: Состояние: {record[0]}\n"
                    f"Погода: {record[1]}\n"
                    f"Диагноз: {record[3]}\n"
                    f"Врач: {record[6]}\n\n"
                )
        
        # 3. Визуализации (только если есть симптомы)
        image_files = []
        if symptom_details:
            symptom_df = pd.DataFrame(symptom_details)
            
            # A. График симптомов по дням
            plt.figure(figsize=(12, 6))
            
            # Группируем по дате и симптому
            daily_symptoms = symptom_df.groupby(['Дата', 'Симптом']).size().unstack().fillna(0)
            
            # Сортируем по частоте симптомов
            top_symptoms = symptom_df['Симптом'].value_counts().index[:5]  # Топ 5 симптомов
            daily_symptoms = daily_symptoms[top_symptoms]
            
            # Построение графика
            daily_symptoms.plot(kind='bar', stacked=True, ax=plt.gca(), width=0.8)
            plt.title(f'Динамика симптомов по дням\n{start_date} - {end_date}')
            plt.ylabel('Количество упоминаний')
            plt.xlabel('Дата')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend(title='Симптомы', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            buf1 = BytesIO()
            plt.savefig(buf1, format='png', dpi=120, bbox_inches='tight')
            buf1.seek(0)
            plt.close()
            image_files.append(('symptoms_by_day.png', buf1))
            
            # B. Улучшенная тепловая карта
            plt.figure(figsize=(10, 6))
            
            # Группируем по категориям и датам
            heatmap_data = symptom_df.groupby(['Дата', 'Категория']).size().unstack().fillna(0)
            
            # Нормализуем для лучшей визуализации
            heatmap_data = heatmap_data.div(heatmap_data.sum(axis=1), axis=0)
            
            sns.heatmap(
                heatmap_data.T,
                cmap='YlOrRd',
                annot=True, fmt='.1%',
                linewidths=.5,
                cbar_kws={'label': 'Доля симптомов'}
            )
            plt.title('Распределение симптомов по категориям и дням')
            plt.xlabel('Дата')
            plt.ylabel('Категория симптомов')
            plt.tight_layout()
            
            buf2 = BytesIO()
            plt.savefig(buf2, format='png', dpi=120)
            buf2.seek(0)
            plt.close()
            image_files.append(('symptoms_heatmap.png', buf2))
        
        # 4. CSV отчет (как раньше)
        csv_filename = f"{base_filename}.csv"
        df.to_csv(csv_filename, index=False, encoding="utf-8")
        
        return (
            InputFile(csv_filename),
            InputFile(txt_filename),
            *[InputFile(buf, filename=name) for name, buf in image_files]
        )
        
    except Exception as e:
        logger.error(f"Ошибка генерации отчета: {e}", exc_info=True)
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