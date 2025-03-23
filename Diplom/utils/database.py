import sqlite3
import logging
import asyncio
import pytz
from datetime import datetime
from config import DB_PATH,SERVER_TIMEZONE

# Инициализация логирования
logging.basicConfig(level=logging.INFO)

# Инициализация базы данных
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        # Создание таблицы для напоминаний о приеме таблеток
        cursor.execute('''CREATE TABLE IF NOT EXISTS medicine_reminders (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER,
                            name TEXT,
                            notes TEXT,
                            duration TEXT,
                            days TEXT,
                            dose_count INTEGER,
                            times TEXT,
                            dosage INTEGER,
                            start_date TEXT,
                            end_date TEXT
                        )''')

        # Создание таблицы для учета запаса таблеток
        cursor.execute('''CREATE TABLE IF NOT EXISTS medicine_stock (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER,
                            name TEXT,
                            tablet_count INTEGER,
                            pack_count INTEGER
                        )''')

        # Создание таблицы для личных данных пользователя
        cursor.execute('''CREATE TABLE IF NOT EXISTS user_info (
                            user_id INTEGER PRIMARY KEY,
                            role TEXT DEFAULT 'user',
                            timezone TEXT,
                            health_info TEXT
                        )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS user_condition_analysis (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER NOT NULL,
                            user_condition TEXT NOT NULL,
                            weather_condition TEXT,
                            analysis_results TEXT DEFAULT '-',
                            date_an TEXT NOT NULL
                        )''')

        conn.commit()

# Блокировка для синхронизации доступа к базе данных
db_lock = asyncio.Lock()

# Функция для сохранения напоминания о приеме таблеток
async def save_medicine_reminder(data, user_id):
    # Получение данных о напоминании
    name = data.get('name', '')
    notes = data.get('notes', '')
    duration = data.get('duration', '')
    days = ','.join(data.get('days_of_week', []))  # Преобразуем список дней в строку
    dose_count = data.get('doses_per_day', 0)
    times = data.get('times', [])  # Получаем список времен
    dosage = data.get('dosage', 1)  # Получаем значение дозировки, по умолчанию 1
    start_date = data.get('start_date', '')
    end_date = data.get('end_date', '')

    # Проверка данных
    if not name or not duration or not days or not times or not start_date or not end_date:
        logging.error("Недостаточно данных для сохранения напоминания.")
        return

    async with db_lock:
        # Соединение с базой данных и запись данных
        try:
            with sqlite3.connect(DB_PATH) as connection:
                cursor = connection.cursor()

                # Получаем временную зону пользователя
                cursor.execute("SELECT timezone FROM user_info WHERE user_id = ?", (user_id,))
                timezone_result = cursor.fetchone()
                if timezone_result:
                    user_timezone = pytz.timezone(timezone_result[0])
                else:
                    user_timezone = pytz.utc  # По умолчанию используем UTC, если временная зона не указана

                # Преобразуем время приема в UTC, а затем в часовой пояс сервера
                server_times = []
                for time_str in times:
                    # Создаем объект datetime с учетом временной зоны пользователя
                    local_time = datetime.strptime(time_str, '%H:%M').time()
                    local_datetime = datetime.combine(datetime.now().date(), local_time)
                    local_datetime = user_timezone.localize(local_datetime) if local_datetime.tzinfo is None else local_datetime
                    # Преобразуем в UTC
                    utc_datetime = local_datetime.astimezone(pytz.utc)
                    # Преобразуем в часовой пояс сервера
                    server_datetime = utc_datetime.astimezone(SERVER_TIMEZONE)
                    server_times.append(server_datetime.strftime('%H:%M'))

                # Преобразуем список времен в строку
                times_str = ','.join(server_times)

                cursor.execute("""
                    INSERT INTO medicine_reminders (user_id, name, notes, duration, days, dose_count, times, dosage, start_date, end_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (user_id, name, notes, duration, days, dose_count, times_str, dosage, start_date, end_date))
                connection.commit()
            logging.info(f"Напоминание для пользователя {user_id} успешно сохранено.")
        except sqlite3.Error as e:
            logging.error(f"Ошибка при сохранении напоминания: {e}")

# Функция для получения активных напоминаний
def get_active_reminders(user_id=None):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute("""
                    SELECT user_id, name, notes, duration, days, dose_count, times, dosage, start_date, end_date 
                    FROM medicine_reminders 
                    WHERE user_id = ?
                """, (user_id,))
            else:
                cursor.execute("""
                    SELECT user_id, name, notes, duration, days, dose_count, times, dosage, start_date, end_date 
                    FROM medicine_reminders
                """)
            reminders = cursor.fetchall()

        # Логирование данных для проверки
        logging.info(f"Полученные напоминания: {reminders}")

        # Преобразование данных в словарь для удобства
        active_reminders = []
        for reminder in reminders:
            active_reminders.append({
                'user_id': reminder[0],
                'name': reminder[1],
                'notes': reminder[2],
                'duration': reminder[3],
                'days': reminder[4],
                'dose_count': reminder[5],
                'times': reminder[6],
                'dosage': reminder[7],
                'start_date': reminder[8],
                'end_date': reminder[9]
            })

        return active_reminders
    except sqlite3.Error as e:
        logging.error(f"Ошибка при получении активных напоминаний: {e}")
        return []

# Функция для удаления напоминания
def delete_reminder_from_db(user_id, name, notes, duration, days, dose_count, times, dosage):
    try:
        # Убедимся, что days и times передаются в правильном формате
        if isinstance(days, list):
            days = ",".join(days)  # Преобразуем список в строку без пробелов
        if isinstance(times, list):
            times = ",".join(times)  # Преобразуем список в строку без пробелов

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM medicine_reminders
                WHERE user_id = ? AND name = ? AND notes = ? AND duration = ? AND days = ? AND dose_count = ? AND times = ? AND dosage = ?
            """, (user_id, name, notes, duration, days, dose_count, times, dosage))
            conn.commit()
        logging.info(f"Напоминание для пользователя {user_id} успешно удалено.")
    except sqlite3.Error as e:
        logging.error(f"Ошибка при удалении напоминания: {e}")

# Функция для добавления запаса таблеток в базу данных
async def add_medicine_stock(user_id, name, tablet_count, pack_count):
    async with db_lock:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO medicine_stock (user_id, name, tablet_count, pack_count)
                    VALUES (?, ?, ?, ?)
                """, (user_id, name, tablet_count, pack_count))
                conn.commit()
            logging.info(f"Запас для {name} успешно добавлен для пользователя {user_id}.")
        except sqlite3.Error as e:
            logging.error(f"Ошибка при добавлении запаса: {e}")

# Функция для получения запасов таблеток
async def get_medicine_stock(user_id):
    async with db_lock:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, name, tablet_count, pack_count FROM medicine_stock WHERE user_id = ?", (user_id,))
                stock = cursor.fetchall()

            # Логирование данных для проверки
            logging.info(f"Полученные запасы для пользователя {user_id}: {stock}")

            # Преобразование данных в словарь для удобства
            medicine_stock = []
            for item in stock:
                medicine_stock.append({
                    'id': item[0],
                    'name': item[1],
                    'tablet_count': item[2],
                    'pack_count': item[3]
                })

            return medicine_stock
        except sqlite3.Error as e:
            logging.error(f"Ошибка при получении запасов: {e}")
            return []

# Функция для обновления запаса таблеток
async def update_medicine_stock(stock_id, name, tablet_count, pack_count):
    async with db_lock:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE medicine_stock
                    SET name = ?, tablet_count = ?, pack_count = ?
                    WHERE id = ?
                """, (name, tablet_count, pack_count, stock_id))
                conn.commit()
            logging.info(f"Запас с ID {stock_id} успешно обновлен.")
        except sqlite3.Error as e:
            logging.error(f"Ошибка при обновлении запаса: {e}")

# Функция для удаления запаса таблеток
async def delete_medicine_stock(stock_id):
    async with db_lock:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM medicine_stock WHERE id = ?", (stock_id,))
                conn.commit()
            logging.info(f"Запас с ID {stock_id} успешно удален.")
        except sqlite3.Error as e:
            logging.error(f"Ошибка при удалении запаса: {e}")

async def save_user_info(user_id: int, timezone: str, health_info: str):
    async with db_lock:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO user_info (user_id, timezone, health_info)
                    VALUES (?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        timezone = excluded.timezone,
                        health_info = excluded.health_info
                """, (user_id, timezone, health_info))
                conn.commit()
            logging.info(f"Личные данные пользователя {user_id} сохранены.")
        except sqlite3.Error as e:
            logging.error(f"Ошибка при сохранении личных данных: {e}")
        
async def delete_old_user_info(user_id: int):
    async with db_lock:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                # Удаляем все записи, кроме последней
                cursor.execute("""
                    DELETE FROM user_info
                    WHERE user_id = ? AND rowid NOT IN (
                        SELECT rowid FROM user_info
                        WHERE user_id = ?
                        ORDER BY rowid DESC
                        LIMIT 1
                    )
                """, (user_id, user_id))
                conn.commit()
            logging.info(f"Старые записи пользователя {user_id} удалены.")
        except sqlite3.Error as e:
            logging.error(f"Ошибка при удалении старых записей: {e}")