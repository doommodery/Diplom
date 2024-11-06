import sqlite3
import logging
import asyncio

# Инициализация логирования
logging.basicConfig(level=logging.INFO)

# Инициализация базы данных
def init_db():
    with sqlite3.connect("data/medicine_reminders.db") as conn:
        cursor = conn.cursor()

        # Создание таблицы для напоминаний о приеме таблеток (с добавлением дозировки)
        cursor.execute('''CREATE TABLE IF NOT EXISTS medicine_reminders (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER,
                            name TEXT,
                            notes TEXT,
                            duration TEXT,
                            days TEXT,
                            dose_count INTEGER,
                            times TEXT,
                            dosage INTEGER  -- Новое поле для дозировки
                        )''')

        # Создание новой таблицы для учета запаса таблеток
        cursor.execute('''CREATE TABLE IF NOT EXISTS medicine_stock (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER,
                            name TEXT,
                            tablet_count INTEGER,
                            pack_count INTEGER
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
    times = ','.join(data.get('times', []))  # Преобразуем список времен в строку
    dosage = data.get('dosage', 1)  # Получаем значение дозировки, по умолчанию 1

    # Проверка данных
    if not name or not duration or not days or not times:
        logging.error("Недостаточно данных для сохранения напоминания.")
        return

    async with db_lock:
        # Соединение с базой данных и запись данных
        try:
            with sqlite3.connect("data/medicine_reminders.db") as connection:
                cursor = connection.cursor()
                cursor.execute("""
                    INSERT INTO medicine_reminders (user_id, name, notes, duration, days, dose_count, times, dosage)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (user_id, name, notes, duration, days, dose_count, times, dosage))
                connection.commit()
            logging.info(f"Напоминание для пользователя {user_id} успешно сохранено.")
        except sqlite3.Error as e:
            logging.error(f"Ошибка при сохранении напоминания: {e}")

# Функция для добавления запаса таблеток в базу данных
async def add_medicine_stock(user_id, name, tablet_count, pack_count):
    async with db_lock:
        try:
            with sqlite3.connect("data/medicine_reminders.db") as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO medicine_stock (user_id, name, tablet_count, pack_count)
                    VALUES (?, ?, ?, ?)
                """, (user_id, name, tablet_count, pack_count))
                conn.commit()
            logging.info(f"Запас для {name} успешно добавлен для пользователя {user_id}.")
        except sqlite3.Error as e:
            logging.error(f"Ошибка при добавлении запаса: {e}")

# Функция для получения активных напоминаний
def get_active_reminders(user_id):
    try:
        with sqlite3.connect("data/medicine_reminders.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id, name, notes, duration, days, dose_count, times, dosage FROM medicine_reminders WHERE user_id = ?", (user_id,))
            reminders = cursor.fetchall()

        # Логирование данных для проверки
        logging.info(f"Полученные напоминания для пользователя {user_id}: {reminders}")

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
                'dosage': reminder[7]  # Новое поле дозировки
            })

        return active_reminders
    except sqlite3.Error as e:
        logging.error(f"Ошибка при получении активных напоминаний: {e}")
        return []

# Функция для удаления напоминания
def delete_reminder_from_db(reminder_id, name, notes, duration, days, dose_count, times, dosage):
    try:
        with sqlite3.connect("data/medicine_reminders.db") as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM medicine_reminders
                WHERE user_id = ? AND name = ? AND notes = ? AND duration = ? AND days = ? AND dose_count = ? AND times = ? AND dosage = ?
            """, (reminder_id, name, notes, duration, days, dose_count, times, dosage))
            conn.commit()
        logging.info(f"Напоминание с ID {reminder_id} успешно удалено.")
    except sqlite3.Error as e:
        logging.error(f"Ошибка при удалении напоминания: {e}")

# Функция для получения запасов таблеток
async def get_medicine_stock(user_id):
    async with db_lock:
        try:
            with sqlite3.connect("data/medicine_reminders.db") as conn:
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
            with sqlite3.connect("data/medicine_reminders.db") as conn:
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
            with sqlite3.connect("data/medicine_reminders.db") as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM medicine_stock WHERE id = ?", (stock_id,))
                conn.commit()
            logging.info(f"Запас с ID {stock_id} успешно удален.")
        except sqlite3.Error as e:
            logging.error(f"Ошибка при удалении запаса: {e}")
