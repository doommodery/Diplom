import sqlite3

def update_db():
    with sqlite3.connect("data/medicine_reminders.db") as conn:
        cursor = conn.cursor()
        
        # Добавление нового поля "dosage" в таблицу medicine_reminders, если оно не существует
        try:
            cursor.execute("ALTER TABLE medicine_reminders ADD COLUMN dosage INTEGER DEFAULT 1")
            print("Поле 'dosage' добавлено в таблицу 'medicine_reminders'.")
        except sqlite3.OperationalError:
            print("Поле 'dosage' уже существует в таблице 'medicine_reminders'.")

        # Создание таблицы medicine_stock, если она не существует
        cursor.execute('''CREATE TABLE IF NOT EXISTS medicine_stock (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER,
                            name TEXT,
                            tablet_count INTEGER,
                            pack_count INTEGER
                        )''')
        print("Таблица 'medicine_stock' создана или уже существует.")

        conn.commit()

# Выполнение функции для обновления базы данных
update_db()
