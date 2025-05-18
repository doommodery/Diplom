import logging
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup, KeyboardButton
from datetime import datetime, date
import pytz
from config import SERVER_TIMEZONE

# Инициализация логирования
logging.basicConfig(level=logging.INFO)

def create_reply_keyboard(buttons, row_width=2, resize=True):
    return ReplyKeyboardMarkup(resize_keyboard=resize, row_width=row_width).add(*buttons)

def create_inline_keyboard(buttons, row_width=2):
    keyboard = InlineKeyboardMarkup(row_width=row_width)
    keyboard.add(*buttons)
    return keyboard

# Главное меню
def get_start_keyboard():
    buttons = [
        InlineKeyboardButton("Напоминания", callback_data="menu_reminders"),
        InlineKeyboardButton("Здоровье", callback_data="menu_health"),
        InlineKeyboardButton("Личные данные", callback_data="add_user_inf"),
    ]
    return create_inline_keyboard(buttons)

# Кнопки для раздела "Здоровье"
def get_health_keyboard():
    buttons = [
        InlineKeyboardButton("Сделать запись о состоянии здоровья", callback_data="record_health"),
        InlineKeyboardButton("Проанализировать состояние здоровья", callback_data="analyze_health"),
        InlineKeyboardButton("Подготовить отчёт", callback_data="report_health"),
        InlineKeyboardButton("Назад", callback_data="back_to_main")
    ]
    return create_inline_keyboard(buttons)

# Клавиатура для выбора временного промежутка
def get_report_days_keyboard():
    buttons = [
        InlineKeyboardButton("1 день", callback_data="report_1_day"),
        InlineKeyboardButton("7 дней", callback_data="report_7_days"),
        InlineKeyboardButton("14 дней", callback_data="report_14_days"),
        InlineKeyboardButton("30 дней", callback_data="report_30_days"),
        InlineKeyboardButton("Назад", callback_data="back_to_health")
    ]
    return create_inline_keyboard(buttons)

# Кнопки для раздела отчётов
def get_report_keyboard():
    buttons = [
        InlineKeyboardButton("Отправить отчёт", callback_data="record_health"),
        InlineKeyboardButton("Загрузить отчёт", callback_data="analyze_health"),
        InlineKeyboardButton("Назад", callback_data="back_to_health") 
        ]
    return create_inline_keyboard(buttons)
    
# Подменю для работы с напоминаниями
def get_reminders_menu_keyboard():
    buttons = [
        InlineKeyboardButton("Сделать напоминание", callback_data="create_reminder"),
        InlineKeyboardButton("Просмотреть напоминания", callback_data="view_reminders"),
        InlineKeyboardButton("Назад", callback_data="back_to_main")
    ]
    return create_inline_keyboard(buttons)

def get_reminders_keyboard(reminders, user_timezone):
    buttons = []
    for reminder in reminders:
        # Получаем строку с временами (например, "12:00,17:00")
        times = reminder['times']
        
        # Разделяем строку на отдельные времена
        time_list = times.split(",")
        
        # Преобразуем каждое время с учетом временной зоны пользователя
        formatted_times = []
        for time_str in time_list:
            # Преобразуем время в объект datetime
            time_obj = datetime.strptime(time_str.strip(), "%H:%M")
            
            # Локализуем время в серверной временной зоне
            server_time = SERVER_TIMEZONE.localize(
                datetime.now().replace(hour=time_obj.hour, minute=time_obj.minute)
            )
            
            # Преобразуем время в временную зону пользователя
            user_time = server_time.astimezone(user_timezone)
            
            # Форматируем время для вывода
            formatted_time = user_time.strftime("%H:%M")
            formatted_times.append(formatted_time)
        
        # Объединяем преобразованные времена обратно в строку
        formatted_times_str = ", ".join(formatted_times)
        
        # Создаем кнопку с преобразованным временем
        button_text = f"{reminder['name']} - {formatted_times_str} - {reminder['days']}"
        buttons.append(InlineKeyboardButton(button_text, callback_data=f"confirm_delete_{reminder['user_id']}"))
    
    # Добавляем кнопку "Назад"
    buttons.append(InlineKeyboardButton("Назад", callback_data="back_reminders"))
    
    return create_inline_keyboard(buttons)

# Клавиатуры для личных данных
def get_birth_year_keyboard():
    current_year = datetime.now().year
    years = [str(y) for y in range(current_year - 100, current_year - 10)]
    buttons = [KeyboardButton(y) for y in years[-20:]]  # Последние 20 лет
    buttons.append(KeyboardButton("Другие года..."))
    return create_reply_keyboard(buttons, row_width=5)

def get_full_years_keyboard():
    current_year = datetime.now().year
    years = [str(y) for y in range(current_year - 100, current_year - 10)]
    buttons = [KeyboardButton(y) for y in years]
    return create_reply_keyboard([buttons[i:i+10] for i in range(0, len(buttons), 10)], row_width=5)

def get_birth_month_keyboard():
    months = ["Январь", "Февраль", "Март", "Апрель", "Май", "Июнь", 
              "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"]
    return create_reply_keyboard([KeyboardButton(m) for m in months], row_width=4)

def get_birth_day_keyboard():
    return create_reply_keyboard([KeyboardButton(str(d)) for d in range(1, 32)], row_width=7)

def get_duration_keyboard():
    options = ["1 неделя", "2 недели", "1 месяц", "3 месяца"]
    buttons = [InlineKeyboardButton(text=option, callback_data=option) for option in options]
    buttons.append(InlineKeyboardButton("Отменить создание", callback_data="cancel_reminders"))
    return create_inline_keyboard(buttons)

def get_days_keyboard(selected_days=None, confirm_only=False):
    days = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"]
    if selected_days is None:
        selected_days = set()

    buttons = []
    for day in days:
        button_text = f"✅ {day}" if day in selected_days else f"❌ {day}"
        if not confirm_only:
            buttons.append(InlineKeyboardButton(button_text, callback_data=day))

    if not confirm_only:
        buttons.extend([
            InlineKeyboardButton("Подтвердить", callback_data="confirm"),
            InlineKeyboardButton("Выбрать все дни", callback_data="all_days"),
            InlineKeyboardButton("Отменить создание", callback_data="cancel_reminders")
        ])

    return create_inline_keyboard(buttons)

def get_time_selection_keyboard(selected_times):
    available_times = ["01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", "08:00", "09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"]
    buttons = [InlineKeyboardButton(time, callback_data=f"time_{time}") for time in available_times if time not in selected_times]
    buttons.append(InlineKeyboardButton("Отменить создание", callback_data="cancel_reminders"))
    return create_inline_keyboard(buttons, row_width=4)

def get_dosage_keyboard():
    doses = ["1 таблетка", "2 таблетки", "3 таблетки", "4 таблетки"]
    buttons = [InlineKeyboardButton(dose, callback_data=f"dosage_{dose.split()[0]}") for dose in doses]
    buttons.append(InlineKeyboardButton("Отменить создание", callback_data="cancel_reminders"))
    return create_inline_keyboard(buttons)