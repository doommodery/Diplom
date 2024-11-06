import logging
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

# Инициализация логирования
logging.basicConfig(level=logging.INFO)

def create_inline_keyboard(buttons, row_width=2):
    keyboard = InlineKeyboardMarkup(row_width=row_width)
    keyboard.add(*buttons)
    return keyboard

def get_start_keyboard():
    buttons = [
        InlineKeyboardButton("Сделать напоминание по таблеткам", callback_data="create_reminder"),
        InlineKeyboardButton("Просмотреть напоминания", callback_data="view_reminders"),
        InlineKeyboardButton("Добавить запас таблеток", callback_data="add_stock"),
        InlineKeyboardButton("Просмотреть запас таблеток", callback_data="view_stock")
    ]
    return create_inline_keyboard(buttons)

def get_duration_keyboard():
    options = ["1 неделя", "2 недели", "1 месяц", "3 месяца"]
    buttons = [InlineKeyboardButton(text=option, callback_data=option) for option in options]
    buttons.append(InlineKeyboardButton("Отменить создание", callback_data="cancel"))
    return create_inline_keyboard(buttons)

def get_dosage_keyboard():
    doses = ["1 таблетка", "2 таблетки", "3 таблетки", "4 таблетки"]
    buttons = [InlineKeyboardButton(dose, callback_data=f"dosage_{dose.split()[0]}") for dose in doses]
    buttons.append(InlineKeyboardButton("Отменить создание", callback_data="cancel"))
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
            InlineKeyboardButton("Отменить создание", callback_data="cancel")
        ])

    return create_inline_keyboard(buttons)

def get_time_selection_keyboard(selected_times):
    available_times = ["01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", "08:00", "09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"]
    buttons = [InlineKeyboardButton(time, callback_data=f"time_{time}") for time in available_times if time not in selected_times]
    buttons.append(InlineKeyboardButton("Отменить создание", callback_data="cancel"))
    return create_inline_keyboard(buttons, row_width=4)

def get_reminders_keyboard(reminders):
    buttons = [InlineKeyboardButton(f"{reminder['name']} - {reminder['times']} - {reminder['days']}", callback_data=f"confirm_delete_{reminder['user_id']}") for reminder in reminders]
    buttons.append(InlineKeyboardButton("Назад", callback_data="back"))
    return create_inline_keyboard(buttons)

def get_stock_keyboard():
    buttons = [
        InlineKeyboardButton("Добавить название", callback_data="add_name"),
        InlineKeyboardButton("Добавить количество таблеток", callback_data="add_tablet_count"),
        InlineKeyboardButton("Добавить количество упаковок", callback_data="add_pack_count"),
        InlineKeyboardButton("Подтвердить", callback_data="confirm_stock"),
        InlineKeyboardButton("Отменить создание", callback_data="cancel")
    ]
    return create_inline_keyboard(buttons)

def get_view_stock_keyboard(stock):
    buttons = [InlineKeyboardButton(f"{item['name']} - {item['tablet_count']} таблеток - {item['pack_count']} упаковок", callback_data=f"edit_stock_{item['id']}") for item in stock]
    buttons.append(InlineKeyboardButton("Назад", callback_data="back"))
    return create_inline_keyboard(buttons)

def get_edit_stock_keyboard():
    buttons = [
        InlineKeyboardButton("Изменить название", callback_data="edit_name"),
        InlineKeyboardButton("Изменить количество таблеток", callback_data="edit_tablet_count"),
        InlineKeyboardButton("Изменить количество упаковок", callback_data="edit_pack_count"),
        InlineKeyboardButton("Назад", callback_data="back")
    ]
    return create_inline_keyboard(buttons)
