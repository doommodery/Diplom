from aiogram import types
from aiogram.dispatcher import FSMContext
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from core.states import MedicineReminder
from keyboards.keyboards import (
    get_start_keyboard,
    get_duration_keyboard,
    get_days_keyboard,
    get_time_selection_keyboard,
    get_reminders_keyboard,
    get_dosage_keyboard,
    get_reminders_menu_keyboard
)
from utils.database import save_medicine_reminder, get_active_reminders, delete_reminder_from_db
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from config import DB_PATH
import sqlite3, pytz

# Обработчик для кнопки "Напоминания"
async def reminders_menu(callback_query: types.CallbackQuery):
    await callback_query.message.answer("Меню напоминаний:", reply_markup=get_reminders_menu_keyboard())

# Обработчик для создания напоминания
async def initiate_medicine_reminder(callback_query: types.CallbackQuery):
    await callback_query.message.answer("Введите название лекарства:")
    await MedicineReminder.name.set()

# Установка названия лекарства
async def set_medicine_name(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['name'] = message.text
    await message.answer("Добавьте примечание к приему лекарства:")
    await MedicineReminder.notes.set()

# Установка примечания
async def set_medicine_notes(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['notes'] = message.text
    await message.answer("Выберите срок приема лекарства:", reply_markup=get_duration_keyboard())
    await MedicineReminder.duration.set()

# Установка срока приема
async def set_duration(callback_query: types.CallbackQuery, state: FSMContext):
    if callback_query.data == "back":
        await callback_query.message.answer("Что вы хотите сделать?", reply_markup=get_reminders_menu_keyboard())
        await state.finish()
        return
    elif callback_query.data == "cancel":
        await callback_query.message.answer("Создание напоминания отменено.")
        await callback_query.message.answer("Что вы хотите сделать?", reply_markup=get_start_keyboard())
        await state.finish()
        return

    async with state.proxy() as data:
        data['duration'] = callback_query.data
    await callback_query.message.answer(
        "Выберите дни недели для приема (нажмите «Подтвердить», когда выбраны все дни):",
        reply_markup=get_days_keyboard()
    )
    await MedicineReminder.days_of_week.set()

# Установка дней недели
async def set_days_of_week(callback_query: types.CallbackQuery, state: FSMContext):
    if callback_query.data == "back":
        await callback_query.message.answer("Выберите срок приема лекарства:", reply_markup=get_duration_keyboard())
        await MedicineReminder.duration.set()
        return
    elif callback_query.data == "cancel":
        await callback_query.message.answer("Создание напоминания отменено.")
        await callback_query.message.answer("Что вы хотите сделать?", reply_markup=get_reminders_menu_keyboard())
        await state.finish()
        return

    async with state.proxy() as data:
        if 'days_of_week' not in data:
            data['days_of_week'] = set()

        if callback_query.data == "confirm":
            if data['days_of_week']:
                await callback_query.message.answer("Сколько раз в день принимать лекарство?")
                await MedicineReminder.doses_per_day.set()
            else:
                await callback_query.answer("Пожалуйста, выберите хотя бы один день недели.")
                return
        elif callback_query.data == "all_days":
            data['days_of_week'] = {"Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"}
        else:
            if callback_query.data in data['days_of_week']:
                data['days_of_week'].remove(callback_query.data)
            else:
                data['days_of_week'].add(callback_query.data)

        await callback_query.message.edit_reply_markup(
            reply_markup=get_days_keyboard(data['days_of_week'])
        )
        await callback_query.answer(f"День {callback_query.data} {'выбран' if callback_query.data in data['days_of_week'] else 'убран'}.")

# Установка количества приемов
async def set_doses_per_day(message: types.Message, state: FSMContext):
    try:
        doses_per_day = int(message.text)
        if doses_per_day <= 0:
            raise ValueError("Количество приемов должно быть положительным числом.")
    except ValueError as e:
        await message.answer(f"Ошибка: {e}. Пожалуйста, введите корректное количество приемов.")
        return

    async with state.proxy() as data:
        data['doses_per_day'] = doses_per_day
        data['times'] = []
    await message.answer("Выберите дозировку:", reply_markup=get_dosage_keyboard())
    await MedicineReminder.dosage.set()

# Установка дозировки
async def set_dosage(callback_query: types.CallbackQuery, state: FSMContext):
    if callback_query.data == "back":
        await callback_query.message.answer("Сколько раз в день принимать лекарство?")
        await MedicineReminder.doses_per_day.set()
        return
    elif callback_query.data == "cancel":
        await callback_query.message.answer("Создание напоминания отменено.")
        await callback_query.message.answer("Что вы хотите сделать?", reply_markup=get_reminders_menu_keyboard())
        await state.finish()
        return

    async with state.proxy() as data:
        data['dosage'] = int(callback_query.data.split('_')[1])
    await callback_query.message.answer("Выберите время для 1 приема:", reply_markup=get_time_selection_keyboard(data['times']))
    await MedicineReminder.times.set()

# Установка времени приема
async def set_times(callback_query: types.CallbackQuery, state: FSMContext):
    if callback_query.data == "back":
        await callback_query.message.answer("Выберите дозировку:", reply_markup=get_dosage_keyboard())
        await MedicineReminder.dosage.set()
        return
    elif callback_query.data == "cancel":
        await callback_query.message.answer("Создание напоминания отменено.")
        await callback_query.message.answer("Что вы хотите сделать?", reply_markup=get_reminders_menu_keyboard())
        await state.finish()
        return

    async with state.proxy() as data:
        time = callback_query.data.split('_')[1]
        data['times'].append(time)

        if len(data['times']) < data['doses_per_day']:
            await callback_query.message.answer(
                f"Выберите время для {len(data['times']) + 1} приема:",
                reply_markup=get_time_selection_keyboard(data['times'])
            )
        else:
            # Установка start_date и end_date
            start_date = datetime.now().strftime('%Y-%m-%d')
            duration = data['duration']

            # Определяем end_date на основе duration
            if duration == "1 неделя":
                end_date = (datetime.now() + timedelta(weeks=1)).strftime('%Y-%m-%d')
            elif duration == "2 недели":
                end_date = (datetime.now() + timedelta(weeks=2)).strftime('%Y-%m-%d')
            elif duration == "1 месяц":
                end_date = (datetime.now() + relativedelta(months=1)).strftime('%Y-%m-%d')
            elif duration == "3 месяца":
                end_date = (datetime.now() + relativedelta(months=3)).strftime('%Y-%m-%d')
            else:
                end_date = start_date  # На случай, если duration не распознан

            data['start_date'] = start_date
            data['end_date'] = end_date

            await save_medicine_reminder(data, callback_query.from_user.id)
            await callback_query.message.answer("Напоминание успешно сохранено!")
            await state.finish()
            await callback_query.message.answer("Что вы хотите сделать?", reply_markup=get_reminders_menu_keyboard())

# Обработчик для просмотра напоминаний
async def view_reminders(callback_query: types.CallbackQuery):
    user_id = callback_query.from_user.id  # Получаем user_id из callback_query
    reminders = get_active_reminders(user_id)
    
    if not reminders:
        await callback_query.message.answer("У вас нет активных напоминаний.")
        await callback_query.message.answer("Что вы хотите сделать?", reply_markup=get_reminders_menu_keyboard())
        return

    # Получаем временную зону пользователя
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT timezone FROM user_info WHERE user_id = ?", (user_id,))
        timezone_result = cursor.fetchone()
        if timezone_result:
            user_timezone_str = timezone_result[0]  # Получаем строку временной зоны
            user_timezone = pytz.timezone(user_timezone_str)  # Преобразуем строку в объект временной зоны
        else:
            user_timezone = pytz.utc  # По умолчанию используем UTC, если временная зона не указана

    # Создаем клавиатуру с учетом временной зоны пользователя
    keyboard = get_reminders_keyboard(reminders, user_timezone)
    
    await callback_query.message.answer("Ваши активные напоминания:", reply_markup=keyboard)
    await MedicineReminder.view_reminders.set()

# Обработчик для подтверждения удаления
async def confirm_delete_reminder(callback_query: types.CallbackQuery, state: FSMContext):
    reminder_id = int(callback_query.data.split('_')[2])
    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton("Да", callback_data=f"delete_{reminder_id}"))
    keyboard.add(InlineKeyboardButton("Нет", callback_data="cancel_delete"))
    await callback_query.message.answer(f"Вы уверены, что хотите удалить напоминание?", reply_markup=keyboard)
    await MedicineReminder.confirm_delete.set()

# Обработчик для удаления напоминания
async def delete_reminder(callback_query: types.CallbackQuery, state: FSMContext):
    reminder_id = int(callback_query.data.split('_')[1])
    reminders = get_active_reminders()
    reminder = next((r for r in reminders if r['user_id'] == reminder_id), None)

    if reminder:
        delete_reminder_from_db(
            reminder['user_id'],
            reminder['name'],
            reminder['notes'],
            reminder['duration'],
            reminder['days'],
            reminder['dose_count'],
            reminder['times'],
            reminder['dosage']
        )
        await callback_query.message.answer("Напоминание успешно удалено!")
    else:
        await callback_query.message.answer("Напоминание не найдено.")

    await view_reminders(callback_query)
    await state.finish()

# Обработчик для отмены удаления
async def cancel_delete_reminder(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.message.answer("Удаление отменено.")
    await view_reminders(callback_query)
    await state.finish()

# Обработчик для кнопки "Назад" в просмотре напоминаний
async def back_to_start(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.message.answer("Выберите действие:", reply_markup=get_start_keyboard())
    await state.finish()

# Обработчик для кнопки "Назад" в просмотре таблеток
async def back_to_reminders(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.message.answer("Меню напоминаний:", reply_markup=get_reminders_menu_keyboard())
    await state.finish()
