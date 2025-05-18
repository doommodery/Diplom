from aiogram import types, Dispatcher
from core.health_condition import cancel_health_entry, health_keyboard, load_report, process_days_input, process_health_analysis, process_health_entry, report_14_days, report_1_day, report_30_days, report_7_days, start_health_analysis, start_health_entry
from core.states import HealthConditionState, ReportState
from keyboards.keyboards import get_start_keyboard
from core.medicine_reminder import MedicineReminder,initiate_medicine_reminder,set_medicine_name,set_medicine_notes,set_duration,set_days_of_week,set_doses_per_day,set_dosage,set_times,view_reminders,confirm_delete_reminder,delete_reminder,cancel_delete_reminder,back_to_start,reminders_menu,back_to_reminders
from core.user_inf import (
    UserInfo,
    handle_personal_data,
    handle_birth_year,
    handle_birth_month, 
    handle_birth_day,
    handle_chronic_conditions,
    handle_city
)
import logging  # Для логирования

# Обработчик команды /start
async def start(message: types.Message):
    try:
        logging.info(f"Команда /start вызвана пользователем: {message.from_user.id}")
        await message.answer(
            "Бот запущен и готов к работе! Выберите действие:",
            reply_markup=get_start_keyboard()  # Отправляем клавиатуру
        
        )

    except Exception as e:
        logging.error(f"Ошибка при обработке команды /start: {e}")

# Регистрация обработчика
def register_handlers_start(dp: Dispatcher):
    dp.register_message_handler(start, commands=["start"])
    dp.register_callback_query_handler(initiate_medicine_reminder, text="create_reminder", state="*")
    
    dp.register_callback_query_handler(back_to_start, text="back_to_main", state="*")
    dp.register_callback_query_handler(reminders_menu, text="menu_reminders", state="*")
    dp.register_callback_query_handler(back_to_reminders, text="back_reminders", state="*")
    #user_inf
    dp.register_callback_query_handler(handle_personal_data, text="add_user_inf")
    dp.register_message_handler(handle_city, state=UserInfo.waiting_for_city)
    dp.register_callback_query_handler(handle_birth_year, state=UserInfo.waiting_for_year)
    dp.register_callback_query_handler(handle_birth_month, state=UserInfo.waiting_for_month)
    dp.register_callback_query_handler(handle_birth_day, state=UserInfo.waiting_for_day)
    dp.register_message_handler(handle_chronic_conditions, state=UserInfo.waiting_for_conditions)
    #reminders
    dp.register_callback_query_handler(initiate_medicine_reminder, text="create_reminder", state="*")
    dp.register_message_handler(set_medicine_name, state=MedicineReminder.name)
    dp.register_message_handler(set_medicine_notes, state=MedicineReminder.notes)
    dp.register_callback_query_handler(set_duration, state=MedicineReminder.duration)
    dp.register_callback_query_handler(set_days_of_week, state=MedicineReminder.days_of_week)
    dp.register_message_handler(set_doses_per_day, state=MedicineReminder.doses_per_day)
    dp.register_callback_query_handler(set_dosage, state=MedicineReminder.dosage)
    dp.register_callback_query_handler(set_times, state=MedicineReminder.times)
    dp.register_callback_query_handler(view_reminders, text="view_reminders", state="*")
    dp.register_callback_query_handler(confirm_delete_reminder, text_contains="confirm_delete_", state=MedicineReminder.view_reminders)
    dp.register_callback_query_handler(delete_reminder, text_contains="delete_", state=MedicineReminder.confirm_delete)
    dp.register_callback_query_handler(cancel_delete_reminder, text="cancel_delete", state=MedicineReminder.confirm_delete)
    #health_condition
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