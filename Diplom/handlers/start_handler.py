from aiogram import types, Dispatcher
from keyboards.keyboards import get_start_keyboard

from handlers.medicine_reminder import MedicineReminder,initiate_medicine_reminder,set_medicine_name,set_medicine_notes,set_duration,set_days_of_week,set_doses_per_day,set_dosage,set_times,view_reminders,confirm_delete_reminder,delete_reminder,cancel_delete_reminder,back_to_start,reminders_menu,back_to_reminders
from handlers.stock import Stock,initiate_add_stock,set_stock_name,set_tablet_count,set_pack_count,view_stock,edit_stock,edit_stock_name,set_edit_stock_name,edit_tablet_count,set_edit_tablet_count,edit_pack_count,set_edit_pack_count,delete_stock, back_to_pills,pills_menu
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
    dp.register_callback_query_handler(back_to_start, text="back_to_main", state="*")
    dp.register_callback_query_handler(reminders_menu, text="menu_reminders", state="*")
    dp.register_callback_query_handler(initiate_add_stock, text="add_stock", state="*")
    dp.register_message_handler(set_stock_name, state=Stock.stock_name)
    dp.register_message_handler(set_tablet_count, state=Stock.tablet_count)
    dp.register_message_handler(set_pack_count, state=Stock.pack_count)
    dp.register_callback_query_handler(view_stock, text="view_stock", state="*")
    dp.register_callback_query_handler(edit_stock, text_contains="edit_stock_", state=Stock.view_stock)
    dp.register_callback_query_handler(edit_stock_name, text="edit_name", state=Stock.edit_stock_name)
    dp.register_message_handler(set_edit_stock_name, state=Stock.edit_stock_name)
    dp.register_callback_query_handler(edit_tablet_count, text="edit_tablet_count", state=Stock.edit_stock_name)
    dp.register_message_handler(set_edit_tablet_count, state=Stock.edit_tablet_count)
    dp.register_callback_query_handler(edit_pack_count, text="edit_pack_count", state=Stock.edit_stock_name)
    dp.register_message_handler(set_edit_pack_count, state=Stock.edit_pack_count)
    dp.register_callback_query_handler(delete_stock, text_contains="delete_stock_", state=Stock.view_stock)
    dp.register_callback_query_handler(pills_menu, text="menu_pills", state="*")
    dp.register_callback_query_handler(back_to_reminders, text="cancel_reminders", state="*")
    dp.register_callback_query_handler(back_to_pills, text="cancel_pills", state="*")
    dp.register_callback_query_handler(back_to_pills, text="back_pills", state="*")
    dp.register_callback_query_handler(back_to_reminders, text="back_reminders", state="*")