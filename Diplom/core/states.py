from aiogram.dispatcher.filters.state import State, StatesGroup

class HealthConditionState(StatesGroup):
    waiting_for_condition = State()
    waiting_for_analysis = State()

class ReportState(StatesGroup):
    waiting_for_days = State()

class UserInfo(StatesGroup):
    waiting_for_city = State()
    waiting_for_year = State()
    waiting_for_month = State()
    waiting_for_day = State()
    waiting_for_conditions = State()

class MedicineReminder(StatesGroup):
    name = State()
    notes = State()
    duration = State()
    days_of_week = State()
    doses_per_day = State()
    times = State()
    dosage = State()
    view_reminders = State()
    confirm_delete = State()