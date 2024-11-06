from aiogram import types, Dispatcher
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from keyboards.medicine_keyboards import (
    get_start_keyboard,
    get_view_stock_keyboard,
    get_edit_stock_keyboard
)
from utils.database import add_medicine_stock, get_medicine_stock, update_medicine_stock, delete_medicine_stock

# Определяем состояния
class Stock(StatesGroup):
    stock_name = State()
    tablet_count = State()
    pack_count = State()
    view_stock = State()
    edit_stock_name = State()
    edit_tablet_count = State()
    edit_pack_count = State()

# Обработчик для добавления запаса таблеток
async def initiate_add_stock(callback_query: types.CallbackQuery):
    await callback_query.message.answer("Введите название лекарства:")
    await Stock.stock_name.set()

# Установка названия запаса
async def set_stock_name(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['stock_name'] = message.text
    await message.answer("Введите количество таблеток:")
    await Stock.tablet_count.set()

# Установка количества таблеток
async def set_tablet_count(message: types.Message, state: FSMContext):
    try:
        tablet_count = int(message.text)
        if tablet_count <= 0:
            raise ValueError("Количество таблеток должно быть положительным числом.")
    except ValueError as e:
        await message.answer(f"Ошибка: {e}. Пожалуйста, введите корректное количество таблеток.")
        return

    async with state.proxy() as data:
        data['tablet_count'] = tablet_count
    await message.answer("Введите количество упаковок:")
    await Stock.pack_count.set()

# Установка количества упаковок
async def set_pack_count(message: types.Message, state: FSMContext):
    try:
        pack_count = int(message.text)
        if pack_count <= 0:
            raise ValueError("Количество упаковок должно быть положительным числом.")
    except ValueError as e:
        await message.answer(f"Ошибка: {e}. Пожалуйста, введите корректное количество упаковок.")
        return

    async with state.proxy() as data:
        data['pack_count'] = pack_count

    await add_medicine_stock(message.from_user.id, data['stock_name'], data['tablet_count'], data['pack_count'])
    await message.answer("Запас успешно добавлен!")
    await state.finish()
    await message.answer("Что вы хотите сделать?", reply_markup=get_start_keyboard())

# Обработчик для просмотра запаса таблеток
async def view_stock(callback_query: types.CallbackQuery):
    stock = await get_medicine_stock(callback_query.from_user.id)
    if not stock:
        await callback_query.message.answer("У вас нет запасов таблеток.")
        await callback_query.message.answer("Что вы хотите сделать?", reply_markup=get_start_keyboard())
        return

    stock_summary = {}
    for item in stock:
        name = item['name']
        if name not in stock_summary:
            stock_summary[name] = 0
        stock_summary[name] += item['tablet_count'] * item['pack_count']

    summary_message = "\n".join([f"{name}: {count} таблеток" for name, count in stock_summary.items()])
    await callback_query.message.edit_text(f"Общее количество таблеток по названиям:\n{summary_message}")
    await callback_query.message.answer("Ваши запасы таблеток:", reply_markup=get_view_stock_keyboard(stock))
    await Stock.view_stock.set()

# Обработчик для редактирования запаса таблеток
async def edit_stock(callback_query: types.CallbackQuery, state: FSMContext):
    stock_id = int(callback_query.data.split('_')[2])
    stock = await get_medicine_stock(callback_query.from_user.id)
    stock_item = next((item for item in stock if item['id'] == stock_id), None)

    if not stock_item:
        await callback_query.message.answer("Запас не найден.")
        return

    async with state.proxy() as data:
        data['stock_id'] = stock_id
        data['stock_name'] = stock_item['name']
        data['tablet_count'] = stock_item['tablet_count']
        data['pack_count'] = stock_item['pack_count']

    await callback_query.message.answer(f"Изменение запаса для {stock_item['name']}:", reply_markup=get_edit_stock_keyboard())
    await Stock.edit_stock_name.set()

async def edit_stock_name(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.message.answer("Введите новое название лекарства:")
    await Stock.edit_stock_name.set()

async def edit_tablet_count(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.message.answer("Введите новое количество таблеток:")
    await Stock.edit_tablet_count.set()

async def edit_pack_count(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.message.answer("Введите новое количество упаковок:")
    await Stock.edit_pack_count.set()

# Обработчик для установки нового названия
async def set_edit_stock_name(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['stock_name'] = message.text
        stock_id = data['stock_id']
        tablet_count = data['tablet_count']
        pack_count = data['pack_count']

    await update_medicine_stock(stock_id, data['stock_name'], tablet_count, pack_count)
    updated_stock = await get_medicine_stock(message.from_user.id)
    updated_item = next((item for item in updated_stock if item['id'] == stock_id), None)

    if updated_item:
        await message.answer(f"Название успешно изменено!\n{updated_item['name']} - {updated_item['tablet_count']} таблеток - {updated_item['pack_count']} упаковок")
    else:
        await message.answer("Ошибка при обновлении записи.")

    await message.answer("Что вы хотите сделать дальше?", reply_markup=get_edit_stock_keyboard())
    await Stock.edit_stock_name.set()

# Обработчик для установки нового количества таблеток
async def set_edit_tablet_count(message: types.Message, state: FSMContext):
    try:
        tablet_count = int(message.text)
        if tablet_count <= 0:
            raise ValueError("Количество таблеток должно быть положительным числом.")
    except ValueError as e:
        await message.answer(f"Ошибка: {e}. Пожалуйста, введите корректное количество таблеток.")
        return

    async with state.proxy() as data:
        data['tablet_count'] = tablet_count
        stock_id = data['stock_id']
        stock_name = data['stock_name']
        pack_count = data['pack_count']

    await update_medicine_stock(stock_id, stock_name, tablet_count, pack_count)
    updated_stock = await get_medicine_stock(message.from_user.id)
    updated_item = next((item for item in updated_stock if item['id'] == stock_id), None)

    if updated_item:
        await message.answer(f"Количество таблеток успешно изменено!\n{updated_item['name']} - {updated_item['tablet_count']} таблеток - {updated_item['pack_count']} упаковок")
    else:
        await message.answer("Ошибка при обновлении записи.")

    await message.answer("Что вы хотите сделать дальше?", reply_markup=get_edit_stock_keyboard())
    await Stock.edit_stock_name.set()

# Обработчик для установки нового количества упаковок
async def set_edit_pack_count(message: types.Message, state: FSMContext):
    try:
        pack_count = int(message.text)
        if pack_count <= 0:
            raise ValueError("Количество упаковок должно быть положительным числом.")
    except ValueError as e:
        await message.answer(f"Ошибка: {e}. Пожалуйста, введите корректное количество упаковок.")
        return

    async with state.proxy() as data:
        data['pack_count'] = pack_count
        stock_id = data['stock_id']
        stock_name = data['stock_name']
        tablet_count = data['tablet_count']

    await update_medicine_stock(stock_id, stock_name, tablet_count, pack_count)
    updated_stock = await get_medicine_stock(message.from_user.id)
    updated_item = next((item for item in updated_stock if item['id'] == stock_id), None)

    if updated_item:
        await message.answer(f"Количество упаковок успешно изменено!\n{updated_item['name']} - {updated_item['tablet_count']} таблеток - {updated_item['pack_count']} упаковок")
    else:
        await message.answer("Ошибка при обновлении записи.")

    await message.answer("Что вы хотите сделать дальше?", reply_markup=get_edit_stock_keyboard())
    await Stock.edit_stock_name.set()

# Обработчик для удаления запаса таблеток
async def delete_stock(callback_query: types.CallbackQuery, state: FSMContext):
    stock_id = int(callback_query.data.split('_')[2])
    stock = get_medicine_stock(callback_query.from_user.id)
    stock_item = next((item for item in stock if item['id'] == stock_id), None)

    if not stock_item:
        await callback_query.message.answer("Запас не найден.")
        return

    await delete_medicine_stock(stock_id)
    await callback_query.message.answer("Запас успешно удален!")
    await view_stock(callback_query)
    await state.finish()

# Регистрация обработчиков
def register_handlers(dp: Dispatcher):
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