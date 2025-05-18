import random
import pandas as pd
import csv
from collections import defaultdict
import numpy as np

# ================== DATA POOLS ==================
SYMPTOMS = {
    # Без симптомов
    "Чувствую себя хорошо": {"category": "general", "severity": 0},
    
    # Общие симптомы
    "Слабость": {"category": "general", "severity": 1},
    "Усталость": {"category": "general", "severity": 1},
    "Потливость": {"category": "general", "severity": 1},
    "Озноб": {"category": "general", "severity": 1},
    "Ломота в теле": {"category": "general", "severity": 1},
    "Потеря аппетита": {"category": "general", "severity": 1},
    "Жажда": {"category": "general", "severity": 1},
    "Головокружение": {"category": "neurological", "severity": 2},
    "Потеря веса": {"category": "general", "severity": 2},
    "Увеличение веса": {"category": "general", "severity": 1},
    "Температура 37-38": {"category": "general", "severity": 1},
    "Температура 38.5+": {"category": "general", "severity": 2},
    
    # Неврологические
    "Головная боль": {"category": "neurological", "severity": 1},
    "Мигрень": {"category": "neurological", "severity": 2},
    "Нарушение координации": {"category": "neurological", "severity": 3},
    "Тремор": {"category": "neurological", "severity": 2},
    "Судороги": {"category": "neurological", "severity": 4},
    "Онемение конечностей": {"category": "neurological", "severity": 2},
    "Потеря сознания": {"category": "neurological", "severity": 4},
    "Спутанность сознания": {"category": "neurological", "severity": 4},
    "Проблемы со сном": {"category": "neurological", "severity": 1},
    "Раздражительность": {"category": "neurological", "severity": 1},
    "Апатия": {"category": "neurological", "severity": 2},
    "Депрессия": {"category": "neurological", "severity": 2},
    "Тревожность": {"category": "neurological", "severity": 2},
    "Панические атаки": {"category": "neurological", "severity": 3},
    
    # Респираторные
    "Кашель сухой": {"category": "respiratory", "severity": 1},
    "Кашель с мокротой": {"category": "respiratory", "severity": 1},
    "Одышка": {"category": "respiratory", "severity": 2},
    "Хрипы в легких": {"category": "respiratory", "severity": 2},
    "Боль в груди при дыхании": {"category": "respiratory", "severity": 3},
    "Кровохарканье": {"category": "respiratory", "severity": 4},
    "Заложенность носа": {"category": "respiratory", "severity": 1},
    "Насморк": {"category": "respiratory", "severity": 1},
    "Чихание": {"category": "respiratory", "severity": 1},
    "Потеря обоняния": {"category": "respiratory", "severity": 2},
    
    # Сердечно-сосудистые
    "Боль в груди": {"category": "cardiovascular", "severity": 3},
    "Тяжесть в груди": {"category": "cardiovascular", "severity": 2},
    "Учащенное сердцебиение": {"category": "cardiovascular", "severity": 2},
    "Аритмия": {"category": "cardiovascular", "severity": 3},
    "Повышенное давление": {"category": "cardiovascular", "severity": 2},
    "Пониженное давление": {"category": "cardiovascular", "severity": 2},
    "Отеки ног": {"category": "cardiovascular", "severity": 2},
    
    # ЖКТ
    "Тошнота": {"category": "gastrointestinal", "severity": 1},
    "Рвота": {"category": "gastrointestinal", "severity": 2},
    "Изжога": {"category": "gastrointestinal", "severity": 1},
    "Боль в животе": {"category": "gastrointestinal", "severity": 2},
    "Боль в желудке": {"category": "gastrointestinal", "severity": 2},
    "Вздутие живота": {"category": "gastrointestinal", "severity": 1},
    "Диарея": {"category": "gastrointestinal", "severity": 2},
    "Запор": {"category": "gastrointestinal", "severity": 1},
    "Кровь в стуле": {"category": "gastrointestinal", "severity": 4},
    "Черный стул": {"category": "gastrointestinal", "severity": 3},
    "Кровь в рвоте": {"category": "gastrointestinal", "severity": 4},
    
    # Мочеполовая система
    "Частое мочеиспускание": {"category": "urological", "severity": 2},
    "Боль при мочеиспускании": {"category": "urological", "severity": 2},
    "Кровь в моче": {"category": "urological", "severity": 4},
    "Боль в почках": {"category": "urological", "severity": 3},
    "Боль внизу живота": {"category": "gynecological", "severity": 2},
    
    # Опорно-двигательная система
    "Боль в спине": {"category": "musculoskeletal", "severity": 2},
    "Боль в пояснице": {"category": "musculoskeletal", "severity": 2},
    "Боль в суставах": {"category": "musculoskeletal", "severity": 2},
    "Боль в мышцах": {"category": "musculoskeletal", "severity": 1},
    "Боль в шее": {"category": "musculoskeletal", "severity": 1},
    "Ограничение подвижности": {"category": "musculoskeletal", "severity": 2},
    
    # Кожные
    "Сыпь": {"category": "dermatological", "severity": 2},
    "Зуд кожи": {"category": "dermatological", "severity": 1},
    "Сухость кожи": {"category": "dermatological", "severity": 1},
    "Желтушность": {"category": "dermatological", "severity": 4},
    "Бледность": {"category": "dermatological", "severity": 2},
    "Выпадение волос": {"category": "dermatological", "severity": 2},
    
    # ЛОР
    "Боль в горле": {"category": "ent", "severity": 1},
    "Боль в ухе": {"category": "ent", "severity": 1},
    "Заложенность уха": {"category": "ent", "severity": 1},
    "Потеря слуха": {"category": "ent", "severity": 2},
    "Шум в ушах": {"category": "ent", "severity": 1},
    
    # Глазные
    "Покраснение глаз": {"category": "ophthalmological", "severity": 1},
    "Боль в глазах": {"category": "ophthalmological", "severity": 2},
    "Ухудшение зрения": {"category": "ophthalmological", "severity": 3},
    "Двоение в глазах": {"category": "ophthalmological", "severity": 3},
    
    # Другие
    "Увеличение лимфоузлов": {"category": "other", "severity": 3},
    "Кровоточивость десен": {"category": "other", "severity": 1}
}

CHRONIC_CONDITIONS = {
    "-": {"category": "none", "severity": 0},
    
    # Сердечно-сосудистые
    "Гипертония": {"category": "cardiovascular", "severity": 2},
    "ИБС": {"category": "cardiovascular", "severity": 3},
    "Сердечная недостаточность": {"category": "cardiovascular", "severity": 3},
    "Аритмия": {"category": "cardiovascular", "severity": 3},
    "Атеросклероз": {"category": "cardiovascular", "severity": 3},
    "Варикоз": {"category": "cardiovascular", "severity": 2},
    
    # Эндокринные
    "Диабет 1 типа": {"category": "endocrine", "severity": 3},
    "Диабет 2 типа": {"category": "endocrine", "severity": 2},
    "Гипотиреоз": {"category": "endocrine", "severity": 2},
    "Гипертиреоз": {"category": "endocrine", "severity": 2},
    
    # Респираторные
    "Астма": {"category": "respiratory", "severity": 2},
    "ХОБЛ": {"category": "respiratory", "severity": 3},
    "Хронический бронхит": {"category": "respiratory", "severity": 2},
    
    # ЖКТ
    "Гастрит": {"category": "gastrointestinal", "severity": 1},
    "Язва желудка": {"category": "gastrointestinal", "severity": 3},
    "Холецистит": {"category": "gastrointestinal", "severity": 2},
    "Панкреатит": {"category": "gastrointestinal", "severity": 3},
    "СРК": {"category": "gastrointestinal", "severity": 1},
    
    # Неврологические
    "Эпилепсия": {"category": "neurological", "severity": 4},
    "Болезнь Паркинсона": {"category": "neurological", "severity": 5},
    "Рассеянный склероз": {"category": "neurological", "severity": 5},
    "Мигрень": {"category": "neurological", "severity": 2},
    
    # Опорно-двигательные
    "Остеохондроз": {"category": "musculoskeletal", "severity": 2},
    "Артрит": {"category": "musculoskeletal", "severity": 2},
    "Артроз": {"category": "musculoskeletal", "severity": 2},
    "Остеопороз": {"category": "musculoskeletal", "severity": 2},
    
    # Психические
    "Депрессия": {"category": "psychiatric", "severity": 2},
    "Тревожное расстройство": {"category": "psychiatric", "severity": 2},
    "Биполярное расстройство": {"category": "psychiatric", "severity": 4},
    
    # Другие
    "Хроническая болезнь почек": {"category": "urological", "severity": 4},
    "Анемия": {"category": "hematological", "severity": 2},
    "Аллергия": {"category": "immunological", "severity": 1},
    "Псориаз": {"category": "dermatological", "severity": 2},
    "Гепатит": {"category": "hepatological", "severity": 4},
    "ВИЧ": {"category": "immunological", "severity": 5}
}

# Врачи по категориям
DOCTORS_BY_CATEGORY = {
    "cardiovascular": "Кардиолог",
    "neurological": "Невролог",
    "gastrointestinal": "Гастроэнтеролог",
    "endocrine": "Эндокринолог",
    "respiratory": "Пульмонолог",
    "ent": "ЛОР",
    "ophthalmological": "Офтальмолог",
    "dermatological": "Дерматолог",
    "musculoskeletal": "Ревматолог",
    "urological": "Уролог",
    "gynecological": "Гинеколог",
    "psychiatric": "Психиатр",
    "general": "Терапевт"
}

# Безобидные симптомы (не требующие обращения к врачу)
HARMLESS_SYMPTOMS = {
    "Чувствую себя хорошо",
    "Усталость",  # после работы/учебы
    "Потливость",  # в жару/при стрессе
    "Легкая головная боль",  # от усталости
    "Проблемы со сном",  # временные
    "Заложенность носа",  # легкая аллергия
    "Чихание",  # аллергическая реакция
    "Сухость кожи",  # сезонное
    "Легкий зуд кожи",  # без сыпи
    "Вздутие живота",  # после еды
    "Запор",  # временный
    "Раздражительность",  # ситуативная
    "Мышечная усталость"  # после нагрузки
}

# Нормальные комбинации симптомов и хронических заболеваний
NORMAL_COMBINATIONS = {
    # Диабет + жажда/потливость
    ("Жажда", "Диабет 1 типа"): True,
    ("Жажда", "Диабет 2 типа"): True,
    ("Потливость", "Диабет 1 типа"): True,
    
    # Гипертония + легкое головокружение
    ("Головокружение", "Гипертония"): True,
    
    # Астма + легкая одышка
    ("Одышка", "Астма"): True,
    
    # Хронический бронхит + кашель
    ("Кашель с мокротой", "Хронический бронхит"): True,
    
    # Остеохондроз + боль в спине
    ("Боль в спине", "Остеохондроз"): True,
    
    # Депрессия + апатия
    ("Апатия", "Депрессия"): True,
    
    # Сезонная аллергия + насморк/чихание
    ("Чихание", "Аллергия"): True,
    ("Насморк", "Аллергия"): True,
    
    # СРК + вздутие/дискомфорт
    ("Вздутие живота", "СРК"): True,
    ("Диарея", "СРК"): True
}

# ================== CONFIG ==================
CONFIG = {
    "NUM_RECORDS": 86042,
    "HEALTHY_PROB": 0.15,
    "SYMPTOM_NUM_PROBS": [0.3, 0.25, 0.2, 0.15, 0.07, 0.03],
    "CHRONIC_NUM_PROBS": [0.6, 0.3, 0.08, 0.02],
    "AGE_GROUPS": {
        "young": (10, 25, 0.25),
        "adult": (26, 60, 0.6), 
        "elderly": (61, 90, 0.15)
    }
}

# ================== CORE FUNCTIONS ==================
def generate_weather():
    season = random.choice(["winter", "spring", "summer", "autumn"])
    if season == "winter":
        return f"{random.choice(['Ясно', 'Снег', 'Пасмурно'])}, {random.uniform(-20, 0):.1f}°C"
    # ... по аналогии для других сезонов

def generate_age():
    group = random.choices(
        list(CONFIG["AGE_GROUPS"].keys()),
        weights=[v[2] for v in CONFIG["AGE_GROUPS"].values()]
    )[0]
    min_age, max_age, _ = CONFIG["AGE_GROUPS"][group]
    return random.randint(min_age, max_age)

def is_normal_condition(symptoms, chronic):
    if not symptoms or all(s in HARMLESS_SYMPTOMS for s in symptoms):
        return True
    
    for s in symptoms:
        for c in chronic:
            if (s, c) in NORMAL_COMBINATIONS:
                return True
    return False

def select_doctor(symptoms, chronic):
    if not symptoms or set(symptoms) == {"Чувствую себя хорошо"}:
        return "-"
    
    category_weights = defaultdict(float)
    
    # Учет симптомов
    for s in symptoms:
        if s in SYMPTOMS:
            cat = SYMPTOMS[s]["category"]
            category_weights[cat] += SYMPTOMS[s]["severity"] * 1.2
    
    # Учет хронических заболеваний
    for c in chronic:
        if c in CHRONIC_CONDITIONS:
            cat = CHRONIC_CONDITIONS[c]["category"]
            category_weights[cat] += CHRONIC_CONDITIONS[c]["severity"] * 0.7
    
    if not category_weights:
        return "Терапевт"
    
    dominant = max(category_weights.items(), key=lambda x: x[1])
    sorted_cats = sorted(category_weights.items(), key=lambda x: -x[1])
    
    if len(sorted_cats) > 1 and (sorted_cats[0][1] - sorted_cats[1][1]) < 2.5:
        return "Терапевт"
    
    return DOCTORS_BY_CATEGORY.get(dominant[0], "Терапевт")

def generate_record():
    if random.random() < CONFIG["HEALTHY_PROB"]:
        health_status = "Чувствую себя хорошо"
        chronic = "-"
    else:
        num_symptoms = random.choices(
            range(1, len(CONFIG["SYMPTOM_NUM_PROBS"])+1),
            weights=CONFIG["SYMPTOM_NUM_PROBS"]
        )[0]
        symptoms = random.sample(
            [s for s in SYMPTOMS if s != "Чувствую себя хорошо"], 
            num_symptoms
        )
        health_status = ", ".join(symptoms)
        
        num_chronic = random.choices(
            range(len(CONFIG["CHRONIC_NUM_PROBS"])),
            weights=CONFIG["CHRONIC_NUM_PROBS"]
        )[0]
        chronic = ", ".join(random.sample(
            [c for c in CHRONIC_CONDITIONS if c != "-"], 
            num_chronic
        )) if num_chronic > 0 else "-"
    
    age = generate_age()
    weather = generate_weather()
    symptoms_list = health_status.split(", ") if health_status != "Чувствую себя хорошо" else []
    chronic_list = chronic.split(", ") if chronic != "-" else []
    
    if is_normal_condition(symptoms_list, chronic_list):
        return {
            "Состояние здоровья": health_status,
            "Возраст": age,
            "Хронические состояния": chronic,
            "Погодные условия": weather,
            "Рекомендация": "Нет повода для беспокойства",
            "Рекомендуемый врач": "-"
        }
    else:
        return {
            "Состояние здоровья": health_status,
            "Возраст": age,
            "Хронические состояния": chronic,
            "Погодные условия": weather,
            "Рекомендация": "Рекомендуется обратиться к специалисту",
            "Рекомендуемый врач": select_doctor(symptoms_list, chronic_list)
        }

# ================== MAIN GENERATION ==================
def generate_dataset():
    data = [generate_record() for _ in range(CONFIG["NUM_RECORDS"])]
    df = pd.DataFrame(data)
    
    # Валидация
    df = df[~((df["Рекомендация"] == "Нет повода для беспокойства") & 
             (df["Рекомендуемый врач"] != "-"))]
    
    df.to_csv("medical_dataset.csv", index=False, encoding="utf-8-sig")
    print(f"Сгенерировано {len(df)} записей")

if __name__ == "__main__":
    generate_dataset()