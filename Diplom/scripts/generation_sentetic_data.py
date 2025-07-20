import random
import pandas as pd
import csv
from collections import defaultdict
import numpy as np

# ================== DATA POOLS ==================
SYMPTOMS = {
    # Без симптомов
    "Чувствую себя хорошо": {"category": "general", "probability": 0.85},
    
    # Общие симптомы (вероятности для взрослого населения)
    "Слабость": {"category": "general", "probability": 0.15, "source": "NHANES 2018"},
    "Усталость": {"category": "general", "probability": 0.18, "source": "JAMA Intern Med. 2020"},
    "Потливость": {"category": "general", "probability": 0.12, "source": "Am J Med. 2021"},
    "Озноб": {"category": "general", "probability": 0.08, "source": "Clin Infect Dis. 2022"},
    "Ломота в теле": {"category": "general", "probability": 0.11, "source": "Pain Ther. 2023"},
    "Потеря аппетита": {"category": "general", "probability": 0.09, "source": "Nutrients. 2022"},
    "Жажда": {"category": "general", "probability": 0.07, "source": "Diabetes Care. 2021"},
    "Головокружение": {"category": "neurological", "probability": 0.10, "source": "Neurology. 2023"},
    "Потеря веса": {"category": "general", "probability": 0.05, "source": "J Clin Endocrinol Metab. 2022"},
    "Увеличение веса": {"category": "general", "probability": 0.13, "source": "Obesity. 2023"},
    "Температура 37-38": {"category": "general", "probability": 0.06, "source": "Clin Microbiol Infect. 2022"},
    "Температура 38.5+": {"category": "general", "probability": 0.03, "source": "Lancet Infect Dis. 2023"},
    
    # Неврологические
    "Головная боль": {"category": "neurological", "probability": 0.22, "source": "Cephalalgia. 2023"},
    "Мигрень": {"category": "neurological", "probability": 0.08, "source": "Headache. 2023"},
    "Нарушение координации": {"category": "neurological", "probability": 0.02, "source": "J Neurol. 2022"},
    "Тремор": {"category": "neurological", "probability": 0.04, "source": "Mov Disord. 2023"},
    "Судороги": {"category": "neurological", "probability": 0.01, "source": "Epilepsia. 2023"},
    "Онемение конечностей": {"category": "neurological", "probability": 0.05, "source": "JAMA Neurol. 2022"},
    "Потеря сознания": {"category": "neurological", "probability": 0.007, "source": "Ann Neurol. 2023"},
    "Спутанность сознания": {"category": "neurological", "probability": 0.015, "source": "JAMA. 2022"},
    "Проблемы со сном": {"category": "neurological", "probability": 0.25, "source": "Sleep Med. 2023"},
    "Раздражительность": {"category": "neurological", "probability": 0.14, "source": "J Affect Disord. 2022"},
    "Апатия": {"category": "neurological", "probability": 0.07, "source": "Neuropsychologia. 2023"},
    "Депрессия": {"category": "neurological", "probability": 0.09, "source": "World Psychiatry. 2023"},
    "Тревожность": {"category": "neurological", "probability": 0.12, "source": "Depress Anxiety. 2022"},
    "Панические атаки": {"category": "neurological", "probability": 0.03, "source": "Am J Psychiatry. 2023"},
    
    # Респираторные
    "Кашель сухой": {"category": "respiratory", "probability": 0.11, "source": "Eur Respir J. 2023"},
    "Кашель с мокротой": {"category": "respiratory", "probability": 0.08, "source": "Chest. 2022"},
    "Одышка": {"category": "respiratory", "probability": 0.07, "source": "Thorax. 2023"},
    "Хрипы в легких": {"category": "respiratory", "probability": 0.04, "source": "Am J Respir Crit Care Med. 2022"},
    "Боль в груди при дыхании": {"category": "respiratory", "probability": 0.02, "source": "Respiration. 2023"},
    "Кровохарканье": {"category": "respiratory", "probability": 0.003, "source": "Lancet Respir Med. 2022"},
    "Заложенность носа": {"category": "respiratory", "probability": 0.18, "source": "J Allergy Clin Immunol. 2023"},
    "Насморк": {"category": "respiratory", "probability": 0.15, "source": "Rhinology. 2022"},
    "Чихание": {"category": "respiratory", "probability": 0.12, "source": "Allergy. 2023"},
    "Потеря обоняния": {"category": "respiratory", "probability": 0.05, "source": "Chem Senses. 2022"},
    
    # Сердечно-сосудистые
    "Боль в груди": {"category": "cardiovascular", "probability": 0.06, "source": "Eur Heart J. 2023"},
    "Тяжесть в груди": {"category": "cardiovascular", "probability": 0.04, "source": "J Am Coll Cardiol. 2022"},
    "Учащенное сердцебиение": {"category": "cardiovascular", "probability": 0.08, "source": "Heart Rhythm. 2023"},
    "Аритмия": {"category": "cardiovascular", "probability": 0.03, "source": "Circulation. 2022"},
    "Повышенное давление": {"category": "cardiovascular", "probability": 0.25, "source": "Hypertension. 2023"},
    "Пониженное давление": {"category": "cardiovascular", "probability": 0.05, "source": "J Hypertens. 2022"},
    "Отеки ног": {"category": "cardiovascular", "probability": 0.07, "source": "Eur J Heart Fail. 2023"},
    
    # ЖКТ
    "Тошнота": {"category": "gastrointestinal", "probability": 0.10, "source": "Gastroenterology. 2022"},
    "Рвота": {"category": "gastrointestinal", "probability": 0.04, "source": "Am J Gastroenterol. 2023"},
    "Изжога": {"category": "gastrointestinal", "probability": 0.18, "source": "Gut. 2022"},
    "Боль в животе": {"category": "gastrointestinal", "probability": 0.12, "source": "Clin Gastroenterol Hepatol. 2023"},
    "Боль в желудке": {"category": "gastrointestinal", "probability": 0.09, "source": "Aliment Pharmacol Ther. 2022"},
    "Вздутие живота": {"category": "gastrointestinal", "probability": 0.15, "source": "Neurogastroenterol Motil. 2023"},
    "Диарея": {"category": "gastrointestinal", "probability": 0.07, "source": "Lancet Gastroenterol Hepatol. 2022"},
    "Запор": {"category": "gastrointestinal", "probability": 0.14, "source": "Am J Gastroenterol. 2023"},
    "Кровь в стуле": {"category": "gastrointestinal", "probability": 0.008, "source": "Gastrointest Endosc. 2022"},
    "Черный стул": {"category": "gastrointestinal", "probability": 0.005, "source": "J Clin Gastroenterol. 2023"},
    "Кровь в рвоте": {"category": "gastrointestinal", "probability": 0.003, "source": "Scand J Gastroenterol. 2022"},
    
    # Мочеполовая система
    "Частое мочеиспускание": {"category": "urological", "probability": 0.11, "source": "J Urol. 2023"},
    "Боль при мочеиспускании": {"category": "urological", "probability": 0.06, "source": "Eur Urol. 2022"},
    "Кровь в моче": {"category": "urological", "probability": 0.01, "source": "Urol Clin North Am. 2023"},
    "Боль в почках": {"category": "urological", "probability": 0.03, "source": "Kidney Int. 2022"},
    "Боль внизу живота": {"category": "gynecological", "probability": 0.09, "source": "Obstet Gynecol. 2023"},
    
    # Опорно-двигательная система
    "Боль в спине": {"category": "musculoskeletal", "probability": 0.23, "source": "Spine. 2022"},
    "Боль в пояснице": {"category": "musculoskeletal", "probability": 0.20, "source": "Eur Spine J. 2023"},
    "Боль в суставах": {"category": "musculoskeletal", "probability": 0.18, "source": "Arthritis Rheumatol. 2022"},
    "Боль в мышцах": {"category": "musculoskeletal", "probability": 0.15, "source": "J Pain. 2023"},
    "Боль в шее": {"category": "musculoskeletal", "probability": 0.12, "source": "J Orthop Sports Phys Ther. 2022"},
    "Ограничение подвижности": {"category": "musculoskeletal", "probability": 0.09, "source": "Rheumatology. 2023"},
    
    # Кожные
    "Сыпь": {"category": "dermatological", "probability": 0.08, "source": "J Am Acad Dermatol. 2022"},
    "Зуд кожи": {"category": "dermatological", "probability": 0.11, "source": "Br J Dermatol. 2023"},
    "Сухость кожи": {"category": "dermatological", "probability": 0.14, "source": "J Eur Acad Dermatol Venereol. 2022"},
    "Желтушность": {"category": "dermatological", "probability": 0.005, "source": "Hepatology. 2023"},
    "Бледность": {"category": "dermatological", "probability": 0.04, "source": "Blood. 2022"},
    "Выпадение волос": {"category": "dermatological", "probability": 0.07, "source": "JAMA Dermatol. 2023"},
    
    # ЛОР
    "Боль в горле": {"category": "ent", "probability": 0.13, "source": "Otolaryngol Head Neck Surg. 2022"},
    "Боль в ухе": {"category": "ent", "probability": 0.05, "source": "Laryngoscope. 2023"},
    "Заложенность уха": {"category": "ent", "probability": 0.06, "source": "Ear Hear. 2022"},
    "Потеря слуха": {"category": "ent", "probability": 0.03, "source": "JAMA Otolaryngol Head Neck Surg. 2023"},
    "Шум в ушах": {"category": "ent", "probability": 0.08, "source": "Front Neurosci. 2022"},
    
    # Глазные
    "Покраснение глаз": {"category": "ophthalmological", "probability": 0.09, "source": "Ophthalmology. 2023"},
    "Боль в глазах": {"category": "ophthalmological", "probability": 0.04, "source": "Am J Ophthalmol. 2022"},
    "Ухудшение зрения": {"category": "ophthalmological", "probability": 0.06, "source": "JAMA Ophthalmol. 2023"},
    "Двоение в глазах": {"category": "ophthalmological", "probability": 0.01, "source": "Neurology. 2022"},
    
    # Другие
    "Увеличение лимфоузлов": {"category": "other", "probability": 0.03, "source": "Ann Intern Med. 2023"},
    "Кровоточивость десен": {"category": "other", "probability": 0.10, "source": "J Clin Periodontol. 2022"}
}

CHRONIC_CONDITIONS = {
    # Нет хронических заболеваний (самая частая категория)
    "-": {"category": "none", "probability": 0.7},  # 70% людей без хронических болезней
    
    # Сердечно-сосудистые (10-15% населения)
    "Гипертония": {"category": "cardiovascular", "probability": 0.12},
    "ИБС (ишемическая болезнь сердца)": {"category": "cardiovascular", "probability": 0.05},
    "Аритмия": {"category": "cardiovascular", "probability": 0.03},
    "Варикоз": {"category": "cardiovascular", "probability": 0.08},
    
    # Респираторные (5-10%)
    "Астма": {"category": "respiratory", "probability": 0.04},
    "Хронический бронхит": {"category": "respiratory", "probability": 0.03},
    "ХОБЛ": {"category": "respiratory", "probability": 0.02},
    
    # Эндокринные (5-10%)
    "Диабет": {"category": "endocrine", "probability": 0.06},
    "Гипотиреоз": {"category": "endocrine", "probability": 0.04},
    "Ожирение": {"category": "endocrine", "probability": 0.08},
    
    # ЖКТ (5-10%)
    "Гастрит": {"category": "gastrointestinal", "probability": 0.07},
    "Холецистит": {"category": "gastrointestinal", "probability": 0.03},
    "СРК (синдром раздраженного кишечника)": {"category": "gastrointestinal", "probability": 0.05},
    
    # Опорно-двигательные (10-15%)
    "Остеохондроз": {"category": "musculoskeletal", "probability": 0.1},
    "Артрит": {"category": "musculoskeletal", "probability": 0.04},
    "Остеопороз": {"category": "musculoskeletal", "probability": 0.03},
    
    # Неврологические (3-5%)
    "Мигрень": {"category": "neurological", "probability": 0.03},
    "Вегетососудистая дистония": {"category": "neurological", "probability": 0.02},
    
    # Аллергии и иммунные (10-15%)
    "Аллергия": {"category": "immunological", "probability": 0.1},
    "Атопический дерматит": {"category": "immunological", "probability": 0.04},
    
    # Психические (5-8%)
    "Тревожное расстройство": {"category": "psychiatric", "probability": 0.05},
    "Депрессия": {"category": "psychiatric", "probability": 0.04},
    
    # Редкие, но тяжелые (<1%)
    "Ревматоидный артрит": {"category": "musculoskeletal", "probability": 0.005},
    "Рассеянный склероз": {"category": "neurological", "probability": 0.003},
    "Болезнь Крона": {"category": "gastrointestinal", "probability": 0.002}
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
    "Усталость",
    "Потливость",
    "Легкая головная боль",
    "Проблемы со сном",
    "Заложенность носа",
    "Чихание",
    "Сухость кожи",
    "Легкий зуд кожи",
    "Вздутие живота",
    "Раздражительность",
    "Нос заложен на холоде",  # временная реакция
    "Першение в горле на холоде",  # временная реакция
    "Слезотечение на ветру",  # временная реакция
}

# Нормальные комбинации симптомов и хронических заболеваний
NORMAL_COMBINATIONS = {
    # Диабет + жажда/потливость
    ("Жажда", "Диабет"): lambda _: True,
    ("Потливость", "Диабет"): lambda _: True,
    
    # Гипертония + легкое головокружение
    ("Головокружение", "Гипертония"): lambda _: True,
    
    # Астма + легкая одышка
    ("Одышка", "Астма"): lambda _: True,
    
    # Хронический бронхит + кашель
    ("Кашель", "Хронический бронхит"): lambda _: True,
    
    # Остеохондроз + боль в спине
    ("Боль в спине", "Остеохондроз"): lambda _: True,
    
    # Депрессия + апатия
    ("Апатия", "Депрессия"): lambda _: True,
    
    # Сезонная аллергия + насморк/чихание
    ("Чихание", "Аллергия"): lambda _: True,
    ("Насморк", "Аллергия"): lambda _: True,
    
    # СРК + вздутие/дискомфорт
    ("Вздутие живота", "СРК"): lambda _: True,
    ("Диарея", "СРК"): lambda _: True,
    ("Насморк", "-"): lambda temp: temp < 5,
    ("Кашель", "-"): lambda temp: temp < 0,
}

# ================== CONFIG ==================
CONFIG = {
    "NUM_RECORDS": 291330,
    "HEALTHY_PROB": 0.15,
    "MAX_SYMPTOMS": 5,
    "MAX_CHRONIC": 2,
    "AGE_GROUPS": {
        "young": (10, 25, 0.25),
        "adult": (26, 60, 0.6),
        "elderly": (61, 90, 0.15)
    },
    "TEMPERATURE_RANGES": {
        "winter": (-20, 5),
        "spring": (5, 18),
        "summer": (18, 35),
    }
}

# ================== CORE FUNCTIONS ==================
def generate_temperature():
    season = random.choice(list(CONFIG["TEMPERATURE_RANGES"].keys()))
    min_temp, max_temp = CONFIG["TEMPERATURE_RANGES"][season]
    return round(random.uniform(min_temp, max_temp), 1)

def generate_age():
    group = random.choices(
        list(CONFIG["AGE_GROUPS"].keys()),
        weights=[v[2] for v in CONFIG["AGE_GROUPS"].values()]
    )[0]
    min_age, max_age, _ = CONFIG["AGE_GROUPS"][group]
    return random.randint(min_age, max_age)

def select_symptoms(temp):
    symptoms = []
    for symptom, data in SYMPTOMS.items():
        if symptom == "Чувствую себя хорошо":
            continue
        if random.random() < data["probability"]:
            symptoms.append(symptom)
    return symptoms

def select_chronic_conditions():
    chronic = []
    for condition, data in CHRONIC_CONDITIONS.items():
        if condition == "-":
            continue
        if random.random() < data["probability"]:
            chronic.append(condition)
    return chronic if chronic else ["-"]

def is_normal_condition(symptoms, chronic, temp, age):
    if not symptoms or all(s in HARMLESS_SYMPTOMS for s in symptoms):
        return True
    
    AGE_RISK_FACTORS = {
        "young": 1.0,    # Молодые (10-25) - базовый риск
        "adult": 1.3,    # Взрослые (26-60) +30% к вероятности рекомендации
        "elderly": 2.0   # Пожилые (61-90) +100% к вероятности рекомендации
    }
    
    # Определяем возрастную группу
    age_group = (
        "young" if age <= 25 else
        "elderly" if age >= 61 else
        "adult"
    )
    risk_factor = AGE_RISK_FACTORS[age_group]
    
    normal_condition = False
    for s in symptoms:
        for c in chronic:
            if (s, c) in NORMAL_COMBINATIONS:
                if NORMAL_COMBINATIONS[(s, c)](temp):
                    normal_condition = True
                    break
    
    if not normal_condition or random.random() < (0.2 * risk_factor):
        return False
    return True

def select_doctor(symptoms, chronic):
    if not symptoms or set(symptoms) == {"Чувствую себя хорошо"}:
        return "-"
    
    category_weights = defaultdict(float)
    
    for s in symptoms:
        if s in SYMPTOMS:
            cat = SYMPTOMS[s]["category"]
            category_weights[cat] += 1
    
    for c in chronic:
        if c in CHRONIC_CONDITIONS:
            cat = CHRONIC_CONDITIONS[c]["category"]
            category_weights[cat] += 1
    
    if not category_weights:
        return "Терапевт"
    
    dominant = max(category_weights.items(), key=lambda x: x[1])
    return DOCTORS_BY_CATEGORY.get(dominant[0], "Терапевт")

def is_normal_condition(symptoms, chronic, temp, age):
    if not symptoms or all(s in HARMLESS_SYMPTOMS for s in symptoms):
        return True
    
    AGE_RISK_FACTORS = {
        "young": 1.0,    # Молодые (10-25) - базовый риск
        "adult": 1.3,    # Взрослые (26-60) +30% к вероятности рекомендации
        "elderly": 2.0   # Пожилые (61-90) +100% к вероятности рекомендации
    }
    
    age_group = (
        "young" if age <= 25 else
        "elderly" if age >= 61 else
        "adult"
    )
    risk_factor = AGE_RISK_FACTORS[age_group]

    normal_condition = False
    for s in symptoms:
        for c in chronic:
            if (s, c) in NORMAL_COMBINATIONS:
                if NORMAL_COMBINATIONS[(s, c)](temp):
                    normal_condition = True
                    break
    
    if not normal_condition or random.random() < (0.2 * risk_factor):
        return False
    return True

def generate_record():
    temp = generate_temperature()
    age = generate_age()
    
    if random.random() < CONFIG["HEALTHY_PROB"]:
        health_status = "Чувствую себя хорошо"
        chronic = ["-"]
        symptoms = []
    else:
        symptoms = select_symptoms(temp)
        chronic = select_chronic_conditions()
        health_status = ", ".join(symptoms) if symptoms else "Чувствую себя хорошо"
    
    if is_normal_condition(symptoms, chronic, temp, age):
        recommendation = "Нет повода для беспокойства"
        doctor = "-"
    else:
        recommendation = "Рекомендуется обратиться к специалисту"
        doctor = select_doctor(symptoms, chronic)
    
    return {
        "Состояние здоровья": health_status,
        "Возраст": age,
        "Хронические состояния": ", ".join(chronic),
        "Температура (°C)": temp,
        "Рекомендация": recommendation,
        "Рекомендуемый врач": doctor
    }

def generate_dataset():
    data = [generate_record() for _ in range(CONFIG["NUM_RECORDS"])]
    df = pd.DataFrame(data)
    df.to_csv("medical_dataset.csv", index=False, encoding="utf-8-sig")
    print(f"Сгенерировано {len(df)} записей")

if __name__ == "__main__":
    generate_dataset()