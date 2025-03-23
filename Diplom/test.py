import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertPreTrainedModel, BertModel, Trainer, TrainingArguments, BertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # Добавленный импорт

# Определение многозадачной модели
class MultiTaskBERT(BertPreTrainedModel):
    def __init__(self, config, num_diagnosis_labels, num_doctor_labels):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.diagnosis_classifier = nn.Linear(config.hidden_size, num_diagnosis_labels)
        self.doctor_classifier = nn.Linear(config.hidden_size, num_doctor_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = self.dropout(outputs.pooler_output)

        diagnosis_logits = self.diagnosis_classifier(pooled_output)
        doctor_logits = self.doctor_classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            diagnosis_loss = loss_fct(diagnosis_logits, labels[:, 0])
            doctor_loss = loss_fct(doctor_logits, labels[:, 1])
            loss = diagnosis_loss + doctor_loss
            return loss, diagnosis_logits, doctor_logits  # Возвращаем loss, чтобы Trainer мог его использовать

        return diagnosis_logits, doctor_logits


# Создание класса датасета
class HealthDataset(Dataset):
    def __init__(self, encodings, diagnosis_labels, doctor_labels):
        self.encodings = encodings
        self.diagnosis_labels = diagnosis_labels
        self.doctor_labels = doctor_labels
    
    def __len__(self):
        return len(self.diagnosis_labels)
    
    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx].clone().detach(),
            'attention_mask': self.encodings['attention_mask'][idx].clone().detach(),
            'labels': torch.tensor([self.diagnosis_labels[idx], self.doctor_labels[idx]])
        }
        return item

# Функция предобработки данных
def preprocess_data(file_path, tokenizer):
    df = pd.read_csv(file_path)

    # Преобразуем текстовые метки в числовые
    label_encoder_diagnosis = LabelEncoder()
    df['Метка (Диагноз)'] = label_encoder_diagnosis.fit_transform(df['Метка (Диагноз)'])

    label_encoder_doctor = LabelEncoder()
    df['Рекомендуемый врач'] = label_encoder_doctor.fit_transform(df['Рекомендуемый врач'])

    # Проверка уникальных меток
    print(f"Unique diagnosis labels: {df['Метка (Диагноз)'].nunique()}")
    print(f"Unique doctor labels: {df['Рекомендуемый врач'].nunique()}")

    # Проверка диапазона меток
    print(f"Min diagnosis label: {df['Метка (Диагноз)'].min()}")
    print(f"Max diagnosis label: {df['Метка (Диагноз)'].max()}")
    print(f"Min doctor label: {df['Рекомендуемый врач'].min()}")
    print(f"Max doctor label: {df['Рекомендуемый врач'].max()}")

    encodings = tokenizer(df['Состояние здоровья'].tolist(), padding=True, truncation=True, return_tensors="pt")

    encodings['diagnosis_labels'] = df['Метка (Диагноз)'].tolist()
    encodings['doctor_labels'] = df['Рекомендуемый врач'].tolist()

    # Шаг отладки: Печатаем первые несколько меток, чтобы убедиться, что они правильно добавлены
    print(f"Diagnosis labels: {encodings['diagnosis_labels'][:5]}")
    print(f"Doctor labels: {encodings['doctor_labels'][:5]}")

    return encodings

# Функция для преобразования CSV в Dataset
def csv_to_dataset(csv_path, tokenizer, max_length=128):
    """
    Преобразует CSV-файл в Dataset.
    
    Параметры:
        csv_path: Путь к CSV-файлу.
        tokenizer: Токенизатор для обработки текста.
        max_length: Максимальная длина последовательности.
    
    Возвращает:
        Объект типа Dataset.
    """
    # Загрузка данных из CSV
    df = pd.read_csv(csv_path)
    
    # Преобразование текстовых меток в числовые
    label_encoder_diagnosis = LabelEncoder()
    label_encoder_doctor = LabelEncoder()
    
    df['Метка (Диагноз)'] = label_encoder_diagnosis.fit_transform(df['Метка (Диагноз)'])
    df['Рекомендуемый врач'] = label_encoder_doctor.fit_transform(df['Рекомендуемый врач'])
    
    # Токенизация текста
    encodings = tokenizer(
        df['Состояние здоровья'].tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Преобразование меток в числовой формат
    diagnosis_labels = df['Метка (Диагноз)'].tolist()
    doctor_labels = df['Рекомендуемый врач'].tolist()
    
    # Создание датасета
    dataset = HealthDataset(encodings, diagnosis_labels, doctor_labels)
    return dataset

# Функция для оценки модели
def evaluate_model(model, dataset, tokenizer, device="cuda"):
    """
    Оценивает модель на предоставленном датасете.
    
    Параметры:
        model: Модель для оценки.
        dataset: Датасет (объект типа Dataset).
        tokenizer: Токенизатор для обработки текста.
        device: Устройство для вычислений ("cuda" или "cpu").
    
    Возвращает:
        Словарь с метриками: accuracy, f1, precision, recall.
    """
    model.to(device)
    model.eval()  # Переводим модель в режим оценки

    # Списки для хранения предсказаний и истинных меток
    all_diagnosis_preds = []
    all_doctor_preds = []
    all_diagnosis_labels = []
    all_doctor_labels = []

    with torch.no_grad():  # Отключаем вычисление градиентов
        for item in dataset:
            # Перемещаем данные на устройство
            input_ids = item['input_ids'].to(device)
            attention_mask = item['attention_mask'].to(device)
            labels = item['labels'].to(device)

            # Получаем предсказания модели
            outputs = model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
            
            # Модель возвращает два выхода: diagnosis_logits и doctor_logits
            diagnosis_logits, doctor_logits = outputs[0], outputs[1]

            # Преобразуем logits в предсказанные классы
            diagnosis_preds = torch.argmax(diagnosis_logits, dim=-1).cpu().numpy()
            doctor_preds = torch.argmax(doctor_logits, dim=-1).cpu().numpy()

            # Сохраняем предсказания и истинные метки
            all_diagnosis_preds.extend(diagnosis_preds)
            all_doctor_preds.extend(doctor_preds)
            all_diagnosis_labels.append(labels[0].cpu().numpy())
            all_doctor_labels.append(labels[1].cpu().numpy())

    # Вычисляем метрики для диагнозов
    diagnosis_accuracy = accuracy_score(all_diagnosis_labels, all_diagnosis_preds)
    diagnosis_f1 = f1_score(all_diagnosis_labels, all_diagnosis_preds, average="weighted")
    diagnosis_precision = precision_score(all_diagnosis_labels, all_diagnosis_preds, average="weighted")
    diagnosis_recall = recall_score(all_diagnosis_labels, all_diagnosis_preds, average="weighted")

    # Вычисляем метрики для врачей
    doctor_accuracy = accuracy_score(all_doctor_labels, all_doctor_preds)
    doctor_f1 = f1_score(all_doctor_labels, all_doctor_preds, average="weighted")
    doctor_precision = precision_score(all_doctor_labels, all_doctor_preds, average="weighted")
    doctor_recall = recall_score(all_doctor_labels, all_doctor_preds, average="weighted")

    # Возвращаем результаты
    return {
        "diagnosis_accuracy": diagnosis_accuracy,
        "diagnosis_f1": diagnosis_f1,
        "diagnosis_precision": diagnosis_precision,
        "diagnosis_recall": diagnosis_recall,
        "doctor_accuracy": doctor_accuracy,
        "doctor_f1": doctor_f1,
        "doctor_precision": doctor_precision,
        "doctor_recall": doctor_recall,
    }

# Путь к модели и токенизатору
model_path = "saved_model/"
num_diagnosis_labels = 10  # Количество классов для диагнозов
num_doctor_labels = 5  # Количество классов для врачей

# Загрузка модели
model = MultiTaskBERT.from_pretrained(model_path, num_diagnosis_labels=num_diagnosis_labels, num_doctor_labels=num_doctor_labels)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Путь к тестовому CSV-файлу
test_csv_path = "test_final.csv"

# Преобразование CSV в Dataset
test_dataset = csv_to_dataset(test_csv_path, tokenizer)

# Устройство для вычислений
device = "cuda" if torch.cuda.is_available() else "cpu"

# Оценка модели
metrics = evaluate_model(model, test_dataset, tokenizer, device=device)

# Вывод результатов
print("Diagnosis Metrics:")
print(f"Accuracy: {metrics['diagnosis_accuracy']:.4f}")
print(f"F1 Score: {metrics['diagnosis_f1']:.4f}")
print(f"Precision: {metrics['diagnosis_precision']:.4f}")
print(f"Recall: {metrics['diagnosis_recall']:.4f}")

print("\nDoctor Metrics:")
print(f"Accuracy: {metrics['doctor_accuracy']:.4f}")
print(f"F1 Score: {metrics['doctor_f1']:.4f}")
print(f"Precision: {metrics['doctor_precision']:.4f}")
print(f"Recall: {metrics['doctor_recall']:.4f}")