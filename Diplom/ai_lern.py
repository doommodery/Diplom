import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertPreTrainedModel, BertModel, Trainer, TrainingArguments, BertTokenizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F 

# Определение многозадачной модели
class MultiTaskBERT(BertPreTrainedModel):
    def __init__(self, config, num_diagnosis_labels, num_doctor_labels, num_additional_features):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # Классификаторы с учетом дополнительных признаков
        self.diagnosis_classifier = nn.Linear(config.hidden_size + num_additional_features, num_diagnosis_labels)
        self.doctor_classifier = nn.Linear(config.hidden_size + num_additional_features, num_doctor_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, additional_features=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = self.dropout(outputs.pooler_output)

        print(f"pooled_output shape: {pooled_output.shape}")  # Debug
        print(f"additional_features shape: {additional_features.shape}")  # Debug

        if additional_features is not None:
            expected_size = 28351  # Или реальный максимум из твоих данных
            if additional_features.shape[1] != expected_size:
                additional_features = F.pad(additional_features, (0, expected_size - additional_features.shape[1]))

            pooled_output = torch.cat([pooled_output, additional_features], dim=1)

        print(f"Concatenated pooled_output shape: {pooled_output.shape}")  # Debug

        diagnosis_logits = self.diagnosis_classifier(pooled_output)
        doctor_logits = self.doctor_classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            diagnosis_loss = loss_fct(diagnosis_logits, labels[:, 0])
            doctor_loss = loss_fct(doctor_logits, labels[:, 1])
            loss = diagnosis_loss + doctor_loss
            return loss, diagnosis_logits, doctor_logits

        return diagnosis_logits, doctor_logits


# Создание класса датасета
class HealthDataset(Dataset):
    def __init__(self, encodings, additional_features, diagnosis_labels, doctor_labels):
        self.encodings = encodings
        self.additional_features = additional_features
        self.diagnosis_labels = diagnosis_labels
        self.doctor_labels = doctor_labels

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx].clone().detach(),
            'attention_mask': self.encodings['attention_mask'][idx].clone().detach(),
            'additional_features': torch.tensor(self.additional_features[idx], dtype=torch.float),
            'labels': torch.tensor([self.diagnosis_labels[idx], self.doctor_labels[idx]], dtype=torch.long)
        }
        return item


# Кастомный тренер
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if 'labels' not in inputs:
            print("Ошибка: отсутствуют метки в inputs")
            return None

        labels = inputs.pop("labels")
        diagnosis_labels = labels[:, 0]
        doctor_labels = labels[:, 1]

        outputs = model(**inputs)

        if len(outputs) == 3:  # Если модель вернула loss
            loss = outputs[0]
            logits_diagnosis, logits_doctor = outputs[1], outputs[2]
        else:
            logits_diagnosis, logits_doctor = outputs  # Распаковываем корректно
            loss_fct = nn.CrossEntropyLoss()
            loss_diagnosis = loss_fct(logits_diagnosis, diagnosis_labels)
            loss_doctor = loss_fct(logits_doctor, doctor_labels)
            loss = loss_diagnosis + loss_doctor

        if return_outputs:
            return loss, (logits_diagnosis, logits_doctor)
        return loss


# Функция предобработки данных
def preprocess_data(file_path, tokenizer):
    df = pd.read_csv(file_path)

    # Преобразуем текстовые метки в числовые
    label_encoder_diagnosis = LabelEncoder()
    df['Метка (Диагноз)'] = label_encoder_diagnosis.fit_transform(df['Метка (Диагноз)'])

    label_encoder_doctor = LabelEncoder()
    df['Рекомендуемый врач'] = label_encoder_doctor.fit_transform(df['Рекомендуемый врач'])

    # Токенизация текста
    encodings = tokenizer(df['Состояние здоровья'].tolist(), padding=True, truncation=True, return_tensors="pt")

    # Обработка дополнительных признаков
    numeric_features = ['Возраст']
    categorical_features = ['Хронические состояния', 'Погодные условия']

    # Препроцессинг для числовых и категориальных признаков
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Применяем предобработку ко всему датасету
    additional_features = preprocessor.fit_transform(df[numeric_features + categorical_features]).toarray()

    return encodings, additional_features, df['Метка (Диагноз)'].tolist(), df['Рекомендуемый врач'].tolist()


# Разделение основного датасета на тренировочный, валидационный и тестовый
full_data = pd.read_csv("medical_data.csv")
train_data = full_data.iloc[:-2000]  # Все кроме последних 2000 строк
val_data = train_data.sample(frac=0.1, random_state=42)  # 10% данных для валидации
train_data = train_data.drop(val_data.index)

# Сохранение разделенных датасетов
train_data.to_csv("train.csv", index=False)
val_data.to_csv("validation.csv", index=False)
full_data.iloc[-2000:].to_csv("test_final.csv", index=False)  # Последние 2000 строк

# Инициализация токенизатора
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Предобработка данных
train_encodings, train_additional_features, train_diagnosis_labels, train_doctor_labels = preprocess_data("train.csv", tokenizer)
eval_encodings, eval_additional_features, eval_diagnosis_labels, eval_doctor_labels = preprocess_data("validation.csv", tokenizer)

# Проверка данных перед обучением
num_diagnosis_labels = len(set(train_diagnosis_labels))
num_doctor_labels = len(set(train_doctor_labels))
num_additional_features = train_additional_features.shape[1]  # Количество дополнительных признаков

# Инициализация модели
model = MultiTaskBERT.from_pretrained('bert-base-uncased', num_diagnosis_labels=num_diagnosis_labels, num_doctor_labels=num_doctor_labels, num_additional_features=num_additional_features)

# Настройки тренировки
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Создание экземпляра Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=HealthDataset(train_encodings, train_additional_features, train_diagnosis_labels, train_doctor_labels),
    eval_dataset=HealthDataset(eval_encodings, eval_additional_features, eval_diagnosis_labels, eval_doctor_labels)
)

# Запуск тренировки
trainer.train()

# Сохранение модели и токенизатора
model.save_pretrained("./saved_model/")
tokenizer.save_pretrained("./saved_model/")

# Финальное тестирование
test_encodings, test_additional_features, test_diagnosis_labels, test_doctor_labels = preprocess_data("test_final.csv", tokenizer)
test_dataset = HealthDataset(test_encodings, test_additional_features, test_diagnosis_labels, test_doctor_labels)

# Оценка модели
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def evaluate_model(model, dataset, tokenizer, device="cuda"):
    model.to(device)
    model.eval()  # Переводим модель в режим оценки

    all_diagnosis_preds = []
    all_doctor_preds = []
    all_diagnosis_labels = []
    all_doctor_labels = []

    with torch.no_grad():  # Отключаем вычисление градиентов
        for item in dataset:
            input_ids = item['input_ids'].to(device)
            attention_mask = item['attention_mask'].to(device)
            additional_features = item['additional_features'].to(device)
            labels = item['labels'].to(device)

            outputs = model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0), additional_features=additional_features.unsqueeze(0))
            diagnosis_logits, doctor_logits = outputs[0], outputs[1]

            diagnosis_preds = torch.argmax(diagnosis_logits, dim=-1).cpu().numpy()
            doctor_preds = torch.argmax(doctor_logits, dim=-1).cpu().numpy()

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


# Оценка модели
device = "cuda" if torch.cuda.is_available() else "cpu"
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