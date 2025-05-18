import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertPreTrainedModel, BertModel, Trainer, TrainingArguments, BertTokenizer, AutoConfig
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch.nn.functional as F
import joblib
import os
import logging
from sklearn.model_selection import train_test_split

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiTaskBERT(BertPreTrainedModel):
    def __init__(self, config, num_diagnosis_labels, num_doctor_labels, num_additional_features):
        super().__init__(config)
        self.num_diagnosis_labels = num_diagnosis_labels
        self.num_doctor_labels = num_doctor_labels
        self.num_additional_features = num_additional_features
        
        # Сохраняем параметры в конфиг для последующей загрузки
        config.num_diagnosis_labels = num_diagnosis_labels
        config.num_doctor_labels = num_doctor_labels
        config.num_additional_features = num_additional_features
        
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.diagnosis_classifier = nn.Linear(config.hidden_size + num_additional_features, num_diagnosis_labels)
        self.doctor_classifier = nn.Linear(config.hidden_size + num_additional_features, num_doctor_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, additional_features=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = self.dropout(outputs.pooler_output)

        if additional_features is not None:
            if additional_features.shape[1] != self.num_additional_features:
                if additional_features.shape[1] < self.num_additional_features:
                    pad_size = self.num_additional_features - additional_features.shape[1]
                    additional_features = F.pad(additional_features, (0, pad_size))
                else:
                    additional_features = additional_features[:, :self.num_additional_features]
            
            pooled_output = torch.cat([pooled_output, additional_features], dim=1)

        diagnosis_logits = self.diagnosis_classifier(pooled_output)
        doctor_logits = self.doctor_classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            diagnosis_loss = loss_fct(diagnosis_logits, labels[:, 0])
            doctor_loss = loss_fct(doctor_logits, labels[:, 1])
            loss = diagnosis_loss + doctor_loss
            return loss, diagnosis_logits, doctor_logits

        return diagnosis_logits, doctor_logits

class HealthDataset(Dataset):
    def __init__(self, encodings, additional_features, diagnosis_labels, doctor_labels):
        self.encodings = encodings
        self.additional_features = additional_features
        self.diagnosis_labels = diagnosis_labels
        self.doctor_labels = doctor_labels

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx].clone().detach(),
            'attention_mask': self.encodings['attention_mask'][idx].clone().detach(),
            'additional_features': torch.tensor(self.additional_features[idx], dtype=torch.float),
            'labels': torch.tensor([self.diagnosis_labels[idx], self.doctor_labels[idx]], dtype=torch.long)
        }

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if 'labels' not in inputs:
            logger.error("Отсутствуют метки в inputs")
            return None

        labels = inputs.pop("labels")
        outputs = model(**inputs)

        if len(outputs) == 3:
            loss, logits_diagnosis, logits_doctor = outputs
        else:
            logits_diagnosis, logits_doctor = outputs
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits_diagnosis, labels[:, 0]) + loss_fct(logits_doctor, labels[:, 1])

        return (loss, (logits_diagnosis, logits_doctor)) if return_outputs else loss

def preprocess_data(df, tokenizer, preprocessor=None, label_encoders=None):
    # Инициализация или использование существующих энкодеров
    if label_encoders is None:
        label_encoder_diagnosis = LabelEncoder()
        label_encoder_doctor = LabelEncoder()
        df['Рекомендация'] = label_encoder_diagnosis.fit_transform(df['Рекомендация'])
        df['Рекомендуемый врач'] = label_encoder_doctor.fit_transform(df['Рекомендуемый врач'])
    else:
        label_encoder_diagnosis, label_encoder_doctor = label_encoders
        df['Рекомендация'] = label_encoder_diagnosis.transform(df['Рекомендация'])
        df['Рекомендуемый врач'] = label_encoder_doctor.transform(df['Рекомендуемый врач'])

    # Токенизация текста
    encodings = tokenizer(df['Состояние здоровья'].tolist(), padding=True, truncation=True, return_tensors="pt")

    # Обработка дополнительных признаков
    numeric_features = ['Возраст']
    categorical_features = ['Хронические состояния', 'Погодные условия']

    if preprocessor is None:
        preprocessor = ColumnTransformer([ 
            ('num', StandardScaler(), numeric_features), 
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        additional_features = preprocessor.fit_transform(df[numeric_features + categorical_features])
    else:
        additional_features = preprocessor.transform(df[numeric_features + categorical_features])

    return (
        encodings, 
        additional_features.toarray(), 
        df['Рекомендация'].tolist(), 
        df['Рекомендуемый врач'].tolist(),
        preprocessor,
        (label_encoder_diagnosis, label_encoder_doctor)
    )

def evaluate_model(model, dataset, device="cuda"):
    model.to(device).eval()
    all_preds = {'diagnosis': [], 'doctor': []}
    all_labels = {'diagnosis': [], 'doctor': []}

    with torch.no_grad():
        for item in dataset:
            inputs = {k: v.unsqueeze(0).to(device) for k, v in item.items() if k != 'labels'}
            labels = item['labels'].to(device)
            
            logits_diagnosis, logits_doctor = model(**inputs)
            
            all_preds['diagnosis'].extend(torch.argmax(logits_diagnosis, -1).cpu().numpy())
            all_preds['doctor'].extend(torch.argmax(logits_doctor, -1).cpu().numpy())
            all_labels['diagnosis'].append(labels[0].cpu().numpy())
            all_labels['doctor'].append(labels[1].cpu().numpy())

    metrics = {}
    for task in ['diagnosis', 'doctor']:
        metrics[f'{task}_accuracy'] = accuracy_score(all_labels[task], all_preds[task])
        metrics[f'{task}_f1'] = f1_score(all_labels[task], all_preds[task], average='weighted')
        metrics[f'{task}_precision'] = precision_score(all_labels[task], all_preds[task], average='weighted')
        metrics[f'{task}_recall'] = recall_score(all_labels[task], all_preds[task], average='weighted')
    
    return metrics

def save_components(model, tokenizer, preprocessor, label_encoders, config, output_dir):
    """Сохраняет все компоненты модели"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Сохраняем модель и токенизатор
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Сохраняем конфиг (уже содержит все необходимые параметры)
    config.save_pretrained(output_dir)
    
    # Сохраняем препроцессор и энкодеры
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.pkl'))
    joblib.dump(label_encoders[0], os.path.join(output_dir, 'diagnosis_encoder.pkl'))
    joblib.dump(label_encoders[1], os.path.join(output_dir, 'doctor_encoder.pkl'))
    
    # Сохраняем информацию о признаках для ясности
    feature_info = {
        'numeric_features': ['Возраст'],
        'categorical_features': ['Хронические состояния', 'Погодные условия'],
        'text_feature': 'Состояние здоровья'
    }
    joblib.dump(feature_info, os.path.join(output_dir, 'feature_info.pkl'))
    
    logger.info(f"Все компоненты модели сохранены в {output_dir}")

def main():
    # Инициализация
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Загрузка и разделение данных
    full_data = pd.read_csv("medical_dataset.csv ")
    
    # Разделение данных: последние 2000 строк - тестовый набор
    train_val_data = full_data.iloc[:-2000]
    test_data = full_data.iloc[-2000:]
    
    # Разделение train_val на train и validation (90%/10%)
    train_data, val_data = train_test_split(train_val_data, test_size=0.1, random_state=42)
    
    # Предобработка данных
    (train_encodings, train_features, train_diag_labels, 
     train_doc_labels, preprocessor, label_encoders) = preprocess_data(train_data, tokenizer)
    
    (val_encodings, val_features, val_diag_labels, 
     val_doc_labels, _, _) = preprocess_data(val_data, tokenizer, preprocessor, label_encoders)
    
    (test_encodings, test_features, test_diag_labels, 
     test_doc_labels, _, _) = preprocess_data(test_data, tokenizer, preprocessor, label_encoders)

    # Создание датасетов
    train_dataset = HealthDataset(train_encodings, train_features, train_diag_labels, train_doc_labels)
    val_dataset = HealthDataset(val_encodings, val_features, val_diag_labels, val_doc_labels)
    test_dataset = HealthDataset(test_encodings, test_features, test_diag_labels, test_doc_labels)

    # Инициализация модели
    config = AutoConfig.from_pretrained('bert-base-uncased')
    model = MultiTaskBERT.from_pretrained(
        'bert-base-uncased',
        config=config,
        num_diagnosis_labels=len(label_encoders[0].classes_),
        num_doctor_labels=len(label_encoders[1].classes_),
        num_additional_features=train_features.shape[1]
    )

    # Настройка обучения
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    # Обучение
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train()

    # Сохранение всех компонентов модели
    print("Размерность дополнительных признаков:", train_features.shape[1])
    print("Количество классов диагнозов:", len(label_encoders[0].classes_))
    print("Количество классов врачей:", len(label_encoders[1].classes_))
    config.num_diagnosis_labels = len(label_encoders[0].classes_)
    config.num_doctor_labels = len(label_encoders[1].classes_)
    config.num_additional_features = train_features.shape[1]
    config.save_pretrained("./saved_model")  # Явно сохраняем обновленный конфиг
    save_components(
        model=model,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        label_encoders=label_encoders,
        config=config,
        output_dir="./saved_model"
    )

    # Оценка
    metrics = evaluate_model(model, test_dataset)
    for task in ['diagnosis', 'doctor']:
        print(f"\n{task.capitalize()} Metrics:")
        print(f"Accuracy: {metrics[f'{task}_accuracy']:.4f}")
        print(f"F1: {metrics[f'{task}_f1']:.4f}")
        print(f"Precision: {metrics[f'{task}_precision']:.4f}")
        print(f"Recall: {metrics[f'{task}_recall']:.4f}")

if __name__ == "__main__":
    main()  