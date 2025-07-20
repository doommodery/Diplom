import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertPreTrainedModel, BertModel, Trainer, TrainingArguments, BertTokenizer, AutoConfig
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch.nn.functional as F
import joblib
import os
import logging
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiTaskBERT(BertPreTrainedModel):
    def __init__(self, config, num_diagnosis_labels, num_doctor_labels, num_additional_features):
        super().__init__(config)
        self.num_diagnosis_labels = num_diagnosis_labels
        self.num_doctor_labels = num_doctor_labels
        self.num_additional_features = num_additional_features

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_history = {
            'training_start': datetime.now().isoformat(),
            'steps': [],
            'validation': [],
            'test_metrics': None
        }

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        if len(outputs) == 3:
            loss, logits_diagnosis, logits_doctor = outputs
        else:
            logits_diagnosis, logits_doctor = outputs
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits_diagnosis, labels[:, 0]) + loss_fct(logits_doctor, labels[:, 1])

        if self.state.global_step % self.args.logging_steps == 0:
            preds_diagnosis = torch.argmax(logits_diagnosis, dim=-1)
            accuracy_diagnosis = (preds_diagnosis == labels[:, 0]).float().mean()

            preds_doctor = torch.argmax(logits_doctor, dim=-1)
            accuracy_doctor = (preds_doctor == labels[:, 1]).float().mean()

            step_metrics = {
                'step': self.state.global_step,
                'loss': loss.item(),
                'diagnosis_accuracy': accuracy_diagnosis.item(),
                'doctor_accuracy': accuracy_doctor.item(),
                'timestamp': datetime.now().isoformat()
            }
            self.training_history['steps'].append(step_metrics)

        return (loss, (logits_diagnosis, logits_doctor)) if return_outputs else loss

    def compute_metrics(self, eval_pred):
        logits_diagnosis, logits_doctor = eval_pred.predictions
        labels_diagnosis = eval_pred.label_ids[:, 0]
        labels_doctor = eval_pred.label_ids[:, 1]

        preds_diagnosis = np.argmax(logits_diagnosis, axis=-1)
        preds_doctor = np.argmax(logits_doctor, axis=-1)

        metrics = {
            'eval_loss': None,
            'eval_diagnosis_accuracy': accuracy_score(labels_diagnosis, preds_diagnosis),
            'eval_doctor_accuracy': accuracy_score(labels_doctor, preds_doctor),
            'eval_diagnosis_f1': f1_score(labels_diagnosis, preds_diagnosis, average='weighted'),
            'eval_doctor_f1': f1_score(labels_doctor, preds_doctor, average='weighted')
        }

        eval_metrics = {
            'step': self.state.global_step,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        self.training_history['validation'].append(eval_metrics)

        return metrics

    def save_training_history(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        history_file = os.path.join(output_dir, 'training_history.json')

        self.training_history['training_params'] = {
            'batch_size': self.args.per_device_train_batch_size,
            'epochs': self.args.num_train_epochs,
            'learning_rate': self.args.learning_rate,
            'weight_decay': self.args.weight_decay,
            'total_steps': self.state.max_steps
        }

        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        logger.info(f"История обучения сохранена в {history_file}")

def preprocess_data(df, tokenizer, label_encoders=None, chronic_encoder=None, age_scaler=None, temp_scaler=None):
    df = df.copy()

    if age_scaler is None:
        age_scaler = StandardScaler()
        df['age_normalized'] = age_scaler.fit_transform(df[['Возраст']])
    else:
        df['age_normalized'] = age_scaler.transform(df[['Возраст']])
    
    if temp_scaler is None:
        temp_scaler = StandardScaler()
        df['temp_normalized'] = temp_scaler.fit_transform(df[['Температура (°C)']].fillna(0))
    else:
        df['temp_normalized'] = temp_scaler.transform(df[['Температура (°C)']].fillna(0))

    df['Хронические состояния'] = df['Хронические состояния'].fillna('-')
    chronic_list = df['Хронические состояния'].apply(lambda x: [s.strip() for s in x.split(',')] if x != '-' else [])

    if chronic_encoder is None:
        chronic_encoder = MultiLabelBinarizer()
        chronic_features = chronic_encoder.fit_transform(chronic_list)
    else:
        chronic_features = chronic_encoder.transform(chronic_list)

    additional_features = np.hstack([
        df[['age_normalized', 'temp_normalized']].values,
        chronic_features
    ]).astype(np.float32)

    df['Рекомендуемый врач'] = df['Рекомендуемый врач'].fillna('-').replace('-', '(нет врача)')

    if label_encoders is None:
        label_encoder_diagnosis = LabelEncoder()
        label_encoder_doctor = LabelEncoder()
        df['Рекомендация'] = label_encoder_diagnosis.fit_transform(df['Рекомендация'])
        df['Рекомендуемый врач'] = label_encoder_doctor.fit_transform(df['Рекомендуемый врач'])
    else:
        label_encoder_diagnosis, label_encoder_doctor = label_encoders
        df['Рекомендация'] = label_encoder_diagnosis.transform(df['Рекомендация'])
        df['Рекомендуемый врач'] = label_encoder_doctor.transform(df['Рекомендуемый врач'])

    encodings = tokenizer(df['Состояние здоровья'].tolist(), padding=True, truncation=True, return_tensors="pt")

    return (
        encodings,
        additional_features,
        df['Рекомендация'].tolist(),
        df['Рекомендуемый врач'].tolist(),
        (label_encoder_diagnosis, label_encoder_doctor),
        chronic_encoder,
        age_scaler,
        temp_scaler
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

def save_components(model, tokenizer, label_encoders, config, output_dir, chronic_encoder, age_scaler, temp_scaler):
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    config.save_pretrained(output_dir)

    joblib.dump(label_encoders[0], os.path.join(output_dir, 'diagnosis_encoder.pkl'))
    joblib.dump(label_encoders[1], os.path.join(output_dir, 'doctor_encoder.pkl'))
    joblib.dump(chronic_encoder, os.path.join(output_dir, 'chronic_encoder.pkl'))
    joblib.dump(age_scaler, os.path.join(output_dir, 'age_scaler.pkl'))
    joblib.dump(temp_scaler, os.path.join(output_dir, 'temp_scaler.pkl'))

    logger.info(f"Все компоненты модели сохранены в {output_dir}")

def main():
    logger.info("Загрузка токенизатора...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    logger.info("Загрузка данных из medical_dataset.csv...")
    full_data = pd.read_csv("medical_dataset.csv")

    logger.info("Разделение данных на train/val/test...")
    train_val_data = full_data.iloc[:-15000]
    test_data = full_data.iloc[-15000:]
    train_data, val_data = train_test_split(train_val_data, test_size=0.1, random_state=42)

    logger.info("Предобработка тренировочных данных...")
    (train_encodings, train_features, train_diag_labels, train_doc_labels, 
     label_encoders, chronic_encoder, age_scaler, temp_scaler) = preprocess_data(train_data, tokenizer)

    logger.info("Предобработка валидационных данных...")
    (val_encodings, val_features, val_diag_labels, val_doc_labels, 
     _, _, _, _) = preprocess_data(val_data, tokenizer, label_encoders, chronic_encoder, age_scaler, temp_scaler)

    logger.info("Предобработка тестовых данных...")
    (test_encodings, test_features, test_diag_labels, test_doc_labels, 
     _, _, _, _) = preprocess_data(test_data, tokenizer, label_encoders, chronic_encoder, age_scaler, temp_scaler)

    logger.info("Создание датасетов...")
    train_dataset = HealthDataset(train_encodings, train_features, train_diag_labels, train_doc_labels)
    val_dataset = HealthDataset(val_encodings, val_features, val_diag_labels, val_doc_labels)
    test_dataset = HealthDataset(test_encodings, test_features, test_diag_labels, test_doc_labels)

    logger.info("Создание конфигурации и модели...")
    config = AutoConfig.from_pretrained('bert-base-uncased')
    model = MultiTaskBERT.from_pretrained(
        'bert-base-uncased',
        config=config,
        num_diagnosis_labels=len(label_encoders[0].classes_),
        num_doctor_labels=len(label_encoders[1].classes_),
        num_additional_features=train_features.shape[1]
    )

    logger.info("Установка параметров обучения...")
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="steps",
        eval_steps=2000,
        save_strategy="steps",
        save_steps=2000,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        report_to="none",
        save_total_limit=1,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=5e-5,  
    )

    logger.info("Инициализация кастомного Trainer...")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    logger.info("Запуск обучения...")
    trainer.train()
    logger.info("Обучение завершено.")

    logger.info("Сохранение истории обучения...")
    trainer.save_training_history(training_args.output_dir)

    logger.info("Оценка модели на тестовом наборе...")
    test_metrics = evaluate_model(model, test_dataset)
    trainer.training_history['test_metrics'] = test_metrics
    trainer.save_training_history(training_args.output_dir)

    logger.info("Сохранение компонентов модели...")
    config.num_diagnosis_labels = len(label_encoders[0].classes_)
    config.num_doctor_labels = len(label_encoders[1].classes_)
    config.num_additional_features = train_features.shape[1]
    save_components(model, tokenizer, label_encoders, config, "./saved_model", chronic_encoder, age_scaler, temp_scaler)

    logger.info("Результаты тестирования:")
    for task in ['diagnosis', 'doctor']:
        logger.info(f"{task.capitalize()} Metrics:")
        logger.info(f"Accuracy: {test_metrics[f'{task}_accuracy']:.4f}")
        logger.info(f"F1: {test_metrics[f'{task}_f1']:.4f}")
        logger.info(f"Precision: {test_metrics[f'{task}_precision']:.4f}")
        logger.info(f"Recall: {test_metrics[f'{task}_recall']:.4f}")

if __name__ == "__main__":
    main()