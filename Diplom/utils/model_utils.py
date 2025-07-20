import torch
import joblib
import logging
from transformers import BertTokenizer, AutoConfig, BertModel
from safetensors.torch import load_file
from torch import nn
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class MultiTaskBERT(nn.Module):
    def __init__(self, config, num_diagnosis_labels, num_doctor_labels, num_additional_features):
        super().__init__()
        self.config = config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.diagnosis_classifier = nn.Linear(config.hidden_size + num_additional_features, num_diagnosis_labels)
        self.doctor_classifier = nn.Linear(config.hidden_size + num_additional_features, num_doctor_labels)

    def forward(self, input_ids, attention_mask=None, additional_features=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        
        if additional_features is not None:
            pooled_output = torch.cat([pooled_output, additional_features], dim=1)
        
        return self.diagnosis_classifier(pooled_output), self.doctor_classifier(pooled_output)

class AIModel:
    _instance = None
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.diagnosis_encoder = None
        self.doctor_encoder = None
        self.feature_info = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_ready = False
    
    @classmethod
    async def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            await cls._instance.load_models()
        return cls._instance
    
    async def load_models(self):
        """Загрузка всех компонентов модели"""
        try:
            model_path = "./saved_model/"
            
            logger.info("Загрузка кодировщиков меток...")
            self.diagnosis_encoder = joblib.load(f'{model_path}diagnosis_encoder.pkl')
            self.doctor_encoder = joblib.load(f'{model_path}doctor_encoder.pkl')
            self.feature_info = joblib.load(f'{model_path}feature_info.pkl')
            
            config = AutoConfig.from_pretrained(model_path)
            self.model = MultiTaskBERT(
                config=config,
                num_diagnosis_labels=len(self.diagnosis_encoder.classes_),
                num_doctor_labels=len(self.doctor_encoder.classes_),
                num_additional_features=len(self.feature_info['age_groups']) + len(self.feature_info['temp_groups'])
            )
            
            state_dict = load_file(f"{model_path}model.safetensors")
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.is_ready = True
            logger.info(f"Модель загружена на {self.device}")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки: {e}", exc_info=True)
            self.is_ready = False
            raise

    def preprocess_input(self, input_data):
        """Подготовка входных данных для модели"""
        df = pd.DataFrame([input_data] if isinstance(input_data, dict) else input_data)
        
        df['age_group'] = pd.cut(df['Возраст'], 
                                bins=[0, 25, 60, 90],
                                labels=self.feature_info['age_groups'])
        
        df['temp_group'] = pd.cut(df['Температура (°C)'], 
                                 bins=[-float('inf'), 5, 18, float('inf')],
                                 labels=self.feature_info['temp_groups'])
        
        age_dummies = pd.get_dummies(df['age_group'], prefix='age')
        temp_dummies = pd.get_dummies(df['temp_group'], prefix='temp')
        
        expected_age_cols = [f'age_{g}' for g in self.feature_info['age_groups']]
        expected_temp_cols = [f'temp_{g}' for g in self.feature_info['temp_groups']]
        
        for col in expected_age_cols:
            if col not in age_dummies:
                age_dummies[col] = 0
        for col in expected_temp_cols:
            if col not in temp_dummies:
                temp_dummies[col] = 0
                
        additional_features = pd.concat([age_dummies, temp_dummies], axis=1)
        additional_features = additional_features[expected_age_cols + expected_temp_cols]
        
        encodings = self.tokenizer(
            df['Состояние здоровья'].tolist(), 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        
        return encodings, additional_features.values.astype(np.float32)