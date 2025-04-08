import torch
import joblib
import logging
from transformers import BertTokenizer, AutoConfig, BertModel
from safetensors.torch import load_file
from torch import nn
import pandas as pd

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
        self.preprocessor = None
        self.diagnosis_encoder = None
        self.doctor_encoder = None
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
            
            logger.info("Загрузка предобработчиков...")
            self.preprocessor = joblib.load(f'{model_path}preprocessor.pkl')
            self.diagnosis_encoder = joblib.load(f'{model_path}diagnosis_encoder.pkl')
            self.doctor_encoder = joblib.load(f'{model_path}doctor_encoder.pkl')
            
            # Тестовый DataFrame для определения размерности
            test_df = pd.DataFrame({
                'Возраст': [30],
                'Хронические состояния': ['нет'],
                'Погодные условия': ['ясно']
            })
            num_additional_features = self.preprocessor.transform(test_df).shape[1]
            
            # Загрузка модели
            config = AutoConfig.from_pretrained(model_path)
            self.model = MultiTaskBERT(
                config=config,
                num_diagnosis_labels=len(self.diagnosis_encoder.classes_),
                num_doctor_labels=len(self.doctor_encoder.classes_),
                num_additional_features=num_additional_features
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