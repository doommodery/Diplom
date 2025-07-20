import torch
import joblib
import logging
import pandas as pd
import numpy as np
from transformers import BertTokenizer, AutoConfig, BertModel
from safetensors.torch import load_file
from torch import nn
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class MultiTaskBERT(nn.Module):
    def __init__(self, config, num_diagnosis_labels: int, num_doctor_labels: int, 
                 num_additional_features: int, dropout_rate: float = 0.1):
        super().__init__()
        self.config = config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout_rate)
        self.diagnosis_classifier = nn.Linear(config.hidden_size + num_additional_features, num_diagnosis_labels)
        self.doctor_classifier = nn.Linear(config.hidden_size + num_additional_features, num_doctor_labels)
        
        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов классификаторов"""
        nn.init.xavier_uniform_(self.diagnosis_classifier.weight)
        nn.init.zeros_(self.diagnosis_classifier.bias)
        nn.init.xavier_uniform_(self.doctor_classifier.weight)
        nn.init.zeros_(self.doctor_classifier.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                additional_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        
        if additional_features is not None:
            # Проверка и нормализация дополнительных признаков
            if additional_features.dim() == 1:
                additional_features = additional_features.unsqueeze(0)
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
        self.confidence_threshold = 0.7
    
    @classmethod
    async def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            await cls._instance.load_models()
        return cls._instance
    
    async def load_models(self):
        """Асинхронная загрузка всех компонентов модели"""
        try:
            model_path = "./saved_model/"
            
            # 1. Загрузка вспомогательных данных
            self.feature_info = joblib.load(f'{model_path}feature_info.pkl')
            self.diagnosis_encoder = joblib.load(f'{model_path}diagnosis_encoder.pkl')
            self.doctor_encoder = joblib.load(f'{model_path}doctor_encoder.pkl')
            
            # Проверка структуры feature_info
            required_keys = ['age_groups', 'temp_groups', 'chronic_conditions', 
                           'num_additional_features', 'feature_names']
            if not all(key in self.feature_info for key in required_keys):
                raise ValueError("Неполная информация о признаках в feature_info.pkl")
            
            # 2. Загрузка конфигурации модели
            config = AutoConfig.from_pretrained(model_path)
            
            # 3. Инициализация модели
            self.model = MultiTaskBERT(
                config=config,
                num_diagnosis_labels=len(self.diagnosis_encoder.classes_),
                num_doctor_labels=len(self.doctor_encoder.classes_),
                num_additional_features=self.feature_info['num_additional_features']
            )
            
            # 4. Загрузка весов модели
            state_dict = load_file(f"{model_path}model.safetensors")
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            # 5. Загрузка токенизатора
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            
            self.is_ready = True
            logger.info(f"Модель успешно загружена. Доп. признаки: {self.feature_info['num_additional_features']}")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {str(e)}", exc_info=True)
            self.is_ready = False
            raise

    def preprocess_input(self, input_data: Dict) -> Tuple[Dict, torch.Tensor]:
        """Подготовка входных данных в соответствии с feature_info"""
        df = pd.DataFrame([input_data] if isinstance(input_data, dict) else input_data)
        
        # 1. Инициализация признаков
        additional_features = []
        
        # 2. Обработка возраста (нормализация)
        age = df['Возраст'].values[0] if 'Возраст' in df else 30
        age_scaler = StandardScaler()
        age_normalized = age_scaler.fit_transform([[age]])[0][0]
        additional_features.append(age_normalized)
        
        # 3. Обработка температуры (нормализация)
        temp = df['Температура (°C)'].values[0] if 'Температура (°C)' in df else 36.6
        temp_scaler = StandardScaler()
        temp_normalized = temp_scaler.fit_transform([[temp]])[0][0]
        additional_features.append(temp_normalized)
        
        # 4. Обработка хронических состояний
        chronic_conditions = df['Хронические состояния'].values[0] if 'Хронические состояния' in df else '-'
        chronic_list = [s.strip() for s in chronic_conditions.split(',')] if chronic_conditions != '-' else []
        
        chronic_encoded = np.zeros(len(self.feature_info['chronic_conditions']))
        for condition in chronic_list:
            if condition in self.feature_info['chronic_conditions']:
                idx = self.feature_info['chronic_conditions'].index(condition)
                chronic_encoded[idx] = 1
        
        additional_features.extend(chronic_encoded)
        
        # 5. Проверка и корректировка размерности
        if len(additional_features) != self.feature_info['num_additional_features']:
            logger.warning(
                f"Корректировка размерности признаков: было {len(additional_features)}, "
                f"требуется {self.feature_info['num_additional_features']}"
            )
            if len(additional_features) < self.feature_info['num_additional_features']:
                additional_features.extend([0.0] * (self.feature_info['num_additional_features'] - len(additional_features)))
            else:
                additional_features = additional_features[:self.feature_info['num_additional_features']]
        
        # 6. Токенизация текста
        encodings = self.tokenizer(
            df['Состояние здоровья'].tolist(),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        return encodings, torch.tensor(additional_features, dtype=torch.float32)

    def get_confidence(self, logits: torch.Tensor) -> float:
        """Вычисляет уверенность модели в предсказании"""
        probabilities = torch.softmax(logits, dim=-1)
        top_prob = torch.max(probabilities).item()
        return top_prob

    def analyze_health(self, input_data: Dict) -> Dict:
        """Основной метод для анализа состояния здоровья"""
        if not self.is_ready:
            raise RuntimeError("Модель не загружена")
            
        try:
            encodings, additional_features = self.preprocess_input(input_data)
            
            # Проверка размерности перед передачей в модель
            if additional_features.shape[0] != self.feature_info['num_additional_features']:
                raise ValueError(
                    f"Несоответствие размерности признаков. "
                    f"Ожидалось {self.feature_info['num_additional_features']}, "
                    f"получено {additional_features.shape[0]}"
                )
            
            inputs = {
                'input_ids': encodings['input_ids'].to(self.device),
                'attention_mask': encodings['attention_mask'].to(self.device),
                'additional_features': additional_features.unsqueeze(0).to(self.device)
            }

            with torch.no_grad():
                diagnosis_logits, doctor_logits = self.model(**inputs)

            # Получаем предсказания и уверенность
            diagnosis_idx = torch.argmax(diagnosis_logits).item()
            doctor_idx = torch.argmax(doctor_logits).item()
            
            diagnosis = self.diagnosis_encoder.inverse_transform([diagnosis_idx])[0]
            doctor = self.doctor_encoder.inverse_transform([doctor_idx])[0]
            
            diagnosis_confidence = self.get_confidence(diagnosis_logits)
            doctor_confidence = self.get_confidence(doctor_logits)
            
            # Проверяем уверенность модели
            if diagnosis_confidence < self.confidence_threshold:
                diagnosis = "Не удалось определить с достаточной уверенностью"
                
            if doctor_confidence < self.confidence_threshold:
                doctor = "Рекомендуется консультация терапевта"

            return {
                'diagnosis': diagnosis,
                'doctor': doctor,
                'confidence': {
                    'diagnosis': diagnosis_confidence,
                    'doctor': doctor_confidence
                },
                'input_features': {
                    'age': input_data.get('Возраст'),
                    'temperature': input_data.get('Температура (°C)')
                }
            }
            
        except Exception as e:
            logger.error(f"Ошибка при анализе: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'diagnosis': 'Не удалось выполнить анализ',
                'doctor': 'Обратитесь к специалисту',
                'confidence': {
                    'diagnosis': 0.0,
                    'doctor': 0.0
                }
            }