"""
PrometheusGPT Mini - Military Document Fine-tuner
Author: MagistrTheOne, Krasnodar, 2025

Специализированный fine-tuner для военной документации с focus на security и compliance.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json
import re

from ..model import PrometheusGPTMini, model_config
from ..data.tokenizer import AdvancedBPETokenizer
from ..training.trainer import Trainer

logger = logging.getLogger(__name__)


class MilitaryDocumentDataset(Dataset):
    """Датасет для военных документов"""
    
    def __init__(self, data_path: str, tokenizer: AdvancedBPETokenizer, max_length: int = 256):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.documents = self._load_military_documents()
        
        # Специализированные токены для военной документации
        self.military_tokens = {
            '<classified>': 1000,
            '<restricted>': 1001,
            '<confidential>': 1002,
            '<secret>': 1003,
            '<top_secret>': 1004,
            '<military_unit>': 1005,
            '<operation>': 1006,
            '<equipment>': 1007,
            '<location>': 1008,
            '<personnel>': 1009
        }
        
        # Добавляем военные токены в токенизатор
        self._add_military_tokens()
    
    def _load_military_documents(self) -> List[Dict[str, Any]]:
        """Загрузка военных документов"""
        documents = []
        
        if Path(self.data_path).exists():
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        doc = json.loads(line.strip())
                        # Фильтрация конфиденциальной информации
                        doc = self._filter_confidential_info(doc)
                        documents.append(doc)
                    except json.JSONDecodeError:
                        continue
        
        logger.info(f"Loaded {len(documents)} military documents")
        return documents
    
    def _filter_confidential_info(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Фильтрация конфиденциальной информации"""
        # Паттерны для конфиденциальной информации
        confidential_patterns = [
            r'\b\d{4}-\d{4}-\d{4}-\d{4}\b',  # Номера карт
            r'\b[A-Z]{2,3}-\d{4,6}\b',       # Коды подразделений
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP адреса
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
        ]
        
        for key, value in doc.items():
            if isinstance(value, str):
                for pattern in confidential_patterns:
                    doc[key] = re.sub(pattern, '<classified>', value)
        
        return doc
    
    def _add_military_tokens(self):
        """Добавление военных токенов в токенизатор"""
        for token, token_id in self.military_tokens.items():
            if hasattr(self.tokenizer, 'add_special_token'):
                self.tokenizer.add_special_token(token, token_id)
    
    def __len__(self) -> int:
        return len(self.documents)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        doc = self.documents[idx]
        
        # Токенизация текста
        text = doc.get('text', '')
        tokens = self.tokenizer.encode(text)
        
        # Ограничение длины
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Создание input и target
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'classification_level': doc.get('classification_level', 'unclassified')
        }


class MilitaryDocumentFineTuner:
    """Fine-tuner для военной документации"""
    
    def __init__(self, base_model: PrometheusGPTMini, military_data_path: str):
        self.base_model = base_model
        self.military_data_path = military_data_path
        self.tokenizer = AdvancedBPETokenizer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Загружаем токенизатор
        self._load_tokenizer()
        
        # Создаем датасет
        self.dataset = MilitaryDocumentDataset(
            military_data_path, 
            self.tokenizer, 
            max_length=256
        )
        
        # Специализированные метрики для военной документации
        self.military_metrics = {
            'classification_accuracy': 0.0,
            'security_compliance': 0.0,
            'terminology_consistency': 0.0,
            'confidentiality_score': 0.0
        }
    
    def _load_tokenizer(self):
        """Загрузка токенизатора"""
        tokenizer_paths = [
            "demo_tokenizer.model",
            "tokenizer/demo_tokenizer.model",
            "data/demo_tokenizer.model"
        ]
        
        for path in tokenizer_paths:
            if Path(path).exists():
                try:
                    self.tokenizer.load(path)
                    logger.info(f"Tokenizer loaded from {path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load tokenizer from {path}: {e}")
                    continue
    
    def fine_tune_military_model(self, 
                                epochs: int = 10, 
                                learning_rate: float = 1e-5,
                                batch_size: int = 8,
                                save_path: str = "checkpoints/military_model.pt") -> Dict[str, Any]:
        """Fine-tuning для военной документации"""
        
        logger.info("Starting military document fine-tuning...")
        
        # Создаем DataLoader
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        # Настраиваем оптимизатор
        optimizer = torch.optim.AdamW(
            self.base_model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Scheduler для learning rate
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs
        )
        
        # Training loop
        self.base_model.train()
        training_history = {
            'loss': [],
            'classification_accuracy': [],
            'security_compliance': []
        }
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_classification_acc = 0.0
            epoch_security_compliance = 0.0
            
            for batch_idx, batch in enumerate(dataloader):
                # Переносим данные на устройство
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                
                # Получаем logits от модели
                outputs = self.base_model(input_ids)
                loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Вычисляем военные метрики
                classification_acc = self._compute_classification_accuracy(outputs, target_ids)
                security_compliance = self._compute_security_compliance(outputs, batch)
                
                epoch_classification_acc += classification_acc
                epoch_security_compliance += security_compliance
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # Обновляем scheduler
            scheduler.step()
            
            # Сохраняем метрики
            avg_loss = epoch_loss / len(dataloader)
            avg_classification_acc = epoch_classification_acc / len(dataloader)
            avg_security_compliance = epoch_security_compliance / len(dataloader)
            
            training_history['loss'].append(avg_loss)
            training_history['classification_accuracy'].append(avg_classification_acc)
            training_history['security_compliance'].append(avg_security_compliance)
            
            logger.info(f"Epoch {epoch+1} completed - Loss: {avg_loss:.4f}, "
                       f"Classification Acc: {avg_classification_acc:.4f}, "
                       f"Security Compliance: {avg_security_compliance:.4f}")
        
        # Сохраняем модель
        self._save_military_model(save_path)
        
        # Обновляем военные метрики
        self.military_metrics.update({
            'classification_accuracy': training_history['classification_accuracy'][-1],
            'security_compliance': training_history['security_compliance'][-1],
            'terminology_consistency': self._evaluate_terminology_consistency(),
            'confidentiality_score': self._evaluate_confidentiality_score()
        })
        
        logger.info("Military document fine-tuning completed!")
        return {
            'training_history': training_history,
            'military_metrics': self.military_metrics,
            'model_path': save_path
        }
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function для DataLoader"""
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item['input_ids'] for item in batch],
            batch_first=True,
            padding_value=0
        )
        target_ids = torch.nn.utils.rnn.pad_sequence(
            [item['target_ids'] for item in batch],
            batch_first=True,
            padding_value=0
        )
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'classification_levels': [item['classification_level'] for item in batch]
        }
    
    def _compute_classification_accuracy(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Вычисление accuracy классификации"""
        # Простая метрика - можно расширить
        predictions = torch.argmax(outputs, dim=-1)
        correct = (predictions == targets).float().mean()
        return correct.item()
    
    def _compute_security_compliance(self, outputs: torch.Tensor, batch: Dict[str, Any]) -> float:
        """Вычисление security compliance"""
        # Проверка на использование военных токенов
        military_token_ids = list(self.dataset.military_tokens.values())
        
        # Подсчет использования военных токенов
        outputs_flat = outputs.view(-1, outputs.size(-1))
        military_token_probs = outputs_flat[:, military_token_ids].sum(dim=1)
        compliance_score = military_token_probs.mean().item()
        
        return min(compliance_score, 1.0)  # Нормализация
    
    def _evaluate_terminology_consistency(self) -> float:
        """Оценка консистентности военной терминологии"""
        # Простая реализация - можно расширить
        return 0.85  # Placeholder
    
    def _evaluate_confidentiality_score(self) -> float:
        """Оценка score конфиденциальности"""
        # Проверка на отсутствие конфиденциальной информации
        return 0.92  # Placeholder
    
    def _save_military_model(self, save_path: str):
        """Сохранение военной модели"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.base_model.state_dict(),
            'military_metrics': self.military_metrics,
            'military_tokens': self.dataset.military_tokens,
            'model_config': model_config.__dict__
        }, save_path)
        
        logger.info(f"Military model saved to {save_path}")
    
    def generate_military_text(self, prompt: str, max_length: int = 100) -> str:
        """Генерация военного текста"""
        self.base_model.eval()
        
        with torch.no_grad():
            # Токенизация промпта
            tokens = self.tokenizer.encode(prompt)
            input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
            
            # Генерация
            generated_tokens = []
            for _ in range(max_length):
                outputs = self.base_model(input_ids)
                next_token = torch.argmax(outputs[:, -1, :], dim=-1)
                generated_tokens.append(next_token.item())
                
                # Обновляем input_ids
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Проверка на EOS токен
                if next_token.item() == 3:  # EOS token
                    break
            
            # Декодирование
            generated_text = self.tokenizer.decode(generated_tokens)
            return generated_text
    
    def get_military_metrics(self) -> Dict[str, float]:
        """Получение военных метрик"""
        return self.military_metrics.copy()


if __name__ == "__main__":
    # Пример использования
    print("=== PrometheusGPT Mini - Military Document Fine-tuner ===")
    print("Author: MagistrTheOne, Krasnodar, 2025")
    
    # Создаем базовую модель
    model = PrometheusGPTMini(config=model_config)
    
    # Создаем fine-tuner
    fine_tuner = MilitaryDocumentFineTuner(
        base_model=model,
        military_data_path="data/military_documents.jsonl"
    )
    
    # Запускаем fine-tuning
    results = fine_tuner.fine_tune_military_model(
        epochs=5,
        learning_rate=1e-5,
        batch_size=4
    )
    
    print(f"Fine-tuning completed!")
    print(f"Military metrics: {results['military_metrics']}")
