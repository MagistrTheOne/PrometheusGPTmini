"""
Подготовка мультиязычных данных для PrometheusGPT Mini
Загрузка и предобработка датасетов для обучения с нуля
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets
from typing import Dict, List, Optional, Tuple
import logging
from config import ModelConfig
from src.tokenizer import MultilingualTokenizer
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultilingualDataset(Dataset):
    """
    Мультиязычный датасет для обучения PrometheusGPT Mini
    Поддерживает параллельные корпуса и монолингвальные данные
    """

    def __init__(self, config: ModelConfig, tokenizer: MultilingualTokenizer,
                 split: str = "train", max_length: Optional[int] = None):
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length or config.max_seq_len

        # Загружаем и подготавливаем данные
        self.data = self._load_and_prepare_data()

        logger.info(f"Dataset {split} loaded with {len(self.data)} samples")

    def _load_and_prepare_data(self) -> List[Dict]:
        """Загрузка и подготовка данных"""
        prepared_data = []

        # Загружаем датасеты для каждого языка
        for lang in self.config.languages:
            lang_data = self._load_language_data(lang, self.split)
            prepared_data.extend(lang_data)

        # Балансировка данных по языкам
        prepared_data = self._balance_languages(prepared_data)

        # Перемешиваем
        np.random.shuffle(prepared_data)

        return prepared_data

    def _load_language_data(self, lang: str, split: str) -> List[Dict]:
        """Загрузка данных для конкретного языка"""
        try:
            if lang == 'en':
                # Wikipedia на английском
                dataset = load_dataset("wikipedia", "20220301.en", split=split)
                texts = [item['text'] for item in dataset]
            elif lang == 'ru':
                # Wikipedia на русском
                dataset = load_dataset("wikipedia", "20220301.ru", split=split)
                texts = [item['text'] for item in dataset]
            else:
                # Для других языков используем CC-100 или аналогичные
                # Пока используем синтетические данные для демонстрации
                texts = self._generate_sample_data(lang, 1000 if split == "train" else 100)

            # Обрабатываем тексты
            processed_data = []
            for text in texts:
                # Токенизация с языковым маркером
                tokens = self.tokenizer.encode(text, lang, add_special_tokens=True)

                # Для обучения на next-token prediction используем causal LM
                if len(tokens) > self.max_length:
                    # Разбиваем длинные тексты на чанки
                    for i in range(0, len(tokens) - self.max_length + 1, self.max_length // 2):
                        chunk = tokens[i:i + self.max_length]
                        if len(chunk) >= 10:  # Минимум 10 токенов
                            processed_data.append({
                                'input_ids': chunk,
                                'attention_mask': [1] * len(chunk),
                                'labels': chunk.copy(),  # Для causal LM labels = input_ids
                                'language': lang
                            })
                else:
                    if len(tokens) >= 10:
                        processed_data.append({
                            'input_ids': tokens,
                            'attention_mask': [1] * len(tokens),
                            'labels': tokens.copy(),
                            'language': lang
                        })

            logger.info(f"Loaded {len(processed_data)} samples for {lang} ({split})")
            return processed_data

        except Exception as e:
            logger.warning(f"Failed to load {lang} data: {e}")
            # Возвращаем пустой список если загрузка не удалась
            return []

    def _generate_sample_data(self, lang: str, num_samples: int) -> List[str]:
        """Генерация примерных данных для языков (для демонстрации)"""
        # В реальном проекте здесь будут загружаться настоящие датасеты
        # Например, для испанского: load_dataset("cc100", lang="es")
        # Для французского: load_dataset("cc100", lang="fr")
        # Для немецкого: load_dataset("cc100", lang="de")

        sample_texts = {
            'es': [
                "La inteligencia artificial está revolucionando el mundo de la tecnología.",
                "El procesamiento del lenguaje natural permite a las máquinas entender el texto humano.",
                "Los modelos de lenguaje grandes pueden generar respuestas coherentes y útiles.",
                "La tokenización es un paso fundamental en el procesamiento de texto.",
                "El aprendizaje automático requiere grandes cantidades de datos de calidad.",
            ],
            'fr': [
                "L'intelligence artificielle révolutionne le monde de la technologie.",
                "Le traitement du langage naturel permet aux machines de comprendre le texte humain.",
                "Les grands modèles de langage peuvent générer des réponses cohérentes et utiles.",
                "La tokenisation est une étape fondamentale dans le traitement du texte.",
                "L'apprentissage automatique nécessite de grandes quantités de données de qualité.",
            ],
            'de': [
                "Künstliche Intelligenz revolutioniert die Welt der Technologie.",
                "Die Verarbeitung natürlicher Sprache ermöglicht Maschinen, menschlichen Text zu verstehen.",
                "Große Sprachmodelle können kohärente und nützliche Antworten generieren.",
                "Die Tokenisierung ist ein grundlegender Schritt bei der Textverarbeitung.",
                "Maschinelles Lernen erfordert große Mengen hochwertiger Daten.",
            ]
        }

        base_texts = sample_texts.get(lang, sample_texts['en'])
        # Генерируем вариации для увеличения размера датасета
        texts = []
        for i in range(num_samples):
            text = base_texts[i % len(base_texts)]
            # Добавляем небольшие вариации
            if i % 3 == 0:
                text = text + " Esto es un ejemplo adicional."
            texts.append(text)

        return texts

    def _balance_languages(self, data: List[Dict]) -> List[Dict]:
        """Балансировка данных по языкам"""
        lang_counts = {}
        for item in data:
            lang = item['language']
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        logger.info(f"Language distribution before balancing: {lang_counts}")

        # Находим минимальное количество сэмплов
        min_samples = min(lang_counts.values()) if lang_counts else 0

        # Ограничиваем каждый язык до min_samples
        balanced_data = []
        lang_samples = {lang: 0 for lang in self.config.languages}

        for item in data:
            lang = item['language']
            if lang_samples[lang] < min_samples:
                balanced_data.append(item)
                lang_samples[lang] += 1

        logger.info(f"Language distribution after balancing: {dict(lang_samples)}")
        return balanced_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(item['labels'], dtype=torch.long),
            'language': item['language']
        }


class MultilingualDataLoader:
    """
    DataLoader с поддержкой мультиязычных данных и кастомной collate функцией
    """

    def __init__(self, config: ModelConfig, tokenizer: MultilingualTokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def get_data_loader(self, split: str = "train", batch_size: Optional[int] = None,
                       shuffle: bool = True) -> DataLoader:
        """
        Создание DataLoader для указанного сплита

        Args:
            split: train/val/test
            batch_size: Размер батча (по умолчанию из config)
            shuffle: Перемешивать ли данные

        Returns:
            DataLoader
        """
        batch_size = batch_size or self.config.batch_size

        dataset = MultilingualDataset(self.config, self.tokenizer, split)

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
            num_workers=0,  # Для Windows лучше 0
            pin_memory=True if torch.cuda.is_available() else False
        )

        return data_loader

    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Кастомная collate функция для padding последовательностей
        """
        # Находим максимальную длину в батче
        max_len = max(len(item['input_ids']) for item in batch)

        # Padding
        pad_token_id = self.tokenizer.special_tokens["<pad>"]

        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            # Padding input_ids
            padded_input_ids = item['input_ids'].tolist()
            while len(padded_input_ids) < max_len:
                padded_input_ids.append(pad_token_id)
            input_ids.append(padded_input_ids)

            # Padding attention_mask
            padded_mask = item['attention_mask'].tolist()
            while len(padded_mask) < max_len:
                padded_mask.append(0)  # 0 для padded токенов
            attention_masks.append(padded_mask)

            # Padding labels
            padded_labels = item['labels'].tolist()
            while len(padded_labels) < max_len:
                padded_labels.append(-100)  # -100 для игнорирования в loss
            labels.append(padded_labels)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


def create_data_loaders(config: ModelConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Создание DataLoader'ов для train/val/test

    Returns:
        Кортеж (train_loader, val_loader, test_loader)
    """
    tokenizer = MultilingualTokenizer(config)
    data_loader = MultilingualDataLoader(config, tokenizer)

    train_loader = data_loader.get_data_loader("train", shuffle=True)
    val_loader = data_loader.get_data_loader("val", shuffle=False)
    test_loader = data_loader.get_data_loader("test", shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Тест датасета
    config = ModelConfig()
    tokenizer = MultilingualTokenizer(config)

    # Создаем датасет
    dataset = MultilingualDataset(config, tokenizer, split="train")

    print(f"Dataset size: {len(dataset)}")

    # Проверяем несколько примеров
    for i in range(min(3, len(dataset))):
        item = dataset[i]
        print(f"Sample {i}:")
        print(f"  Language: {item['language']}")
        print(f"  Input shape: {item['input_ids'].shape}")
        print(f"  Text preview: {tokenizer.decode(item['input_ids'][:20].tolist())}")
        print()

    # Тест DataLoader
    data_loader = MultilingualDataLoader(config, tokenizer)
    train_loader = data_loader.get_data_loader("train", batch_size=2)

    batch = next(iter(train_loader))
    print("Batch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  labels: {batch['labels'].shape}")

    print("Data loading test completed!")
