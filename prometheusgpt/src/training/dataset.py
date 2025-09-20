"""
PrometheusGPT Mini - Dataset Preparation
Author: MagistrTheOne, Krasnodar, 2025

Подготовка датасетов для обучения модели с авторством.
"""

import os
import json
import torch
from typing import List, Dict, Tuple, Optional, Iterator
from torch.utils.data import Dataset, DataLoader
import hashlib

from src.tokenizer.bpe_tokenizer import BPETokenizer


class TextDataset(Dataset):
    """Датасет для текстовых данных"""

    def __init__(self, texts: List[str], tokenizer: BPETokenizer, max_length: int = 256):
        """
        Args:
            texts: список текстов
            tokenizer: BPE токенизатор
            max_length: максимальная длина последовательности
        """

        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Токенизируем все тексты заранее
        self.tokenized_texts = []
        for text in texts:
            # Добавляем информацию об авторе
            author_text = self.tokenizer.prepare_text_with_author(text)
            tokens = self.tokenizer.encode(author_text)
            self.tokenized_texts.append(tokens)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Получить элемент датасета"""

        tokens = self.tokenized_texts[idx]

        # Обрезаем или дополняем до max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            # Дополняем паддингом
            tokens = tokens + [self.tokenizer.special_tokens['<pad>']] * (self.max_length - len(tokens))

        # Создаем входные и целевые последовательности
        input_tokens = torch.tensor(tokens[:-1], dtype=torch.long)  # все кроме последнего
        target_tokens = torch.tensor(tokens[1:], dtype=torch.long)   # все кроме первого

        return {
            'input_ids': input_tokens,
            'target_ids': target_tokens,
            'attention_mask': (input_tokens != self.tokenizer.special_tokens['<pad>']).long()
        }


class TranslationDataset(Dataset):
    """Датасет для параллельных текстов (RU/EN)"""

    def __init__(self, src_texts: List[str], tgt_texts: List[str],
                 src_tokenizer: BPETokenizer, tgt_tokenizer: BPETokenizer,
                 max_length: int = 256):
        """
        Args:
            src_texts: исходные тексты
            tgt_texts: целевые тексты
            src_tokenizer: токенизатор для исходного языка
            tgt_tokenizer: токенизатор для целевого языка
            max_length: максимальная длина последовательности
        """

        assert len(src_texts) == len(tgt_texts), "Количество текстов должно совпадать"

        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.src_texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Получить элемент датасета"""

        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        # Добавляем информацию об авторе к обоим текстам
        src_author_text = self.src_tokenizer.prepare_text_with_author(src_text)
        tgt_author_text = self.tgt_tokenizer.prepare_text_with_author(tgt_text)

        # Токенизируем
        src_tokens = self.src_tokenizer.encode(src_author_text)
        tgt_tokens = self.tgt_tokenizer.encode(tgt_author_text)

        # Обрезаем или дополняем
        src_tokens = self._pad_or_truncate(src_tokens, self.max_length)
        tgt_tokens = self._pad_or_truncate(tgt_tokens, self.max_length)

        # Создаем входные и целевые последовательности
        src_input = torch.tensor(src_tokens[:-1], dtype=torch.long)
        src_target = torch.tensor(src_tokens[1:], dtype=torch.long)

        tgt_input = torch.tensor(tgt_tokens[:-1], dtype=torch.long)
        tgt_target = torch.tensor(tgt_tokens[1:], dtype=torch.long)

        return {
            'src_input_ids': src_input,
            'src_target_ids': src_target,
            'tgt_input_ids': tgt_input,
            'tgt_target_ids': tgt_target,
            'src_attention_mask': (src_input != self.src_tokenizer.special_tokens['<pad>']).long(),
            'tgt_attention_mask': (tgt_input != self.tgt_tokenizer.special_tokens['<pad>']).long()
        }

    def _pad_or_truncate(self, tokens: List[int], max_length: int) -> List[int]:
        """Обрезать или дополнить последовательность"""
        if len(tokens) > max_length:
            return tokens[:max_length]
        else:
            pad_id = self.src_tokenizer.special_tokens['<pad>']
            return tokens + [pad_id] * (max_length - len(tokens))


def create_demo_dataset() -> Tuple[List[str], List[str], List[str], List[str]]:
    """Создать демо датасет с параллельными текстами RU/EN"""

    # Русские тексты
    ru_texts = [
        "Привет, мир! Это демонстрация работы модели.",
        "Меня зовут MagistrTheOne, я из Краснодара.",
        "Сегодня 2025 год, и мы создаем собственную LLM модель.",
        "Transformer - это мощная архитектура для обработки текстов.",
        "Искусственный интеллект помогает решать сложные задачи.",
        "Программирование - это творческий процесс создания решений.",
        "Машинное обучение позволяет компьютерам учиться на данных.",
        "Нейронные сети моделируют работу человеческого мозга.",
        "Естественная обработка языка - это сложная область ИИ.",
        "Глубокое обучение использует многослойные нейронные сети.",
        "Большие языковые модели требуют огромных вычислительных ресурсов.",
        "Трансформеры изменили подход к обработке последовательностей.",
        "Внимание - это механизм фокусировки на важных частях данных.",
        "Генеративные модели могут создавать новый контент.",
        "Предобученные модели можно дообучать для конкретных задач."
    ]

    # Английские переводы
    en_texts = [
        "Hello, world! This is a model demonstration.",
        "My name is MagistrTheOne, I'm from Krasnodar.",
        "Today is 2025, and we're building our own LLM model.",
        "Transformer is a powerful architecture for text processing.",
        "Artificial intelligence helps solve complex problems.",
        "Programming is a creative process of creating solutions.",
        "Machine learning allows computers to learn from data.",
        "Neural networks model the work of the human brain.",
        "Natural language processing is a complex area of AI.",
        "Deep learning uses multi-layer neural networks.",
        "Large language models require enormous computational resources.",
        "Transformers changed the approach to sequence processing.",
        "Attention is a mechanism for focusing on important parts of data.",
        "Generative models can create new content.",
        "Pre-trained models can be fine-tuned for specific tasks."
    ]

    # Дополнительные тексты для обучения токенизатора
    extra_ru = [
        "Функция потерь измеряет разницу между предсказаниями и реальностью.",
        "Обратное распространение ошибки обновляет веса нейронной сети.",
        "Оптимизатор градиентного спуска минимизирует функцию потерь.",
        "Регуляризация предотвращает переобучение модели.",
        "Кросс-энтропия часто используется как функция потерь для классификации.",
        "Градиентный клиппинг предотвращает взрыв градиентов.",
        "Пакетная нормализация стабилизирует обучение.",
        "Dropout случайно отключает нейроны для регуляризации.",
        "Learning rate определяет размер шага оптимизации.",
        "Эпоха - это один полный проход через датасет.",
        "Батч - это подмножество данных для одного шага обучения.",
        "Итерация - это один шаг обновления весов.",
        "Валидация проверяет качество модели на новых данных.",
        "Переобучение происходит когда модель слишком хорошо запоминает данные.",
        "Недообучение когда модель не может выучить даже обучающие данные."
    ]

    extra_en = [
        "Loss function measures the difference between predictions and reality.",
        "Backpropagation updates the weights of the neural network.",
        "Gradient descent optimizer minimizes the loss function.",
        "Regularization prevents model overfitting.",
        "Cross-entropy is often used as a loss function for classification.",
        "Gradient clipping prevents gradient explosion.",
        "Batch normalization stabilizes training.",
        "Dropout randomly disables neurons for regularization.",
        "Learning rate determines the step size of optimization.",
        "Epoch is one complete pass through the dataset.",
        "Batch is a subset of data for one training step.",
        "Iteration is one step of weight update.",
        "Validation checks model quality on new data.",
        "Overfitting occurs when model memorizes data too well.",
        "Underfitting when model cannot learn even training data."
    ]

    return ru_texts + extra_ru, en_texts + extra_en, ru_texts, en_texts


def create_dataloader(dataset: Dataset, batch_size: int = 8,
                     shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    """Создать DataLoader для датасета"""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # для ускорения передачи на GPU
        collate_fn=collate_batch
    )


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Объединить батч элементов датасета"""

    # Находим максимальную длину в батче
    max_src_len = max(item['src_input_ids'].size(0) for item in batch)
    max_tgt_len = max(item['tgt_input_ids'].size(0) for item in batch)

    # Инициализируем тензоры для батча
    batch_size = len(batch)
    src_input_ids = torch.zeros(batch_size, max_src_len, dtype=torch.long)
    src_target_ids = torch.zeros(batch_size, max_src_len, dtype=torch.long)
    src_attention_mask = torch.zeros(batch_size, max_src_len, dtype=torch.long)

    tgt_input_ids = torch.zeros(batch_size, max_tgt_len, dtype=torch.long)
    tgt_target_ids = torch.zeros(batch_size, max_tgt_len, dtype=torch.long)
    tgt_attention_mask = torch.zeros(batch_size, max_tgt_len, dtype=torch.long)

    # Заполняем тензоры
    for i, item in enumerate(batch):
        src_len = item['src_input_ids'].size(0)
        tgt_len = item['tgt_input_ids'].size(0)

        src_input_ids[i, :src_len] = item['src_input_ids']
        src_target_ids[i, :src_len] = item['src_target_ids']
        src_attention_mask[i, :src_len] = item['src_attention_mask']

        tgt_input_ids[i, :tgt_len] = item['tgt_input_ids']
        tgt_target_ids[i, :tgt_len] = item['tgt_target_ids']
        tgt_attention_mask[i, :tgt_len] = item['tgt_attention_mask']

    return {
        'src_input_ids': src_input_ids,
        'src_target_ids': src_target_ids,
        'src_attention_mask': src_attention_mask,
        'tgt_input_ids': tgt_input_ids,
        'tgt_target_ids': tgt_target_ids,
        'tgt_attention_mask': tgt_attention_mask
    }


def save_dataset_info(dataset_path: str, info: Dict):
    """Сохранить информацию о датасете"""
    info_path = os.path.join(dataset_path, 'dataset_info.json')

    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def load_dataset_info(dataset_path: str) -> Dict:
    """Загрузить информацию о датасете"""
    info_path = os.path.join(dataset_path, 'dataset_info.json')

    if not os.path.exists(info_path):
        return {}

    with open(info_path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    # Демонстрация работы с датасетом
    print("=== PrometheusGPT Mini Dataset Demo ===")
    print("Author: MagistrTheOne, Krasnodar, 2025")

    # Создаем демо датасет
    ru_texts, en_texts, train_ru, train_en = create_demo_dataset()

    print(f"\nDataset sizes:")
    print(f"Full RU: {len(ru_texts)} texts")
    print(f"Full EN: {len(en_texts)} texts")
    print(f"Train RU: {len(train_ru)} texts")
    print(f"Train EN: {len(train_en)} texts")

    # Создаем токенизаторы
    print("\nCreating tokenizers...")
    ru_tokenizer = BPETokenizer()
    en_tokenizer = BPETokenizer()

    # Обучаем токенизаторы на соответствующих текстах
    ru_tokenizer.train(ru_texts, "ru_tokenizer", vocab_size=1000)
    en_tokenizer.train(en_texts, "en_tokenizer", vocab_size=1000)

    # Создаем датасет
    print("\nCreating dataset...")
    dataset = TranslationDataset(train_ru, train_en, ru_tokenizer, en_tokenizer)

    print(f"Dataset size: {len(dataset)}")

    # Пример элемента датасета
    sample = dataset[0]
    print(f"\nSample shapes:")
    for key, value in sample.items():
        print(f"  {key}: {value.shape}")

    # Создаем DataLoader
    print(f"\nCreating DataLoader...")
    dataloader = create_dataloader(dataset, batch_size=4)

    # Пример батча
    batch = next(iter(dataloader))
    print(f"\nBatch shapes:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")

    print(f"\n✅ Dataset demo completed!")
