"""
Мультиязычная система токенизации для PrometheusGPT Mini
Использует SentencePiece с BPE для создания единого словаря 32k токенов
"""

import os
import sentencepiece as spm
from typing import List, Dict, Optional, Union
from config import ModelConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultilingualTokenizer:
    """
    Мультиязычный токенизатор на базе SentencePiece
    Поддерживает 5 языков: en, ru, es, fr, de
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_path = os.path.join(config.model_dir, "tokenizer.model")
        self.vocab_path = os.path.join(config.model_dir, "tokenizer.vocab")

        # Специальные токены
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
            "<mask>": 4,
        }

        # Языковые маркеры
        self.lang_tokens = {}
        for i, lang in enumerate(config.languages):
            self.lang_tokens[f"<{lang}>"] = len(self.special_tokens) + i

        self.sp_model = None
        self._load_or_create_tokenizer()

    def _load_or_create_tokenizer(self):
        """Загрузка существующего токенизатора или создание нового"""
        if os.path.exists(self.model_path):
            logger.info(f"Loading existing tokenizer from {self.model_path}")
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.load(self.model_path)
        else:
            logger.info("Creating new multilingual tokenizer")
            self._create_tokenizer()

    def _create_tokenizer(self):
        """Создание нового мультиязычного токенизатора"""
        # Создаем временные файлы с обучающими данными
        training_files = self._prepare_training_data()

        # Настройки SentencePiece
        spm_args = [
            f"--input={','.join(training_files)}",
            f"--model_prefix={os.path.join(self.config.model_dir, 'tokenizer')}",
            f"--vocab_size={self.config.vocab_size}",
            "--model_type=bpe",
            "--character_coverage=1.0",  # Полное покрытие символов для мультиязычности
            "--byte_fallback=true",     # Fallback для неизвестных символов
            "--add_dummy_prefix=false",
            "--normalization_rule_name=nmt_nfkc",  # Нормализация для разных языков
            "--remove_extra_whitespaces=true",
            "--split_digits=true",
            "--user_defined_symbols=<pad>,<unk>,<bos>,<eos>,<mask>",
        ]

        # Добавляем языковые маркеры
        for lang in self.config.languages:
            spm_args.append(f"--user_defined_symbols=<{lang}>")

        # Обучаем токенизатор
        spm.SentencePieceTrainer.train(" ".join(spm_args))

        # Загружаем обученную модель
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(self.model_path)

        # Очищаем временные файлы
        for file in training_files:
            if os.path.exists(file):
                os.remove(file)

        logger.info(f"Tokenizer created with vocab size: {self.sp_model.get_piece_size()}")

    def _prepare_training_data(self) -> List[str]:
        """Подготовка обучающих данных для токенизатора"""
        training_files = []

        # Создаем директорию для данных если не существует
        os.makedirs(self.config.data_dir, exist_ok=True)

        # Для каждого языка создаем файл с примерами текста
        lang_samples = {
            'en': [
                "Hello world! This is a sample English text for tokenizer training.",
                "Machine learning and artificial intelligence are transforming our world.",
                "Natural language processing enables computers to understand human language.",
                "Large language models can generate coherent and contextually relevant text.",
                "Tokenization is the process of breaking text into smaller units called tokens.",
            ],
            'ru': [
                "Привет мир! Это пример русского текста для обучения токенизатора.",
                "Машинное обучение и искусственный интеллект меняют наш мир.",
                "Обработка естественного языка позволяет компьютерам понимать человеческий язык.",
                "Большие языковые модели могут генерировать связный и контекстуально релевантный текст.",
                "Токенизация - это процесс разбиения текста на более мелкие единицы, называемые токенами.",
            ],
            'es': [
                "¡Hola mundo! Este es un ejemplo de texto en español para el entrenamiento del tokenizador.",
                "El aprendizaje automático y la inteligencia artificial están transformando nuestro mundo.",
                "El procesamiento del lenguaje natural permite a las computadoras entender el lenguaje humano.",
                "Los grandes modelos de lenguaje pueden generar texto coherente y contextualmente relevante.",
                "La tokenización es el proceso de dividir el texto en unidades más pequeñas llamadas tokens.",
            ],
            'fr': [
                "Bonjour le monde! Ceci est un exemple de texte français pour l'entraînement du tokeniseur.",
                "L'apprentissage automatique et l'intelligence artificielle transforment notre monde.",
                "Le traitement du langage naturel permet aux ordinateurs de comprendre le langage humain.",
                "Les grands modèles de langage peuvent générer du texte cohérent et contextuellement pertinent.",
                "La tokenisation est le processus de division du texte en unités plus petites appelées tokens.",
            ],
            'de': [
                "Hallo Welt! Dies ist ein Beispiel deutscher Text für das Training des Tokenizers.",
                "Maschinelles Lernen und künstliche Intelligenz verändern unsere Welt.",
                "Die Verarbeitung natürlicher Sprache ermöglicht es Computern, menschliche Sprache zu verstehen.",
                "Große Sprachmodelle können zusammenhängenden und kontextuell relevanten Text generieren.",
                "Tokenisierung ist der Prozess der Aufteilung von Text in kleinere Einheiten namens Tokens.",
            ]
        }

        for lang, samples in lang_samples.items():
            file_path = os.path.join(self.config.data_dir, f"tokenizer_train_{lang}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(sample + '\n')
            training_files.append(file_path)

        return training_files

    def encode(self, text: str, lang: Optional[str] = None,
               add_special_tokens: bool = True) -> List[int]:
        """
        Кодирование текста в токены

        Args:
            text: Входной текст
            lang: Код языка (опционально)
            add_special_tokens: Добавлять ли BOS/EOS токены

        Returns:
            Список токенов
        """
        if lang and lang in self.config.languages:
            # Добавляем языковой маркер в начало
            text = f"<{lang}> {text}"

        # Токенизация
        tokens = self.sp_model.encode(text, out_type=int)

        if add_special_tokens:
            tokens = [self.special_tokens["<bos>"]] + tokens + [self.special_tokens["<eos>"]]

        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """
        Декодирование токенов в текст

        Args:
            tokens: Список токенов
            skip_special_tokens: Пропускать ли специальные токены

        Returns:
            Декодированный текст
        """
        if skip_special_tokens:
            # Удаляем специальные токены
            special_token_ids = set(self.special_tokens.values())
            tokens = [t for t in tokens if t not in special_token_ids]

        return self.sp_model.decode(tokens)

    def encode_batch(self, texts: List[str], langs: Optional[List[str]] = None,
                     add_special_tokens: bool = True, max_length: Optional[int] = None,
                     padding: bool = True, truncation: bool = True) -> Dict[str, List[List[int]]]:
        """
        Пакетное кодирование текстов

        Args:
            texts: Список текстов
            langs: Список кодов языков (опционально)
            add_special_tokens: Добавлять ли BOS/EOS
            max_length: Максимальная длина
            padding: Дополнять ли до max_length
            truncation: Обрезать ли до max_length

        Returns:
            Словарь с токенами и маской внимания
        """
        batch_tokens = []

        for i, text in enumerate(texts):
            lang = langs[i] if langs else None
            tokens = self.encode(text, lang, add_special_tokens)

            if truncation and max_length and len(tokens) > max_length:
                tokens = tokens[:max_length]

            batch_tokens.append(tokens)

        # Padding
        if padding and max_length:
            pad_token = self.special_tokens["<pad>"]
            for tokens in batch_tokens:
                while len(tokens) < max_length:
                    tokens.append(pad_token)

        # Создание маски внимания
        attention_masks = []
        for tokens in batch_tokens:
            mask = [1 if token != self.special_tokens["<pad>"] else 0 for token in tokens]
            attention_masks.append(mask)

        return {
            "input_ids": batch_tokens,
            "attention_mask": attention_masks
        }

    def get_vocab_size(self) -> int:
        """Получение размера словаря"""
        return self.sp_model.get_piece_size()

    def get_special_tokens_map(self) -> Dict[str, int]:
        """Получение карты специальных токенов"""
        return {**self.special_tokens, **self.lang_tokens}

    def save_vocab(self, path: str):
        """Сохранение словаря в файл"""
        vocab = []
        for i in range(self.get_vocab_size()):
            piece = self.sp_model.id_to_piece(i)
            vocab.append(f"{piece}\t{i}")

        with open(path, 'w', encoding='utf-8') as f:
            f.write("\n".join(vocab))

    def mask_tokens(self, input_ids: List[int], mask_prob: float = 0.15) -> tuple:
        """
        Маскировка токенов для MLM задачи

        Args:
            input_ids: Список токенов
            mask_prob: Вероятность маскировки

        Returns:
            Кортеж (masked_input_ids, labels)
        """
        import random

        masked_input_ids = input_ids.copy()
        labels = [-100] * len(input_ids)  # -100 для игнорирования в loss

        for i, token_id in enumerate(input_ids):
            if token_id in self.special_tokens.values():
                continue  # Не маскируем специальные токены

            if random.random() < mask_prob:
                # 80% - маскируем
                if random.random() < 0.8:
                    masked_input_ids[i] = self.special_tokens["<mask>"]
                # 10% - случайный токен
                elif random.random() < 0.5:
                    masked_input_ids[i] = random.randint(5, self.get_vocab_size() - 1)
                # 10% - оставляем как есть
                labels[i] = token_id
            else:
                labels[i] = token_id

        return masked_input_ids, labels


if __name__ == "__main__":
    # Тест токенизатора
    config = ModelConfig()
    tokenizer = MultilingualTokenizer(config)

    # Тест кодирования/декодирования
    test_texts = [
        "Hello world!",
        "Привет мир!",
        "¡Hola mundo!",
        "Bonjour le monde!",
        "Hallo Welt!"
    ]

    test_langs = ['en', 'ru', 'es', 'fr', 'de']

    print("Testing multilingual tokenization:")
    for text, lang in zip(test_texts, test_langs):
        tokens = tokenizer.encode(text, lang)
        decoded = tokenizer.decode(tokens)
        print(f"{lang}: {text} -> {len(tokens)} tokens -> {decoded}")

    # Тест пакетной обработки
    batch = tokenizer.encode_batch(test_texts, test_langs, max_length=20)
    print(f"\nBatch encoding result:")
    print(f"Input IDs shape: {[len(ids) for ids in batch['input_ids']]}")
    print(f"Attention masks shape: {[len(mask) for mask in batch['attention_mask']]}")

    # Сохранение словаря
    vocab_path = os.path.join(config.model_dir, "vocab.txt")
    tokenizer.save_vocab(vocab_path)
    print(f"\nVocabulary saved to {vocab_path}")

    print("Tokenizer test completed!")
